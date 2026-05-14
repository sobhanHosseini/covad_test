"""Novelty detection experiment — Protocol 2.

For each holdout defect (crack, hole, cut, print):
  1. Train on holdout CSV (all images EXCEPT holdout defect normals filtered
     to train/good, plus all known defect images)
  2. Evaluate on holdout defect test images → Level 1/2/3 distribution
  3. Run CONCIL update with holdout defect concept labels (new concept heads)
  4. Re-evaluate → defect should now be Level 2 (explained)
  5. Verify zero-forgetting: known concept weights must not change

Holdout CSVs already exist:
    annotations/hazelnut/hazelnut_holdout_{defect}.csv
Each CSV has all image rows but WITHOUT the holdout defect's tier-3 concept
columns (6 columns for crack/hole, 2 for cut, 4 for print).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from features.dinov2_extractor import DINOv2Extractor
from features.patchcore_memory import PatchCoreMemory
from models.concept_heads       import ConceptHeads
from models.linear_head         import LinearAnomalyHead
from solvers.concil             import ConcilSolver

_META_COLS = frozenset(
    ["image_path", "label_index", "mask_path", "anomaly_type", "split", "view"]
)
_BATCH = 16


# ── result dataclasses ────────────────────────────────────────────────────────

@dataclass
class LevelCounts:
    n_level1: int       # missed  (s_novel < τ)
    n_level2: int       # explained  (s_novel ≥ τ AND max(c) > θ)
    n_level3: int       # novel       (s_novel ≥ τ AND max(c) ≤ θ)
    total:    int
    avg_s_novel: float
    avg_max_c:   float


@dataclass
class HoldoutResult:
    holdout_defect:    str
    n_known_concepts:  int       # concept heads before update
    n_new_concepts:    int       # concept heads added at update
    before:            LevelCounts
    after:             LevelCounts
    c_auc_new:         float     # mean AUROC on new concept columns (after update)
    weight_delta_max:  float     # max |W_after[:K_old] - W_before| ≈ 0 if CONCIL correct


# ── leakage check ─────────────────────────────────────────────────────────────

def leakage_check(
    holdout_defect:   str,
    holdout_csv_path: str,
    full_csv_path:    str,
    mvtec_root:       str,
    category:         str = "hazelnut",
) -> bool:
    """Verify three leakage conditions for the holdout experiment.

    1. No test/good/ paths appear in training normals
       (held-out normals stay held-out — we filter to train/good/).
    2. No test/{holdout_defect}/ paths appear in training rows
       (holdout defect images are never seen during training).
    3. Holdout-defect-specific concept columns are absent from holdout CSV
       (concept leakage prevented — model can't see defect-specific labels).
    """
    holdout_df = pd.read_csv(holdout_csv_path)
    full_df    = pd.read_csv(full_csv_path)
    mvtec_cat  = Path(mvtec_root) / category

    holdout_cols = set(c for c in holdout_df.columns if c not in _META_COLS)
    full_cols    = set(c for c in full_df.columns    if c not in _META_COLS)
    missing_cols = full_cols - holdout_cols

    # Derive training rows (what _train_model uses)
    is_train_normal = (
        (holdout_df["label_index"] == 0) &
        holdout_df["image_path"].str.contains("train/good", regex=False)
    )
    is_known_defect = (
        (holdout_df["label_index"] == 1) &
        (holdout_df["anomaly_type"] != holdout_defect)
    )
    train_df  = holdout_df[is_train_normal | is_known_defect]
    train_paths = set(train_df["image_path"].tolist())

    violations: list[str] = []

    # Check 1: no test/good/ in training normals
    test_good_in_train = [
        p for p in train_paths if "test/good" in p
    ]
    if test_good_in_train:
        violations.append(
            f"  CHECK 1 FAIL: {len(test_good_in_train)} test/good paths in training normals"
        )

    # Check 2: no test/{holdout_defect}/ in training rows
    test_defect_paths = {
        str(p) for p in (mvtec_cat / "test" / holdout_defect).glob("*.png")
    }
    leaked_defect = train_paths & test_defect_paths
    if leaked_defect:
        violations.append(
            f"  CHECK 2 FAIL: {len(leaked_defect)} test/{holdout_defect} paths in training"
        )

    # Check 3: holdout concept columns absent from holdout CSV
    # (checked indirectly — missing_cols should be non-empty)
    if not missing_cols:
        violations.append(
            "  CHECK 3 FAIL: holdout CSV has ALL concept columns — no concept leakage prevention"
        )

    print(f"\nLeakage check — holdout_{holdout_defect}:")
    print(f"  Training rows     : {len(train_df)}  "
          f"({is_train_normal.sum()} normal, {is_known_defect.sum()} defect)")
    print(f"  Absent columns    : {sorted(missing_cols)}")
    print(f"  Test holdout imgs : {len(test_defect_paths)}")

    if violations:
        for v in violations:
            print(v)
        return False

    print("  Leakage check PASSED ✓")
    return True


# ── main experiment class ─────────────────────────────────────────────────────

class NoveltyExperiment:
    """Full holdout novelty detection protocol."""

    HOLDOUT_DEFECTS = ["crack", "hole", "cut", "print"]

    def __init__(self, config: dict):
        self.config         = config
        self._category      = config.get("category", "hazelnut")
        self._mvtec_root    = config["mvtec_root"]
        self._ann_dir       = Path(config["annotations_dir"])
        self._full_csv      = self._ann_dir / f"{self._category}.csv"
        self._device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.lambda_c     = float(config.get("lambda_concept",  1e-4))
        self.lambda_a     = float(config.get("lambda_anomaly",  1e-4))
        self.coreset_size = int(config.get("coreset_size",      10_000))
        self.tau_pct      = float(config.get("tau_percentile",  95.0))
        self.theta        = float(config.get("theta_concept",   0.5))

        print(f"  Loading DINOv2 extractor (shared across all holdouts) ...")
        self.extractor = DINOv2Extractor(device=self._device)

    # ── feature extraction ────────────────────────────────────────────────────

    @torch.no_grad()
    def _extract(self, paths: list[str | Path], desc: str = "") -> tuple[torch.Tensor, torch.Tensor]:
        patch_all, pool_all = [], []
        imgs = [Image.open(p).convert("RGB") for p in paths]
        for i in tqdm(range(0, len(imgs), _BATCH), desc=f"  {desc}", leave=False):
            p, z = self.extractor.extract_both(imgs[i : i + _BATCH])
            patch_all.append(p.cpu()); pool_all.append(z.cpu())
        return torch.cat(patch_all), torch.cat(pool_all)

    # ── level counting ────────────────────────────────────────────────────────

    @torch.no_grad()
    def _count_levels(
        self,
        holdout_defect: str,
        concept_heads:  ConceptHeads,
        anomaly_head:   LinearAnomalyHead,
        memory:         PatchCoreMemory,
        tau:            float,
    ) -> LevelCounts:
        """Run inference on all test/{holdout_defect}/ images and count levels."""
        mvtec_cat = Path(self._mvtec_root) / self._category
        paths = sorted((mvtec_cat / "test" / holdout_defect).glob("*.png"))
        imgs  = [Image.open(p).convert("RGB") for p in paths]

        s_novels, max_cs = [], []
        n1 = n2 = n3 = 0

        for i in range(0, len(imgs), _BATCH):
            batch_imgs = imgs[i : i + _BATCH]
            p, z = self.extractor.extract_both(batch_imgs)
            s, _  = memory.score(p)               # (b,)
            c     = concept_heads(z)              # (b, K)
            for j in range(len(batch_imgs)):
                sv   = float(s[j].cpu())
                mc   = float(c[j].cpu().max())
                s_novels.append(sv); max_cs.append(mc)
                if sv < tau:
                    n1 += 1
                elif mc > self.theta:
                    n2 += 1
                else:
                    n3 += 1

        return LevelCounts(
            n_level1    = n1,
            n_level2    = n2,
            n_level3    = n3,
            total       = len(imgs),
            avg_s_novel = float(np.mean(s_novels)),
            avg_max_c   = float(np.mean(max_cs)),
        )

    # ── main holdout protocol ─────────────────────────────────────────────────

    def run_holdout(self, holdout_defect: str) -> HoldoutResult:
        holdout_csv = self._ann_dir / f"{self._category}_holdout_{holdout_defect}.csv"
        holdout_df  = pd.read_csv(holdout_csv)
        full_df     = pd.read_csv(self._full_csv)

        # ── concept column bookkeeping ────────────────────────────────────────
        holdout_cnames = [c for c in holdout_df.columns if c not in _META_COLS]
        full_cnames    = [c for c in full_df.columns    if c not in _META_COLS]
        new_cnames     = [c for c in full_cnames if c not in holdout_cnames]

        print(f"\n{'='*60}")
        print(f"HOLDOUT EXPERIMENT — {holdout_defect.upper()}")
        print(f"  Known concepts  : {len(holdout_cnames)}")
        print(f"  New concepts    : {new_cnames}")
        print(f"{'='*60}")

        # ── Step 1: training data ─────────────────────────────────────────────
        is_train_normal = (
            (holdout_df["label_index"] == 0) &
            holdout_df["image_path"].str.contains("train/good", regex=False)
        )
        is_known_defect = (
            (holdout_df["label_index"] == 1) &
            (holdout_df["anomaly_type"] != holdout_defect)
        )
        train_df = holdout_df[is_train_normal | is_known_defect].reset_index(drop=True)
        print(f"  Training: {(train_df['label_index']==0).sum()} normal + "
              f"{(train_df['label_index']==1).sum()} known-defect images")

        train_paths = train_df["image_path"].tolist()

        # ── Step 2: feature extraction ────────────────────────────────────────
        print("  Extracting training features ...")
        patch_tokens, pooled_z = self._extract(train_paths, "train feats")

        # ── Step 3: memory bank (from train/good only) ────────────────────────
        memory = PatchCoreMemory(coreset_size=self.coreset_size, device=self._device)
        normal_mask = torch.tensor(train_df["label_index"].values == 0)
        memory.build(patch_tokens[normal_mask])

        # Compute τ from normal s_novel
        s_norm, _ = memory.score(patch_tokens[normal_mask])
        tau = float(np.percentile(s_norm.cpu().numpy(), self.tau_pct))
        print(f"  Memory: {memory.coreset_size:,} patches  τ={tau:.4f}")

        # ── Step 4: CONCIL training on known data ─────────────────────────────
        solver        = ConcilSolver(lambda_concept=self.lambda_c, lambda_anomaly=self.lambda_a)
        concept_heads = ConceptHeads(holdout_cnames)
        anomaly_head  = LinearAnomalyHead(n_concepts=len(holdout_cnames))

        Z = pooled_z.float()
        C = torch.tensor(train_df[holdout_cnames].values.astype("float32"))
        y = torch.tensor(train_df["label_index"].values.astype("float32"))

        W_c, b_c = solver.update_concept_heads(Z, C, new_concept_cols=len(holdout_cnames))
        concept_heads.set_weights(W_c, b_c)

        with torch.no_grad():
            C_act = concept_heads(pooled_z.float())
        W_a, b_a = solver.update_anomaly_head(C_act, y)
        anomaly_head.set_weights(W_a, b_a)
        print(f"  CONCIL training done  K={solver.K}")

        # Snapshot weights before update (for zero-forgetting check)
        W_before = concept_heads.linear.weight.data.clone().numpy()

        # ── Step 5: evaluate BEFORE update ───────────────────────────────────
        print(f"  Evaluating {holdout_defect} test images BEFORE update ...")
        counts_before = self._count_levels(
            holdout_defect, concept_heads, anomaly_head, memory, tau
        )
        print(f"    Level 1 (missed)  : {counts_before.n_level1}")
        print(f"    Level 2 (explained): {counts_before.n_level2}")
        print(f"    Level 3 (novel)   : {counts_before.n_level3}")
        print(f"    avg s_novel={counts_before.avg_s_novel:.3f}  avg max(c)={counts_before.avg_max_c:.3f}")

        # ── Step 6: CONCIL update with holdout defect concepts ────────────────
        # Load holdout defect rows from FULL CSV (has all 42 concept columns)
        new_def_df = full_df[full_df["anomaly_type"] == holdout_defect].reset_index(drop=True)
        print(f"\n  CONCIL update — adding {holdout_defect} ({len(new_def_df)} images, "
              f"{len(new_cnames)} new concept columns) ...")

        # Expand heads for new concept columns
        if new_cnames:
            solver.expand_for_new_concepts(len(new_cnames))
            concept_heads.add_concepts(new_cnames)
            anomaly_head.expand(len(holdout_cnames) + len(new_cnames))

        # Extract features for new defect images
        new_paths = new_def_df["image_path"].tolist()
        _, new_z  = self._extract(new_paths, "new defect feats")

        Z_new = new_z.float()
        C_new = torch.tensor(
            new_def_df[full_cnames].values.astype("float32")
        )
        y_new = torch.tensor(new_def_df["label_index"].values.astype("float32"))

        W_c2, b_c2 = solver.update_concept_heads(Z_new, C_new)
        concept_heads.set_weights(W_c2, b_c2)

        with torch.no_grad():
            C_act2 = concept_heads(new_z.float())
        W_a2, b_a2 = solver.update_anomaly_head(
            C_act2, y_new, vocabulary_expanded=bool(new_cnames)
        )
        anomaly_head.set_weights(W_a2, b_a2)
        print(f"  Update done  K={solver.K}")

        # ── Step 7: zero-forgetting check ─────────────────────────────────────
        W_after    = concept_heads.linear.weight.data[:len(holdout_cnames)].numpy()
        weight_delta = float(np.abs(W_after - W_before).max())
        print(f"  Zero-forgetting check: max |ΔW_known| = {weight_delta:.2e}  "
              f"({'✓' if weight_delta < 1e-4 else '✗ WARNING'})")

        # ── Step 8: evaluate AFTER update ─────────────────────────────────────
        print(f"  Evaluating {holdout_defect} test images AFTER update ...")
        counts_after = self._count_levels(
            holdout_defect, concept_heads, anomaly_head, memory, tau
        )
        print(f"    Level 1 (missed)  : {counts_after.n_level1}")
        print(f"    Level 2 (explained): {counts_after.n_level2}")
        print(f"    Level 3 (novel)   : {counts_after.n_level3}")
        print(f"    avg s_novel={counts_after.avg_s_novel:.3f}  avg max(c)={counts_after.avg_max_c:.3f}")

        # ── Step 9: C-AUC on new concept columns ──────────────────────────────
        c_auc_new = float("nan")
        if new_cnames:
            mvtec_cat    = Path(self._mvtec_root) / self._category
            test_paths   = sorted((mvtec_cat / "test" / holdout_defect).glob("*.png"))
            test_imgs    = [Image.open(p).convert("RGB") for p in test_paths]

            # Load test normal + holdout_defect labels from full CSV
            test_normal_df  = full_df[full_df["image_path"].str.contains("test/good", regex=False)]
            test_defect_df  = full_df[full_df["anomaly_type"] == holdout_defect]
            test_label_df   = pd.concat([test_normal_df, test_defect_df], ignore_index=True)

            eval_paths = (
                sorted(Path(self._mvtec_root, self._category, "test", "good").glob("*.png")) +
                test_paths
            )
            _, eval_z = self._extract(eval_paths, "C-AUC eval")
            with torch.no_grad():
                c_eval = concept_heads(eval_z.float()).cpu().numpy()

            # Only the new concept columns
            new_start = len(holdout_cnames)
            aucs = []
            for j, cname in enumerate(new_cnames):
                if cname not in test_label_df.columns:
                    continue
                gt = test_label_df[cname].values.astype(float)
                pred = c_eval[:, new_start + j]
                if len(np.unique(gt)) < 2:
                    continue
                try:
                    aucs.append(roc_auc_score(gt, pred))
                except ValueError:
                    pass
            c_auc_new = float(np.mean(aucs)) if aucs else float("nan")
            print(f"  C-AUC on new concepts: {c_auc_new:.4f}")

        return HoldoutResult(
            holdout_defect   = holdout_defect,
            n_known_concepts = len(holdout_cnames),
            n_new_concepts   = len(new_cnames),
            before           = counts_before,
            after            = counts_after,
            c_auc_new        = c_auc_new,
            weight_delta_max = weight_delta,
        )

    # ── run all holdouts ──────────────────────────────────────────────────────

    def run_all(self) -> list[HoldoutResult]:
        results = []
        for defect in self.HOLDOUT_DEFECTS:
            holdout_csv = self._ann_dir / f"{self._category}_holdout_{defect}.csv"
            if not holdout_csv.exists():
                print(f"  Skipping {defect} — holdout CSV not found")
                continue
            r = self.run_holdout(defect)
            results.append(r)
        return results

    # ── summary table ─────────────────────────────────────────────────────────

    @staticmethod
    def print_table(results: list[HoldoutResult]) -> None:
        W = 82
        print("\n" + "=" * W)
        print("NOVELTY DETECTION EXPERIMENT — CONCIL holdout results (hazelnut)")
        print("=" * W)
        hdr = (
            f"{'Defect':<8}  {'K_old':>5}  {'K_new':>5}  "
            f"{'Lvl1':>4}  {'Lvl2':>4}  {'Lvl3':>4}  "
            f"{'→ Lvl2':>6}  {'s_nov':>6}  "
            f"{'C-AUC':>6}  {'ΔW_max':>8}"
        )
        print(hdr)
        print("-" * W)
        for r in results:
            b, a = r.before, r.after
            print(
                f"{r.holdout_defect:<8}  {r.n_known_concepts:>5}  {r.n_new_concepts:>5}  "
                f"{b.n_level1:>4}  {b.n_level2:>4}  {b.n_level3:>4}  "
                f"{a.n_level2:>6}  {b.avg_s_novel:>6.3f}  "
                f"{r.c_auc_new:>6.3f}  {r.weight_delta_max:>8.2e}"
            )
        print("=" * W)
        print("Columns: Defect | K_old | K_new | Before(L1/L2/L3) | →L2(after) | avg s_novel | C-AUC(new) | ΔW")
        print("Level 3 before → Level 2 after = defect correctly explained after CONCIL update")
        print("ΔW ≈ 0 confirms CONCIL zero-forgetting on known concept heads")


# ── __main__ ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from pathlib import Path

    # Derive MVTec root from annotation CSV
    _sample_csv = "annotations/hazelnut/hazelnut.csv"
    _first_img  = pd.read_csv(_sample_csv, nrows=1)["image_path"].iloc[0]
    _mvtec_root = str(Path(_first_img).parents[3])

    config = {
        "mvtec_root":      _mvtec_root,
        "annotations_dir": "annotations/hazelnut",
        "category":        "hazelnut",
        "lambda_concept":  1e-4,
        "lambda_anomaly":  1e-4,
        "coreset_size":    10_000,
        "tau_percentile":  95.0,
        "theta_concept":   0.5,
    }

    # ── leakage checks ────────────────────────────────────────────────────────
    print("=" * 60)
    print("LEAKAGE CHECKS")
    print("=" * 60)
    exp = NoveltyExperiment(config)
    all_passed = True
    for defect in NoveltyExperiment.HOLDOUT_DEFECTS:
        holdout_csv = f"annotations/hazelnut/hazelnut_holdout_{defect}.csv"
        ok = leakage_check(
            defect, holdout_csv,
            "annotations/hazelnut/hazelnut.csv",
            _mvtec_root,
        )
        all_passed = all_passed and ok

    if not all_passed:
        print("\n⚠ Some leakage checks failed — investigate before trusting results.")
    else:
        print("\nAll leakage checks PASSED ✓")

    # ── run all holdout experiments ───────────────────────────────────────────
    print("\nRunning all holdout experiments ...")
    results = exp.run_all()
    NoveltyExperiment.print_table(results)
