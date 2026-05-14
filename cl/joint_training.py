"""Joint training upper bound for the CONCIL comparison table.

Trains CONCIL on all task data simultaneously (no sequential constraint)
and evaluates identically to CLTrainer.  This is the theoretical upper bound:
if sequential training perfectly matched joint training, CONCIL BWT would be
exactly 0.000.  In practice CONCIL approaches this bound; the naive baseline
does not.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from features.dinov2_extractor import DINOv2Extractor
from features.patchcore_memory import PatchCoreMemory
from models.concept_heads       import ConceptHeads
from models.linear_head         import LinearAnomalyHead
from solvers.concil             import ConcilSolver
from evaluators.evaluator_cl    import CLEvaluator, ContinualLog, TaskEvalResult

_META = frozenset(
    ["image_path", "label_index", "mask_path", "anomaly_type", "split", "view"]
)
_BATCH = 16


class JointTrainer:
    """Train CONCIL on all data at once — theoretical upper bound."""

    def __init__(self, config: dict):
        self.config         = config
        self._category      = config.get("category", "hazelnut")
        self._ann_dir       = Path(config["annotations_dir"])
        self._mvtec_root    = config["mvtec_root"]
        self._ckpt_dir      = Path(config["checkpoint_dir"])
        self._full_csv      = self._ann_dir / f"{self._category}.csv"
        self._device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._lambda_c  = float(config.get("lambda_concept",  1e-4))
        self._lambda_a  = float(config.get("lambda_anomaly",  1e-4))
        self._coreset   = int(config.get("coreset_size",      10_000))
        self._tau_pct   = float(config.get("tau_percentile",  95.0))

    # ── helpers ───────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _extract(self, paths: list[str], desc: str = "") -> tuple[torch.Tensor, torch.Tensor]:
        imgs = [Image.open(p).convert("RGB") for p in paths]
        p_all, z_all = [], []
        for i in tqdm(range(0, len(imgs), _BATCH), desc=f"  {desc}", leave=False):
            p, z = extractor.extract_both(imgs[i : i + _BATCH])
            p_all.append(p.cpu()); z_all.append(z.cpu())
        return torch.cat(p_all), torch.cat(z_all)

    # ── main run ──────────────────────────────────────────────────────────────

    def run(self, task_sequence_path: str) -> dict:
        """Train jointly on all task CSVs, evaluate, return metrics dict."""

        with open(task_sequence_path) as f:
            tasks = json.load(f)

        print(f"\n{'='*60}")
        print(f"JOINT TRAINING — {self._category.upper()}  (upper bound)")
        print(f"{'='*60}")
        print(f"  Device: {self._device}")

        # ── Step 1: load and concatenate all task CSVs ────────────────────────
        frames = []
        for t in tasks:
            df = pd.read_csv(t["csv_path"])
            # Keep same train/good normal filter as CLTrainer
            is_train_normal = (
                (df["label_index"] == 0) &
                df["image_path"].str.contains("/train/good/", regex=False)
            )
            is_defect = df["label_index"] == 1
            frames.append(df[is_train_normal | is_defect])

        all_df = pd.concat(frames, ignore_index=True).drop_duplicates("image_path")
        concept_names = [c for c in all_df.columns if c not in _META]

        n_normal   = int((all_df["label_index"] == 0).sum())
        n_defect   = int((all_df["label_index"] == 1).sum())
        n_total    = len(all_df)
        print(f"\n  Training data: {n_normal} normal + {n_defect} defect = {n_total} images")
        print(f"  Concepts     : {len(concept_names)}")

        # ── Step 2: feature extraction ────────────────────────────────────────
        print("\n  Extracting features ...")
        global extractor
        extractor = DINOv2Extractor(device=self._device)

        paths = all_df["image_path"].tolist()
        patch_tokens, pooled_z = self._extract(paths, "joint feats")
        print(f"  pooled_z {tuple(pooled_z.shape)}")

        # ── Step 3: PatchCore memory bank ─────────────────────────────────────
        normal_mask = torch.tensor(all_df["label_index"].values == 0)
        memory = PatchCoreMemory(coreset_size=self._coreset, device=self._device)
        memory.build(patch_tokens[normal_mask])
        s_norm, _ = memory.score(patch_tokens[normal_mask])
        tau = float(np.percentile(s_norm.cpu().numpy(), self._tau_pct))
        print(f"  Memory: {memory.coreset_size:,} patches  τ={tau:.4f}")

        # ── Step 4: single CONCIL solve ───────────────────────────────────────
        print("\n  Running single CONCIL solve ...")
        solver = ConcilSolver(lambda_concept=self._lambda_c, lambda_anomaly=self._lambda_a)
        concept_heads = ConceptHeads(concept_names, input_dim=DINOv2Extractor.POOLED_DIM)
        anomaly_head  = LinearAnomalyHead(n_concepts=len(concept_names))

        Z = pooled_z.float()
        C = torch.tensor(all_df[concept_names].values.astype("float32"))
        y = torch.tensor(all_df["label_index"].values.astype("float32"))

        W_c, b_c = solver.update_concept_heads(Z, C, new_concept_cols=len(concept_names))
        concept_heads.set_weights(W_c, b_c)

        with torch.no_grad():
            C_act = concept_heads(Z)
        W_a, b_a = solver.update_anomaly_head(C_act, y)
        anomaly_head.set_weights(W_a, b_a)
        print(f"  CONCIL done  K={solver.K}")

        # ── Step 5: evaluate per defect ───────────────────────────────────────
        print("\n  Evaluating per defect ...")
        evaluator  = CLEvaluator(extractor, memory, concept_heads, anomaly_head)
        full_df    = pd.read_csv(self._full_csv)
        meta_cols  = {c for c in _META}
        concept_col_names = [c for c in full_df.columns if c not in meta_cols]

        log = ContinualLog()
        defect_names = [t["defect"] for t in tasks]

        for tid, defect in enumerate(defect_names, 1):
            mvtec_cat = Path(self._mvtec_root) / self._category
            normal_paths = sorted((mvtec_cat / "test" / "good").glob("*.png"))
            defect_paths = sorted((mvtec_cat / "test" / defect).glob("*.png"))
            normal_imgs  = [Image.open(p).convert("RGB") for p in normal_paths]
            defect_imgs  = [Image.open(p).convert("RGB") for p in defect_paths]

            n_lab = full_df[full_df["image_path"].str.contains("test/good",  regex=False)][concept_col_names].reset_index(drop=True)
            d_lab = full_df[full_df["image_path"].str.contains(f"test/{defect}", regex=False)][concept_col_names].reset_index(drop=True)
            concept_labels = pd.concat([n_lab, d_lab], ignore_index=True)

            result = evaluator.evaluate(
                task_id=tid, defect_name=defect, evaluated_after_task=tid,
                normal_images=normal_imgs, defect_images=defect_imgs,
                concept_labels=concept_labels, tau=tau,
            )
            log.append(result)
            print(f"    {defect:<8} I-AUC(novel)={result.i_auc_novel:.4f}  "
                  f"I-AUC(concept)={result.i_auc_concept:.4f}  C-AUC={result.c_auc_mean:.4f}")

        # ── Step 6: comparison table ──────────────────────────────────────────
        joint_cauc = {r.defect_name: r.c_auc_mean  for r in log.results}
        joint_iauc = {r.defect_name: r.i_auc_novel for r in log.results}

        results_dict = {
            "joint": {d: {"c_auc": joint_cauc[d], "i_auc_novel": joint_iauc[d]} for d in defect_names},
            "tau": tau,
        }

        # Derive log paths: checkpoints/{category}/log_final.json
        # The joint checkpoint_dir is checkpoints/{category}/joint, so go up two levels
        ckpt_root   = Path(self.config["checkpoint_dir"]).parent.parent
        _print_comparison_table(
            defect_names, joint_cauc,
            ckpt_concil = ckpt_root / self._category / "log_final.json",
            ckpt_naive  = ckpt_root / f"{self._category}-baseline" / "log_final.json",
        )

        # ── Step 7: save ──────────────────────────────────────────────────────
        self._ckpt_dir.mkdir(parents=True, exist_ok=True)
        out_path = self._ckpt_dir / "joint_training_results.json"
        with open(out_path, "w") as f:
            json.dump({
                "joint_c_auc":    joint_cauc,
                "joint_i_auc_novel": joint_iauc,
                "tau":            tau,
                "n_train_normal": n_normal,
                "n_train_defect": n_defect,
            }, f, indent=2)
        print(f"\n  Results saved → {out_path}")

        return results_dict


# ── comparison table helper ───────────────────────────────────────────────────

def _load_final_cauc(log_path: Path) -> dict[str, float]:
    """Load per-defect C-AUC from the final evaluation in a log file."""
    if not log_path.exists():
        return {}
    with open(log_path) as f:
        results = json.load(f)["results"]
    best: dict[str, dict] = {}
    for r in results:
        d = r["defect_name"]
        if d not in best or r["evaluated_after_task"] > best[d]["evaluated_after_task"]:
            best[d] = r
    return {d: r["c_auc_mean"] for d, r in best.items()}


def _load_bwt(log_path: Path) -> float:
    """Load mean Concept BWT from a log file."""
    if not log_path.exists():
        return float("nan")
    from evaluators.evaluator_cl import TaskEvalResult
    with open(log_path) as f:
        data = json.load(f)
    log = ContinualLog(results=[TaskEvalResult(**r) for r in data["results"]])
    return log.mean_concept_bwt()


def _print_comparison_table(
    defect_names: list[str],
    joint_cauc:   dict[str, float],
    ckpt_concil:  Path,
    ckpt_naive:   Path,
) -> None:
    concil_cauc = _load_final_cauc(ckpt_concil)
    naive_cauc  = _load_final_cauc(ckpt_naive)

    concil_bwt = _load_bwt(ckpt_concil)
    naive_bwt  = _load_bwt(ckpt_naive)

    W = 66
    print(f"\n{'='*W}")
    print("  JOINT TRAINING vs SEQUENTIAL CL vs NAIVE BASELINE")
    print(f"{'='*W}")

    print(f"\n  C-AUC per defect (final evaluation on held-out test set):\n")
    print(f"  {'Defect':<10} {'Joint':>7}  {'CONCIL-CL':>10}  {'Naive-CL':>9}  {'CL Gap':>7}")
    print(f"  {'-'*8:<10} {'-'*6:>7}  {'-'*9:>10}  {'-'*8:>9}  {'-'*6:>7}")
    joint_vals, concil_vals, naive_vals = [], [], []
    for d in defect_names:
        jv = joint_cauc.get(d, float("nan"))
        cv = concil_cauc.get(d, float("nan"))
        nv = naive_cauc.get(d, float("nan"))
        gap = cv - nv if not (np.isnan(cv) or np.isnan(nv)) else float("nan")
        print(f"  {d:<10} {jv:>7.4f}  {cv:>10.4f}  {nv:>9.4f}  {gap:>+7.4f}")
        joint_vals.append(jv); concil_vals.append(cv); naive_vals.append(nv)

    jmean = float(np.nanmean(joint_vals))
    cmean = float(np.nanmean(concil_vals))
    nmean = float(np.nanmean(naive_vals))
    print(f"  {'─'*8:<10} {'─'*6:>7}  {'─'*9:>10}  {'─'*8:>9}  {'─'*6:>7}")
    print(f"  {'Average':<10} {jmean:>7.4f}  {cmean:>10.4f}  {nmean:>9.4f}  {cmean-nmean:>+7.4f}")

    print(f"\n  {'Method':<25} {'Avg C-AUC':>10}  {'BWT':>8}  {'vs Joint':>10}")
    print(f"  {'-'*24:<25} {'-'*9:>10}  {'-'*7:>8}  {'-'*9:>10}")
    print(f"  {'Joint training':<25} {jmean:>10.4f}  {'0.0000':>8}  {'baseline':>10}")
    vs_concil = f"{(cmean/jmean - 1)*100:+.2f}%" if jmean > 0 else "N/A"
    vs_naive  = f"{(nmean/jmean - 1)*100:+.2f}%" if jmean > 0 else "N/A"
    bwt_c = f"{concil_bwt:+.4f}" if not np.isnan(concil_bwt) else "  N/A "
    bwt_n = f"{naive_bwt:+.4f}"  if not np.isnan(naive_bwt)  else "  N/A "
    print(f"  {'CONCIL sequential':<25} {cmean:>10.4f}  {bwt_c:>8}  {vs_concil:>10}")
    print(f"  {'Naive sequential':<25} {nmean:>10.4f}  {bwt_n:>8}  {vs_naive:>10}")
    print(f"  {'─'*24:<25} {'─'*9:>10}  {'─'*7:>8}  {'─'*9:>10}")
    print(f"{'='*W}\n")

    if jmean > 0:
        print(f"  CONCIL sequential achieves {cmean/jmean*100:.1f}% of joint training C-AUC")
        print(f"  Naive  sequential achieves {nmean/jmean*100:.1f}% of joint training C-AUC")


# ── __main__ ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _sample = pd.read_csv("annotations/hazelnut/hazelnut.csv", nrows=1)
    mvtec_root = str(Path(_sample["image_path"].iloc[0]).parents[3])

    config = {
        "mvtec_root":      mvtec_root,
        "annotations_dir": "annotations/hazelnut",
        "category":        "hazelnut",
        "checkpoint_dir":  "checkpoints/hazelnut/joint",
        "lambda_concept":  1e-4,
        "lambda_anomaly":  1e-4,
        "coreset_size":    10_000,
        "tau_percentile":  95.0,
    }

    trainer = JointTrainer(config)
    trainer.run("annotations/hazelnut/cl_tasks/task_sequence.json")
