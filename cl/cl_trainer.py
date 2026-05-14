"""CL trainer — orchestrates the full Scenario B sequential experiment.

Connects: DINOv2Extractor → PatchCoreMemory → ConceptHeads → ConcilSolver
          → LinearAnomalyHead → CLEvaluator → ContinualLog

Constraints enforced:
  - DINOv2 is NEVER set to train mode
  - Memory bank is built ONCE at Task 1 and never updated
  - CONCIL uses NO torch.optim, NO loss.backward()
  - ConceptDeduplicator prevents vocabulary bloat across tasks
"""

from __future__ import annotations

import dataclasses
import json
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from features.dinov2_extractor  import DINOv2Extractor
from features.patchcore_memory  import PatchCoreMemory
from models.concept_heads        import ConceptHeads
from models.linear_head          import LinearAnomalyHead
from solvers.concil              import ConcilSolver
from cl.concept_deduplicator     import ConceptDeduplicator
from evaluators.evaluator_cl     import CLEvaluator, ContinualLog, TaskEvalResult, compute_tau

_META_COLS = frozenset(
    ["image_path", "label_index", "mask_path", "anomaly_type", "split", "view"]
)


class CLTrainer:
    """Scenario B sequential continual learning for one MVTec category."""

    def __init__(self, config: dict):
        self.config         = config
        self.batch_size     = int(config.get("batch_size",    16))
        self.tau_percentile = float(config.get("tau_percentile", 95.0))
        self.checkpoint_dir = Path(config["checkpoint_dir"])

        self._annotations_dir = Path(config["annotations_dir"])
        self._category        = config["category"]
        self._full_csv_path   = self._annotations_dir / f"{self._category}.csv"
        self._cl_tasks_dir    = self._annotations_dir / "cl_tasks"

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ── model components ──────────────────────────────────────────────────
        self.extractor    = DINOv2Extractor(device=self._device)
        self.memory       = PatchCoreMemory(
            coreset_size=int(config.get("coreset_size", 10_000)),
            device=self._device,
        )
        self.concept_heads: Optional[ConceptHeads]     = None
        self.anomaly_head:  Optional[LinearAnomalyHead] = None
        self.solver        = ConcilSolver(
            input_dim      = DINOv2Extractor.POOLED_DIM,
            lambda_concept = float(config.get("lambda_concept", 1e-4)),
            lambda_anomaly = float(config.get("lambda_anomaly", 1e-4)),
        )
        self.deduplicator = ConceptDeduplicator()
        self.evaluator:   Optional[CLEvaluator] = None
        self.log          = ContinualLog()
        self.tau:         Optional[float] = None

    # ── private helpers ───────────────────────────────────────────────────────

    def _concept_cols(self, df: pd.DataFrame) -> list[str]:
        return [c for c in df.columns if c not in _META_COLS]

    def _load_task_data(
        self, task_csv_path: str
    ) -> tuple[list[Image.Image], np.ndarray, np.ndarray, dict]:
        """Load CONCIL training data from a per-task CSV.

        Normal images are filtered to MVTec train/good/ paths ONLY.
        This prevents data leakage: test/good/ images must stay
        held-out for evaluation and must not enter the memory bank.

        Returns:
            images:         list of PIL Images in CSV row order
            concept_matrix: (N, K) float32 array of concept labels
            y_labels:       (N,)   float32 array (0=normal, 1=anomalous)
            meta:           {'n_normal', 'n_anomalous', 'defect_name', 'concept_names'}
        """
        df = pd.read_csv(task_csv_path)

        # Keep all anomalous rows + only normals whose path is under train/good/
        # (MVTec test/good/ images are held-out for evaluation — never for training)
        normal_mask      = df["label_index"] == 0
        train_normal_mask = normal_mask & df["image_path"].str.contains(
            "/train/good/", regex=False
        )
        anomaly_mask = df["label_index"] == 1
        df = df[train_normal_mask | anomaly_mask].reset_index(drop=True)

        images = [
            Image.open(p).convert("RGB")
            for p in tqdm(df["image_path"], desc="  loading images", leave=False)
        ]
        concept_names  = self._concept_cols(df)
        concept_matrix = df[concept_names].values.astype(np.float32)
        y_labels       = df["label_index"].values.astype(np.float32)

        defect_types = df[df["label_index"] == 1]["anomaly_type"].unique().tolist()
        defect_name  = defect_types[0] if defect_types else "unknown"

        meta = {
            "n_normal":      int((y_labels == 0).sum()),
            "n_anomalous":   int((y_labels == 1).sum()),
            "defect_name":   defect_name,
            "concept_names": concept_names,
        }
        return images, concept_matrix, y_labels, meta

    @torch.no_grad()
    def _extract_features_batched(
        self, images: list[Image.Image], desc: str = "extracting"
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract patch tokens and pooled z for all images in batches."""
        patch_list, pool_list = [], []
        bar = tqdm(range(0, len(images), self.batch_size), desc=f"  {desc}", leave=False)
        for i in bar:
            batch     = images[i : i + self.batch_size]
            p, z      = self.extractor.extract_both(batch)
            patch_list.append(p.cpu())
            pool_list.append(z.cpu())
        return torch.cat(patch_list, dim=0), torch.cat(pool_list, dim=0)

    def _load_test_data_for_defect(
        self, defect_name: str
    ) -> tuple[list[Image.Image], list[Image.Image], pd.DataFrame]:
        """Load held-out test images and their concept labels for evaluation.

        Normal images : MVTec test/good/ — never seen by CONCIL or memory bank.
        Defect images : MVTec test/{defect}/ — all MVTec defects live in test/.
        Concept labels: loaded from hazelnut.csv by matching image paths.
                        hazelnut.csv covers all 501 MVTec hazelnut images,
                        including test/good/ and all test defect images.
        """
        mvtec_cat = Path(self.config["mvtec_root"]) / self._category

        # ── images ────────────────────────────────────────────────────────────
        normal_paths  = sorted((mvtec_cat / "test" / "good").glob("*.png"))
        defect_paths  = sorted((mvtec_cat / "test" / defect_name).glob("*.png"))
        normal_images = [Image.open(p).convert("RGB") for p in normal_paths]
        defect_images = [Image.open(p).convert("RGB") for p in defect_paths]

        # ── concept labels from annotation CSV ────────────────────────────────
        full_df = pd.read_csv(self._full_csv_path)
        meta_cols   = {"image_path", "label_index", "split", "anomaly_type",
                       "mask_path", "view"}
        concept_cols = [c for c in full_df.columns if c not in meta_cols]

        normal_labels = full_df[
            full_df["image_path"].str.contains("test/good", regex=False)
        ][concept_cols].reset_index(drop=True)

        defect_labels = full_df[
            full_df["image_path"].str.contains(
                f"test/{defect_name}", regex=False
            )
        ][concept_cols].reset_index(drop=True)

        concept_labels = pd.concat(
            [normal_labels, defect_labels], ignore_index=True
        )

        return normal_images, defect_images, concept_labels

    def _save_checkpoint(self, task_id: int) -> None:
        ckpt_dir = self.checkpoint_dir / f"task_{task_id}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.concept_heads.save(ckpt_dir / "concept_heads.pt")
        self.anomaly_head.save(ckpt_dir / "anomaly_head.pt")
        self.solver.save(ckpt_dir / "concil_state.pt")

        if task_id == 1:
            self.memory.save(ckpt_dir / "memory.pt")

        with open(ckpt_dir / "tau.json", "w") as f:
            json.dump({"tau": self.tau, "percentile": self.tau_percentile}, f)

        log_dicts = [dataclasses.asdict(r) for r in self.log.results]
        with open(self.checkpoint_dir / "log.json", "w") as f:
            json.dump({"results": log_dicts}, f, indent=2)

        print(f"  Checkpoint saved → {ckpt_dir}/")

    def _load_checkpoint(self, task_id: int) -> None:
        ckpt_dir = self.checkpoint_dir / f"task_{task_id}"

        self.concept_heads = ConceptHeads.load(ckpt_dir / "concept_heads.pt")
        self.anomaly_head  = LinearAnomalyHead.load(ckpt_dir / "anomaly_head.pt")
        self.solver        = ConcilSolver.load(ckpt_dir / "concil_state.pt")

        mem_path = self.checkpoint_dir / "task_1" / "memory.pt"
        self.memory = PatchCoreMemory.load(mem_path, device=self._device)

        with open(ckpt_dir / "tau.json") as f:
            self.tau = json.load(f)["tau"]

        log_path = self.checkpoint_dir / "log.json"
        if log_path.exists():
            with open(log_path) as f:
                data = json.load(f)
            self.log = ContinualLog(
                results=[TaskEvalResult(**r) for r in data["results"]]
            )

        self.evaluator = CLEvaluator(
            extractor    = self.extractor,
            memory       = self.memory,
            concept_heads = self.concept_heads,
            anomaly_head  = self.anomaly_head,
        )
        print(f"  Checkpoint restored from {ckpt_dir}/")

    # ── Task 1 ────────────────────────────────────────────────────────────────

    def run_task_1(self, task_info: dict) -> None:
        """Execute Task 1: build memory bank, initialise all heads via CONCIL."""
        print(f"\n{'='*60}")
        print(f"TASK 1 — {task_info['defect'].upper()}")
        print(f"{'='*60}")

        # Step 1 — load data
        images, C_labels, y, meta = self._load_task_data(task_info["csv_path"])
        print(
            f"  Data: {meta['n_normal']} normal + {meta['n_anomalous']} "
            f"defect images  (K={len(meta['concept_names'])} concepts)"
        )

        # Step 2 — extract features
        print("  Extracting features ...")
        patch_tokens, pooled_z = self._extract_features_batched(images, "features T1")
        print(
            f"  patch_tokens {tuple(patch_tokens.shape)}, "
            f"pooled_z {tuple(pooled_z.shape)}"
        )

        # Step 3 — build PatchCore memory bank (ONCE ONLY)
        normal_mask    = torch.tensor(y == 0)
        normal_patches = patch_tokens[normal_mask]
        self.memory.build(normal_patches)
        print(
            f"  Memory bank: {self.memory.coreset_size:,} patches "
            f"({self.memory.memory_mb:.1f} MB) — FROZEN"
        )

        # Step 4 — calibrate τ from normal image s_novel
        s_novel_normal, _ = self.memory.score(normal_patches)
        self.tau = float(
            np.percentile(s_novel_normal.cpu().numpy(), self.tau_percentile)
        )
        print(
            f"  τ = {self.tau:.4f}  "
            f"({self.tau_percentile}th percentile of normal s_novel)"
        )

        # Step 5 — initialise concept and anomaly heads
        concept_names = meta["concept_names"]
        self.concept_heads = ConceptHeads(concept_names, input_dim=DINOv2Extractor.POOLED_DIM)
        self.anomaly_head  = LinearAnomalyHead(n_concepts=len(concept_names))
        self.deduplicator.register_vocabulary(concept_names)
        print(f"  Concept heads initialised: K={len(concept_names)}")

        # Step 6 — CONCIL: concept heads
        Z = pooled_z.float()
        C = torch.tensor(C_labels, dtype=torch.float32)
        W_c, b_c = self.solver.update_concept_heads(
            Z, C, new_concept_cols=len(concept_names)
        )
        self.concept_heads.set_weights(W_c, b_c)
        print(f"  Concept heads updated via CONCIL  (K={self.solver.K})")

        # Step 7 — CONCIL: anomaly head
        with torch.no_grad():
            C_activated = self.concept_heads.forward(pooled_z.float())
        W_a, b_a = self.solver.update_anomaly_head(
            C_activated, torch.tensor(y, dtype=torch.float32)
        )
        self.anomaly_head.set_weights(W_a, b_a)
        print("  Anomaly head updated via CONCIL")

        # Step 8 — evaluate on test set
        self.evaluator = CLEvaluator(
            extractor    = self.extractor,
            memory       = self.memory,
            concept_heads = self.concept_heads,
            anomaly_head  = self.anomaly_head,
        )
        print("  Evaluating on test set ...")
        test_normals, test_defects, test_labels = self._load_test_data_for_defect(
            meta["defect_name"]
        )
        result = self.evaluator.evaluate(
            task_id              = 1,
            defect_name          = meta["defect_name"],
            evaluated_after_task = 1,
            normal_images        = test_normals,
            defect_images        = test_defects,
            concept_labels       = test_labels,
            tau                  = self.tau,
        )
        self.log.append(result)
        print(
            f"  T1 result: I-AUC(novel)={result.i_auc_novel:.4f}  "
            f"I-AUC(concept)={result.i_auc_concept:.4f}  "
            f"C-AUC={result.c_auc_mean:.4f}"
        )

        # Step 9 — checkpoint
        self._save_checkpoint(task_id=1)

    # ── Tasks 2 … T ──────────────────────────────────────────────────────────

    def run_task_n(self, task_info: dict, previous_task_id: int) -> None:
        """Execute task N > 1: update heads, evaluate ALL seen defects."""
        tid      = task_info["task_id"]
        defect   = task_info["defect"]

        print(f"\n{'='*60}")
        print(f"TASK {tid} — {defect.upper()}")
        print(f"{'='*60}")

        # Step 1 — load data (defect images only)
        images, C_labels, y, meta = self._load_task_data(task_info["csv_path"])
        print(f"  Data: {meta['n_anomalous']} defect images")

        # Step 2 — extract pooled features only (memory bank is frozen)
        print("  Extracting pooled features ...")
        _, pooled_z = self._extract_features_batched(images, f"features T{tid}")
        print(f"  pooled_z {tuple(pooled_z.shape)}")

        # Step 3 — handle new concepts via deduplicator
        csv_concepts     = meta["concept_names"]
        current_vocab    = set(self.concept_heads.concept_names)
        new_in_csv       = [c for c in csv_concepts if c not in current_vocab]
        dedup_result     = self.deduplicator.process_new_concepts(new_in_csv)
        vocabulary_expanded = bool(dedup_result.genuinely_new)

        if vocabulary_expanded:
            n_new = len(dedup_result.genuinely_new)
            print(f"  New concepts detected: {dedup_result.genuinely_new}")
            self.solver.expand_for_new_concepts(n_new)
            self.concept_heads.add_concepts(dedup_result.genuinely_new)
            self.anomaly_head.expand(self.concept_heads.n_concepts)
        else:
            print(
                f"  No new concepts. Updating existing "
                f"{self.concept_heads.n_concepts} heads."
            )

        # Step 4 — CONCIL: concept heads
        Z = pooled_z.float()
        C = torch.tensor(C_labels, dtype=torch.float32)
        n_new_cols = len(dedup_result.genuinely_new) if vocabulary_expanded else 0
        W_c, b_c = self.solver.update_concept_heads(Z, C, new_concept_cols=n_new_cols)
        self.concept_heads.set_weights(W_c, b_c)
        print(f"  Concept heads updated  (K={self.solver.K}, n_tasks_seen={self.solver.n_tasks_seen})")

        # Step 5 — CONCIL: anomaly head
        with torch.no_grad():
            C_activated = self.concept_heads.forward(pooled_z.float())
        W_a, b_a = self.solver.update_anomaly_head(
            C_activated,
            torch.tensor(y, dtype=torch.float32),
            vocabulary_expanded=vocabulary_expanded,
        )
        self.anomaly_head.set_weights(W_a, b_a)
        print("  Anomaly head updated")

        # Step 6 — evaluate ALL seen defects (measures forgetting)
        print("  Evaluating all seen defects ...")
        task_sequence = self._cached_task_sequence

        for seen_task in task_sequence:
            if seen_task["task_id"] > tid:
                break
            d = seen_task["defect"]
            test_normals, test_defects, test_labels = self._load_test_data_for_defect(d)
            result = self.evaluator.evaluate(
                task_id              = seen_task["task_id"],
                defect_name          = d,
                evaluated_after_task = tid,
                normal_images        = test_normals,
                defect_images        = test_defects,
                concept_labels       = test_labels,
                tau                  = self.tau,
            )
            self.log.append(result)
            print(
                f"    {d:<8} I-AUC(novel)={result.i_auc_novel:.4f}  "
                f"I-AUC(concept)={result.i_auc_concept:.4f}  "
                f"C-AUC={result.c_auc_mean:.4f}"
            )

        # Step 7 — checkpoint
        self._save_checkpoint(task_id=tid)

    # ── orchestration ─────────────────────────────────────────────────────────

    def run(self, task_sequence_path: str) -> ContinualLog:
        """Run the full sequential CL experiment from Task 1 to T.

        Args:
            task_sequence_path: path to task_sequence.json

        Returns:
            Completed ContinualLog.
        """
        with open(task_sequence_path) as f:
            tasks = json.load(f)

        self._cached_task_sequence = tasks

        print(f"\nStarting CONVAD-CL Scenario B — {self._category}")
        print(f"Task sequence: {[t['defect'] for t in tasks]}")
        print(f"Device: {self._device}")

        self.run_task_1(tasks[0])

        for task in tasks[1:]:
            self.run_task_n(task, previous_task_id=task["task_id"] - 1)

        # Final summary
        print(f"\n{'='*60}")
        print("FINAL RESULTS")
        print(f"{'='*60}")
        print(self.log.summary_table())

        print("\nI-AUC(novel) matrix:")
        print(self.log.i_auc_matrix().to_string(float_format="{:.4f}".format))
        print("\nC-AUC(mean) matrix:")
        print(self.log.c_auc_matrix().to_string(float_format="{:.4f}".format))
        bwt = self.log.mean_concept_bwt()
        print(f"\nMean Concept BWT : {bwt:+.4f}")
        print("(Expected ≈ 0.00 with CONCIL zero-forgetting guarantee)")

        # Save final log
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        log_dicts = [dataclasses.asdict(r) for r in self.log.results]
        with open(self.checkpoint_dir / "log_final.json", "w") as f:
            json.dump({"results": log_dicts}, f, indent=2)
        print(f"\nFinal log saved → {self.checkpoint_dir}/log_final.json")

        return self.log

    def resume(self, from_task: int, task_sequence_path: str) -> ContinualLog:
        """Resume a partially completed run from a checkpoint.

        Args:
            from_task:           task_id to resume FROM (already completed)
            task_sequence_path:  path to task_sequence.json
        """
        print(f"Resuming from Task {from_task} checkpoint ...")
        self._load_checkpoint(from_task)

        with open(task_sequence_path) as f:
            tasks = json.load(f)
        self._cached_task_sequence = tasks

        for task in tasks:
            if task["task_id"] <= from_task:
                continue
            self.run_task_n(task, previous_task_id=task["task_id"] - 1)

        return self.log

    def dry_run(self, task_sequence_path: str) -> bool:
        """Verify data files and component state without loading images or running inference.

        Returns True if all checks pass, False otherwise.
        """
        with open(task_sequence_path) as f:
            tasks = json.load(f)

        print(f"\nTask sequence: {len(tasks)} tasks")
        for t in tasks:
            print(
                f"  T{t['task_id']} {t['defect']:<8} "
                f"{t['n_images']:>4} images  "
                f"{len(t['new_concepts']):>2} new concepts"
            )

        print("\nVerifying data files ...")
        issues: list[str] = []

        if not self._full_csv_path.exists():
            issues.append(f"  MISSING: {self._full_csv_path}")

        for t in tasks:
            csv_path = Path(t["csv_path"])
            if not csv_path.exists():
                issues.append(f"  MISSING csv: {csv_path}")
            else:
                df_row = pd.read_csv(csv_path, nrows=1)
                img = Path(df_row["image_path"].iloc[0])
                if not img.exists():
                    issues.append(f"  MISSING image: {img}")

        mvtec_dir = Path(self.config["mvtec_root"]) / self._category
        for subdir in ["train/good", "test/good", "test/crack",
                        "test/hole", "test/cut", "test/print"]:
            p = mvtec_dir / subdir
            if not p.exists():
                issues.append(f"  MISSING dir: {p}")

        if issues:
            for s in issues:
                print(s)
            return False

        print("  All data files found ✓")

        n_params = sum(p.numel() for p in self.extractor.model.parameters())
        frozen   = all(not p.requires_grad for p in self.extractor.model.parameters())
        print("\nComponent status:")
        print(f"  DINOv2Extractor    : {n_params:,} params  frozen={frozen}")
        print(f"  PatchCoreMemory    : will build at Task 1")
        print(f"  ConcilSolver       : D={self.solver.D}  λ_c={self.solver.lambda_c}  λ_a={self.solver.lambda_a}")
        print(f"  ConceptDeduplicator: threshold={self.deduplicator.threshold}")
        print(f"  Device             : {self._device}")
        print("  All components OK ✓")
        return True


# ── __main__ — dry run (wiring verification, no inference) ───────────────────

if __name__ == "__main__":
    import sys

    print("=" * 62)
    print("CLTrainer — dry run (wiring verification)")
    print("=" * 62)

    # ── config ────────────────────────────────────────────────────────────────
    BASE = Path(__file__).resolve().parent.parent

    # Derive MVTec root from the first image path in the annotation CSV
    # (avoids hardcoding the absolute path — works on any machine)
    _ann_dir = BASE / "annotations/hazelnut"
    _sample_csv = _ann_dir / "hazelnut.csv"
    _first_img  = pd.read_csv(_sample_csv, nrows=1)["image_path"].iloc[0]
    # image path: .../mvtec/hazelnut/train/good/000.png  → mvtec_root = .../mvtec
    _mvtec_root = str(Path(_first_img).parents[3])

    config = {
        "mvtec_root":      _mvtec_root,
        "annotations_dir": str(_ann_dir),
        "category":        "hazelnut",
        "checkpoint_dir":  str(BASE / "checkpoints/hazelnut"),
        "batch_size":      16,
        "lambda_concept":  1e-4,
        "lambda_anomaly":  1e-4,
        "coreset_size":    10_000,
        "tau_percentile":  95.0,
    }

    # ── 1. load task sequence ─────────────────────────────────────────────────
    task_seq_path = Path(config["annotations_dir"]) / "cl_tasks" / "task_sequence.json"
    if not task_seq_path.exists():
        sys.exit(f"task_sequence.json not found: {task_seq_path}\nRun cl.task_csv_builder first.")

    with open(task_seq_path) as f:
        tasks = json.load(f)

    print(f"\nTask sequence loaded: {len(tasks)} tasks")
    for t in tasks:
        print(
            f"  Task {t['task_id']}: {t['defect']:<8}  "
            f"{t['n_images']:>4} images  "
            f"{len(t['new_concepts']):>2} new concepts"
        )

    # ── 2. verify data files ──────────────────────────────────────────────────
    print("\nVerifying data files ...")
    issues: list[str] = []
    full_csv = Path(config["annotations_dir"]) / "hazelnut.csv"
    if not full_csv.exists():
        issues.append(f"  MISSING: {full_csv}")

    for t in tasks:
        csv_path = Path(t["csv_path"])
        if not csv_path.exists():
            issues.append(f"  MISSING csv: {csv_path}")
        else:
            # spot-check first image path in csv
            df = pd.read_csv(csv_path, nrows=1)
            img_path = Path(df["image_path"].iloc[0])
            if not img_path.exists():
                issues.append(f"  MISSING image: {img_path}")

    mvtec_hazeln = Path(config["mvtec_root"]) / "hazelnut"
    if not mvtec_hazeln.exists():
        issues.append(f"  MISSING MVTec dir: {mvtec_hazeln}")
    else:
        for defect_dir in ["test/good", "test/crack", "test/hole", "test/cut", "test/print"]:
            p = mvtec_hazeln / defect_dir
            if not p.exists():
                issues.append(f"  MISSING MVTec subdir: {p}")

    if issues:
        print("  Issues found:")
        for s in issues:
            print(s)
        sys.exit(1)

    print("  All data files found ✓")

    # ── 3. initialise components ──────────────────────────────────────────────
    print("\nInitialising model components ...")
    print("  Loading DINOv2 ViT-B/14 ...")
    trainer = CLTrainer(config)

    print(f"  DINOv2Extractor   : {sum(p.numel() for p in trainer.extractor.model.parameters()):,} params  frozen={all(not p.requires_grad for p in trainer.extractor.model.parameters())}")
    print(f"  PatchCoreMemory   : is_built={trainer.memory.is_built}  (will build at Task 1)")
    print(f"  ConcilSolver      : D={trainer.solver.D}  λ_c={trainer.solver.lambda_c}  λ_a={trainer.solver.lambda_a}")
    print(f"  ConceptDeduplicator: model={trainer.deduplicator.model_name}  threshold={trainer.deduplicator.threshold}")
    print(f"  ConceptHeads      : None (initialised at Task 1)")
    print(f"  LinearAnomalyHead : None (initialised at Task 1)")
    print(f"  ContinualLog      : {len(trainer.log.results)} results so far")
    print(f"  Device            : {trainer._device}")
    print("  All components initialised ✓")

    # ── 4. summary ────────────────────────────────────────────────────────────
    print(f"\n{'='*62}")
    print("Ready to run.")
    print(f"  Call: trainer.run('{task_seq_path}')")
    print(f"  Or:   python -m main_scripts.run_cl --config <cfg.yaml>")
    print(f"{'='*62}")
