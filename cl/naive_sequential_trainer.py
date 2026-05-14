"""Naive sequential trainer — catastrophic forgetting baseline.

Same architecture as CLTrainer (frozen DINOv2 + PatchCore memory + concept
heads + linear anomaly head) but updated with gradient descent on current-task
data ONLY.  No replay buffer, no closed-form accumulation.

Expected behaviour: C-AUC for earlier defects drops sharply after later tasks
are trained — textbook catastrophic forgetting.  Used as the negative baseline
against which CONCIL's zero-forgetting guarantee is demonstrated.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm

from features.dinov2_extractor import DINOv2Extractor
from features.patchcore_memory import PatchCoreMemory
from models.concept_heads       import ConceptHeads
from models.linear_head         import LinearAnomalyHead
from evaluators.evaluator_cl    import CLEvaluator, ContinualLog, TaskEvalResult

_META_COLS = frozenset(
    ["image_path", "label_index", "mask_path", "anomaly_type", "split", "view"]
)


class NaiveSequentialTrainer:
    """Catastrophic-forgetting baseline: gradient descent, no replay.

    Identical data loading and evaluation protocol to CLTrainer.
    The ONLY difference: CONCIL is replaced by Adam + BCE on current-task
    images only.  At Task 2 the model never sees Task 1 data again.
    """

    def __init__(self, config: dict):
        self.config         = config
        self.batch_size     = int(config.get("batch_size",    16))
        self.tau_percentile = float(config.get("tau_percentile", 95.0))
        self.checkpoint_dir = Path(config["checkpoint_dir"])
        self.n_epochs       = int(config.get("naive_epochs", 50))
        self.lr             = float(config.get("naive_lr",   1e-3))

        self._annotations_dir = Path(config["annotations_dir"])
        self._category        = config["category"]
        self._full_csv_path   = self._annotations_dir / f"{self._category}.csv"
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.extractor = DINOv2Extractor(device=self._device)
        self.memory    = PatchCoreMemory(
            coreset_size=int(config.get("coreset_size", 10_000)),
            device=self._device,
        )
        self.concept_heads: Optional[ConceptHeads]      = None
        self.anomaly_head:  Optional[LinearAnomalyHead] = None
        self.evaluator:     Optional[CLEvaluator]       = None
        self.log            = ContinualLog()
        self.tau:           Optional[float] = None
        self._cached_task_sequence: list[dict] = []

    # ── shared helpers (mirror of CLTrainer) ─────────────────────────────────

    def _concept_cols(self, df: pd.DataFrame) -> list[str]:
        return [c for c in df.columns if c not in _META_COLS]

    def _load_task_data(self, task_csv_path: str):
        df = pd.read_csv(task_csv_path)
        train_normal  = (df["label_index"] == 0) & df["image_path"].str.contains(
            "/train/good/", regex=False
        )
        anomaly_mask = df["label_index"] == 1
        df = df[train_normal | anomaly_mask].reset_index(drop=True)

        images = [
            Image.open(p).convert("RGB")
            for p in tqdm(df["image_path"], desc="  loading images", leave=False)
        ]
        cnames = self._concept_cols(df)
        C      = df[cnames].values.astype(np.float32)
        y      = df["label_index"].values.astype(np.float32)
        defect = df[df["label_index"] == 1]["anomaly_type"].iloc[0]
        meta   = {"n_normal": int((y==0).sum()), "n_anomalous": int((y==1).sum()),
                  "defect_name": defect, "concept_names": cnames}
        return images, C, y, meta

    @torch.no_grad()
    def _extract_features_batched(self, images, desc="extracting"):
        patch_list, pool_list = [], []
        for i in tqdm(range(0, len(images), self.batch_size), desc=f"  {desc}", leave=False):
            p, z = self.extractor.extract_both(images[i : i + self.batch_size])
            patch_list.append(p.cpu()); pool_list.append(z.cpu())
        return torch.cat(patch_list), torch.cat(pool_list)

    def _load_test_data_for_defect(self, defect_name: str):
        mvtec = Path(self.config["mvtec_root"]) / self._category
        normal_paths = sorted((mvtec / "test" / "good").glob("*.png"))
        defect_paths = sorted((mvtec / "test" / defect_name).glob("*.png"))
        normal_images = [Image.open(p).convert("RGB") for p in normal_paths]
        defect_images = [Image.open(p).convert("RGB") for p in defect_paths]

        full_df      = pd.read_csv(self._full_csv_path)
        concept_cols = [c for c in full_df.columns if c not in _META_COLS]
        n_lab = full_df[full_df["image_path"].str.contains("test/good",      regex=False)][concept_cols].reset_index(drop=True)
        d_lab = full_df[full_df["image_path"].str.contains(f"test/{defect_name}", regex=False)][concept_cols].reset_index(drop=True)
        concept_labels = pd.concat([n_lab, d_lab], ignore_index=True)
        return normal_images, defect_images, concept_labels

    def dry_run(self, task_sequence_path: str) -> bool:
        """Delegate to CLTrainer.dry_run — identical data/component checks."""
        from cl.cl_trainer import CLTrainer
        _t = CLTrainer(self.config)
        return _t.dry_run(task_sequence_path)

    def _save_checkpoint(self, task_id: int) -> None:
        ckpt = self.checkpoint_dir / f"task_{task_id}"
        ckpt.mkdir(parents=True, exist_ok=True)
        self.concept_heads.save(ckpt / "concept_heads.pt")
        self.anomaly_head.save(ckpt  / "anomaly_head.pt")
        if task_id == 1:
            self.memory.save(ckpt / "memory.pt")
        with open(ckpt / "tau.json", "w") as f:
            json.dump({"tau": self.tau}, f)
        with open(self.checkpoint_dir / "log.json", "w") as f:
            json.dump({"results": [dataclasses.asdict(r) for r in self.log.results]}, f, indent=2)

    # ── gradient training ─────────────────────────────────────────────────────

    def _train_heads_gradient(
        self,
        pooled_z: torch.Tensor,
        C_labels: np.ndarray,
        y_labels: np.ndarray,
    ) -> float:
        """Train concept heads + anomaly head with Adam/BCE on current data only.

        Operates directly on the underlying nn.Linear weights, bypassing the
        @torch.no_grad() decorators on ConceptHeads.forward() and
        LinearAnomalyHead.forward().  No replay: only current-task images.
        """
        Z = pooled_z.float()
        C = torch.tensor(C_labels, dtype=torch.float32)
        y = torch.tensor(y_labels, dtype=torch.float32)

        c_lin = self.concept_heads.linear   # nn.Linear(1536, K)
        a_lin = self.anomaly_head.linear    # nn.Linear(K, 1)

        # Temporarily enable gradients
        for p in list(c_lin.parameters()) + list(a_lin.parameters()):
            p.requires_grad_(True)

        optimizer = torch.optim.Adam(
            list(c_lin.parameters()) + list(a_lin.parameters()), lr=self.lr
        )
        c_crit = nn.BCEWithLogitsLoss()
        a_crit = nn.BCEWithLogitsLoss()

        bar = tqdm(range(self.n_epochs), desc="  gradient train", leave=False)
        final_loss = 0.0
        for _ in bar:
            optimizer.zero_grad()
            c_logits = c_lin(Z)                               # (N, K)
            a_logits = a_lin(torch.sigmoid(c_logits)).squeeze(1)  # (N,)
            loss = c_crit(c_logits, C) + a_crit(a_logits, y)
            loss.backward()
            optimizer.step()
            final_loss = loss.item()
            bar.set_postfix(loss=f"{final_loss:.4f}")

        # Re-freeze
        for p in list(c_lin.parameters()) + list(a_lin.parameters()):
            p.requires_grad_(False)

        return final_loss

    # ── task runners ─────────────────────────────────────────────────────────

    def run_task_1(self, task_info: dict) -> None:
        print(f"\n{'='*60}\nTASK 1 — {task_info['defect'].upper()}  [NAIVE BASELINE]\n{'='*60}")

        images, C_labels, y, meta = self._load_task_data(task_info["csv_path"])
        print(f"  Data: {meta['n_normal']} normal + {meta['n_anomalous']} defect  K={len(meta['concept_names'])}")

        patch_tokens, pooled_z = self._extract_features_batched(images, "features T1")
        print(f"  patch_tokens {tuple(patch_tokens.shape)}, pooled_z {tuple(pooled_z.shape)}")

        # Memory bank (identical to CLTrainer — fair comparison)
        normal_mask    = torch.tensor(y == 0)
        self.memory.build(patch_tokens[normal_mask])
        print(f"  Memory: {self.memory.coreset_size:,} patches ({self.memory.memory_mb:.1f} MB) — FROZEN")

        s_nov_normal, _ = self.memory.score(patch_tokens[normal_mask])
        self.tau = float(np.percentile(s_nov_normal.cpu().numpy(), self.tau_percentile))
        print(f"  τ = {self.tau:.4f}")

        # Initialise heads
        self.concept_heads = ConceptHeads(meta["concept_names"], input_dim=DINOv2Extractor.POOLED_DIM)
        self.anomaly_head  = LinearAnomalyHead(n_concepts=len(meta["concept_names"]))

        # Gradient update (no replay — but T1 has all the data)
        final_loss = self._train_heads_gradient(pooled_z, C_labels, y)
        print(f"  Gradient training done. Final loss: {final_loss:.4f}")

        self.evaluator = CLEvaluator(self.extractor, self.memory, self.concept_heads, self.anomaly_head)
        test_n, test_d, test_labels = self._load_test_data_for_defect(meta["defect_name"])
        result = self.evaluator.evaluate(1, meta["defect_name"], 1, test_n, test_d, test_labels, self.tau)
        self.log.append(result)
        print(f"  T1: I-AUC(novel)={result.i_auc_novel:.4f}  I-AUC(concept)={result.i_auc_concept:.4f}  C-AUC={result.c_auc_mean:.4f}")
        self._save_checkpoint(1)

    def run_task_n(self, task_info: dict) -> None:
        tid    = task_info["task_id"]
        defect = task_info["defect"]
        print(f"\n{'='*60}\nTASK {tid} — {defect.upper()}  [NAIVE BASELINE]\n{'='*60}")

        images, C_labels, y, meta = self._load_task_data(task_info["csv_path"])
        print(f"  Data: {meta['n_anomalous']} defect images (NO replay of previous tasks)")

        _, pooled_z = self._extract_features_batched(images, f"features T{tid}")

        # Handle new concept columns (same as CLTrainer)
        csv_concepts  = meta["concept_names"]
        current_vocab = set(self.concept_heads.concept_names)
        new_in_csv    = [c for c in csv_concepts if c not in current_vocab]
        if new_in_csv:
            self.concept_heads.add_concepts(new_in_csv)
            self.anomaly_head.expand(self.concept_heads.n_concepts)
            print(f"  Vocabulary expanded: +{len(new_in_csv)} concepts  K={self.concept_heads.n_concepts}")
        else:
            print(f"  No new concepts. Overwriting {self.concept_heads.n_concepts} heads (NO memory of past tasks).")

        # Gradient update on current task only — this is where forgetting happens
        final_loss = self._train_heads_gradient(pooled_z, C_labels, y)
        print(f"  Gradient training done. Final loss: {final_loss:.4f}")

        print("  Evaluating all seen defects ...")
        for seen in self._cached_task_sequence:
            if seen["task_id"] > tid:
                break
            test_n, test_d, test_labels = self._load_test_data_for_defect(seen["defect"])
            result = self.evaluator.evaluate(
                seen["task_id"], seen["defect"], tid, test_n, test_d, test_labels, self.tau
            )
            self.log.append(result)
            print(f"    {seen['defect']:<8} I-AUC(novel)={result.i_auc_novel:.4f}  "
                  f"I-AUC(concept)={result.i_auc_concept:.4f}  C-AUC={result.c_auc_mean:.4f}")

        self._save_checkpoint(tid)

    # ── orchestration ─────────────────────────────────────────────────────────

    def run(self, task_sequence_path: str) -> ContinualLog:
        """Run the full naive sequential experiment from Task 1 to T."""
        with open(task_sequence_path) as f:
            tasks = json.load(f)
        self._cached_task_sequence = tasks

        print(f"\nStarting NAIVE SEQUENTIAL BASELINE — {self._category}")
        print(f"Gradient descent: {self.n_epochs} epochs, lr={self.lr}, NO replay")
        print(f"Device: {self._device}")

        self.run_task_1(tasks[0])
        for task in tasks[1:]:
            self.run_task_n(task)

        print(f"\n{'='*60}\nFINAL RESULTS — NAIVE BASELINE\n{'='*60}")
        print(self.log.summary_table())
        print("\nI-AUC(novel) matrix:")
        print(self.log.i_auc_matrix().to_string(float_format="{:.4f}".format))
        print("\nC-AUC(mean) matrix:")
        print(self.log.c_auc_matrix().to_string(float_format="{:.4f}".format))
        bwt = self.log.mean_concept_bwt()
        print(f"\nMean Concept BWT : {bwt:+.4f}")
        print("(Expected strongly negative — catastrophic forgetting)")

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        with open(self.checkpoint_dir / "log_final.json", "w") as f:
            json.dump({"results": [dataclasses.asdict(r) for r in self.log.results]}, f, indent=2)

        return self.log
