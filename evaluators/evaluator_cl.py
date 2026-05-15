"""CL evaluation — all metrics for the thesis experiment.

Provides:
  TaskEvalResult   — per-task evaluation snapshot
  ContinualLog     — accumulates results, computes AUC matrices and Concept-BWT
  CLEvaluator      — runs inference and returns TaskEvalResult
  compute_tau()    — calibrate s_novel threshold from normal images
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

_BATCH_SIZE = 16
_NAN = float("nan")


# ── dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class TaskEvalResult:
    task_id: int
    defect_name: str
    evaluated_after_task: int

    # Detection branch (s_novel from PatchCore)
    i_auc_novel: float
    tau: float

    # Concept branch (linear anomaly head)
    i_auc_concept: float
    c_auc_mean: float
    c_auc_per_concept: dict[str, float]

    # Image counts
    n_normal: int
    n_anomalous: int


@dataclass
class ContinualLog:
    """Accumulates TaskEvalResults and computes CL metrics."""

    results: list[TaskEvalResult] = field(default_factory=list)

    def append(self, r: TaskEvalResult) -> None:
        self.results.append(r)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _ordered_defects(self) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for r in sorted(self.results, key=lambda x: x.task_id):
            if r.defect_name not in seen:
                out.append(r.defect_name)
                seen.add(r.defect_name)
        return out

    def _eval_times(self) -> list[int]:
        return sorted(set(r.evaluated_after_task for r in self.results))

    # ── matrices ──────────────────────────────────────────────────────────────

    def i_auc_matrix(self) -> pd.DataFrame:
        """I-AUC(novel) matrix: rows=defect, columns=evaluated_after_task.

        Entry [defect, t] = i_auc_novel for that defect evaluated after task t.
        Entries where t < task_id are NaN (defect not yet learned).
        """
        defects = self._ordered_defects()
        times   = self._eval_times()
        mat = pd.DataFrame(_NAN, index=defects, columns=times)
        mat.index.name   = "defect"
        mat.columns.name = "after_task"
        for r in self.results:
            mat.loc[r.defect_name, r.evaluated_after_task] = r.i_auc_novel
        return mat

    def c_auc_matrix(self) -> pd.DataFrame:
        """Mean C-AUC matrix — same structure as i_auc_matrix."""
        defects = self._ordered_defects()
        times   = self._eval_times()
        mat = pd.DataFrame(_NAN, index=defects, columns=times)
        mat.index.name   = "defect"
        mat.columns.name = "after_task"
        for r in self.results:
            mat.loc[r.defect_name, r.evaluated_after_task] = r.c_auc_mean
        return mat

    # ── Concept-BWT (Lopez-Paz & Ranzato 2017) ───────────────────────────────

    def per_defect_bwt(self) -> dict[str, float]:
        """Standard Backward Transfer per defect (Lopez-Paz & Ranzato 2017).

        BWT_d = R(T, d) - R(t_first(d), d)

        where:
          R(t, d) = C-AUC(mean) of defect d evaluated after training task t
          t_first(d) = the task at which defect d was first evaluated (= task_id)
          T = index of the final task

        Only defined for defects where t_first < T (i.e., all but the last).
        Negative = forgetting, Positive = forward transfer.
        """
        if len(self.results) < 2:
            return {}

        T = max(r.evaluated_after_task for r in self.results)

        # Build lookup: (defect, evaluated_after_task) → c_auc_mean
        lookup: dict[tuple[str, int], float] = {}
        for r in self.results:
            lookup[(r.defect_name, r.evaluated_after_task)] = r.c_auc_mean

        # For each defect, find its first evaluation time (= task_id)
        t_first_for: dict[str, int] = {}
        for r in self.results:
            d = r.defect_name
            if d not in t_first_for or r.task_id < t_first_for[d]:
                t_first_for[d] = r.task_id   # task_id == t_first by protocol

        bwt: dict[str, float] = {}
        for d, t_first in t_first_for.items():
            if t_first >= T:
                continue   # last task: no future point to measure forgetting
            r_first = lookup.get((d, t_first), _NAN)
            r_final = lookup.get((d, T), _NAN)
            if not (np.isnan(r_first) or np.isnan(r_final)):
                bwt[d] = float(r_final - r_first)
            else:
                bwt[d] = _NAN

        return bwt

    def concept_bwt(self) -> dict[str, float]:
        """Alias for per_defect_bwt() — kept for backward compatibility."""
        return self.per_defect_bwt()

    def mean_concept_bwt(self) -> float:
        """Mean BWT across all defects with valid values (Lopez-Paz formula)."""
        vals = [v for v in self.per_defect_bwt().values() if not np.isnan(v)]
        return float(np.mean(vals)) if vals else _NAN

    # ── summary ───────────────────────────────────────────────────────────────

    def summary_table(self) -> str:
        W = 82
        sep = "─" * W
        header = (
            f"{'T':>2}  {'defect':<10} {'@':>2}  "
            f"{'I-AUC(novel)':>13} {'I-AUC(concept)':>15} "
            f"{'C-AUC(mean)':>12}  {'τ':>8}"
        )
        lines = [
            "=" * W,
            "  CONVAD-CL Continual Learning Evaluation",
            "=" * W,
            header,
            sep,
        ]
        for r in sorted(self.results, key=lambda x: (x.evaluated_after_task, x.task_id)):
            lines.append(
                f"{r.task_id:>2}  {r.defect_name:<10} {r.evaluated_after_task:>2}  "
                f"{r.i_auc_novel:>13.4f} {r.i_auc_concept:>15.4f} "
                f"{r.c_auc_mean:>12.4f}  {r.tau:>8.4f}"
            )
        lines.append(sep)
        bwt_d = self.per_defect_bwt()
        mean_bwt = self.mean_concept_bwt()
        lines.append(f"  Standard BWT (Lopez-Paz 2017): {mean_bwt:+.4f}")
        if bwt_d:
            for d, v in sorted(bwt_d.items()):
                tag = "" if np.isnan(v) else ("  ← forgetting" if v < -0.01 else "  ← stable")
                lines.append(f"    BWT[{d}] = {v:+.4f}{tag}")
        lines.append("=" * W)
        return "\n".join(lines)


# ── standalone utility ────────────────────────────────────────────────────────

def compute_tau(
    memory: Any,
    normal_images: list,
    extractor: Any,
    percentile: float = 95.0,
) -> float:
    """Compute s_novel threshold as a percentile of normal image scores.

    Called by CLTrainer after Task 1 memory build.  Normal images are
    held-out (not used to build the memory bank — use val split).

    Args:
        memory:        built PatchCoreMemory
        normal_images: list of PIL Images (normal, held-out val set)
        extractor:     DINOv2Extractor instance
        percentile:    score percentile to use as τ (default 95)

    Returns:
        tau: float threshold
    """
    all_scores: list[torch.Tensor] = []
    for i in range(0, len(normal_images), _BATCH_SIZE):
        batch = normal_images[i : i + _BATCH_SIZE]
        patches = extractor.extract_patch_tokens(batch)
        scores, _ = memory.score(patches)
        all_scores.append(scores.cpu())
    all_scores_np = torch.cat(all_scores).numpy()
    return float(np.percentile(all_scores_np, percentile))


# ── evaluator ─────────────────────────────────────────────────────────────────

class CLEvaluator:
    """Runs inference with current model state and computes all metrics.

    Always uses current weights — call after CONCIL has updated them.
    """

    def __init__(
        self,
        extractor: Any,
        memory: Any,
        concept_heads: Any,
        anomaly_head: Any,
    ):
        self.extractor    = extractor
        self.memory       = memory
        self.concept_heads = concept_heads
        self.anomaly_head  = anomaly_head

    @torch.no_grad()
    def evaluate(
        self,
        task_id: int,
        defect_name: str,
        evaluated_after_task: int,
        normal_images: list,
        defect_images: list,
        concept_labels: pd.DataFrame,
        tau: float,
    ) -> TaskEvalResult:
        """Run inference on normal + defect images; compute all metrics.

        Args:
            task_id:              which task this defect belongs to (1-indexed)
            defect_name:          name of the defect type (e.g. "crack")
            evaluated_after_task: which CL task just completed
            normal_images:        list of PIL Images — normal (label=0)
            defect_images:        list of PIL Images — defect (label=1)
            concept_labels:       DataFrame with N rows and one column per concept
                                  (ground-truth binary labels from VLM annotation).
                                  Pass None for held-out test evaluation where
                                  ground-truth labels are not available.
                                  When None: c_auc_per_concept={}, c_auc_mean=NaN.
            tau:                  current s_novel threshold (for logging only)

        Returns:
            TaskEvalResult with all metrics filled in.

        Evaluation protocol note:
            I-AUC(novel)   — PatchCore branch; uses held-out test/good/ normals.
            I-AUC(concept) — linear head branch; uses held-out test/good/ normals.
            C-AUC(mean)    — concept head quality; requires VLM labels, so it is
                             NaN for held-out evaluation and must be measured
                             separately on the training set (see compute_cauc_train).
        """
        all_images = normal_images + defect_images
        N_normal  = len(normal_images)
        N_defect  = len(defect_images)
        N_total   = N_normal + N_defect
        y_true    = np.array([0] * N_normal + [1] * N_defect, dtype=np.float32)

        # ── batch feature extraction ──────────────────────────────────────────
        patch_list: list[torch.Tensor] = []
        pool_list:  list[torch.Tensor] = []
        for i in range(0, N_total, _BATCH_SIZE):
            batch = all_images[i : i + _BATCH_SIZE]
            p, z  = self.extractor.extract_both(batch)
            patch_list.append(p.cpu())
            pool_list.append(z.cpu())

        patch_tokens = torch.cat(patch_list, dim=0)   # (N, 256, 768)
        pooled_z     = torch.cat(pool_list,  dim=0)   # (N, 1536)

        # ── detection branch: s_novel from memory bank ────────────────────────
        s_novel, _ = self.memory.score(patch_tokens)  # (N,)
        s_novel_np = s_novel.cpu().numpy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            i_auc_novel = float(roc_auc_score(y_true, s_novel_np))

        # ── concept branch: linear anomaly head ───────────────────────────────
        c     = self.concept_heads(pooled_z)           # (N, K)
        y_hat = self.anomaly_head(c)                   # (N,)
        y_hat_np = y_hat.cpu().numpy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            i_auc_concept = float(roc_auc_score(y_true, y_hat_np))

        # ── per-concept AUC ───────────────────────────────────────────────────
        # concept_labels always provided — hazelnut.csv covers all 501 images
        # including test/good/ and all test defect images.
        c_np = c.cpu().numpy()                         # (N, K)
        c_auc_per_concept: dict[str, float] = {}

        for k_idx, k_name in enumerate(self.concept_heads.concept_names):
            if k_name not in concept_labels.columns:
                c_auc_per_concept[k_name] = _NAN
                continue
            c_true_k = concept_labels[k_name].values.astype(np.float32)
            c_pred_k = c_np[:, k_idx]
            if len(np.unique(c_true_k)) < 2:
                c_auc_per_concept[k_name] = _NAN
                continue
            try:
                c_auc_per_concept[k_name] = float(roc_auc_score(c_true_k, c_pred_k))
            except ValueError:
                c_auc_per_concept[k_name] = _NAN

        valid_caucs = [v for v in c_auc_per_concept.values() if not np.isnan(v)]
        c_auc_mean  = float(np.mean(valid_caucs)) if valid_caucs else _NAN

        return TaskEvalResult(
            task_id              = task_id,
            defect_name          = defect_name,
            evaluated_after_task = evaluated_after_task,
            i_auc_novel          = i_auc_novel,
            tau                  = tau,
            i_auc_concept        = i_auc_concept,
            c_auc_mean           = c_auc_mean,
            c_auc_per_concept    = c_auc_per_concept,
            n_normal             = N_normal,
            n_anomalous          = N_defect,
        )

    def evaluate_all_seen_defects(
        self,
        after_task: int,
        task_sequence: list[dict],
        tau: float,
        full_csv: str,
        batch_size: int = _BATCH_SIZE,
    ) -> list[TaskEvalResult]:
        """Evaluate all defects seen so far (task_id <= after_task).

        Loads images and concept labels from the full annotation CSV.
        Returns one TaskEvalResult per defect type seen so far.

        Args:
            after_task:    current CL task index (1-indexed)
            task_sequence: list of task dicts from task_sequence.json
            tau:           current s_novel threshold
            full_csv:      path to annotations/hazelnut/hazelnut.csv
            batch_size:    images per extraction batch
        """
        from PIL import Image as PILImage

        df = pd.read_csv(full_csv)
        # Use only test/good/ images (H-5: excludes train/good/ from evaluation normals)
        test_good_dir = Path(mvtec_root) / category / "test" / "good"
        normal_paths  = sorted(test_good_dir.glob("*.png"))
        normal_images = [PILImage.open(p).convert("RGB") for p in normal_paths]

        results: list[TaskEvalResult] = []
        for task in task_sequence:
            if task["task_id"] > after_task:
                break
            defect = task["defect"]
            defect_paths = df[df["anomaly_type"] == defect]["image_path"].tolist()
            defect_images = [PILImage.open(p).convert("RGB") for p in defect_paths]

            # Build concept_labels for this evaluation set
            normal_df = df[df["anomaly_type"] == "good"]
            defect_df = df[df["anomaly_type"] == defect]
            eval_df   = pd.concat([normal_df, defect_df], ignore_index=True)
            meta_cols = {"image_path", "label_index", "mask_path",
                         "anomaly_type", "split", "view"}
            concept_cols = [c for c in eval_df.columns if c not in meta_cols]
            concept_labels = eval_df[concept_cols].reset_index(drop=True)

            result = self.evaluate(
                task_id              = task["task_id"],
                defect_name          = defect,
                evaluated_after_task = after_task,
                normal_images        = normal_images,
                defect_images        = defect_images,
                concept_labels       = concept_labels,
                tau                  = tau,
            )
            results.append(result)

        return results


# ── __main__ — synthetic correctness test ─────────────────────────────────────

if __name__ == "__main__":
    import sys

    from models.concept_heads import ConceptHeads
    from models.linear_head   import LinearAnomalyHead

    print("=" * 68)
    print("evaluator_cl — synthetic correctness test")
    print("=" * 68)

    rng = np.random.default_rng(42)

    # ── mock components ───────────────────────────────────────────────────────
    class _MockExtractor:
        """Returns random tensors — shape matches DINOv2 output."""
        def __init__(self, seed: int = 0):
            self._g = torch.Generator().manual_seed(seed)

        def extract_both(self, images):
            B = len(images)
            p = torch.rand(B, 256, 768, generator=self._g)
            z = torch.rand(B, 1536,     generator=self._g)
            return p, z

    class _MockMemory:
        """Returns pre-planned s_novel scores in order."""
        is_built = True

        def __init__(self, scores: np.ndarray):
            self._scores = torch.tensor(scores, dtype=torch.float32)
            self._ptr    = 0

        def score(self, patch_tokens):
            B   = patch_tokens.shape[0]
            out = self._scores[self._ptr : self._ptr + B]
            self._ptr += B
            maps = torch.zeros(B, 16, 16)
            return out, maps

    # ── Part 1: test evaluate() ───────────────────────────────────────────────
    print("\n── Part 1: evaluate() with mock components ──────────────────")

    N_NORMAL, N_DEFECT = 20, 10
    N_TOTAL = N_NORMAL + N_DEFECT
    CONCEPTS = ["rough_surface", "color_change", "crack_visible", "hole_present", "print_mark"]
    K = len(CONCEPTS)

    # Controlled s_novel: normal images LOW, defect images HIGH → should give high AUROC
    normal_scores = rng.uniform(0.20, 0.45, N_NORMAL)
    defect_scores = rng.uniform(0.70, 1.20, N_DEFECT)
    all_scores    = np.concatenate([normal_scores, defect_scores])

    concept_heads = ConceptHeads(CONCEPTS, input_dim=1536)
    anomaly_head  = LinearAnomalyHead(n_concepts=K)

    # Random dummy weights (CONCIL not run — just testing metric pipeline)
    W_dummy = rng.standard_normal((K, 1536)).astype(np.float32)
    b_dummy = np.zeros(K, dtype=np.float32)
    concept_heads.set_weights(W_dummy, b_dummy)
    anomaly_head.set_weights(
        np.ones(K, dtype=np.float32) / K,
        np.zeros(1, dtype=np.float32),
    )

    # Concept labels: random binary, ensure both classes present in each column
    c_label_data = rng.integers(0, 2, (N_TOTAL, K)).astype(float)
    for j in range(K):          # guarantee variance
        c_label_data[0, j] = 0
        c_label_data[1, j] = 1
    concept_labels = pd.DataFrame(c_label_data, columns=CONCEPTS)

    evaluator = CLEvaluator(
        extractor     = _MockExtractor(seed=7),
        memory        = _MockMemory(all_scores),
        concept_heads = concept_heads,
        anomaly_head  = anomaly_head,
    )

    normal_imgs = ["n"] * N_NORMAL   # placeholder strings — mock extractor uses len()
    defect_imgs = ["d"] * N_DEFECT

    result = evaluator.evaluate(
        task_id              = 1,
        defect_name          = "crack",
        evaluated_after_task = 1,
        normal_images        = normal_imgs,
        defect_images        = defect_imgs,
        concept_labels       = concept_labels,
        tau                  = 0.55,
    )

    print(f"task_id              : {result.task_id}")
    print(f"defect_name          : {result.defect_name}")
    print(f"evaluated_after_task : {result.evaluated_after_task}")
    print(f"n_normal             : {result.n_normal}")
    print(f"n_anomalous          : {result.n_anomalous}")
    print(f"i_auc_novel          : {result.i_auc_novel:.4f}   (expect > 0.90 — controlled scores)")
    print(f"i_auc_concept        : {result.i_auc_concept:.4f}  (random weights — near 0.50)")
    print(f"c_auc_mean           : {result.c_auc_mean:.4f}")
    print(f"tau                  : {result.tau}")
    print(f"c_auc_per_concept:")
    for k, v in result.c_auc_per_concept.items():
        print(f"  {k:<20} {v:.4f}")

    assert result.i_auc_novel > 0.90, f"Expected high I-AUC(novel), got {result.i_auc_novel:.4f}"
    assert len(result.c_auc_per_concept) == K
    print("  assertions passed ✓")

    # ── Part 2: ContinualLog with 3-task synthetic history ────────────────────
    print("\n── Part 2: ContinualLog — 3 tasks, 6 results ────────────────")

    def _make_result(tid, defect, after, i_nov, i_con, c_per):
        return TaskEvalResult(
            task_id              = tid,
            defect_name          = defect,
            evaluated_after_task = after,
            i_auc_novel          = i_nov,
            tau                  = 0.50,
            i_auc_concept        = i_con,
            c_auc_mean           = float(np.nanmean(list(c_per.values()))),
            c_auc_per_concept    = c_per,
            n_normal             = 20,
            n_anomalous          = 10,
        )

    # c_auc values show gradual forgetting as more tasks arrive
    cp = lambda v0,v1,v2,v3,v4: dict(zip(CONCEPTS,[v0,v1,v2,v3,v4]))

    log = ContinualLog()
    log.append(_make_result(1,"crack",1, 0.90,0.78, cp(0.75,0.70,0.68,0.80,0.65)))
    log.append(_make_result(1,"crack",2, 0.88,0.76, cp(0.73,0.68,0.65,0.78,0.62)))
    log.append(_make_result(2,"hole", 2, 0.85,0.74, cp(0.72,0.69,0.66,0.79,0.63)))
    log.append(_make_result(1,"crack",3, 0.87,0.75, cp(0.71,0.66,0.63,0.76,0.60)))
    log.append(_make_result(2,"hole", 3, 0.84,0.73, cp(0.70,0.67,0.64,0.77,0.61)))
    log.append(_make_result(3,"cut",  3, 0.82,0.71, cp(0.69,0.65,0.62,0.75,0.59)))

    print("\nI-AUC(novel) matrix:")
    print(log.i_auc_matrix().to_string(float_format="{:.4f}".format))

    print("\nC-AUC(mean) matrix:")
    print(log.c_auc_matrix().to_string(float_format="{:.4f}".format))

    print("\nConcept BWT (per concept):")
    bwt = log.concept_bwt()
    for k, v in bwt.items():
        print(f"  {k:<20} {v:+.4f}")

    print(f"\nMean Concept BWT: {log.mean_concept_bwt():+.4f}")

    print(f"\n{log.summary_table()}")

    # ── assertions ────────────────────────────────────────────────────────────
    i_mat = log.i_auc_matrix()
    c_mat = log.c_auc_matrix()

    # NaN above diagonal
    assert np.isnan(i_mat.loc["hole",  1]), "hole@T1 should be NaN"
    assert np.isnan(i_mat.loc["cut",   1]), "cut@T1 should be NaN"
    assert np.isnan(i_mat.loc["cut",   2]), "cut@T2 should be NaN"
    # Filled on/below diagonal
    assert not np.isnan(i_mat.loc["crack", 1])
    assert not np.isnan(i_mat.loc["hole",  2])
    assert not np.isnan(i_mat.loc["cut",   3])
    print("\nMatrix structure (NaN above diagonal) : ✓")

    # All BWT values should be negative (forgetting in this synthetic log)
    assert all(v < 0 for v in bwt.values()), f"Expected forgetting: {bwt}"
    print("All BWT values < 0 (forgetting confirmed) : ✓")
    print("\nAll assertions passed.")
