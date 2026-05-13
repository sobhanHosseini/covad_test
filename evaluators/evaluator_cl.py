"""CL-specific evaluation: I-AUC, C-AUC, Concept-BWT, s_novel calibration curve.

Placeholder — implement after CLTrainer is complete.
"""

from __future__ import annotations
import numpy as np


class CLEvaluator:
    """Evaluate continual learning metrics across all tasks.

    TODO — implement:
      - compute_concept_bwt(auc_matrix):
          auc_matrix[i, j] = C-AUC of concept j evaluated after task i
          BWT_j = auc_matrix[T-1, j] - auc_matrix[j, j]   (AUROC drop)
          mean_bwt = mean over all concepts seen before final task

      - compute_snivel_calibration(s_novel_scores, labels, taus):
          plot FPR/TPR at each τ across tasks

      - summarise(task_results):
          print table: Task | I-AUC | C-AUC | Concept-BWT
    """

    def __init__(self, concept_names: list[str]):
        raise NotImplementedError("CLEvaluator not yet implemented")

    def compute_concept_bwt(self, auc_matrix: np.ndarray) -> tuple[np.ndarray, float]:
        """Per-concept BWT and mean BWT."""
        raise NotImplementedError

    def summarise(self, task_results: list[dict]) -> None:
        raise NotImplementedError
