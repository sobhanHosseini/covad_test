"""CBM evaluator — metric logic only.

Inference path (TODO): replace gradient model with DINOv2 + concept heads.
The functions compute_iauc() and compute_cauc() are the reusable core;
they operate on pre-collected numpy arrays, not on model objects.
"""

import numpy as np
from sklearn.metrics import roc_auc_score

from utils.metrics import compute_image_f1


def compute_iauc(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    """Image-level AUC and best-F1 for anomaly detection.

    Args:
        y_true: (N,) integer array, 0=normal 1=anomalous
        y_prob: (N,) float array, predicted anomaly probability

    Returns:
        (i_auc, f1_max)
    """
    i_auc = roc_auc_score(y_true, y_prob)
    f1_max, _ = compute_image_f1(y_true, y_prob)
    return i_auc, f1_max


def compute_cauc(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    concept_names: list[str] | None = None,
) -> tuple[float, list[float]]:
    """Per-concept and mean C-AUC.

    Args:
        y_true: (N, K) binary array of ground-truth concept labels
        y_prob: (N, K) float array of predicted concept probabilities
        concept_names: optional list of K names for per-concept reporting

    Returns:
        (mean_cauc, per_concept_aucs)  — NaN inserted for all-zero columns
    """
    K = y_true.shape[1]
    aucs = []
    for i in range(K):
        try:
            auc = roc_auc_score(y_true[:, i], y_prob[:, i])
        except ValueError:
            auc = float("nan")
        aucs.append(auc)

    mean_cauc = float(np.nanmean(aucs))

    if concept_names is not None:
        for name, auc in zip(concept_names, aucs):
            print(f"  C-AUC  {name}: {auc:.4f}")

    return mean_cauc, aucs


# TODO: implement CLBMEvaluator class that wires DINOv2Extractor + ConceptHeads
# into the inference loop and then calls compute_iauc / compute_cauc above.
