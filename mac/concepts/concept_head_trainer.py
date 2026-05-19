"""
concept_head_trainer.py — Train binary linear concept heads on full SAE codes.

Design (corrected):
  Features   : full 4096-dim SAE code for every patch.
  Labels     : member atoms define WHICH patches are positive, NOT which
               features are used.  Every head sees the same 4096-dim input.
  Shared atoms: allowed and expected.  Two concepts can share member atoms;
               their heads learn different linear combinations of the full code.

Label assignment for defect concept k:
  For each member atom a_i, compute threshold_i = 70th-percentile of a_i
  activation across all anomaly patches.
  Positive  → anomaly patches where ANY member atom exceeds its threshold.
  Negative  → normal patches  +  anomaly patches where ALL member atoms
               are below their thresholds.

Label assignment for the normality head:
  Positive  → normal patches (balanced sample matching anomaly count).
  Negative  → all anomaly patches.

Model: LogisticRegression(C=1.0, class_weight='balanced', solver='saga').
Split: stratified 80/20 train/test.
"""

from __future__ import annotations

import json
import pickle
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split


# ─────────────────────────────────────────────────────────────────────────────
# Label assignment
# ─────────────────────────────────────────────────────────────────────────────

def _defect_labels(
    concept: dict[str, Any],
    anomaly_codes: torch.Tensor,   # (N_anom, 4096)
    pos_percentile: float = 70.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute positive and negative masks for one defect concept.

    Per-atom thresholds: for each member atom a_i, threshold_i is the
    *pos_percentile*-th percentile of a_i's activations across ALL anomaly
    patches.  A patch is positive when ANY a_i exceeds its threshold.

    Args:
        concept: Dict with ``"atom_ids"`` list.
        anomaly_codes: (N_anom, 4096) float tensor.
        pos_percentile: Percentile threshold per atom (default 70 → top 30%).

    Returns:
        (pos_mask, neg_mask) — boolean numpy arrays of shape (N_anom,).
        ``neg_mask`` is the strict complement: ALL atoms below threshold.
    """
    member_ids = [int(aid) for aid in concept["atom_ids"]]
    acts = anomaly_codes[:, member_ids].float()           # (N_anom, n_members)
    thresholds = torch.quantile(acts, pos_percentile / 100.0, dim=0)  # (n_members,)
    above = acts > thresholds.unsqueeze(0)                # (N_anom, n_members)
    pos_mask = above.any(dim=1).numpy()
    neg_mask  = (~above.any(dim=1)).numpy()               # ALL atoms below threshold
    return pos_mask, neg_mask


def _build_defect_dataset(
    concept: dict[str, Any],
    anomaly_codes: torch.Tensor,
    normal_codes: torch.Tensor,
    pos_percentile: float = 70.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Build (X, y) for one defect concept head.

    Positive (y=1): anomaly patches where ANY member atom > its threshold.
    Negative (y=0): ALL normal patches  +  anomaly patches where ALL member
                    atoms are below threshold.
    Features: full 4096-dim SAE code.
    """
    pos_mask, neg_mask = _defect_labels(concept, anomaly_codes, pos_percentile)

    X_pos     = anomaly_codes[pos_mask].numpy().astype(np.float32)
    X_neg_anom = anomaly_codes[neg_mask].numpy().astype(np.float32)
    X_neg_norm = normal_codes.numpy().astype(np.float32)

    X_neg = np.concatenate([X_neg_anom, X_neg_norm], axis=0)
    X     = np.concatenate([X_pos, X_neg], axis=0)
    y     = np.array([1] * len(X_pos) + [0] * len(X_neg), dtype=np.int32)
    return X, y


def _build_normality_dataset(
    anomaly_codes: torch.Tensor,
    normal_codes: torch.Tensor,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Build (X, y) for the normality head.

    Positive (y=1): normal patches — sampled to match anomaly count for balance.
    Negative (y=0): all anomaly patches.
    Features: full 4096-dim SAE code.
    """
    n_anom = anomaly_codes.shape[0]
    n_norm = normal_codes.shape[0]

    rng = random.Random(seed)
    if n_norm >= n_anom:
        idx = rng.sample(range(n_norm), n_anom)
        X_pos = normal_codes[idx].numpy().astype(np.float32)
    else:
        # Fewer normals than anomalies — use all normals, note imbalance
        X_pos = normal_codes.numpy().astype(np.float32)

    X_neg = anomaly_codes.numpy().astype(np.float32)
    X     = np.concatenate([X_pos, X_neg], axis=0)
    y     = np.array([1] * len(X_pos) + [0] * len(X_neg), dtype=np.int32)
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def _fit_head(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    C: float = 1.0,
    seed: int = 42,
) -> tuple[Any, dict[str, float]]:
    """Fit one LogisticRegression head and return (model, metrics)."""
    clf = LogisticRegression(
        C=C,
        class_weight="balanced",
        max_iter=2000,
        solver="saga",
        n_jobs=-1,
        random_state=seed,
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return clf, {
        "accuracy":  float(accuracy_score(y_test, y_pred)),
        "f1":        float(f1_score(y_test, y_pred, zero_division=0)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_test, y_pred, zero_division=0)),
    }


def train_all_heads(
    vocabulary: dict[str, Any],
    anomaly_patches: list[dict[str, Any]],
    normal_codes: torch.Tensor,
    output_dir: str | Path,
    run_name: str = "cross_category",
    pos_percentile: float = 70.0,
    C: float = 1.0,
    test_size: float = 0.2,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Train K defect heads + 1 normality head, all on full 4096-dim SAE codes.

    Args:
        vocabulary: Dict with ``"concepts"`` list (each has ``"name"``,
            ``"atom_ids"``).
        anomaly_patches: List of patch dicts with ``"sae_code"`` (4096-dim).
        normal_codes: (M, 4096) tensor of normal patch SAE codes.
        output_dir: Where to write JSON metrics and pkl models.
        run_name: Prefix for output file names.
        pos_percentile: Per-atom activation percentile defining positive label
            (default 70 → top 30% patches per atom).
        C: LogisticRegression inverse regularisation (default 1.0).
        test_size: Fraction for test split (default 0.2).
        seed: Random seed.

    Returns:
        List of per-head result dicts (K defect + 1 normality).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    anomaly_codes = torch.stack([p["sae_code"] for p in anomaly_patches]).float()
    normal_codes  = normal_codes.float()

    n_anom, n_feat = anomaly_codes.shape
    n_norm = normal_codes.shape[0]
    n_concepts = len(vocabulary["concepts"])

    print(f"[concept_heads] Anomaly codes  : {n_anom} × {n_feat}")
    print(f"[concept_heads] Normal codes   : {n_norm} × {n_feat}")
    print(f"[concept_heads] Defect heads   : {n_concepts}")
    print(f"[concept_heads] C={C}  pos_pct={pos_percentile}  solver=saga\n")

    all_results: list[dict[str, Any]] = []
    all_models:  dict[str, Any] = {}

    # Store test-set probabilities for correlation analysis (anomaly patches only)
    anom_probs: dict[str, np.ndarray] = {}

    header = (f"{'Head':<38} {'N+':<7} {'N-':<7} "
              f"{'Acc':>7} {'F1':>7} {'Prec':>7} {'Rec':>7}")
    print(header)
    print("─" * len(header))

    # ── K defect heads ────────────────────────────────────────────────────────
    for concept in vocabulary["concepts"]:
        name = concept["name"]
        X, y = _build_defect_dataset(
            concept, anomaly_codes, normal_codes,
            pos_percentile=pos_percentile, seed=seed,
        )
        n_pos, n_neg = int(y.sum()), int((y == 0).sum())

        if n_pos < 10:
            print(f"  {name:<38} SKIP — only {n_pos} positives")
            continue

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=seed
        )
        clf, m = _fit_head(X_tr, y_tr, X_te, y_te, C=C, seed=seed)

        print(f"  {name:<38} {n_pos:<7} {n_neg:<7} "
              f"{m['accuracy']:>7.3f} {m['f1']:>7.3f} "
              f"{m['precision']:>7.3f} {m['recall']:>7.3f}")

        # Probabilities on the full anomaly set (for correlation)
        anom_probs[name] = clf.predict_proba(
            anomaly_codes.numpy().astype(np.float32)
        )[:, 1]

        all_results.append({"head": name, "type": "defect",
                             "n_pos": n_pos, "n_neg": n_neg,
                             **{k: round(v, 4) for k, v in m.items()}})
        all_models[name] = clf

    # ── Normality head ────────────────────────────────────────────────────────
    print()
    X_n, y_n = _build_normality_dataset(anomaly_codes, normal_codes, seed=seed)
    n_pos_n, n_neg_n = int(y_n.sum()), int((y_n == 0).sum())

    if n_norm < n_anom:
        print(f"  [NOTE] Only {n_norm} normal patches vs {n_anom} anomaly — "
              f"class_weight='balanced' compensates.")

    X_tr_n, X_te_n, y_tr_n, y_te_n = train_test_split(
        X_n, y_n, test_size=test_size, stratify=y_n, random_state=seed
    )
    clf_n, m_n = _fit_head(X_tr_n, y_tr_n, X_te_n, y_te_n, C=C, seed=seed)

    print(f"  {'normality':<38} {n_pos_n:<7} {n_neg_n:<7} "
          f"{m_n['accuracy']:>7.3f} {m_n['f1']:>7.3f} "
          f"{m_n['precision']:>7.3f} {m_n['recall']:>7.3f}")

    anom_probs["normality"] = clf_n.predict_proba(
        anomaly_codes.numpy().astype(np.float32)
    )[:, 1]

    all_results.append({"head": "normality", "type": "normality",
                        "n_pos": n_pos_n, "n_neg": n_neg_n,
                        **{k: round(v, 4) for k, v in m_n.items()}})
    all_models["normality"] = clf_n

    print("─" * len(header))
    f1s = [r["f1"] for r in all_results]
    print(f"\n  Mean F1 : {np.mean(f1s):.3f}  ±{np.std(f1s):.3f}")

    # ── Pairwise prediction correlation ───────────────────────────────────────
    print("\n  Pairwise correlation of head predictions on anomaly patches:")
    names = list(anom_probs.keys())
    mat   = np.stack([anom_probs[n] for n in names])          # (K+1, N_anom)
    corr  = np.corrcoef(mat)
    flagged = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            r = corr[i, j]
            flag = "  ← HIGH" if abs(r) > 0.95 else ""
            print(f"    {names[i]:<32} ↔ {names[j]:<32} r={r:+.3f}{flag}")
            if abs(r) > 0.95:
                flagged.append((names[i], names[j], round(r, 3)))

    if flagged:
        print(f"\n  [WARN] {len(flagged)} head pair(s) with |r| > 0.95 — "
              "consider merging those concepts.")
    else:
        print("\n  No head pairs with |r| > 0.95 — all heads are distinct.")

    # ── Save ─────────────────────────────────────────────────────────────────
    metrics_path = output_dir / f"{run_name}_concept_heads.json"
    models_path  = output_dir / f"{run_name}_concept_heads.pkl"

    with open(metrics_path, "w") as f:
        json.dump(all_results, f, indent=2)
    with open(models_path, "wb") as f:
        pickle.dump(all_models, f)

    print(f"\n  Metrics → {metrics_path}")
    print(f"  Models  → {models_path}")
    return all_results


# ── Backwards-compatible alias ────────────────────────────────────────────────
def train_concept_heads(vocabulary, anomaly_patches, normal_codes,
                        output_dir, **kwargs):
    """Alias kept for compatibility with existing scripts."""
    run_name = kwargs.pop("run_name", "cross_category")
    return train_all_heads(vocabulary, anomaly_patches, normal_codes,
                           output_dir, run_name=run_name, **kwargs)
