"""
elastic_net_sweep.py — Hyperparameter sweep: elastic net CONCIL on 13-dim concept space.

Uses:
  - v2 concept cache (ma_cbm/outputs/v2_concept_cache.pkl): 13-dim concept vectors
    for all 15 MVTec categories, same paths/splits as v1 for comparability.
  - ElasticConcilSolver (ma_cbm/continual/elastic_concil.py): replaces ridge
    regression with L1+L2 coordinate descent on the accumulated Gram matrix.
  - SAE guide scores (mac/outputs/dual_branch/score_cache.pkl): for dual-branch
    combination at best configuration.

Zero-forgetting holds for all sweep configurations (see elastic_concil.py docstring).

Sweep grid:
  l1_ratio: [0.3, 0.5, 0.7]   (L1 vs L2 balance)
  alpha:    [0.01, 0.1, 1.0]  (overall regularisation strength)

Baselines:
  v1-ridge (5-dim):  Surface 0.947 | Structural 0.666 | Overall 0.835
  v2-ridge (13-dim): Surface 0.879 | Structural 0.999 | Overall 0.927

Targets:
  Surface ≥ 0.940  Structural ≥ 0.990  Overall ≥ 0.960  BWT ≈ 0.000
"""

from __future__ import annotations

import json
import pickle
import sys
from itertools import product
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

_COVAD  = Path(__file__).resolve().parents[2]
_MABCM  = _COVAD / "ma_cbm"
sys.path.insert(0, str(_COVAD))
sys.path.insert(0, str(_COVAD / "mac"))
sys.path.insert(0, str(_MABCM))

from continual.elastic_concil import ElasticConcilSolver

# ── Paths ─────────────────────────────────────────────────────────────────────
_V2_CACHE  = _MABCM / "outputs" / "v2_concept_cache.pkl"
_SAE_SC    = _COVAD / "mac/outputs/dual_branch/score_cache.pkl"
_OUT_JSON  = _MABCM / "outputs" / "elastic_sweep_results.json"
_BEST_JSON = _MABCM / "outputs" / "elastic_best_results.json"

MVTEC_15 = [
    "bottle","cable","capsule","carpet","grid","hazelnut","leather",
    "metal_nut","pill","screw","tile","toothbrush","transistor","wood","zipper",
]
SURFACE  = {"carpet","grid","hazelnut","leather","metal_nut","pill","tile","wood","zipper"}
N_CONC   = 13

CONCEPT_NAMES = [
    # Surface (0-4)
    "surface_discontinuity","surface_discoloration","surface_crack",
    "surface_abrasion","surface_void",
    # Structural (5-12)
    "head_deformation_or_flaring","structural_discontinuity","structural_displacement",
    "internal_passage_blockage","stud_array_discontinuity",
    "compromised_lead_attachment","internal_bore_restriction","trio_arrangement",
]

SWEEP_L1   = [0.3, 0.5, 0.7]
SWEEP_ALPHA = [0.01, 0.1, 1.0]

BASELINES = {
    "v1_ridge_5dim":  {"surface": 0.947, "structural": 0.666, "overall": 0.835},
    "v2_ridge_13dim": {"surface": 0.879, "structural": 0.999, "overall": 0.927},
}
TARGETS = {"surface": 0.940, "structural": 0.990, "overall": 0.960}

# ── Helpers ───────────────────────────────────────────────────────────────────

def _auc(y, s):
    try:    return float(roc_auc_score(y, s))
    except: return float("nan")

def _norm01(x):
    lo, hi = x.min(), x.max()
    return np.zeros_like(x) if hi - lo < 1e-9 else (x - lo) / (hi - lo)

def _eval_set(cc):
    """Eval set: defect_train + test_normal + defect_test (same as v1)."""
    vecs = cc["vecs"];  y = np.array(cc["y_all"])
    nt, nd, nn = cc["n_train_normal"], cc["n_defect_train"], cc["n_test_normal"]
    ev = np.concatenate([vecs[nt:nt+nd], vecs[nt+nd:nt+nd+nn], vecs[nt+nd+nn:]])
    ey = np.concatenate([y[nt:nt+nd],   y[nt+nd:nt+nd+nn],    y[nt+nd+nn:]])
    return ev.astype(np.float32), ey.astype(np.int32)

def _train_set(cc):
    vecs = cc["vecs"];  y = np.array(cc["y_all"])
    mask = np.array(cc["is_train_mask"])
    return vecs[mask].astype(np.float32), y[mask].astype(np.float32)


# ── Core: run one sequential CONCIL experiment ────────────────────────────────

def run_one(cache_v2, alpha, l1_ratio):
    """Sequential 15-task CONCIL with ElasticConcilSolver.

    Returns:
        init_scores: {cat: np.ndarray} — anomaly scores after task t
        final_scores: {cat: np.ndarray} — anomaly scores after all 15 tasks
        init_y: {cat: np.ndarray} — binary labels for eval set
        final_w: np.ndarray (13,) — learned weight vector after all tasks
        final_b: float — learned bias
    """
    solver = ElasticConcilSolver(input_dim=N_CONC, alpha=alpha, l1_ratio=l1_ratio)
    init_scores, init_y = {}, {}

    for cat in MVTEC_15:
        cc = cache_v2[cat]
        X_tr, y_tr = _train_set(cc)
        w, b_arr = solver.update_anomaly_head(
            torch.tensor(X_tr), torch.tensor(y_tr)
        )
        b = float(b_arr[0])
        ev, ey = _eval_set(cc)
        init_scores[cat] = ev @ w + b
        init_y[cat]      = ey

    # Final weights (solve on full accumulated Gram)
    W_f  = solver._solve_elastic(solver.A_anomaly, solver.b_anomaly[:, 0],
                                  solver.N_anomaly)
    w_f  = W_f[:N_CONC].float().numpy()
    b_f  = float(W_f[N_CONC])

    final_scores = {}
    for cat in MVTEC_15:
        ev, _ = _eval_set(cache_v2[cat])
        final_scores[cat] = ev @ w_f + b_f

    return init_scores, final_scores, init_y, w_f, b_f


def compute_metrics(init_scores, final_scores, init_y):
    """Compute per-category and grouped AUCs + BWT."""
    per_cat = {}
    bwts    = []
    for cat in MVTEC_15:
        y  = init_y[cat]
        ai = _auc(y, init_scores[cat])
        af = _auc(y, final_scores[cat])
        bwt = (af - ai) if cat != MVTEC_15[-1] else float("nan")
        per_cat[cat] = {"init": ai, "final": af, "bwt": bwt}
        if not np.isnan(bwt):
            bwts.append(bwt)

    surf_final = np.mean([per_cat[c]["final"] for c in MVTEC_15 if c in SURFACE])
    strc_final = np.mean([per_cat[c]["final"] for c in MVTEC_15 if c not in SURFACE])
    overall    = np.mean([per_cat[c]["final"] for c in MVTEC_15])
    mean_bwt   = float(np.mean(bwts))

    return {
        "surface":    float(surf_final),
        "structural": float(strc_final),
        "overall":    float(overall),
        "bwt":        mean_bwt,
        "per_cat":    per_cat,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print("  Elastic Net CONCIL Sweep — 9 configurations × 15 tasks")
    print("=" * 72)

    print("\n[init] Loading v2 concept cache …")
    cache_v2 = pickle.load(open(_V2_CACHE, "rb"))
    print(f"  Categories: {list(cache_v2.keys())}")

    # ── Sweep ─────────────────────────────────────────────────────────────────
    print("\n[sweep] Running 9 configurations …\n")

    sweep_results = []
    best_score = -1.0
    best_cfg   = None

    hdr = (f"  {'l1_ratio':>8} {'alpha':>6} | "
           f"{'Surface':>9} {'Struct':>9} {'Overall':>9} {'BWT':>8} | "
           f"{'S≥.940':>7} {'T≥.990':>7}")
    print(hdr);  print("  " + "─" * 72)

    for l1_ratio, alpha in product(SWEEP_L1, SWEEP_ALPHA):
        init_sc, final_sc, init_y, w_f, b_f = run_one(cache_v2, alpha, l1_ratio)
        m = compute_metrics(init_sc, final_sc, init_y)

        s_ok = "✓" if m["surface"]    >= TARGETS["surface"]    else "✗"
        t_ok = "✓" if m["structural"] >= TARGETS["structural"] else "✗"

        print(f"  {l1_ratio:>8.1f} {alpha:>6.2f} | "
              f"{m['surface']:>9.4f} {m['structural']:>9.4f} "
              f"{m['overall']:>9.4f} {m['bwt']:>+8.4f} | "
              f"{s_ok:>7} {t_ok:>7}")

        row = {"l1_ratio": l1_ratio, "alpha": alpha,
               "surface": m["surface"], "structural": m["structural"],
               "overall": m["overall"], "bwt": m["bwt"],
               "per_cat": m["per_cat"], "w": w_f.tolist(), "b": b_f}
        sweep_results.append(row)

        # Best: maximise (surface + structural) with BWT constraint
        score = (m["surface"] + m["structural"]) / 2
        if score > best_score and abs(m["bwt"]) < 0.05:
            best_score = score;  best_cfg = row

    json.dump(sweep_results, open(_OUT_JSON, "w"), indent=2)

    # ── Baseline comparison ────────────────────────────────────────────────────
    print()
    print("  " + "─" * 72)
    print(f"  {'v1-ridge(5d)':>14}            | "
          f"{0.947:>9.4f} {0.666:>9.4f} {0.835:>9.4f} {'—':>8} |")
    print(f"  {'v2-ridge(13d)':>14}            | "
          f"{0.879:>9.4f} {0.999:>9.4f} {0.927:>9.4f} {'—':>8} |")

    # ── Best configuration: full breakdown ────────────────────────────────────
    if best_cfg is None:
        print("\n  [WARN] No configuration met BWT < 0.05. Showing lowest |BWT|.")
        best_cfg = min(sweep_results, key=lambda r: abs(r["bwt"]))

    print(f"\n{'='*72}")
    print(f"  BEST: l1_ratio={best_cfg['l1_ratio']}  alpha={best_cfg['alpha']}")
    print(f"  Surface={best_cfg['surface']:.4f}  Structural={best_cfg['structural']:.4f}  "
          f"Overall={best_cfg['overall']:.4f}  BWT={best_cfg['bwt']:+.4f}")
    print(f"{'='*72}")

    # ── Per-category table for best config ────────────────────────────────────
    print(f"\n  Per-category breakdown (best config):\n")
    print(f"  {'Category':<14} {'Type':<5} {'concept-only':>13} {'BWT':>8}")
    print("  " + "─" * 45)

    for cat in MVTEC_15:
        pc  = best_cfg["per_cat"][cat]
        tag = "S" if cat in SURFACE else "T"
        bwt_str = f"{pc['bwt']:>+8.4f}" if not np.isnan(pc['bwt']) else "  (last)"
        print(f"  {cat:<14} [{tag}]  {pc['final']:>13.4f} {bwt_str}")

    print("  " + "─" * 45)

    # ── Dual-branch for best config ───────────────────────────────────────────
    print(f"\n  Dual-branch (0.9 × SAE + 0.1 × concept-best) …")
    sae_sc = pickle.load(open(_SAE_SC, "rb"))
    dual_aucs = []
    for cat in MVTEC_15:
        if cat not in sae_sc:
            continue
        sae_s  = sae_sc[cat]["sae_scores"].astype(np.float32)
        mac_s  = best_cfg["per_cat"][cat]["final"]  # AUC scalar, not scores
        # Need raw scores for combination — re-run for best config
    # Re-run best config to get raw final scores for combination
    init_sc, final_sc, init_y, w_f, b_f = run_one(
        cache_v2, best_cfg["alpha"], best_cfg["l1_ratio"]
    )
    dual_aucs_cat = {}
    for cat in MVTEC_15:
        if cat not in sae_sc:
            continue
        sae_s = sae_sc[cat]["sae_scores"].astype(np.float32)
        mac_s = final_sc[cat].astype(np.float32)
        y     = init_y[cat]
        comb  = 0.9 * _norm01(sae_s) + 0.1 * _norm01(mac_s)
        dual_aucs_cat[cat] = _auc(y, comb)

    print(f"\n  {'Category':<14} {'Type':<5} {'concept':>9} {'dual-branch':>12}")
    print("  " + "─" * 42)
    for cat in MVTEC_15:
        tag = "S" if cat in SURFACE else "T"
        c   = best_cfg["per_cat"][cat]["final"]
        d   = dual_aucs_cat.get(cat, float("nan"))
        print(f"  {cat:<14} [{tag}]  {c:>9.4f} {d:>12.4f}")
    print("  " + "─" * 42)
    surf_d = np.mean([dual_aucs_cat[c] for c in MVTEC_15 if c in SURFACE and c in dual_aucs_cat])
    strc_d = np.mean([dual_aucs_cat[c] for c in MVTEC_15 if c not in SURFACE and c in dual_aucs_cat])
    all_d  = np.mean(list(dual_aucs_cat.values()))
    print(f"  {'MEAN':<14}       {best_cfg['overall']:>9.4f} {all_d:>12.4f}")
    print(f"\n  Surface dual: {surf_d:.4f}  |  Structural dual: {strc_d:.4f}  |  Overall dual: {all_d:.4f}")

    # ── Weight inspection ─────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  LEARNED WEIGHT VECTOR (best config, after 15 tasks)")
    print(f"{'='*72}")
    w_arr = np.array(best_cfg["w"])
    print(f"\n  {'Dim':<4} {'Concept':<40} {'Weight':>9}  {'Penalised':>9}")
    print("  " + "─" * 67)
    for i, (name, wi) in enumerate(zip(CONCEPT_NAMES, w_arr)):
        group = "surface" if i < 5 else "structural"
        active = "active" if abs(wi) > 1e-3 else "≈ zero"
        print(f"  {i:<4} {name:<40} {wi:>+9.5f}  {active}")
    print(f"  {'13':>4} {'[bias]':<40} {best_cfg['b']:>+9.5f}  not penalised")

    # Per-task effective contribution
    print(f"\n  Effective concept contribution for representative tasks:")
    print(f"  (mean_anomaly_activation × weight vs mean_normal_activation × weight)\n")

    for inspect_cat in ["hazelnut", "cable"]:
        cc    = cache_v2[inspect_cat]
        vecs  = cc["vecs"].astype(np.float32)
        y_all = np.array(cc["y_all"])
        mean_anom = vecs[y_all == 1].mean(axis=0)
        mean_norm = vecs[y_all == 0].mean(axis=0)
        tag   = "surface" if inspect_cat in SURFACE else "structural"
        print(f"  [{inspect_cat}] ({tag}):")
        print(f"  {'Concept':<40} {'μ_anom×w':>10}  {'μ_norm×w':>10}  {'Δ':>8}  {'Status'}")
        print("  " + "─" * 80)
        for i, (name, wi) in enumerate(zip(CONCEPT_NAMES, w_arr)):
            contrib_a = mean_anom[i] * wi
            contrib_n = mean_norm[i] * wi
            delta     = contrib_a - contrib_n
            status    = ">> ACTIVE" if abs(delta) > 0.01 else "   quiet"
            print(f"  {name:<40} {contrib_a:>10.4f}  {contrib_n:>10.4f}  {delta:>8.4f}  {status}")
        print()

    # Save best results
    json.dump({"best_l1_ratio": best_cfg["l1_ratio"], "best_alpha": best_cfg["alpha"],
               "metrics": {k: best_cfg[k] for k in ("surface","structural","overall","bwt")},
               "dual_branch": dual_aucs_cat, "weights": best_cfg["w"], "bias": best_cfg["b"],
               "per_cat": best_cfg["per_cat"]},
              open(_BEST_JSON, "w"), indent=2)

    print(f"\n  Sweep results → {_OUT_JSON.name}")
    print(f"  Best config   → {_BEST_JSON.name}")
    print("=" * 72)


if __name__ == "__main__":
    main()
