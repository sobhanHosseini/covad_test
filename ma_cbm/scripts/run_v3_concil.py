"""
run_v3_concil.py — MA-CBM v3: patch-level surface + all-category image-level concepts.

Unified concept vector per image:
  [0:5]   5 surface concepts  (SAE patch-level heads — from v2 cache)
  [5:5+K] K image-level concepts (DINOv2 max-pool heads — all 15 categories)

The first 5 dimensions are reused directly from the v2 concept cache (no SAE re-run).
The K image-level dimensions come from the allcat_maxpool_cache (no DINOv2 re-run).

Sequential CONCIL experiment: 15 tasks, standard ridge regression (unchanged).
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

_COVAD = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_COVAD))
sys.path.insert(0, str(_COVAD / "mac"))
from solvers.concil import ConcilSolver

_OUT      = Path(__file__).resolve().parent.parent / "outputs"
_V2_CACHE = _OUT / "v2_concept_cache.pkl"
_MP_CACHE = _OUT / "allcat_maxpool_cache.pkl"
_HEADS    = _OUT / "allcat_image_level_heads.pkl"
_VLM_JSON = _OUT / "image_level_vlm_results_allcat.json"
_SAE_SC   = _COVAD / "mac/outputs/dual_branch/score_cache.pkl"
_RES_JSON = _OUT / "v3_results.json"

MVTEC_15 = [
    "bottle","cable","capsule","carpet","grid","hazelnut","leather",
    "metal_nut","pill","screw","tile","toothbrush","transistor","wood","zipper",
]
SURFACE = {"carpet","grid","hazelnut","leather","metal_nut","pill","tile","wood","zipper"}
N_SURF  = 5   # patch-level surface heads (dims 0-4 from v2 cache)

V1_CONCEPT = {
    "bottle":0.669,"cable":0.562,"capsule":0.702,"carpet":0.947,"grid":1.000,
    "hazelnut":0.932,"leather":0.983,"metal_nut":0.883,"pill":0.915,"screw":0.690,
    "tile":0.986,"toothbrush":0.703,"transistor":0.672,"wood":0.887,"zipper":0.994,
}
V2_CONCEPT = {
    "bottle":1.000,"cable":1.000,"capsule":0.997,"carpet":0.922,"grid":1.000,
    "hazelnut":0.773,"leather":0.949,"metal_nut":0.821,"pill":0.777,"screw":1.000,
    "tile":0.900,"toothbrush":1.000,"transistor":1.000,"wood":0.804,"zipper":0.968,
}
V2_DUAL = {
    "bottle":1.000,"cable":0.987,"capsule":0.964,"carpet":0.995,"grid":1.000,
    "hazelnut":0.992,"leather":1.000,"metal_nut":0.995,"pill":0.972,"screw":0.982,
    "tile":1.000,"toothbrush":0.964,"transistor":0.979,"wood":0.978,"zipper":0.997,
}


def _auc(y, s):
    try:    return float(roc_auc_score(y, s))
    except: return float("nan")

def _norm01(x):
    lo, hi = x.min(), x.max()
    return np.zeros_like(x) if hi-lo < 1e-9 else (x-lo)/(hi-lo)


def build_v3_vecs(v2_cache, mp_cache, image_heads, head_names):
    """Build 5+K concept vectors per image for all 15 categories.

    Surface dims (0-4): taken directly from v2 cache (no re-computation).
    Image-level dims (5-5+K): apply image-level LogReg heads to max-pool features.
    """
    K = len(head_names)
    print(f"\n[build] Building {N_SURF + K}-dim concept vectors …")
    cache_v3 = {}

    for cat in MVTEC_15:
        v2  = v2_cache[cat]
        mp  = mp_cache[cat]
        N   = len(v2["all_paths"])

        # Surface dims (0-4) — reuse from v2 cache directly
        surf_vecs = v2["vecs"][:, :N_SURF].astype(np.float32)   # (N, 5)

        # Image-level dims (5-5+K)
        mp_feats = mp["maxpool_feats"].astype(np.float32)        # (N, 1024)
        img_vecs = np.zeros((N, K), dtype=np.float32)
        for k, name in enumerate(head_names):
            clf = image_heads.get(name)
            if clf is not None:
                img_vecs[:, k] = clf.predict_proba(mp_feats)[:, 1]

        vecs = np.concatenate([surf_vecs, img_vecs], axis=1)    # (N, 5+K)

        cache_v3[cat] = {
            "all_paths":    v2["all_paths"],
            "y_all":        v2["y_all"],
            "is_train_mask": v2["is_train_mask"],
            "vecs":         vecs,
            "n_train_normal":  v2["n_train_normal"],
            "n_defect_train":  v2["n_defect_train"],
            "n_test_normal":   v2["n_test_normal"],
            "n_defect_test":   v2["n_defect_test"],
        }
        print(f"  [{cat}]  {N} images  →  {vecs.shape[1]}-dim concept vector")

    return cache_v3


def _eval_set(cc):
    vecs = cc["vecs"];  y = np.array(cc["y_all"])
    nt, nd, nn = cc["n_train_normal"], cc["n_defect_train"], cc["n_test_normal"]
    ev = np.concatenate([vecs[nt:nt+nd], vecs[nt+nd:nt+nd+nn], vecs[nt+nd+nn:]])
    ey = np.concatenate([y[nt:nt+nd],   y[nt+nd:nt+nd+nn],    y[nt+nd+nn:]])
    return ev.astype(np.float32), ey.astype(np.int32)

def _train_set(cc):
    vecs = cc["vecs"];  y = np.array(cc["y_all"])
    mask = np.array(cc["is_train_mask"])
    return vecs[mask].astype(np.float32), y[mask].astype(np.float32)


def run_concil(cache_v3, N_CONC):
    solver = ConcilSolver(input_dim=N_CONC, lambda_anomaly=1e-4)
    init_scores, init_y = {}, {}

    print(f"\n[CONCIL] Sequential 15-task experiment ({N_CONC}-dim concept space) …\n")
    hdr = f"  {'T':<3} {'Category':<14} {'I-AUC':>8}"
    print(hdr);  print("  " + "─" * 30)

    for t, cat in enumerate(MVTEC_15):
        cc = cache_v3[cat]
        X_tr, y_tr = _train_set(cc)
        w, b_arr = solver.update_anomaly_head(
            torch.tensor(X_tr), torch.tensor(y_tr)
        )
        b = float(b_arr[0])
        ev, ey = _eval_set(cc)
        sc = ev @ w + b
        init_scores[cat] = sc;  init_y[cat] = ey
        print(f"  T{t+1:<2} {cat:<14} {_auc(ey, sc):>8.4f}")

    W_f = solver._solve(solver.A_anomaly, solver.b_anomaly[:, 0], solver.lambda_a)
    # _solve returns 1-D tensor when b is 1-D
    w_f = W_f[:N_CONC].float().numpy();  b_f = float(W_f[N_CONC])

    print("\n[CONCIL] Final scoring (after T15) …")
    final_scores = {}
    for cat in MVTEC_15:
        ev, _ = _eval_set(cache_v3[cat])
        final_scores[cat] = ev @ w_f + b_f

    return init_scores, final_scores, init_y


def main():
    print("=" * 72)
    print("  MA-CBM v3 — patch-level (5) + image-level (K) concept heads")
    print("=" * 72)

    print("\n[init] Loading caches and heads …")
    v2_cache    = pickle.load(open(_V2_CACHE, "rb"))
    mp_cache    = pickle.load(open(_MP_CACHE, "rb"))
    image_heads = pickle.load(open(_HEADS, "rb"))
    vlm_results = json.load(open(_VLM_JSON))

    # Build ordered head name list (by cluster_id)
    head_names = [
        next(r["concept_name"] for r in vlm_results if r["cluster_id"] == k)
        for k in range(len(vlm_results))
    ]
    usable = [n for n in head_names if image_heads.get(n) is not None]
    K = len(usable)
    print(f"  Surface heads (patch-level, SAE)    : {N_SURF}")
    print(f"  Image-level heads (max-pool, all-cat): {K}")
    print(f"  Total concept dims: {N_SURF + K}")

    # Build v3 concept cache
    cache_v3 = build_v3_vecs(v2_cache, mp_cache, image_heads, head_names)
    N_CONC   = N_SURF + K

    # Sequential CONCIL
    init_scores, final_scores, init_y = run_concil(cache_v3, N_CONC)

    # Dual-branch
    print("\n[dual] 0.9 × SAE + 0.1 × v3-concept …")
    sae_sc = pickle.load(open(_SAE_SC, "rb"))
    dual = {}
    for cat in MVTEC_15:
        if cat not in sae_sc:
            continue
        sae_s  = sae_sc[cat]["sae_scores"].astype(np.float32)
        mac_s  = final_scores[cat].astype(np.float32)
        y      = init_y[cat]
        comb   = 0.9 * _norm01(sae_s) + 0.1 * _norm01(mac_s)
        dual[cat] = _auc(y, comb)

    # ── Comparison table ──────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("  COMPARISON TABLE — v1 vs v2 vs v3")
    print(f"{'='*80}")
    print(f"\n  {'Category':<13} {'Type':<5} {'v1-conc':>8} {'v2-conc':>8} "
          f"{'v3-conc':>8} {'Δv3-v1':>7} | {'v2-dual':>8} {'v3-dual':>8} | {'BWT-v3':>8}")
    print("  " + "─" * 82)

    bwts, bwts_s, bwts_t = [], [], []
    v3c_s, v3c_t = [], []
    v3d_s, v3d_t = [], []
    results = {}

    for cat in MVTEC_15:
        y    = init_y[cat]
        v3ci = _auc(y, init_scores[cat])
        v3cf = _auc(y, final_scores[cat])
        bwt  = (v3cf - v3ci) if cat != MVTEC_15[-1] else float("nan")
        tag  = "S" if cat in SURFACE else "T"
        v1c  = V1_CONCEPT.get(cat, float("nan"))
        v2c  = V2_CONCEPT.get(cat, float("nan"))
        v2d  = V2_DUAL.get(cat, float("nan"))
        v3d  = dual.get(cat, float("nan"))
        dv   = v3cf - v1c

        bwt_str = f"{bwt:>+8.4f}" if not np.isnan(bwt) else "  (last)"
        print(f"  {cat:<13} [{tag}]  {v1c:>8.3f} {v2c:>8.3f} {v3cf:>8.3f} "
              f"{dv:>+7.3f} | {v2d:>8.3f} {v3d:>8.3f} | {bwt_str}")

        results[cat] = {"v3_concept": v3cf, "v3_dual": v3d, "bwt": bwt, "type": tag}
        if not np.isnan(bwt):
            bwts.append(bwt)
            (bwts_s if cat in SURFACE else bwts_t).append(bwt)
        (v3c_s if cat in SURFACE else v3c_t).append(v3cf)
        (v3d_s if cat in SURFACE else v3d_t).append(v3d)

    print("  " + "─" * 82)
    v3c_all = [results[c]["v3_concept"] for c in MVTEC_15]
    v3d_all = [dual.get(c, float("nan")) for c in MVTEC_15]
    v1_all  = list(V1_CONCEPT.values())
    print(f"  {'MEAN':<13}       {np.mean(v1_all):>8.3f} "
          f"{np.mean(list(V2_CONCEPT.values())):>8.3f} "
          f"{np.mean(v3c_all):>8.3f} "
          f"{np.mean(v3c_all)-np.mean(v1_all):>+7.3f} | "
          f"{np.mean(list(V2_DUAL.values())):>8.3f} "
          f"{np.nanmean(v3d_all):>8.3f} | "
          f"{np.mean(bwts):>+8.4f}")

    print(f"\n  {'─'*60}")
    print(f"  Group breakdown:")
    surf_cats = [c for c in MVTEC_15 if c in SURFACE]
    strc_cats = [c for c in MVTEC_15 if c not in SURFACE]
    print(f"  Surface    [S]: v1={np.mean([V1_CONCEPT[c] for c in surf_cats]):.3f}  "
          f"v2={np.mean([V2_CONCEPT[c] for c in surf_cats]):.3f}  "
          f"v3={np.mean(v3c_s):.3f}  BWT={np.mean(bwts_s):+.4f} | "
          f"v2d={np.mean([V2_DUAL[c] for c in surf_cats]):.3f}  v3d={np.mean(v3d_s):.3f}")
    print(f"  Structural [T]: v1={np.mean([V1_CONCEPT[c] for c in strc_cats]):.3f}  "
          f"v2={np.mean([V2_CONCEPT[c] for c in strc_cats]):.3f}  "
          f"v3={np.mean(v3c_t):.3f}  BWT={np.mean(bwts_t):+.4f} | "
          f"v2d={np.mean([V2_DUAL[c] for c in strc_cats]):.3f}  v3d={np.mean(v3d_t):.3f}")

    # Target check
    tgt_surf = "✓" if np.mean(v3c_s) >= 0.940 else "✗"
    tgt_strc = "✓" if np.mean(v3c_t) >= 0.990 else "✗"
    tgt_all  = "✓" if np.mean(v3c_all) >= 0.960 else "✗"
    tgt_bwt  = "✓" if abs(np.mean(bwts)) < 0.02 else "✗"
    print(f"\n  Targets:")
    print(f"    Surface    ≥ 0.940 : {np.mean(v3c_s):.3f} {tgt_surf}")
    print(f"    Structural ≥ 0.990 : {np.mean(v3c_t):.3f} {tgt_strc}")
    print(f"    Overall    ≥ 0.960 : {np.mean(v3c_all):.3f} {tgt_all}")
    print(f"    BWT        ≈ 0.000 : {np.mean(bwts):+.4f} {tgt_bwt}")
    print("=" * 80)

    json.dump({"results": {c: results[c] for c in MVTEC_15},
               "summary": {"surface": np.mean(v3c_s), "structural": np.mean(v3c_t),
                            "overall": np.mean(v3c_all), "bwt": np.mean(bwts),
                            "dual_overall": np.nanmean(v3d_all)}},
              open(_RES_JSON, "w"), indent=2)
    print(f"\n  Results → {_RES_JSON.name}")


if __name__ == "__main__":
    main()
