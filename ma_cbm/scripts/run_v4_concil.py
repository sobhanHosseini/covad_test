"""
run_v4_concil.py — MA-CBM v4: defect concepts + normality concepts.

Full concept vector per image:
  [0:5]         5 patch-level defect concepts (surface SAE heads, from v2 cache)
  [5:5+D]       D image-level defect concepts (max-pool all-cat heads v2, any splits from Part 1)
  [5+D:5+D+P]   P normality patch concepts (SAE space, from Part 2A)
  [5+D+P:]      M normality image concepts (max-pool space, from Part 2B)

Normality heads output HIGH for normal images, LOW for anomaly.
CONCIL anomaly head learns: w_defect>0, w_normality<0.
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

_COVAD  = Path(__file__).resolve().parents[2]
_MABCM  = _COVAD / "ma_cbm"
_OUT    = _MABCM / "outputs"
sys.path.insert(0, str(_COVAD))
sys.path.insert(0, str(_COVAD / "mac"))

from features.sae import SparseAutoencoder
from solvers.concil import ConcilSolver

# ── Paths ─────────────────────────────────────────────────────────────────────
_V2_CACHE        = _OUT / "v2_concept_cache.pkl"
_MP_CACHE        = _OUT / "allcat_maxpool_cache.pkl"
_SAE_WEIGHTS     = _COVAD / "sae_training/sae_vitl14reg_C4096_k64.pt"

# Part 1 outputs (use v2 if v2 doesn't exist)
_IMG_HEADS_V2    = _OUT / "allcat_image_level_heads_v2.pkl"
_IMG_HEADS_ORIG  = _OUT / "allcat_image_level_heads.pkl"
_VLM_V2          = _OUT / "image_level_vlm_results_allcat_v2.json"
_VLM_ORIG        = _OUT / "image_level_vlm_results_allcat.json"

# Part 2 outputs
_NORM_PATCH_HEADS = _OUT / "norm_patch_concept_heads.pkl"
_NORM_IMG_HEADS   = _OUT / "norm_image_concept_heads.pkl"
_NORM_PATCH_VOCAB = _OUT / "norm_patch_vocabulary.json"
_NORM_IMG_VLM     = _OUT / "norm_image_vlm_results.json"

_SAE_SC          = _COVAD / "mac/outputs/dual_branch/score_cache.pkl"
_RESULTS_JSON    = _OUT / "v4_results.json"

MVTEC_15 = [
    "bottle","cable","capsule","carpet","grid","hazelnut","leather",
    "metal_nut","pill","screw","tile","toothbrush","transistor","wood","zipper",
]
SURFACE = {"carpet","grid","hazelnut","leather","metal_nut","pill","tile","wood","zipper"}
N_SURF  = 5

# v3 baselines for comparison
V3_CONCEPT = {
    "bottle":1.000,"cable":0.998,"capsule":1.000,"carpet":1.000,"grid":1.000,
    "hazelnut":1.000,"leather":1.000,"metal_nut":0.994,"pill":1.000,"screw":1.000,
    "tile":1.000,"toothbrush":0.989,"transistor":0.979,"wood":1.000,"zipper":1.000,
}
V3_DUAL = {
    "bottle":1.000,"cable":0.981,"capsule":0.975,"carpet":0.998,"grid":1.000,
    "hazelnut":0.999,"leather":1.000,"metal_nut":0.998,"pill":0.995,"screw":0.985,
    "tile":1.000,"toothbrush":0.992,"transistor":0.990,"wood":0.996,"zipper":1.000,
}


def _auc(y, s):
    try:    return float(roc_auc_score(y, s))
    except: return float("nan")

def _norm01(x):
    lo, hi = x.min(), x.max()
    return np.zeros_like(x) if hi-lo<1e-9 else (x-lo)/(hi-lo)

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


def build_v4_cache(v2_cache, mp_cache, img_heads, img_head_names,
                   norm_patch_heads, norm_img_heads, sae):
    """Build the full multi-dim concept vector for all 15 categories."""

    D = len(img_head_names)
    P = len(norm_patch_heads)
    M = len(norm_img_heads)
    N_CONC = N_SURF + D + P + M
    print(f"\n[build] Concept vector: {N_SURF} surf + {D} img-defect + "
          f"{P} norm-patch + {M} norm-img = {N_CONC} total dims")

    # Norm patch head names (ordered)
    norm_patch_names = list(norm_patch_heads.keys())
    norm_img_names   = list(norm_img_heads.keys())

    sae_dev = next(sae.parameters()).device
    cache_v4 = {}

    for cat in MVTEC_15:
        v2 = v2_cache[cat]
        mp = mp_cache[cat]
        N  = len(v2["all_paths"])

        # Surface dims (0-4): reuse from v2 cache
        surf_vecs = v2["vecs"][:, :N_SURF].astype(np.float32)   # (N, 5)

        # Defect image dims (5-5+D): apply image-level defect heads to max-pool
        mp_feats = mp["maxpool_feats"].astype(np.float32)        # (N, 1024)
        img_vecs = np.zeros((N, D), dtype=np.float32)
        for k, name in enumerate(img_head_names):
            clf = img_heads.get(name)
            if clf is not None:
                img_vecs[:, k] = clf.predict_proba(mp_feats)[:, 1]

        # Normality patch dims (5+D : 5+D+P): apply norm patch heads to SAE codes
        # Need SAE codes per image → re-extract from patches
        # Use batch DINOv2+SAE: but we only have max-pool cache, not patch cache
        # For efficiency: use the existing v2 cache's SAE-derived surface dims as proxy
        # OR: use the precomputed normal tensor approach
        # Since we don't have per-image patch SAE codes cached for all 15 cats,
        # we approximate normality patch heads using the max-pool features as input
        # (the normality patch heads were trained on 4096-dim SAE codes, so we
        # need proper SAE inference — we skip if patch heads require it)
        # Instead, include normality image heads applied to max-pool feats as a proxy
        # for patch-level normality. A TODO for future work.

        # Normality image dims: apply normality image heads to max-pool feats
        norm_img_vecs = np.zeros((N, M), dtype=np.float32)
        for k, name in enumerate(norm_img_names):
            clf = norm_img_heads.get(name)
            if clf is not None:
                norm_img_vecs[:, k] = clf.predict_proba(mp_feats)[:, 1]

        # Assemble: skip norm_patch for now (needs per-image SAE codes)
        # Full vector = [surf(5) | img_defect(D) | norm_img(M)]
        vecs = np.concatenate([surf_vecs, img_vecs, norm_img_vecs], axis=1)

        cache_v4[cat] = {
            "all_paths":    v2["all_paths"],
            "y_all":        v2["y_all"],
            "is_train_mask": v2["is_train_mask"],
            "vecs":         vecs,
            "n_train_normal":  v2["n_train_normal"],
            "n_defect_train":  v2["n_defect_train"],
            "n_test_normal":   v2["n_test_normal"],
            "n_defect_test":   v2["n_defect_test"],
        }
        print(f"  [{cat}] {N} images → {vecs.shape[1]}-dim")

    return cache_v4, N_CONC - P  # actual dims (without norm_patch)


def run_concil(cache_v4, N_CONC):
    solver = ConcilSolver(input_dim=N_CONC, lambda_anomaly=1e-4)
    init_scores, init_y = {}, {}

    print(f"\n[CONCIL] Sequential 15-task ({N_CONC}-dim concept space) …\n")
    hdr = f"  {'T':<3} {'Category':<14} {'I-AUC':>8}"
    print(hdr);  print("  "+"─"*30)

    for t, cat in enumerate(MVTEC_15):
        cc = cache_v4[cat]
        X_tr, y_tr = _train_set(cc)
        w, b_arr = solver.update_anomaly_head(torch.tensor(X_tr), torch.tensor(y_tr))
        b = float(b_arr[0])
        ev, ey = _eval_set(cc)
        sc = ev @ w + b
        init_scores[cat] = sc;  init_y[cat] = ey
        print(f"  T{t+1:<2} {cat:<14} {_auc(ey,sc):>8.4f}")

    W_f = solver._solve(solver.A_anomaly, solver.b_anomaly[:, 0], solver.lambda_a)
    w_f = W_f[:N_CONC].float().numpy();  b_f = float(W_f[N_CONC])
    final_scores = {}
    for cat in MVTEC_15:
        ev, _ = _eval_set(cache_v4[cat])
        final_scores[cat] = ev @ w_f + b_f
    return init_scores, final_scores, init_y


def main():
    print("=" * 72)
    print("  MA-CBM v4 — Defect + Normality Concept Heads")
    print("=" * 72)

    # ── Load heads (Part 1 outputs preferred, fall back to originals) ─────────
    img_heads_path = _IMG_HEADS_V2 if _IMG_HEADS_V2.exists() else _IMG_HEADS_ORIG
    vlm_path       = _VLM_V2 if _VLM_V2.exists() else _VLM_ORIG
    img_heads  = pickle.load(open(img_heads_path,"rb"))
    vlm_res    = json.load(open(vlm_path))
    # Use concept names in cluster_id order (cluster 3 was split into 12+13 in Part 1)
    img_names = [r["concept_name"] for r in sorted(vlm_res, key=lambda x: x["cluster_id"])]

    print(f"\n  Image-level defect heads: {len(img_names)}")
    print(f"    {'(from ' + img_heads_path.name + ')'}")

    # Load normality heads
    norm_img_heads = {}
    if _NORM_IMG_HEADS.exists():
        norm_img_heads = pickle.load(open(_NORM_IMG_HEADS,"rb"))
        print(f"  Image-level normality heads: {len(norm_img_heads)}")
    else:
        print("  [WARN] No normality image heads found — run upgrade_part2 first")

    norm_patch_heads = {}
    if _NORM_PATCH_HEADS.exists():
        norm_patch_heads = pickle.load(open(_NORM_PATCH_HEADS,"rb"))
        print(f"  Patch-level normality heads: {len(norm_patch_heads)}")
    else:
        print("  [WARN] No normality patch heads found — skipping (need per-image SAE codes)")

    # Load caches
    v2_cache = pickle.load(open(_V2_CACHE,"rb"))
    mp_cache = pickle.load(open(_MP_CACHE,"rb"))
    sae      = SparseAutoencoder.load(str(_SAE_WEIGHTS), device="cpu")

    # ── Build v4 concept cache ────────────────────────────────────────────────
    cache_v4, N_CONC = build_v4_cache(
        v2_cache, mp_cache, img_heads, img_names,
        norm_patch_heads, norm_img_heads, sae
    )

    # ── CONCIL ────────────────────────────────────────────────────────────────
    init_scores, final_scores, init_y = run_concil(cache_v4, N_CONC)

    # ── Dual-branch ───────────────────────────────────────────────────────────
    print("\n[dual] 0.9 × SAE + 0.1 × v4-concept …")
    sae_sc = pickle.load(open(_SAE_SC,"rb"))
    dual = {}
    for cat in MVTEC_15:
        if cat not in sae_sc: continue
        sae_s = sae_sc[cat]["sae_scores"].astype(np.float32)
        mac_s = final_scores[cat].astype(np.float32)
        y     = init_y[cat]
        comb  = 0.9*_norm01(sae_s) + 0.1*_norm01(mac_s)
        dual[cat] = _auc(y, comb)

    # ── Full comparison table ─────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("  COMPARISON — v3 vs v4 (adds normality concepts)")
    print(f"{'='*80}")
    print(f"\n  {'Category':<13} {'Type':<5} {'v3-conc':>8} {'v4-conc':>8} "
          f"{'Δ':>6} | {'v3-dual':>8} {'v4-dual':>8} | {'BWT-v4':>8}")
    print("  "+"─"*82)

    bwts, v4c_s, v4c_t, v4d_s, v4d_t = [], [], [], [], []
    results = {}

    for cat in MVTEC_15:
        y    = init_y[cat]
        v4ci = _auc(y, init_scores[cat])
        v4cf = _auc(y, final_scores[cat])
        bwt  = (v4cf - v4ci) if cat != MVTEC_15[-1] else float("nan")
        tag  = "S" if cat in SURFACE else "T"
        v3c  = V3_CONCEPT.get(cat, float("nan"))
        v3d  = V3_DUAL.get(cat, float("nan"))
        v4d  = dual.get(cat, float("nan"))
        delta = v4cf - v3c

        bwt_str = f"{bwt:>+8.4f}" if not np.isnan(bwt) else "  (last)"
        print(f"  {cat:<13}[{tag}]  {v3c:>8.3f} {v4cf:>8.3f} {delta:>+6.3f} | "
              f"{v3d:>8.3f} {v4d:>8.3f} | {bwt_str}")

        results[cat] = {"v4_concept":v4cf,"v4_dual":v4d,"bwt":bwt,"type":tag}
        if not np.isnan(bwt): bwts.append(bwt)
        if cat in SURFACE:
            v4c_s.append(v4cf); v4d_s.append(v4d)
        else:
            v4c_t.append(v4cf); v4d_t.append(v4d)

    print("  "+"─"*82)
    v4c_all = [results[c]["v4_concept"] for c in MVTEC_15]
    print(f"  {'MEAN':<18}  "
          f"{np.nanmean(list(V3_CONCEPT.values())):>8.3f} "
          f"{np.mean(v4c_all):>8.3f} "
          f"{np.mean(v4c_all)-np.nanmean(list(V3_CONCEPT.values())):>+6.3f} | "
          f"{np.nanmean(list(V3_DUAL.values())):>8.3f} "
          f"{np.nanmean([dual.get(c) for c in MVTEC_15]):>8.3f} | "
          f"{np.mean(bwts):>+8.4f}")

    surf_cats = [c for c in MVTEC_15 if c in SURFACE]
    strc_cats = [c for c in MVTEC_15 if c not in SURFACE]
    print(f"\n  Surface [S]: v3={np.mean([V3_CONCEPT[c] for c in surf_cats]):.3f}  "
          f"v4={np.mean(v4c_s):.3f}  BWT={np.mean([bwts[i] for i,c in enumerate(MVTEC_15[:-1]) if c in SURFACE]):+.4f}")
    print(f"  Struct  [T]: v3={np.mean([V3_CONCEPT[c] for c in strc_cats]):.3f}  "
          f"v4={np.mean(v4c_t):.3f}  BWT={np.mean([bwts[i] for i,c in enumerate(MVTEC_15[:-1]) if c not in SURFACE]):+.4f}")

    print(f"\n  Normality heads added:  {len(norm_img_heads)} image-level + "
          f"{len(norm_patch_heads)} patch-level (patch heads need per-image SAE codes)")

    json.dump({"v4":results,"n_conc":N_CONC,"summary":
               {"surface":np.mean(v4c_s),"structural":np.mean(v4c_t),
                "overall":np.mean(v4c_all),"bwt":np.mean(bwts)}},
              open(_RESULTS_JSON,"w"), indent=2)
    print(f"\n  Saved → {_RESULTS_JSON.name}")
    print("="*80)


if __name__ == "__main__":
    main()
