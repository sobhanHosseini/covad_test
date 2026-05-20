"""
run_v2_concil.py — MA-CBM v2: Unified surface + structural concept branch.

New 13-dim concept vector per image:
  Surface (5-dim) : SAE-based LogReg heads, patch-level, max-pooled
  Structural (8-dim): DINOv2 max-pool LogReg heads, image-level
  = concat([surface_5, structural_8])

Surface heads:  mac/outputs/cross_category/cross_category_6heads_concept_heads.pkl
Structural heads: re-trained from discovery-script clusters (K=8, deterministic)

Eval follows v1 structure (concept_vector_cache.pkl paths/splits) for comparability.
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

_COVAD = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_COVAD))
sys.path.insert(0, str(_COVAD / "mac"))

from features.dinov2_extractor import DINOv2Extractor
from features.sae import SparseAutoencoder
from solvers.concil import ConcilSolver

# ── Paths ─────────────────────────────────────────────────────────────────────
_MVTEC        = Path("/home/sobhan_hosseini/datasets/mvtec")
_SAE_W        = _COVAD / "sae_training" / "sae_vitl14reg_C4096_k64.pt"
_SURF_PKL     = _COVAD / "mac/outputs/cross_category/cross_category_6heads_concept_heads.pkl"
_VLM_JSON     = _COVAD / "ma_cbm/outputs/structural_vlm_results.json"
_V1_CACHE     = _COVAD / "mac/outputs/concil/concept_vector_cache.pkl"
_SAE_SC       = _COVAD / "mac/outputs/dual_branch/score_cache.pkl"
_OUT          = _COVAD / "ma_cbm/outputs"
_STRUCT_PKL   = _OUT / "structural_concept_heads.pkl"
_V2_CACHE     = _OUT / "v2_concept_cache.pkl"
_RESULTS_JSON = _OUT / "v2_results.json"
_OUT.mkdir(parents=True, exist_ok=True)

_DEVICE  = "cuda"
_BATCH   = 4
_K       = 8
_TOP_DIM = 100

STRUCT_CATS = ["cable", "transistor", "toothbrush", "bottle", "capsule", "screw"]
SURFACE     = {"carpet","grid","hazelnut","leather","metal_nut","pill","tile","wood","zipper"}
STRUCT_SET  = set(STRUCT_CATS)

MVTEC_15 = [
    "bottle","cable","capsule","carpet","grid","hazelnut","leather",
    "metal_nut","pill","screw","tile","toothbrush","transistor","wood","zipper",
]

SURF_HEAD_NAMES = [
    "surface_discontinuity","surface_discoloration","surface_crack",
    "surface_abrasion","surface_void",
]
N_SURF = len(SURF_HEAD_NAMES)   # 5
N_STRC = _K                     # 8
N_CONC = N_SURF + N_STRC        # 13

# v1 reference results (for comparison table)
V1_CONCEPT = {
    "bottle":0.669,"cable":0.562,"capsule":0.702,"carpet":0.947,"grid":1.000,
    "hazelnut":0.932,"leather":0.983,"metal_nut":0.883,"pill":0.915,"screw":0.690,
    "tile":0.986,"toothbrush":0.703,"transistor":0.672,"wood":0.887,"zipper":0.994,
}
V1_DUAL = {
    "bottle":0.998,"cable":0.911,"capsule":0.917,"carpet":0.995,"grid":1.000,
    "hazelnut":0.991,"leather":1.000,"metal_nut":0.993,"pill":0.976,"screw":0.905,
    "tile":1.000,"toothbrush":0.919,"transistor":0.941,"wood":0.985,"zipper":0.997,
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def _auc(y, s):
    try:    return float(roc_auc_score(y, s))
    except: return float("nan")

def _norm01(x):
    lo, hi = x.min(), x.max()
    return np.zeros_like(x) if hi - lo < 1e-9 else (x - lo) / (hi - lo)


# ── Phase A: Structural concept heads ────────────────────────────────────────

def _collect_struct() -> tuple[list[dict], np.ndarray, np.ndarray]:
    """Collect anomalous and normal images for 6 structural categories."""
    anom_items, norm_items = [], []
    for cat in STRUCT_CATS:
        root = _MVTEC / cat
        for split in ("train", "test"):
            good = root / split / "good"
            if good.exists():
                for p in sorted(good.glob("*.png")):
                    norm_items.append({"path": str(p), "category": cat})
        for d in sorted((root / "test").iterdir()):
            if d.name == "good" or not d.is_dir():
                continue
            for p in sorted(d.glob("*.png")):
                anom_items.append({"path": str(p), "category": cat,
                                   "defect_type": d.name})
    return anom_items, norm_items


@torch.no_grad()
def _maxpool_batch(paths, dino, batch_size=_BATCH):
    """Extract max-pooled DINOv2 patch features. Returns (N, 1024) float32."""
    parts = []
    for i in range(0, len(paths), batch_size):
        imgs = [Image.open(p).convert("RGB") for p in paths[i:i+batch_size]]
        x = dino._prepare(imgs)
        _, patches = dino._run(x)                      # (b, 256, 1024)
        parts.append(patches.max(dim=1).values.cpu().float().numpy())
        print(f"    {min(i+batch_size,len(paths))}/{len(paths)}", end="\r", flush=True)
    print()
    return np.concatenate(parts, axis=0)


def build_structural_heads(dino, vlm_results):
    """Re-derive K-means clusters + train structural LogReg heads. Returns dict."""
    print("[Phase A] Building structural concept heads …")
    anom_items, norm_items = _collect_struct()
    print(f"  Structural: {len(anom_items)} anomalous, {len(norm_items)} normal")

    anom_paths = [it["path"] for it in anom_items]
    norm_paths  = [it["path"] for it in norm_items]

    print("  Extracting max-pool features for anomalous images …")
    feats_anom = _maxpool_batch(anom_paths, dino)
    print("  Extracting max-pool features for normal images …")
    feats_norm = _maxpool_batch(norm_paths, dino)

    # Discriminative dims
    disc     = feats_anom.mean(0) - feats_norm.mean(0)
    top_dims = np.argsort(np.abs(disc))[::-1][:_TOP_DIM]

    # K-means (same seed as discovery script → same cluster assignments)
    km      = KMeans(n_clusters=_K, random_state=42, n_init=10)
    labels  = km.fit_predict(feats_anom[:, top_dims])

    # Build ordered concept names from VLM results
    cluster_names = {r["cluster_id"]: r["concept_name"] for r in vlm_results}

    heads = {}
    for k in range(_K):
        mask   = labels == k
        name   = cluster_names.get(k, f"cluster_{k}")
        X_pos  = feats_anom[mask]
        X_neg  = np.concatenate([feats_anom[~mask], feats_norm], axis=0)
        X      = np.concatenate([X_pos, X_neg], axis=0)
        y      = np.array([1]*len(X_pos) + [0]*len(X_neg), dtype=np.int32)
        clf    = LogisticRegression(C=1.0, class_weight="balanced",
                                    max_iter=1000, solver="lbfgs", random_state=42)
        clf.fit(X, y)
        heads[name] = clf
        print(f"  Head '{name}': {mask.sum()} pos / {len(X_neg)} neg")

    pickle.dump(heads, open(_STRUCT_PKL, "wb"))
    print(f"  Saved → {_STRUCT_PKL.name}")
    return heads, list(cluster_names[k] for k in range(_K))


# ── Phase B: 13-dim concept vectors for all 15 categories ────────────────────

@torch.no_grad()
def extract_concept_vectors_batch(
    paths: list[str],
    dino, sae, surf_heads, struct_heads, struct_head_names,
    batch_size=_BATCH,
) -> np.ndarray:
    """Extract 13-dim concept vectors for a list of image paths.

    Dim 0-4  : max-pool of per-patch SAE-head probabilities (surface, 5-dim)
    Dim 5-12 : structural LogReg probabilities on max-pooled DINOv2 (8-dim)
    """
    N    = len(paths)
    vecs = np.zeros((N, N_CONC), dtype=np.float32)

    for start in range(0, N, batch_size):
        batch = paths[start:start + batch_size]
        imgs  = [Image.open(p).convert("RGB") for p in batch]
        b     = len(imgs)

        x = dino._prepare(imgs)
        _, patches = dino._run(x)                          # (b, 256, 1024)

        # — Surface heads (SAE codes, patch-level) ——————————————
        flat      = patches.reshape(b * 256, 1024)
        sae_codes = sae.encode(flat.to(sae.b_dec.device))  # (b*256, 4096)
        sae_np    = sae_codes.detach().cpu().float().numpy().reshape(b, 256, 4096)

        for k, name in enumerate(SURF_HEAD_NAMES):
            clf = surf_heads[name]
            for bi in range(b):
                probs = clf.predict_proba(sae_np[bi])[:, 1]   # (256,)
                vecs[start + bi, k] = probs.max()

        # — Structural heads (max-pool, image-level) ——————————
        mp_np = patches.max(dim=1).values.detach().cpu().float().numpy()  # (b, 1024)
        for k, name in enumerate(struct_head_names):
            clf = struct_heads[name]
            probs = clf.predict_proba(mp_np)[:, 1]             # (b,)
            vecs[start:start + b, N_SURF + k] = probs

        print(f"    {min(start + batch_size, N)}/{N}", end="\r", flush=True)

    print()
    return vecs


def build_v2_cache(dino, sae, surf_heads, struct_heads, struct_head_names, v1_cache):
    """Build v2 concept vector cache using same paths/splits as v1."""
    print("\n[Phase B] Extracting 13-dim concept vectors for all 15 categories …")
    cache_v2 = {}
    for cat in MVTEC_15:
        v1 = v1_cache[cat]
        paths = v1["all_paths"]
        print(f"  [{cat}] {len(paths)} images …")
        vecs = extract_concept_vectors_batch(
            paths, dino, sae, surf_heads, struct_heads, struct_head_names,
        )
        cache_v2[cat] = {
            "all_paths":    paths,
            "y_all":        v1["y_all"],
            "is_train_mask": v1["is_train_mask"],
            "vecs":         vecs,
            "n_train_normal":  v1["n_train_normal"],
            "n_defect_train":  v1["n_defect_train"],
            "n_test_normal":   v1["n_test_normal"],
            "n_defect_test":   v1["n_defect_test"],
        }
    pickle.dump(cache_v2, open(_V2_CACHE, "wb"))
    print(f"  Saved → {_V2_CACHE.name}")
    return cache_v2


# ── Eval helpers ──────────────────────────────────────────────────────────────

def _eval_set(cc):
    """Same eval indices as script 06 (defect_train + test_normal + defect_test)."""
    vecs = cc["vecs"];  y = np.array(cc["y_all"])
    nt, nd, nn = cc["n_train_normal"], cc["n_defect_train"], cc["n_test_normal"]
    ev = np.concatenate([vecs[nt:nt+nd], vecs[nt+nd:nt+nd+nn], vecs[nt+nd+nn:]])
    ey = np.concatenate([y[nt:nt+nd],   y[nt+nd:nt+nd+nn],    y[nt+nd+nn:]])
    return ev.astype(np.float32), ey.astype(np.int32)


def _train_set(cc):
    vecs = cc["vecs"];  y = np.array(cc["y_all"])
    mask = np.array(cc["is_train_mask"])
    return vecs[mask].astype(np.float32), y[mask].astype(np.float32)


# ── Phase C+D: Sequential CONCIL ─────────────────────────────────────────────

def run_concil(cache_v2):
    print("\n[Phase C] Sequential CONCIL (13-dim concept space) …\n")
    solver = ConcilSolver(input_dim=N_CONC, lambda_anomaly=1e-4)
    init_scores, init_y = {}, {}

    hdr = f"  {'T':<3} {'Category':<14} {'I-AUC':>8}"
    print(hdr);  print("  " + "─" * 30)

    for t, cat in enumerate(MVTEC_15):
        cc = cache_v2[cat]
        X_tr, y_tr = _train_set(cc)
        C_t = torch.tensor(X_tr);  y_t = torch.tensor(y_tr)
        w, b_arr = solver.update_anomaly_head(C_t, y_t)
        b = float(b_arr[0])

        ev, ey = _eval_set(cc)
        scores = ev @ w + b
        auc = _auc(ey, scores)
        init_scores[cat] = scores;  init_y[cat] = ey

        print(f"  T{t+1:<2} {cat:<14} {auc:>8.4f}")

    # Final weights
    W_f = solver._solve(solver.A_anomaly, solver.b_anomaly, solver.lambda_a)
    w_f = W_f[:N_CONC, 0].float().numpy();  b_f = float(W_f[N_CONC, 0])

    print("\n[Phase D] Final scoring (after all 15 tasks) …")
    final_scores = {}
    for cat in MVTEC_15:
        ev, _ = _eval_set(cache_v2[cat])
        final_scores[cat] = ev @ w_f + b_f

    return init_scores, final_scores, init_y


# ── Phase E: Dual-branch (v2 concept + v1 SAE scores) ────────────────────────

def run_dual_branch(cache_v2, sae_score_cache, init_scores_v2, final_scores_v2, init_y):
    """Combine 0.9 * SAE + 0.1 * MA-CBM-v2 for eval set of each category."""
    # We need SAE scores for the eval set (same indices as _eval_set)
    # sae_score_cache[cat]['sae_scores'] already covers this eval set
    dual_init, dual_final = {}, {}
    for cat in MVTEC_15:
        if cat not in sae_score_cache:
            dual_init[cat] = dual_final[cat] = float("nan")
            continue
        sae_s = sae_score_cache[cat]["sae_scores"].astype(np.float32)
        mac_i  = init_scores_v2[cat].astype(np.float32)
        mac_f  = final_scores_v2[cat].astype(np.float32)
        comb_i = 0.9 * _norm01(sae_s) + 0.1 * _norm01(mac_i)
        comb_f = 0.9 * _norm01(sae_s) + 0.1 * _norm01(mac_f)
        y = init_y[cat]
        dual_init[cat]  = _auc(y, comb_i)
        dual_final[cat] = _auc(y, comb_f)
    return dual_init, dual_final


# ── Phase F: Full comparison table ───────────────────────────────────────────

def print_table(init_scores, final_scores, init_y, dual_init, dual_final):
    print("\n" + "=" * 80)
    print("  FULL COMPARISON TABLE — v1 vs v2 Concept Branch + Dual-Branch")
    print("=" * 80)
    hdr = (f"  {'Category':<13} {'Type':<5} "
           f"{'v1-conc':>8} {'v2-conc':>8} {'ΔCONC':>6} | "
           f"{'v1-dual':>8} {'v2-dual':>8} {'ΔDUAL':>6} | "
           f"{'BWT-v2':>7}")
    print(hdr);  print("  " + "─" * 77)

    bwts, v2_conc_all, v2_dual_all = [], [], []
    bwts_surf, bwts_strc = [], []
    v2c_surf, v2c_strc   = [], []
    v2d_surf, v2d_strc   = [], []

    results = {}
    for cat in MVTEC_15:
        y = init_y[cat]
        v2c_i = _auc(y, init_scores[cat])
        v2c_f = _auc(y, final_scores[cat])
        bwt   = v2c_f - v2c_i if cat != MVTEC_15[-1] else float("nan")

        v1c = V1_CONCEPT.get(cat, float("nan"))
        v1d = V1_DUAL.get(cat, float("nan"))
        v2d = dual_final[cat]

        dc   = v2c_f - v1c
        dd   = v2d   - v1d
        tag  = "S" if cat in SURFACE else "T"

        bwt_str = f"{bwt:>+7.4f}" if not np.isnan(bwt) else "  last"
        print(f"  {cat:<13} [{tag}]  "
              f"{v1c:>8.3f} {v2c_f:>8.3f} {dc:>+6.3f} | "
              f"{v1d:>8.3f} {v2d:>8.3f} {dd:>+6.3f} | "
              f"{bwt_str}")

        results[cat] = {"v2_concept": v2c_f, "v2_dual": v2d, "bwt_v2": bwt,
                        "v1_concept": v1c, "v1_dual": v1d, "type": tag}

        v2_conc_all.append(v2c_f);  v2_dual_all.append(v2d)
        if not np.isnan(bwt):
            bwts.append(bwt)
            (bwts_surf if cat in SURFACE else bwts_strc).append(bwt)
        (v2c_surf if cat in SURFACE else v2c_strc).append(v2c_f)
        (v2d_surf if cat in SURFACE else v2d_strc).append(v2d)

    print("  " + "─" * 77)
    print(f"  {'MEAN':<13}       "
          f"{np.nanmean(list(V1_CONCEPT.values())):>8.3f} "
          f"{np.mean(v2_conc_all):>8.3f} "
          f"{np.mean(v2_conc_all)-np.nanmean(list(V1_CONCEPT.values())):>+6.3f} | "
          f"{np.nanmean(list(V1_DUAL.values())):>8.3f} "
          f"{np.mean(v2_dual_all):>8.3f} "
          f"{np.mean(v2_dual_all)-np.nanmean(list(V1_DUAL.values())):>+6.3f} | "
          f"{np.mean(bwts):>+7.4f}")

    print(f"\n  {'─'*55}")
    print(f"  Group breakdown:")
    surf_cats = [c for c in MVTEC_15 if c in SURFACE]
    strc_cats = [c for c in MVTEC_15 if c not in SURFACE]
    v1c_surf = np.mean([V1_CONCEPT[c] for c in surf_cats])
    v1c_strc = np.mean([V1_CONCEPT[c] for c in strc_cats])
    v1d_surf = np.mean([V1_DUAL[c]    for c in surf_cats])
    v1d_strc = np.mean([V1_DUAL[c]    for c in strc_cats])

    print(f"  {'Surface [S]':<14}: "
          f"v1-conc={v1c_surf:.3f}  v2-conc={np.mean(v2c_surf):.3f}  "
          f"BWT={np.mean(bwts_surf):+.4f} | "
          f"v1-dual={v1d_surf:.3f}  v2-dual={np.mean(v2d_surf):.3f}")
    print(f"  {'Structural [T]':<14}: "
          f"v1-conc={v1c_strc:.3f}  v2-conc={np.mean(v2c_strc):.3f}  "
          f"BWT={np.mean(bwts_strc):+.4f} | "
          f"v1-dual={v1d_strc:.3f}  v2-dual={np.mean(v2d_strc):.3f}")

    print(f"\n  Target: surface ≥ 0.942, structural ≥ 0.900")
    surf_ok = "✓" if np.mean(v2c_surf) >= 0.942 else "✗"
    strc_ok = "✓" if np.mean(v2c_strc) >= 0.900 else "✗"
    print(f"  Surface  concept: {np.mean(v2c_surf):.3f}  {surf_ok}")
    print(f"  Struct   concept: {np.mean(v2c_strc):.3f}  {strc_ok}")
    print("=" * 80)

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print("  MA-CBM v2 — 13-dim Concept Branch (Surface 5 + Structural 8)")
    print("=" * 72)

    print("\n[init] Loading models …")
    dino = DINOv2Extractor("dinov2_vitl14_reg", device=torch.device(_DEVICE))
    sae  = SparseAutoencoder.load(str(_SAE_W), device=_DEVICE)
    sae  = sae.to(_DEVICE).eval()
    surf_heads = pickle.load(open(_SURF_PKL, "rb"))
    vlm_results = json.load(open(_VLM_JSON))
    print(f"  DINOv2 embed={dino.EMBED_DIM}  SAE d_hidden={sae.config.d_hidden}")
    print(f"  Surface heads: {SURF_HEAD_NAMES}")

    # ── Phase A: Structural heads ─────────────────────────────────────────────
    if _STRUCT_PKL.exists():
        struct_heads = pickle.load(open(_STRUCT_PKL, "rb"))
        struct_head_names = [vlm_results[k]["concept_name"] for k in range(_K)]
        print(f"\n[Phase A] Loaded structural heads from {_STRUCT_PKL.name}")
        print(f"  Heads: {struct_head_names}")
    else:
        struct_heads, struct_head_names = build_structural_heads(dino, vlm_results)

    # ── Phase B: 13-dim concept vectors ──────────────────────────────────────
    v1_cache = pickle.load(open(_V1_CACHE, "rb"))
    if _V2_CACHE.exists():
        print(f"\n[Phase B] Loading v2 concept cache from {_V2_CACHE.name} …")
        cache_v2 = pickle.load(open(_V2_CACHE, "rb"))
    else:
        cache_v2 = build_v2_cache(
            dino, sae, surf_heads, struct_heads, struct_head_names, v1_cache
        )

    # ── Phase C+D: CONCIL ─────────────────────────────────────────────────────
    init_scores, final_scores, init_y = run_concil(cache_v2)

    # ── Phase E: Dual-branch ──────────────────────────────────────────────────
    print("\n[Phase E] Dual-branch v2 (0.9 × SAE + 0.1 × MA-CBM-v2) …")
    sae_sc = pickle.load(open(_SAE_SC, "rb"))
    dual_init, dual_final = run_dual_branch(
        cache_v2, sae_sc, init_scores, final_scores, init_y
    )

    # ── Phase F: Table ────────────────────────────────────────────────────────
    results = print_table(init_scores, final_scores, init_y, dual_init, dual_final)

    json.dump({
        "v2_concept_final":  {c: results[c]["v2_concept"]  for c in MVTEC_15},
        "v2_dual_final":     {c: results[c]["v2_dual"]     for c in MVTEC_15},
        "bwt_v2":            {c: results[c]["bwt_v2"]      for c in MVTEC_15},
    }, open(_RESULTS_JSON, "w"), indent=2)
    print(f"\n  Results saved → {_RESULTS_JSON.name}")


if __name__ == "__main__":
    main()
