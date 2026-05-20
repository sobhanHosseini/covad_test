"""
compute_cauc.py — Measure C-AUC of MA-CBM concept heads against CONVAD annotations.

For each of 15 MVTec categories:
  1. Load CONVAD automated CSV ({cat}_dataset_automated.csv).
  2. Classify each CONVAD concept as activation-type or suppression-type
     based on whether it fires more on anomalous or normal images.
  3. Map CONVAD image paths to local filesystem paths.
  4. Look up 17-dim MA-CBM concept vectors from pre-built caches
     (no DINOv2/SAE re-run needed — 100% coverage confirmed).
  5. Compute C-AUC:
       For each activation-type concept c_j:
         For each of our 17 heads h_k:
           AUC_jk = AUROC(our_activation_k, binary_label_cj)
         best_AUC_j = max_k AUC_jk
       C-AUC = weighted_mean(best_AUC_j, weight=positive_count_j)
  6. Compute suppression C-AUC analogously using 1 - our_activation_k.

17-dim concept vector per image:
  [0:5]   5 patch-level SAE surface heads (from v2 cache)
  [5:17] 12 image-level max-pool all-category heads

CONVAD paths: /mnt/disk1/borsattifr/datasets/mvtec/{rel}
Local mapping: /home/sobhan_hosseini/datasets/mvtec/{rel}
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

_COVAD = Path(__file__).resolve().parents[2]
_MABCM = _COVAD / "ma_cbm"
sys.path.insert(0, str(_COVAD))

# ── Paths ─────────────────────────────────────────────────────────────────────
_ANN_DIR    = Path("/home/sobhan_hosseini/cbm_data/mvtec")
_MVTEC      = Path("/home/sobhan_hosseini/datasets/mvtec")
_V2_CACHE   = _MABCM / "outputs" / "v2_concept_cache.pkl"
_MP_CACHE   = _MABCM / "outputs" / "allcat_maxpool_cache.pkl"
_IMG_HEADS  = _MABCM / "outputs" / "allcat_image_level_heads.pkl"
_VLM_JSON   = _MABCM / "outputs" / "image_level_vlm_results_allcat.json"
_OUT_DIR    = _MABCM / "outputs" / "cauc"
_RES_JSON   = _OUT_DIR / "cauc_results.json"
_OUT_DIR.mkdir(parents=True, exist_ok=True)

MVTEC_15 = [
    "bottle","cable","capsule","carpet","grid","hazelnut","leather",
    "metal_nut","pill","screw","tile","toothbrush","transistor","wood","zipper",
]
N_SURF = 5    # surface SAE heads (dims 0-4 from v2 cache)
N_IMG  = 12   # image-level heads (dims 5-16)
N_CONC = N_SURF + N_IMG  # 17

META_COLS = {"split", "image_path", "label_index", "mask_path", "anomaly_type"}

# Reference C-AUC from CONVAD paper Table 1 (Fully Supervised column)
CONVAD_CAUC_REF = {
    "bottle": None, "cable": None, "capsule": None, "carpet": None,
    "grid": None, "hazelnut": 0.89, "leather": None, "metal_nut": None,
    "pill": None, "screw": None, "tile": None, "toothbrush": None,
    "transistor": None, "wood": None, "zipper": None,
    "_mean": 0.86,
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return float("nan")


def _map_path(convad_path: str) -> str:
    """Convert CONVAD server path to local path."""
    rel = convad_path.split("mvtec/")[1]
    return str(_MVTEC / rel)


def _classify_concept(col: pd.Series, labels: pd.Series) -> str:
    """activation if fires more on anomaly; suppression if fires more on normal."""
    m_norm = col[labels == 0].mean()
    m_anom = col[labels == 1].mean()
    return "activation" if m_anom >= m_norm else "suppression"


# ── Phase 0: Load caches and build 17-dim concept lookup ─────────────────────

def build_concept_lookup(v2_cache, mp_cache, img_heads, img_head_names):
    """Build {local_path: 17-dim concept vector} lookup for all 15 categories."""
    print("[init] Building 17-dim concept vector lookup …")
    lookup: dict[str, np.ndarray] = {}

    for cat in MVTEC_15:
        v2 = v2_cache[cat]
        mp = mp_cache[cat]
        N  = len(v2["all_paths"])

        # Surface dims (0-4) from v2 cache — already computed from SAE codes
        surf = v2["vecs"][:, :N_SURF].astype(np.float32)  # (N, 5)

        # Image-level dims (5-16) from maxpool features + image heads
        mp_feats = mp["maxpool_feats"].astype(np.float32)  # (N, 1024)
        img_vecs = np.zeros((N, N_IMG), dtype=np.float32)
        for k, name in enumerate(img_head_names):
            clf = img_heads.get(name)
            if clf is not None:
                img_vecs[:, k] = clf.predict_proba(mp_feats)[:, 1]

        vecs = np.concatenate([surf, img_vecs], axis=1)   # (N, 17)

        for path, vec in zip(v2["all_paths"], vecs):
            lookup[path] = vec

        print(f"  [{cat}] {N} images → lookup size={len(lookup)}")

    return lookup


# ── Phase 1: Per-category C-AUC ──────────────────────────────────────────────

def process_category(
    cat: str,
    concept_lookup: dict[str, np.ndarray],
    concept_names_17: list[str],
) -> dict:
    """Compute C-AUC for one MVTec category.

    Returns dict with concept classification, per-concept best AUC,
    weighted C-AUC for activation and suppression types, and coverage stats.
    """
    csv_path = _ANN_DIR / f"{cat}_dataset_automated.csv"
    df = pd.read_csv(csv_path)

    # Prefer test split; fall back to all if too few anomalies
    test_df = df[df["split"] == "test"].copy()
    n_test_anom = (test_df["label_index"] > 0).sum()
    if n_test_anom < 5:
        print(f"  [{cat}] Only {n_test_anom} test anomalies — using all splits")
        eval_df = df.copy()
    else:
        eval_df = test_df

    concept_cols = [c for c in df.columns if c not in META_COLS]

    # Map paths and collect concept vectors
    local_paths   = [_map_path(p) for p in eval_df["image_path"]]
    labels        = eval_df["label_index"].values.astype(int)
    labels_binary = (labels > 0).astype(int)

    matched_idx = [i for i, p in enumerate(local_paths) if p in concept_lookup]
    skipped     = len(local_paths) - len(matched_idx)

    if len(matched_idx) == 0:
        print(f"  [{cat}] No images matched!")
        return {}

    our_vecs  = np.stack([concept_lookup[local_paths[i]] for i in matched_idx])
    conv_labs = eval_df.iloc[matched_idx][concept_cols].values  # (M, n_concepts)
    y         = labels_binary[matched_idx]

    # Classify each CONVAD concept
    activation_concepts, suppression_concepts = [], []
    for j, cname in enumerate(concept_cols):
        m0 = conv_labs[y == 0, j].mean()
        m1 = conv_labs[y == 1, j].mean()
        (activation_concepts if m1 >= m0 else suppression_concepts).append((j, cname))

    # Per-concept best AUC
    def best_auc_for(j: int, invert: bool = False) -> tuple[float, int, str]:
        y_true = conv_labs[:, j]
        if y_true.sum() == 0:
            return float("nan"), -1, "—"
        best, best_k = -1.0, -1
        for k in range(N_CONC):
            scores = (1 - our_vecs[:, k]) if invert else our_vecs[:, k]
            auc = _safe_auc(y_true, scores)
            if not np.isnan(auc) and auc > best:
                best, best_k = auc, k
        return best, best_k, concept_names_17[best_k] if best_k >= 0 else "—"

    act_results, sup_results = [], []

    for j, cname in activation_concepts:
        auc, best_k, best_name = best_auc_for(j, invert=False)
        weight = int(conv_labs[:, j].sum())
        act_results.append({"concept": cname, "type": "activation",
                            "best_auc": auc, "best_head": best_name, "weight": weight})

    for j, cname in suppression_concepts:
        auc, best_k, best_name = best_auc_for(j, invert=True)
        weight = int(conv_labs[:, j].sum())  # positives = normal images with concept=1
        sup_results.append({"concept": cname, "type": "suppression",
                            "best_auc": auc, "best_head": best_name, "weight": weight})

    # Weighted C-AUC
    def weighted_mean(results: list[dict]) -> float:
        valid = [(r["best_auc"], r["weight"]) for r in results
                 if not np.isnan(r["best_auc"]) and r["weight"] > 0]
        if not valid:
            return float("nan")
        aucs, ws = zip(*valid)
        return float(np.average(aucs, weights=ws))

    return {
        "n_images":           len(matched_idx),
        "n_skipped":          skipped,
        "n_anomalous":        int(y.sum()),
        "n_activation":       len(activation_concepts),
        "n_suppression":      len(suppression_concepts),
        "activation_concepts": [r["concept"] for r in act_results],
        "suppression_concepts": [r["concept"] for r in sup_results],
        "c_auc_activation":   weighted_mean(act_results),
        "c_auc_suppression":  weighted_mean(sup_results),
        "per_concept":        act_results + sup_results,
    }


# ── Phase 2: Main ─────────────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print("  MA-CBM C-AUC vs CONVAD concept annotations")
    print("=" * 72)

    # Load caches
    print("\n[init] Loading caches …")
    v2_cache  = pickle.load(open(_V2_CACHE,  "rb"))
    mp_cache  = pickle.load(open(_MP_CACHE,  "rb"))
    img_heads = pickle.load(open(_IMG_HEADS, "rb"))
    vlm_res   = json.load(open(_VLM_JSON))

    img_head_names = [
        next(r["concept_name"] for r in vlm_res if r["cluster_id"] == k)
        for k in range(len(vlm_res))
    ]
    surf_names = [
        "surface_discontinuity","surface_discoloration","surface_crack",
        "surface_abrasion","surface_void",
    ]
    concept_names_17 = surf_names + img_head_names
    print(f"  17-dim heads: {surf_names} + {img_head_names}")

    # Build lookup
    concept_lookup = build_concept_lookup(v2_cache, mp_cache, img_heads, img_head_names)
    print(f"  Total images in lookup: {len(concept_lookup)}")

    # Per-category processing
    print("\n[C-AUC] Processing all 15 categories …\n")
    all_results = {}

    for cat in MVTEC_15:
        act_path = _OUT_DIR / f"activations_{cat}.pkl"
        if act_path.exists():
            print(f"  [{cat}] loading from cache …")
            r = pickle.load(open(act_path, "rb"))
        else:
            print(f"  [{cat}] …", flush=True)
            r = process_category(cat, concept_lookup, concept_names_17)
            pickle.dump(r, open(act_path, "wb"))

        all_results[cat] = r
        if r:
            ref = CONVAD_CAUC_REF.get(cat)
            ref_str = f"{ref:.3f}" if ref else "  —  "
            print(f"    images={r['n_images']}  anom={r['n_anomalous']}  "
                  f"act={r['n_activation']}  sup={r['n_suppression']}  "
                  f"C-AUC(act)={r['c_auc_activation']:.4f}  "
                  f"C-AUC(sup)={r['c_auc_suppression']:.4f}  "
                  f"CONVAD_ref={ref_str}")

    # ── Table 1: Per-category C-AUC ───────────────────────────────────────────
    print(f"\n{'='*80}")
    print("  TABLE 1 — Per-category C-AUC")
    print(f"{'='*80}")
    print(f"  {'Category':<13} {'n_act':>6} {'n_sup':>6} "
          f"{'MA-CBM C-AUC':>13} {'MA-CBM sup':>11} {'CONVAD ref':>11}")
    print("  " + "─" * 64)

    act_aucs, sup_aucs = [], []
    SURFACE = {"carpet","grid","hazelnut","leather","metal_nut","pill","tile","wood","zipper"}

    for cat in MVTEC_15:
        r   = all_results.get(cat, {})
        if not r:
            continue
        tag = "S" if cat in SURFACE else "T"
        ref = CONVAD_CAUC_REF.get(cat)
        ref_str = f"{ref:.3f}" if ref else "  —  "
        a_auc = r.get("c_auc_activation", float("nan"))
        s_auc = r.get("c_auc_suppression", float("nan"))
        print(f"  {cat:<13}[{tag}] {r['n_activation']:>6} {r['n_suppression']:>6} "
              f"{a_auc:>13.4f} {s_auc:>11.4f} {ref_str:>11}")
        if not np.isnan(a_auc): act_aucs.append(a_auc)
        if not np.isnan(s_auc): sup_aucs.append(s_auc)

    print("  " + "─" * 64)
    print(f"  {'MEAN':<17}             "
          f"{np.mean(act_aucs):>13.4f} {np.mean(sup_aucs):>11.4f} "
          f"{'0.860':>11}")

    # ── Table 2: Best concept alignment (hazelnut) ────────────────────────────
    print(f"\n{'='*80}")
    print("  TABLE 2 — Per-concept alignment (hazelnut example)")
    print(f"{'='*80}")
    hn = all_results.get("hazelnut", {})
    if hn.get("per_concept"):
        print(f"  {'CONVAD concept':<30} {'Type':<12} {'Best MA-CBM head':<42} {'AUC':>6}")
        print("  " + "─" * 92)
        for pc in sorted(hn["per_concept"], key=lambda x: -(x["best_auc"] or 0)):
            auc_s = f"{pc['best_auc']:.4f}" if not np.isnan(pc["best_auc"]) else "  nan"
            print(f"  {pc['concept']:<30} {pc['type']:<12} "
                  f"{pc['best_head']:<42} {auc_s:>6}")

    # ── Table 3: Coverage summary ─────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("  TABLE 3 — Coverage summary (all activation concepts)")
    print(f"{'='*80}")
    all_act = [pc for r in all_results.values()
               for pc in r.get("per_concept", [])
               if pc["type"] == "activation" and not np.isnan(pc["best_auc"])]
    well    = sum(1 for p in all_act if p["best_auc"] > 0.75)
    partial = sum(1 for p in all_act if 0.60 <= p["best_auc"] <= 0.75)
    none    = sum(1 for p in all_act if p["best_auc"] < 0.60)
    total   = len(all_act)
    print(f"  Well covered   (AUC > 0.75) : {well:>4}  ({100*well/total:.1f}%)")
    print(f"  Partially      (0.60–0.75)  : {partial:>4}  ({100*partial/total:.1f}%)")
    print(f"  Not covered    (< 0.60)     : {none:>4}  ({100*none/total:.1f}%)")
    print(f"  Total activation concepts   : {total}")

    # ── Verdict ───────────────────────────────────────────────────────────────
    mean_act = float(np.mean(act_aucs))
    mean_sup = float(np.mean(sup_aucs))

    # Identify suppression concepts per category
    sup_per_cat = {cat: all_results[cat].get("suppression_concepts", [])
                   for cat in MVTEC_15 if cat in all_results}

    print(f"\n{'='*80}")
    print("  VERDICT")
    print(f"{'='*80}")
    print(f"\n  MA-CBM overall C-AUC (activation concepts): {mean_act:.4f}")
    print(f"  CONVAD self-reported C-AUC (all concepts):  0.8600")
    print()
    print(f"  Note: CONVAD's C-AUC includes suppression-type concepts")
    print(f"  which MA-CBM explicitly avoids (by design).")
    print(f"  MA-CBM activation-concept C-AUC:             {mean_act:.4f}")
    print(f"  MA-CBM suppression-concept coverage (inv):   {mean_sup:.4f}")
    print()

    # Flag carpet (woven_fabric_integrity had low VLM confidence)
    carpet_r = all_results.get("carpet", {})
    if carpet_r:
        carpet_act = carpet_r.get("c_auc_activation", float("nan"))
        print(f"  [FLAG] carpet: woven_fabric_integrity had low VLM confidence.")
        print(f"         carpet activation C-AUC = {carpet_act:.4f}")

    print(f"\n  Suppression-type concepts per category:")
    for cat, sups in sup_per_cat.items():
        if sups:
            print(f"    {cat:<14}: {', '.join(sups)}")

    # ── Save ──────────────────────────────────────────────────────────────────
    summary = {
        "mean_c_auc_activation":  mean_act,
        "mean_c_auc_suppression": mean_sup,
        "convad_reference":       0.86,
        "per_category": {
            cat: {
                "c_auc_activation":  all_results[cat].get("c_auc_activation"),
                "c_auc_suppression": all_results[cat].get("c_auc_suppression"),
                "n_activation":      all_results[cat].get("n_activation"),
                "n_suppression":     all_results[cat].get("n_suppression"),
                "n_images":          all_results[cat].get("n_images"),
                "n_skipped":         all_results[cat].get("n_skipped"),
                "convad_ref":        CONVAD_CAUC_REF.get(cat),
                "suppression_concepts": all_results[cat].get("suppression_concepts", []),
            }
            for cat in MVTEC_15 if cat in all_results
        },
        "coverage": {"well": well, "partial": partial, "none": none, "total": total},
        "per_category_per_concept": {
            cat: all_results[cat].get("per_concept", [])
            for cat in MVTEC_15 if cat in all_results
        },
    }
    json.dump(summary, open(_RES_JSON, "w"), indent=2)
    print(f"\n  Results saved → {_RES_JSON}")
    print("=" * 80)


if __name__ == "__main__":
    main()
