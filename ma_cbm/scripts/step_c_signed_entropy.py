"""
step_c_signed_entropy.py — Per-category baselines, signed activations, entropy.

Zero new training. Pure inference on cached 40-dim concept vectors.
Uses concept_lookup_v4.pkl (computed in compute_cauc_v4.py).

40-dim layout:
  dims  0- 4 : PATCH DEFECT  (p-def)  — 5 heads
  dims  5-17 : IMAGE DEFECT  (i-def)  — 13 heads
  dims 18-29 : PATCH NORM    (p-norm) — 12 heads
  dims 30-39 : IMAGE NORM    (i-norm) — 10 heads
"""
from __future__ import annotations
import json, math, pickle, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.special import expit as sigmoid
from sklearn.metrics import roc_auc_score

_COVAD = Path(__file__).resolve().parents[2]
_MABCM = _COVAD / "ma_cbm"
_OUT   = _MABCM / "outputs" / "step_c"
_OUT.mkdir(parents=True, exist_ok=True)

_MP_CACHE = _MABCM / "outputs" / "allcat_maxpool_cache.pkl"
_LOOKUP   = _MABCM / "outputs" / "cauc" / "concept_lookup_v4.pkl"
_VLM_V2   = _MABCM / "outputs" / "image_level_vlm_results_allcat_v2.json"

MVTEC_15 = ["bottle","cable","capsule","carpet","grid","hazelnut","leather",
             "metal_nut","pill","screw","tile","toothbrush","transistor","wood","zipper"]

SURF_NAMES = ["surface_discontinuity","surface_discoloration","surface_crack",
              "surface_abrasion","surface_void"]

# ── load concept names ────────────────────────────────────────────────────────

def _img_names():
    vlm = json.load(open(_VLM_V2))
    return [r["concept_name"] for r in sorted(vlm, key=lambda x: x["cluster_id"])]

def _npat_names():
    return list(pickle.load(open(_MABCM/"outputs"/"norm_patch_concept_heads.pkl","rb")).keys())

def _nimg_names():
    return list(pickle.load(open(_MABCM/"outputs"/"norm_image_concept_heads.pkl","rb")).keys())

# Image-level normality head → intended categories
_NIMG_CATS = {
    "perfect_circular_symmetry":             ["bottle"],
    "modular_paneling":                      ["leather","wood"],
    "product_uniformity":                    ["capsule","pill"],
    "structural_integrity":                  ["hazelnut","metal_nut"],
    "manufactured_fasteners":                ["screw"],
    "woven_textile_structure":               ["carpet","grid"],
    "structural_uniformity_and_completeness":["cable","toothbrush"],
    "material_homogeneity":                  ["tile"],
    "parallel_and_uniform_striping":         ["zipper"],
    "engineered_array_pattern":              ["transistor"],
}

# ── helpers ───────────────────────────────────────────────────────────────────

def _bar(s: float, scale: float = 0.05, width: int = 20) -> str:
    n = min(width, round(abs(s) / scale))
    return ("█" * n).ljust(width)

def _direction(name: str, s: float, ctype: str) -> str:
    if abs(s) < 0.08:  return "as expected"
    if ctype in ("p-def","i-def"):
        return "defect present" if s > 0 else "below normal"
    return "normality disrupted" if s < 0 else "extra normal"

def _entropy18(signed_def18: np.ndarray) -> float:
    p = sigmoid(signed_def18 + 0.5)
    s = p.sum()
    if s < 1e-9: return math.log(18)
    q = np.clip(p / s, 1e-12, 1)
    return float(-(q * np.log(q)).sum())

# ── Step 1: Per-category baselines ───────────────────────────────────────────

def compute_baselines(mp_cache, lookup):
    """Mean 40-dim vector over train/good/ images per category."""
    baselines = {}
    for cat in MVTEC_15:
        d = mp_cache[cat]
        n_tn = d["n_train_normal"]
        vecs = [lookup[p] for p in d["all_paths"][:n_tn] if p in lookup]
        baselines[cat] = np.stack(vecs).mean(axis=0)   # (40,)
    return baselines

# ── Step 5 explanation formatter ─────────────────────────────────────────────

def explain(img_path, cat, lookup, baselines, all_names, types,
            thr_normal, thr_novel, y_label="ANOMALOUS"):
    vec    = lookup[img_path]
    signed = vec - baselines[cat]
    ent    = _entropy18(signed[:18])

    if   ent < thr_normal: verdict = "LIKELY NORMAL"
    elif ent < thr_novel:  verdict = "KNOWN ANOMALY TYPE (confident)"
    else:                  verdict = "NOVEL / AMBIGUOUS — flag for review"

    defect_label = Path(img_path).parent.name

    print("\n  " + "═"*68)
    print(f"  MA-CBM EXPLANATION REPORT")
    print("  " + "═"*68)
    print(f"  Image:    …/{'/'.join(img_path.split('/')[-4:])}")
    print(f"  Category: {cat}   Defect: {defect_label}")
    print(f"  Decision: {y_label}   Anomaly score: {signed[:18].max():+.3f}")
    print(f"  Entropy:  {ent:.3f}  →  {verdict}")

    # What is wrong (defect above baseline)
    def_pos = [(all_names[k], float(signed[k]))
               for k in np.argsort(-signed[:18])[:5] if signed[k] > 0.05]
    # Normality disrupted (patch or image normality below baseline)
    nrm_neg = [(all_names[k], float(signed[k]))
               for k in np.argsort(signed[18:])[:5]
               if signed[18+list(np.argsort(signed[18:30]))[:5].index(
                   list(np.argsort(signed[18:30])[:5])[
                       list(np.argsort(signed[18:30])[:5]).index(k)
                       if k in np.argsort(signed[18:30])[:5] else -1]) < -0.05]
               ] if False else []
    # Simpler: directly
    nrm_neg = [(all_names[18+j], float(signed[18+j]))
               for j in np.argsort(signed[18:30])[:4] if signed[18+j] < -0.05]
    nrm_neg += [(all_names[30+j], float(signed[30+j]))
                for j in np.argsort(signed[30:40])[:3] if signed[30+j] < -0.05]
    nrm_neg = sorted(nrm_neg, key=lambda x: x[1])[:4]

    # Intact (near zero)
    intact = [(all_names[k], float(signed[k]))
              for k in np.argsort(np.abs(signed))[:3]]

    if def_pos:
        print(f"\n  WHAT IS WRONG:")
        for n, s in def_pos:
            print(f"    {n:<45} {s:>+7.3f}  {_bar(s)}  defect present")
    if nrm_neg:
        print(f"\n  WHAT IS DISRUPTED:")
        for n, s in nrm_neg:
            print(f"    {n:<45} {s:>+7.3f}  {_bar(s)}  normality lost")
    if intact:
        print(f"\n  WHAT IS INTACT:")
        for n, s in intact[:3]:
            print(f"    {n:<45} {s:>+7.3f}  as expected")

    if ent >= thr_novel:
        best = max([(all_names[k], float(signed[k])) for k in range(18)],
                   key=lambda x: abs(x[1]))
        print(f"\n  VERDICT: Novel/ambiguous → closest concept: {best[0]} ({best[1]:+.3f})")
        print(f"           Confidence: LOW — FLAG FOR HUMAN REVIEW")
    elif y_label == "ANOMALOUS" and def_pos:
        n, s = def_pos[0]
        conf = "HIGH" if s > 0.40 else "MEDIUM"
        print(f"\n  VERDICT: Known anomaly → primary concept: {n} ({s:+.3f})  Confidence: {conf}")
    else:
        print(f"\n  VERDICT: Normal — all concept values near category baseline")
    print("  " + "═"*68)

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    IMG_NAMES  = _img_names()
    NPAT_NAMES = _npat_names()
    NIMG_NAMES = _nimg_names()
    ALL_NAMES  = SURF_NAMES + IMG_NAMES + NPAT_NAMES + NIMG_NAMES
    TYPES      = ["p-def"]*5 + ["i-def"]*13 + ["p-norm"]*12 + ["i-norm"]*10
    assert len(ALL_NAMES) == 40

    print("=" * 72)
    print("  MA-CBM Step C — Per-Category Baselines + Signed Activations")
    print("=" * 72)
    print("\n[init] Loading caches …")
    mp_cache = pickle.load(open(_MP_CACHE,"rb"))
    lookup   = pickle.load(open(_LOOKUP,"rb"))
    print(f"  {len(lookup)} images in v4 lookup")

    # ── Step 1: baselines ─────────────────────────────────────────────────────
    print("\n[Step 1] Per-category baselines from train/good/ images …")
    baselines = compute_baselines(mp_cache, lookup)

    bl_json = {n: {cat: float(baselines[cat][k]) for cat in MVTEC_15}
               for k, n in enumerate(ALL_NAMES)}
    json.dump(bl_json, open(_OUT/"baselines_per_category.json","w"), indent=2)

    print(f"\n  {'Concept':<45} {'Type':<7} {'Min':>6} {'Max':>6} {'Mean':>6}")
    print("  " + "─" * 70)
    for k, (n, t) in enumerate(zip(ALL_NAMES, TYPES)):
        v = np.array([baselines[c][k] for c in MVTEC_15])
        print(f"  {n:<45} {t:<7} {v.min():>6.3f} {v.max():>6.3f} {v.mean():>6.3f}")

    # ── Step 3: normality head confirmation ───────────────────────────────────
    print(f"\n[Step 3] Image-level normality head baselines (intended categories) …\n")
    print(f"  {'Head':<48} {'Intended':>12} {'Others mean':>12}  Status")
    print("  " + "─" * 80)
    for j, n in enumerate(NIMG_NAMES):
        k = 30 + j
        intended = _NIMG_CATS.get(n, [])
        iv = [baselines[c][k] for c in intended if c in baselines]
        ov = [baselines[c][k] for c in MVTEC_15 if c not in intended]
        im = float(np.mean(iv)) if iv else float("nan")
        om = float(np.mean(ov)) if ov else float("nan")
        ok = "✓" if im >= 0.70 else "⚠ WEAK"
        cats_s = "+".join(intended)
        print(f"  {n:<48} {im:>12.3f} {om:>12.3f}  {ok}  [{cats_s}]")

    # ── Step 2: signed activation examples ───────────────────────────────────
    CASES = [
        ("hazelnut", "cut",          1),
        ("hazelnut", "test/good",    0),
        ("cable",    "missing_wire", 1),
        ("cable",    "test/good",    0),
        ("transistor","bent_lead",   1),
        ("transistor","test/good",   0),
    ]
    print(f"\n[Step 2] Signed activation examples (h_k - baseline_k_category) …")
    val_rows = []
    for cat, tag, yi in CASES:
        d = mp_cache[cat]
        y = np.array(d["y_all"])
        paths = d["all_paths"]
        cands = [p for p, lab in zip(paths, y)
                 if lab == yi and (f"/{tag}/" in p) and p in lookup]
        if not cands:
            print(f"  [{cat}/{tag}] no candidates — skip")
            continue
        img_path = cands[0]
        vec    = lookup[img_path]
        signed = vec - baselines[cat]
        top5   = np.argsort(np.abs(signed))[::-1][:5]

        print(f"\n  ── {cat}/{tag} ({'anomaly' if yi else 'normal'}) ──────────────────────")
        print(f"  {'Concept':<45} {'Signed':>8}  {'Bar':20}  Direction")
        print("  " + "─" * 90)
        for k in top5:
            s = float(signed[k])
            print(f"  {ALL_NAMES[k]:<45} {s:>+8.3f}  {_bar(s):<20}  "
                  f"{_direction(ALL_NAMES[k], s, TYPES[k])}")
        max_abs = float(np.abs(signed).max())
        max_def = float(signed[:18].max())
        val_rows.append((cat, tag, yi, max_abs, max_def))

    print(f"\n  Validation (normal ≤ 0.25 max|signed|, defect ≥ +0.20 max signed):")
    print(f"  {'Image':<30} {'Max|signed|':>12}  {'MaxDef':>8}  Pass")
    for cat, tag, yi, ma, md in val_rows:
        passed = ("✓" if (yi==0 and ma<=0.25) or (yi==1 and md>=0.20) else "⚠")
        print(f"  {cat}/{tag:<20} {ma:>12.3f}  {md:>8.3f}  {passed}")

    # ── Step 4: entropy ───────────────────────────────────────────────────────
    print(f"\n[Step 4] Entropy on test images (18 defect heads, per-category signed) …")
    test_recs = []
    for cat in MVTEC_15:
        d = mp_cache[cat]
        y = np.array(d["y_all"])
        paths = d["all_paths"]
        nt, nd, nn = d["n_train_normal"], d["n_defect_train"], d["n_test_normal"]
        for i in range(nt+nd, len(paths)):
            p = paths[i]
            if p not in lookup: continue
            signed = lookup[p] - baselines[cat]
            ent = _entropy18(signed[:18])
            dtype = Path(p).parent.name if y[i]==1 else "good"
            test_recs.append({"cat":cat,"dt":dtype,"y":int(y[i]),
                               "ent":ent,"smax":float(signed[:18].max()),"path":p})

    ne = np.array([r["ent"]  for r in test_recs if r["y"]==0])
    ae = np.array([r["ent"]  for r in test_recs if r["y"]==1])
    ns = np.array([r["smax"] for r in test_recs if r["y"]==0])
    as_ = np.array([r["smax"] for r in test_recs if r["y"]==1])

    print(f"  Normal   ({len(ne):4d}):  entropy {ne.mean():.3f}±{ne.std():.3f}  "
          f"max-signed {ns.mean():.3f}±{ns.std():.3f}")
    print(f"  Anomaly  ({len(ae):4d}):  entropy {ae.mean():.3f}±{ae.std():.3f}  "
          f"max-signed {as_.mean():.3f}±{as_.std():.3f}")
    print(f"  Gap anomaly-normal (entropy): {ae.mean()-ne.mean():+.3f}")
    print(f"  Max entropy log(18): {math.log(18):.3f}")

    ya = np.array([r["y"] for r in test_recs])
    ea = np.array([r["ent"] for r in test_recs])
    sa = np.array([r["smax"] for r in test_recs])
    auc_ent = float(roc_auc_score(ya, ea))
    auc_sgn = float(roc_auc_score(ya, sa))
    print(f"\n  AUROC:  entropy={auc_ent:.4f}  max-signed={auc_sgn:.4f}  raw-CONCIL=1.0000")

    thr_n = float(np.percentile(ne, 95))
    thr_v = float(np.percentile(ae, 95))
    print(f"  Thresholds:  normal_95pct={thr_n:.3f}  novel_95pct={thr_v:.3f}")

    dtype_ent = defaultdict(list)
    for r in test_recs:
        if r["y"]==1: dtype_ent[f"{r['cat']}/{r['dt']}"].append(r["ent"])
    dm = {dt: float(np.mean(v)) for dt,v in dtype_ent.items() if len(v)>=2}
    ranked = sorted(dm.items(), key=lambda x: x[1])
    print(f"\n  Lowest entropy (most confidently explained):")
    for dt,e in ranked[:5]:  print(f"    {dt:<40}  {e:.3f}")
    print(f"  Highest entropy (most novel/ambiguous):")
    for dt,e in ranked[-5:][::-1]: print(f"    {dt:<40}  {e:.3f}")

    # ── Step 5: explanation reports ───────────────────────────────────────────
    print(f"\n[Step 5] Final explanation reports …")
    EXPLAIN = [
        ("hazelnut","cut","test/good",True),
        ("cable","missing_wire","test/good",True),
        ("hazelnut",None,"test/good",False),
    ]
    for cat, defect, norm_tag, is_anom in EXPLAIN:
        d = mp_cache[cat]
        y = np.array(d["y_all"])
        paths = d["all_paths"]
        if is_anom:
            cands = [p for p,yi in zip(paths,y) if yi==1 and f"/{defect}/" in p and p in lookup]
        else:
            cands = [p for p,yi in zip(paths,y) if yi==0 and f"/{norm_tag}/" in p and p in lookup]
        if cands:
            explain(cands[0], cat, lookup, baselines, ALL_NAMES, TYPES,
                    thr_n, thr_v, "ANOMALOUS" if is_anom else "NORMAL")

    anom_recs = [r for r in test_recs if r["y"]==1]
    max_ent_r = max(anom_recs, key=lambda r: r["ent"])
    min_ent_r = min(anom_recs, key=lambda r: r["ent"])
    print(f"\n  [Highest-entropy anomaly — novel/ambiguous]")
    explain(max_ent_r["path"], max_ent_r["cat"], lookup, baselines,
            ALL_NAMES, TYPES, thr_n, thr_v)
    print(f"\n  [Lowest-entropy anomaly — known type, confident]")
    explain(min_ent_r["path"], min_ent_r["cat"], lookup, baselines,
            ALL_NAMES, TYPES, thr_n, thr_v)

    json.dump({"normal_entropy":{"mean":float(ne.mean()),"std":float(ne.std())},
               "anomaly_entropy":{"mean":float(ae.mean()),"std":float(ae.std())},
               "thr_normal":thr_n, "thr_novel":thr_v,
               "auroc_entropy":auc_ent, "auroc_signed":auc_sgn,
               "defect_type_entropy":dm},
              open(_OUT/"entropy_summary.json","w"), indent=2)
    print(f"\n  Saved → {_OUT}/")
    print("=" * 72)

if __name__ == "__main__":
    main()
