"""
discover_allcat_concepts.py — Image-level concept discovery on ALL 15 MVTec categories.

Expands the structural-only discovery (discover_structural_concepts.py, K=8)
to the full cross-category pool (K=12), letting the clustering surface both
structural and surface-type image-level concepts.

Pipeline:
  1. Extract max-pooled DINOv2 features (1024-dim) for ALL 15 categories.
     Follows v1 cache path/split ordering for downstream compatibility.
     Cached to: ma_cbm/outputs/allcat_maxpool_cache.pkl
  2. Top-100 discriminative feature dimensions (|μ_anom - μ_norm|) across all cats.
  3. K-means (K=12) on all anomalous images projected onto top-100 dims.
  4. 3×3 full-image grids saved to: ma_cbm/outputs/image_level_grids/cluster_{k}.png
  5. VLM naming (gemma4:e4b) using same structural-anomaly prompt.
     Results: ma_cbm/outputs/image_level_vlm_results_allcat.json
  6. Train one LogReg head per cluster. Evaluate on 20% hold-out.
  7. Separability check: per-category anomaly/normal activation ratio.

DINOv2 API (unchanged from structural script):
  x = dino._prepare(images)                → (B,3,224,224)
  _, patches = dino._run(x)               → (B,256,1024)
  max_pool = patches.max(dim=1).values    → (B,1024)
"""

from __future__ import annotations

import io
import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

_COVAD = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_COVAD))
from features.dinov2_extractor import DINOv2Extractor   # noqa: E402

_OUT        = Path(__file__).resolve().parent.parent / "outputs"
_GRIDS      = _OUT / "image_level_grids"
_VLM_JSON   = _OUT / "image_level_vlm_results_allcat.json"
_MP_CACHE   = _OUT / "allcat_maxpool_cache.pkl"
_HEADS_PKL  = _OUT / "allcat_image_level_heads.pkl"
_SEP_JSON   = _OUT / "allcat_separability.json"
_GRIDS.mkdir(parents=True, exist_ok=True)

_V1_CACHE  = _COVAD / "mac/outputs/concil/concept_vector_cache.pkl"
_DEVICE    = "cuda"
_BATCH     = 8
_K         = 12
_TOP_DIM   = 100
_N_GRID    = 9
_CELL_PX   = 224
_OLLAMA    = "http://localhost:6000"
_VLM_MDL   = "gemma4:e4b"

MVTEC_15 = [
    "bottle","cable","capsule","carpet","grid","hazelnut","leather",
    "metal_nut","pill","screw","tile","toothbrush","transistor","wood","zipper",
]

_VLM_PROMPT = """\
You are analyzing industrial inspection images showing defects.
These 9 images all show the same type of structural or \
geometric anomaly in an industrial object.

The anomaly involves the OVERALL structure of the object — \
component shape, orientation, presence or absence of parts, \
spatial relationships — NOT surface-level defects like \
cracks or stains.

Step 1: Describe what you see in each image briefly.
Step 2: Identify what structural or geometric anomaly \
pattern is common to most images. What component \
is wrong? How does it deviate from expected? \
What is missing or misaligned?
Step 3: Give a short name (2-5 words) for this structural \
concept.

Output format:
COMMON_PATTERN: [one sentence]
CONCEPT_NAME: [lowercase_underscores]
CONFIDENCE: [high / medium / low]\
"""


# ── Step 1: Extract max-pool features ────────────────────────────────────────

@torch.no_grad()
def _maxpool_batch(paths: list[str], dino, batch_size: int) -> np.ndarray:
    parts = []
    for i in range(0, len(paths), batch_size):
        imgs = [Image.open(p).convert("RGB") for p in paths[i:i+batch_size]]
        x    = dino._prepare(imgs)
        _, patches = dino._run(x)                       # (b, 256, 1024)
        parts.append(patches.max(dim=1).values.detach().cpu().float().numpy())
        print(f"    {min(i+batch_size, len(paths))}/{len(paths)}", end="\r", flush=True)
    print()
    return np.concatenate(parts, axis=0)                # (N, 1024)


def build_maxpool_cache(dino, v1_cache) -> dict:
    """Extract max-pool features for all 15 categories, following v1 path ordering."""
    cache = {}
    for cat in MVTEC_15:
        v1 = v1_cache[cat]
        print(f"  [{cat}] {len(v1['all_paths'])} images …")
        feats = _maxpool_batch(v1["all_paths"], dino, _BATCH)
        cache[cat] = {
            "all_paths":    v1["all_paths"],
            "y_all":        v1["y_all"],
            "is_train_mask": v1["is_train_mask"],
            "maxpool_feats": feats,
            "n_train_normal":  v1["n_train_normal"],
            "n_defect_train":  v1["n_defect_train"],
            "n_test_normal":   v1["n_test_normal"],
            "n_defect_test":   v1["n_defect_test"],
        }
    pickle.dump(cache, open(_MP_CACHE, "wb"))
    print(f"  Saved → {_MP_CACHE.name}")
    return cache


# ── Steps 2-3: Discriminative dims + K-means ─────────────────────────────────

def find_disc_dims(mp_cache) -> np.ndarray:
    all_anom = np.concatenate([
        mp_cache[c]["maxpool_feats"][np.array(mp_cache[c]["y_all"]) == 1]
        for c in MVTEC_15
    ])
    all_norm = np.concatenate([
        mp_cache[c]["maxpool_feats"][np.array(mp_cache[c]["y_all"]) == 0]
        for c in MVTEC_15
    ])
    disc = all_anom.mean(0) - all_norm.mean(0)
    return np.argsort(np.abs(disc))[::-1][:_TOP_DIM]


def collect_anom(mp_cache):
    items, feats = [], []
    for cat in MVTEC_15:
        mp  = mp_cache[cat]["maxpool_feats"]
        y   = np.array(mp_cache[cat]["y_all"])
        pth = mp_cache[cat]["all_paths"]
        for i in np.where(y == 1)[0]:
            defect = Path(pth[i]).parent.name
            items.append({"path": pth[i], "category": cat, "defect_type": defect})
            feats.append(mp[i])
    return items, np.stack(feats)


# ── Step 4: Grid builder ──────────────────────────────────────────────────────

def build_grid(image_paths: list[str]) -> Image.Image:
    n    = min(len(image_paths), _N_GRID)
    grid = Image.new("RGB", (3 * _CELL_PX, 3 * _CELL_PX), (128, 128, 128))
    for i in range(n):
        img = Image.open(image_paths[i]).convert("RGB").resize(
            (_CELL_PX, _CELL_PX), Image.LANCZOS
        )
        row, col = divmod(i, 3)
        grid.paste(img, (col * _CELL_PX, row * _CELL_PX))
    return grid


# ── Step 5: VLM naming ────────────────────────────────────────────────────────

def _pil_to_bytes(img):
    buf = io.BytesIO();  img.save(buf, format="PNG");  return buf.getvalue()

def _parse_vlm(text):
    def _get(tag):
        m = re.search(rf"{tag}:\s*(.+)", text, re.IGNORECASE)
        return m.group(1).strip() if m else "PARSE_ERROR"
    return {"concept_name": _get("CONCEPT_NAME"),
            "common_pattern": _get("COMMON_PATTERN"),
            "confidence": _get("CONFIDENCE").lower(), "raw": text}

def query_vlm(grid, cluster_id):
    from ollama import Client
    client = Client(host=_OLLAMA)
    try:
        resp = client.chat(model=_VLM_MDL, messages=[{
            "role": "user", "content": _VLM_PROMPT,
            "images": [_pil_to_bytes(grid)],
        }])
        text = resp["message"]["content"]
    except Exception as e:
        text = str(e);  print(f"  [WARN] {e}")
    parsed = _parse_vlm(text)
    parsed["cluster_id"] = cluster_id
    return parsed


# ── Step 6: Concept head training ────────────────────────────────────────────

def train_head(feats_anom, feats_norm, mask_pos):
    X_pos      = feats_anom[mask_pos]
    X_neg_anom = feats_anom[~mask_pos]
    X = np.concatenate([X_pos, X_neg_anom, feats_norm])
    y = np.array([1]*len(X_pos) + [0]*(len(X_neg_anom) + len(feats_norm)), dtype=np.int32)
    if len(X_pos) < 5:
        return None, {"f1": float("nan"), "auc": float("nan"),
                      "n_pos": int(mask_pos.sum()), "n_neg": len(X)-int(mask_pos.sum())}
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000,
                             solver="lbfgs", random_state=42)
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te);  y_prob = clf.predict_proba(X_te)[:, 1]
    return clf, {
        "f1":  float(f1_score(y_te, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_te, y_prob)),
        "n_pos": int(mask_pos.sum()), "n_neg": len(X)-int(mask_pos.sum()),
    }


# ── Step 7: Separability check ────────────────────────────────────────────────

def separability_check(heads_dict, mp_cache):
    """Per-category mean activation on anomaly vs normal, and ratio."""
    head_names = list(heads_dict.keys())
    sep = {}
    for cat in MVTEC_15:
        mp = mp_cache[cat]["maxpool_feats"]
        y  = np.array(mp_cache[cat]["y_all"])
        anom_feats = mp[y == 1]
        norm_feats = mp[y == 0]
        cat_sep = {}
        for name, clf in heads_dict.items():
            if clf is None:
                continue
            p_anom = clf.predict_proba(anom_feats)[:, 1].mean() if len(anom_feats) else 0.0
            p_norm = clf.predict_proba(norm_feats)[:, 1].mean() if len(norm_feats) else 0.0
            ratio  = p_anom / (p_norm + 1e-6)
            cat_sep[name] = {"mean_anom": float(p_anom), "mean_norm": float(p_norm),
                             "ratio": float(ratio)}
        sep[cat] = cat_sep
    return sep


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print(f"  All-Category Image-Level Concept Discovery (K={_K}, 15 categories)")
    print("=" * 72)

    print("\n[init] Loading DINOv2 ViT-L/14-reg …")
    dino = DINOv2Extractor("dinov2_vitl14_reg", device=torch.device(_DEVICE))
    v1_cache = pickle.load(open(_V1_CACHE, "rb"))

    # ── Step 1 ────────────────────────────────────────────────────────────────
    if _MP_CACHE.exists():
        print(f"\n[Step 1] Loading max-pool cache from {_MP_CACHE.name} …")
        mp_cache = pickle.load(open(_MP_CACHE, "rb"))
    else:
        print(f"\n[Step 1] Extracting max-pool features for all 15 categories …")
        mp_cache = build_maxpool_cache(dino, v1_cache)

    # ── Step 2 ────────────────────────────────────────────────────────────────
    print(f"\n[Step 2] Finding top {_TOP_DIM} discriminative DINOv2 dimensions …")
    top_dims = find_disc_dims(mp_cache)
    print(f"  dim range: {top_dims.min()}–{top_dims.max()}")

    # ── Step 3 ────────────────────────────────────────────────────────────────
    print(f"\n[Step 3] Collecting anomalous images and running K-means (K={_K}) …")
    anom_items, feats_anom = collect_anom(mp_cache)
    feats_proj = feats_anom[:, top_dims]          # (N_anom, 100)
    print(f"  Total anomalous images: {len(anom_items)}")

    km     = KMeans(n_clusters=_K, random_state=42, n_init=10)
    labels = km.fit_predict(feats_proj)
    centroids = km.cluster_centers_

    for k in range(_K):
        cats_in = sorted(set(anom_items[i]["category"] for i in range(len(anom_items))
                             if labels[i] == k))
        print(f"  Cluster {k:2d}: {(labels==k).sum():>4} images  cats: {cats_in}")

    # ── Steps 4-5: Grids + VLM ───────────────────────────────────────────────
    print(f"\n[Steps 4-5] Building grids and querying VLM …")

    if _VLM_JSON.exists():
        vlm_results = json.load(open(_VLM_JSON))
        done = {r["cluster_id"] for r in vlm_results}
        print(f"  Loaded {len(vlm_results)} existing results")
    else:
        vlm_results, done = [], set()

    for k in range(_K):
        if k in done:
            name = next(r["concept_name"] for r in vlm_results if r["cluster_id"] == k)
            print(f"  Cluster {k:2d}: [cached] → {name!r}")
            continue

        mask = labels == k
        idx  = np.where(mask)[0]
        dists = np.linalg.norm(feats_proj[idx] - centroids[k], axis=1)
        top9  = idx[np.argsort(dists)[:_N_GRID]]
        rep_paths = [anom_items[i]["path"] for i in top9]

        grid = build_grid(rep_paths)
        grid.save(_GRIDS / f"cluster_{k}.png")

        cats_in    = sorted(set(anom_items[i]["category"] for i in idx))
        defects_in = sorted(set(anom_items[i]["defect_type"] for i in idx))

        print(f"  Cluster {k:2d} ({mask.sum()} images) — VLM …", flush=True)
        result = query_vlm(grid, k)
        result["cluster_size"]          = int(mask.sum())
        result["categories"]            = cats_in
        result["defect_types"]          = defects_in
        result["representative_images"] = rep_paths

        vlm_results.append(result)
        json.dump(vlm_results, open(_VLM_JSON, "w"), indent=2)

        conf_sym = "✓" if result["confidence"] in ("high","medium") else "⚠"
        print(f"  → {conf_sym} {result['concept_name']!r}  [{result['confidence']}]"
              f"  cats: {cats_in}")

    # ── Step 6: Vocabulary + Heads ────────────────────────────────────────────
    print(f"\n[Step 6] Training image-level concept heads (1024-dim max-pool) …\n")

    feats_norm_all = np.concatenate([
        mp_cache[c]["maxpool_feats"][np.array(mp_cache[c]["y_all"]) == 0]
        for c in MVTEC_15
    ])

    cluster_names = {r["cluster_id"]: r["concept_name"] for r in vlm_results}
    heads_dict:  dict = {}
    head_metrics: list = []

    hdr = f"  {'#':>3} {'Concept':<40} {'N+':>5} {'N-':>6} {'F1':>7} {'AUC':>7}"
    print(hdr);  print("  " + "─" * 70)

    for k in range(_K):
        name = cluster_names.get(k, f"cluster_{k}")
        r    = next(r for r in vlm_results if r["cluster_id"] == k)
        mask = labels == k
        clf, m = train_head(feats_anom, feats_norm_all, mask)
        heads_dict[name] = clf
        cats_str = ", ".join(r["categories"])
        print(f"  {k:>3} {name:<40} {m['n_pos']:>5} {m['n_neg']:>6} "
              f"{m['f1']:>7.3f} {m['auc']:>7.3f}")
        head_metrics.append({"cluster": k, "concept": name, **m,
                              "categories": r["categories"], "confidence": r["confidence"]})

    valid_f1s = [h["f1"] for h in head_metrics if not np.isnan(h["f1"])]
    print("  " + "─" * 70)
    print(f"\n  Mean F1: {np.mean(valid_f1s):.3f}")

    pickle.dump(heads_dict, open(_HEADS_PKL, "wb"))
    print(f"  Heads saved → {_HEADS_PKL.name}")

    # ── Step 7: Separability ─────────────────────────────────────────────────
    print(f"\n[Step 7] Separability check per category …")
    usable_heads = {k: v for k, v in heads_dict.items() if v is not None}
    sep = separability_check(usable_heads, mp_cache)

    # Print compact separability table
    concept_names = list(usable_heads.keys())
    SURFACE = {"carpet","grid","hazelnut","leather","metal_nut","pill","tile","wood","zipper"}
    print(f"\n  {'Category':<13} {'Type':<5}", end="")
    for n in concept_names:
        print(f"  {n[:8]:>8}", end="")
    print()
    print("  " + "─" * (18 + 10 * len(concept_names)))

    for cat in MVTEC_15:
        tag = "S" if cat in SURFACE else "T"
        print(f"  {cat:<13} [{tag}]", end="")
        for n in concept_names:
            ratio = sep[cat].get(n, {}).get("ratio", 0.0)
            print(f"  {ratio:>8.2f}", end="")
        print()

    json.dump({"separability": sep, "head_metrics": head_metrics}, open(_SEP_JSON, "w"), indent=2)
    print(f"\n  Separability saved → {_SEP_JSON.name}")
    print(f"  Grids → {_GRIDS}/")
    print(f"  VLM   → {_VLM_JSON.name}")
    print("=" * 72)


if __name__ == "__main__":
    main()
