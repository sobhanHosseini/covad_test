"""
discover_structural_concepts.py — Structural concept discovery via DINOv2 + VLM.

Pipeline
────────
  1. Extract max-pooled DINOv2 patch features (1024-dim) for all images in
     6 structural MVTec categories.
  2. Find the 100 most discriminative feature dimensions (|μ_anom - μ_norm|).
  3. K-means (K=8) on anomalous images projected onto those 100 dims.
  4. Build 3×3 grids of the 9 most representative images per cluster.
  5. Query VLM (gemma4:e4b via Ollama) to name each cluster concept.
  6. Train and evaluate one LogisticRegression concept head per cluster.
  7. Print final vocabulary table.

NO SAE ANYWHERE — entirely in 1024-dim max-pooled DINOv2 space.

DINOv2 API (from features/dinov2_extractor.py):
  x = dino._prepare(images)              → (B,3,224,224) normalised tensor
  cls, patches = dino._run(x)            → (B,1024) and (B,256,1024)
  max_pool = patches.max(dim=1).values   → (B,1024)

Ollama model: gemma4:e4b @ http://localhost:6000
"""

from __future__ import annotations

import io
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

# ── Paths ─────────────────────────────────────────────────────────────────────
_COVAD = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_COVAD))
from features.dinov2_extractor import DINOv2Extractor   # noqa: E402

_OUT      = Path(__file__).resolve().parent.parent / "outputs"
_GRIDS    = _OUT / "structural_grids"
_VLM_JSON = _OUT / "structural_vlm_results.json"
_GRIDS.mkdir(parents=True, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
_MVTEC   = Path("/home/sobhan_hosseini/datasets/mvtec")
_DEVICE  = "cuda"
_BATCH   = 8
_K       = 8          # K-means clusters
_TOP_DIM = 100        # discriminative feature dimensions
_N_GRID  = 9         # images per cluster grid (3×3)
_CELL_PX = 224       # each cell in the grid is CELL_PX × CELL_PX
_OLLAMA  = "http://localhost:6000"
_VLM_MDL = "gemma4:e4b"

STRUCT_CATS = ["cable", "transistor", "toothbrush", "bottle", "capsule", "screw"]

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


# ── Step 1 — Data collection ──────────────────────────────────────────────────

def collect_images(cat: str) -> list[dict]:
    """Return list of {path, category, defect_type, label} for one category."""
    root   = _MVTEC / cat
    items: list[dict] = []

    for split in ("train", "test"):
        good = root / split / "good"
        if good.exists():
            for p in sorted(good.glob("*.png")):
                items.append({"path": str(p), "category": cat,
                              "defect_type": "good", "label": 0})

    test = root / "test"
    for d in sorted(test.iterdir()):
        if d.name == "good" or not d.is_dir():
            continue
        for p in sorted(d.glob("*.png")):
            items.append({"path": str(p), "category": cat,
                          "defect_type": d.name, "label": 1})
    return items


# ── Step 1 — Feature extraction ───────────────────────────────────────────────

@torch.no_grad()
def extract_maxpool(
    paths: list[str],
    dino: DINOv2Extractor,
    batch_size: int = 8,
) -> np.ndarray:
    """Extract max-pooled patch features. Returns (N, 1024) float32 array."""
    parts = []
    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start:start + batch_size]
        images = [Image.open(p).convert("RGB") for p in batch_paths]
        x = dino._prepare(images)
        _, patches = dino._run(x)            # (b, 256, 1024)
        max_pool = patches.max(dim=1).values  # (b, 1024)
        parts.append(max_pool.cpu().float().numpy())
        print(f"    {min(start + batch_size, len(paths))}/{len(paths)}", end="\r", flush=True)
    print()
    return np.concatenate(parts, axis=0)


# ── Step 3 — Grid builder ─────────────────────────────────────────────────────

def build_grid(image_paths: list[str], cell_px: int = _CELL_PX) -> Image.Image:
    """Arrange up to 9 full images into a 3×3 grid.  Grey cells for missing."""
    n   = min(len(image_paths), _N_GRID)
    grid = Image.new("RGB", (3 * cell_px, 3 * cell_px), color=(128, 128, 128))
    for i in range(n):
        img = Image.open(image_paths[i]).convert("RGB")
        img = img.resize((cell_px, cell_px), Image.LANCZOS)
        row, col = divmod(i, 3)
        grid.paste(img, (col * cell_px, row * cell_px))
    return grid


# ── Step 4 — VLM query ───────────────────────────────────────────────────────

def _pil_to_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _parse_vlm(text: str) -> dict:
    def _get(tag):
        m = re.search(rf"{tag}:\s*(.+)", text, re.IGNORECASE)
        return m.group(1).strip() if m else "PARSE_ERROR"
    return {
        "concept_name":   _get("CONCEPT_NAME"),
        "common_pattern": _get("COMMON_PATTERN"),
        "confidence":     _get("CONFIDENCE").lower(),
        "raw":            text,
    }


def query_vlm(grid: Image.Image, cluster_id: int) -> dict:
    """Send grid image to VLM, parse response."""
    from ollama import Client
    client = Client(host=_OLLAMA)
    try:
        resp = client.chat(
            model=_VLM_MDL,
            messages=[{
                "role":    "user",
                "content": _VLM_PROMPT,
                "images":  [_pil_to_bytes(grid)],
            }],
        )
        text = resp["message"]["content"]
    except Exception as exc:
        text = str(exc)
        print(f"  [WARN] VLM call failed: {exc}")
    parsed = _parse_vlm(text)
    parsed["cluster_id"] = cluster_id
    return parsed


# ── Step 6 — Concept head training ───────────────────────────────────────────

def train_concept_head(
    feats_anom: np.ndarray,
    feats_norm: np.ndarray,
    mask_pos: np.ndarray,         # bool mask over anom rows
) -> dict:
    """Train one binary LogReg head and evaluate on 20% hold-out.

    Positive: anomalous images where mask_pos is True (in this cluster)
    Negative: all normal images + anomalous images where mask_pos is False
    Features: 1024-dim max-pooled DINOv2 (NO SAE)
    """
    X_pos     = feats_anom[mask_pos]
    X_neg_anom = feats_anom[~mask_pos]
    X = np.concatenate([X_pos, X_neg_anom, feats_norm], axis=0)
    y = np.array([1]*len(X_pos) + [0]*(len(X_neg_anom) + len(feats_norm)), dtype=np.int32)

    if len(X_pos) < 5:
        return {"f1": float("nan"), "accuracy": float("nan"), "auc": float("nan"),
                "n_pos": int(mask_pos.sum()), "n_neg": len(X) - int(mask_pos.sum())}

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000,
                             solver="lbfgs", random_state=42)
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    y_prob = clf.predict_proba(X_te)[:, 1]
    return {
        "f1":       float(f1_score(y_te, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_te, y_pred)),
        "auc":      float(roc_auc_score(y_te, y_prob)),
        "n_pos":    int(mask_pos.sum()),
        "n_neg":    len(X) - int(mask_pos.sum()),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 72)
    print("  Structural Concept Discovery — DINOv2 max-pool features")
    print("=" * 72)

    # Load model (NO SAE)
    print("\n[init] Loading DINOv2 ViT-L/14-reg …")
    dino = DINOv2Extractor("dinov2_vitl14_reg", device=torch.device(_DEVICE))
    print(f"  embed_dim={dino.EMBED_DIM}  (max-pool → 1024-dim per image)")

    # ── Step 1: Collect images and extract features ───────────────────────────
    all_items:  list[dict] = []
    print("\n[Step 1] Collecting images and extracting max-pool features …")
    for cat in STRUCT_CATS:
        items = collect_images(cat)
        n0 = sum(it["label"] == 0 for it in items)
        n1 = sum(it["label"] == 1 for it in items)
        print(f"  {cat:<12}: {n0} normal + {n1} anomalous = {len(items)} total")
        paths = [it["path"] for it in items]
        feats = extract_maxpool(paths, dino, batch_size=_BATCH)
        for it, f in zip(items, feats):
            it["feat"] = f
        all_items.extend(items)

    # Split normal / anomaly
    anom_items = [it for it in all_items if it["label"] == 1]
    norm_items = [it for it in all_items if it["label"] == 0]
    feats_anom = np.stack([it["feat"] for it in anom_items]).astype(np.float32)
    feats_norm = np.stack([it["feat"] for it in norm_items]).astype(np.float32)
    print(f"\n  Total anomalous: {len(anom_items)}  |  Total normal: {len(norm_items)}")

    # ── Step 2: Discriminative dimensions ────────────────────────────────────
    print(f"\n[Step 2] Finding top {_TOP_DIM} discriminative DINOv2 dimensions …")
    disc = feats_anom.mean(axis=0) - feats_norm.mean(axis=0)   # (1024,)
    top_dims = np.argsort(np.abs(disc))[::-1][:_TOP_DIM]        # (100,)
    print(f"  Max |disc| = {np.abs(disc).max():.4f}  "
          f"  dim range: {top_dims.min()}–{top_dims.max()}")

    feats_anom_proj = feats_anom[:, top_dims]   # (N_anom, 100)

    # ── Step 3: K-means clustering ────────────────────────────────────────────
    print(f"\n[Step 3] K-means (K={_K}) on {len(anom_items)} anomalous images …")
    km = KMeans(n_clusters=_K, random_state=42, n_init=10)
    labels = km.fit_predict(feats_anom_proj)
    centroids = km.cluster_centers_   # (K, 100)

    for k in range(_K):
        cats_in = sorted(set(anom_items[i]["category"] for i in range(len(anom_items)) if labels[i] == k))
        print(f"  Cluster {k}: {(labels == k).sum():>3} images  categories: {cats_in}")

    # ── Step 4: Build grids and query VLM ────────────────────────────────────
    print(f"\n[Step 4] Building grids and querying VLM ({_VLM_MDL}) …")

    # Load partial results if script was interrupted
    if _VLM_JSON.exists():
        with open(_VLM_JSON) as f:
            vlm_results: list[dict] = json.load(f)
        done_clusters = {r["cluster_id"] for r in vlm_results}
        print(f"  Loaded {len(vlm_results)} partial results from {_VLM_JSON.name}")
    else:
        vlm_results = []
        done_clusters = set()

    for k in range(_K):
        if k in done_clusters:
            print(f"  Cluster {k}: already done → {next(r['concept_name'] for r in vlm_results if r['cluster_id']==k)}")
            continue

        mask = labels == k
        idx_in_cluster = np.where(mask)[0]

        # 9 closest to centroid
        dists = np.linalg.norm(feats_anom_proj[idx_in_cluster] - centroids[k], axis=1)
        top9_idx = idx_in_cluster[np.argsort(dists)[:_N_GRID]]
        rep_paths = [anom_items[i]["path"] for i in top9_idx]

        # Build and save grid
        grid = build_grid(rep_paths, cell_px=_CELL_PX)
        grid_path = _GRIDS / f"cluster_{k}.png"
        grid.save(grid_path)

        # VLM query
        print(f"  Cluster {k} ({mask.sum()} images) — querying VLM …", flush=True)
        result = query_vlm(grid, cluster_id=k)

        # Annotate with cluster metadata
        cats_in = sorted(set(anom_items[i]["category"] for i in idx_in_cluster))
        defects_in = sorted(set(anom_items[i]["defect_type"] for i in idx_in_cluster))
        result["cluster_size"]   = int(mask.sum())
        result["categories"]     = cats_in
        result["defect_types"]   = defects_in
        result["representative_images"] = rep_paths

        vlm_results.append(result)
        with open(_VLM_JSON, "w") as f:
            json.dump(vlm_results, f, indent=2)

        conf = result["confidence"]
        conf_sym = "✓" if conf in ("high", "medium") else "⚠"
        print(f"  → {conf_sym} concept: {result['concept_name']!r}  "
              f"confidence: {conf}  cats: {cats_in}")

    # ── Step 5: Print vocabulary ──────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  STRUCTURAL CONCEPT VOCABULARY ({_K} clusters)")
    print(f"{'='*72}")
    print(f"\n  {'#':<3} {'Concept':<35} {'Size':>5} {'Conf':<8} {'Categories'}")
    print("  " + "─" * 68)
    for r in sorted(vlm_results, key=lambda x: x["cluster_id"]):
        print(f"  {r['cluster_id']:<3} {r['concept_name']:<35} "
              f"{r['cluster_size']:>5} {r['confidence']:<8} "
              f"{', '.join(r['categories'])}")

    # ── Step 6: Train concept heads ───────────────────────────────────────────
    print(f"\n[Step 6] Training concept heads (LogReg, C=1.0, 1024-dim features, NO SAE)")
    print(f"  Positive: cluster images  |  Negative: normals + other-cluster anomalies\n")

    head_results: list[dict] = []
    used_clusters = [r for r in vlm_results if r["confidence"] in ("high", "medium")]

    hdr = f"  {'Concept':<35} {'N+':>5} {'N-':>6} {'Acc':>7} {'F1':>7} {'AUC':>7}"
    print(hdr);  print("  " + "─" * 65)

    for r in used_clusters:
        k    = r["cluster_id"]
        mask = (labels == k)
        m    = train_concept_head(feats_anom, feats_norm, mask)
        cats_str = ", ".join(r["categories"])
        print(f"  {r['concept_name']:<35} {m['n_pos']:>5} {m['n_neg']:>6} "
              f"{m['accuracy']:>7.3f} {m['f1']:>7.3f} {m['auc']:>7.3f}")
        head_results.append({
            "concept":    r["concept_name"],
            "cluster_id": k,
            "categories": r["categories"],
            "confidence": r["confidence"],
            **m,
        })

    # ── Step 7: Final report ──────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  FINAL REPORT — Structural Concept Heads (1024-dim DINOv2, NO SAE)")
    print(f"{'='*72}")
    print(f"\n  {'Concept':<35} {'Size':>5} {'F1':>7} {'AUC':>7}  Categories")
    print("  " + "─" * 72)
    for h in sorted(head_results, key=lambda x: -x["f1"] if not np.isnan(x["f1"]) else -1):
        cats_str = ", ".join(h["categories"])
        print(f"  {h['concept']:<35} {h['n_pos']:>5} {h['f1']:>7.3f} "
              f"{h['auc']:>7.3f}  {cats_str}")

    valid_f1s = [h["f1"] for h in head_results if not np.isnan(h["f1"])]
    valid_aucs = [h["auc"] for h in head_results if not np.isnan(h["auc"])]
    print("  " + "─" * 72)
    print(f"\n  Mean F1  of structural concept heads : {np.mean(valid_f1s):.3f}")
    print(f"  Mean AUC of structural concept heads : {np.mean(valid_aucs):.3f}")
    print(f"  vs surface concept heads mean F1     : 0.909")
    print(f"\n  Grids saved → {_GRIDS}/")
    print(f"  VLM results → {_VLM_JSON}")
    print("=" * 72)


if __name__ == "__main__":
    main()
