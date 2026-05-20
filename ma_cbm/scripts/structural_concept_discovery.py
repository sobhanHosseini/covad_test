"""
structural_concept_discovery.py — Discover structural anomaly concepts
from DINOv2 max-pooled features (NO SAE — SAE loses structural information).

Validated: max-pooled DINOv2 patches → I-AUC 0.982 on structural categories.
This script finds what structural concepts exist via K-means + VLM naming.

DINOv2 API (from features/dinov2_extractor.py):
  dino._prepare(images)  → (B, 3, 224, 224) tensor
  dino._run(x)           → (cls_token, patch_tokens)
                            cls_token    : (B, 1024)
                            patch_tokens : (B, 256, 1024)
  Max-pool: patch_tokens.max(dim=1).values → (B, 1024)

Structural categories: cable, transistor, toothbrush, bottle, capsule, screw
"""

from __future__ import annotations

import io
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

_COVAD = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_COVAD))

from features.dinov2_extractor import DINOv2Extractor

_MVTEC   = Path("/home/sobhan_hosseini/datasets/mvtec")
_OUT     = Path(__file__).resolve().parent.parent / "outputs"
_GRIDS   = _OUT / "structural_grids"
_OUT.mkdir(parents=True, exist_ok=True)
_GRIDS.mkdir(parents=True, exist_ok=True)

_DEVICE  = "cuda"
_BATCH   = 8
_K       = 8          # K-means clusters
_TOP_DIM = 100        # discriminative dimensions for clustering
_GRID_SZ = 256        # pixels per image in the 3×3 grid

STRUCTURAL_CATS = ["cable", "transistor", "toothbrush", "bottle", "capsule", "screw"]

_VLM_PROMPT = """\
You are analyzing industrial inspection images showing defects.
These 9 images all show the same type of structural or \
geometric anomaly in an industrial object.

The anomaly involves the OVERALL structure of the object —
component shape, orientation, presence or absence of parts,
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


# ── Data collection ───────────────────────────────────────────────────────────

def collect_paths(cat: str) -> tuple[list[dict], list[dict]]:
    """Return (normal_records, anomaly_records) for one category."""
    root = _MVTEC / cat
    normals, anomalies = [], []

    for split in ("train", "test"):
        good = root / split / "good"
        if good.exists():
            for p in sorted(good.glob("*.png")):
                normals.append({"path": str(p), "category": cat, "defect_type": "good"})

    test_dir = root / "test"
    for d in sorted(test_dir.iterdir()):
        if d.name == "good" or not d.is_dir():
            continue
        for p in sorted(d.glob("*.png")):
            anomalies.append({"path": str(p), "category": cat, "defect_type": d.name})

    return normals, anomalies


# ── Feature extraction ────────────────────────────────────────────────────────

@torch.no_grad()
def extract_maxpool(
    records: list[dict],
    dino: DINOv2Extractor,
    feat_key: str = "max_feat",
    batch_size: int = 8,
) -> list[dict]:
    """Add 'max_feat' (1024-dim numpy) to each record in-place."""
    n = len(records)
    for start in range(0, n, batch_size):
        batch = records[start:start + batch_size]
        images = [Image.open(r["path"]).convert("RGB") for r in batch]
        x = dino._prepare(images)
        _, patch_tokens = dino._run(x)          # (b, 256, 1024)
        maxp = patch_tokens.max(dim=1).values   # (b, 1024)
        for r, v in zip(batch, maxp.cpu().float().numpy()):
            r[feat_key] = v
        done = min(start + batch_size, n)
        print(f"    {done}/{n}", end="\r", flush=True)
    print()
    return records


# ── Grid builder ──────────────────────────────────────────────────────────────

def build_grid(image_paths: list[str], grid_sz: int = _GRID_SZ) -> Image.Image:
    """Build a 3×3 PIL grid from 9 image paths (or fewer, grey fill)."""
    cols, rows = 3, 3
    grid = Image.new("RGB", (cols * grid_sz, rows * grid_sz), color=(128, 128, 128))
    for i, p in enumerate(image_paths[:9]):
        img = Image.open(p).convert("RGB").resize((grid_sz, grid_sz), Image.LANCZOS)
        c, r = i % cols, i // cols
        grid.paste(img, (c * grid_sz, r * grid_sz))
    return grid


# ── VLM query ─────────────────────────────────────────────────────────────────

def _parse(text: str) -> dict[str, str]:
    import re
    def _get(tag):
        m = re.search(rf"{tag}:\s*(.+)", text, re.IGNORECASE)
        return m.group(1).strip() if m else "PARSE_ERROR"
    return {
        "concept_name":   _get("CONCEPT_NAME"),
        "common_pattern": _get("COMMON_PATTERN"),
        "confidence":     _get("CONFIDENCE"),
        "raw":            text,
    }


def query_vlm(grid_img: Image.Image, cluster_id: int,
              model: str = "gemma4:e4b",
              host: str = "http://localhost:6000") -> dict[str, str]:
    """Send 3×3 grid to VLM and return parsed concept info."""
    from ollama import Client
    buf = io.BytesIO()
    grid_img.save(buf, format="PNG")
    img_bytes = buf.getvalue()
    client = Client(host=host)
    resp = client.chat(
        model=model,
        messages=[{"role": "user", "content": _VLM_PROMPT, "images": [img_bytes]}],
    )
    parsed = _parse(resp.message.content)
    parsed["cluster_id"] = cluster_id
    return parsed


# ── Concept head evaluation ───────────────────────────────────────────────────

def train_head(X: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Train LogReg on 1024-dim features, evaluate 20% test split."""
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    clf = LogisticRegression(C=1.0, class_weight="balanced",
                             max_iter=1000, solver="lbfgs", random_state=42)
    clf.fit(X_tr, y_tr)
    pred = clf.predict(X_te)
    prob = clf.predict_proba(X_te)[:, 1]
    return {
        "f1":       round(float(f1_score(y_te, pred, zero_division=0)), 4),
        "accuracy": round(float(accuracy_score(y_te, pred)), 4),
        "auc":      round(float(roc_auc_score(y_te, prob)), 4),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 68)
    print("  Structural Concept Discovery — DINOv2 max-pool features (no SAE)")
    print("=" * 68)

    print("\n[init] Loading DINOv2 ViT-L/14-reg …")
    dino = DINOv2Extractor("dinov2_vitl14_reg", device=torch.device(_DEVICE))

    # ── STEP 1: Extract max-pooled features ──────────────────────────────────
    cache_path = _OUT / "structural_maxpool_cache.pkl"
    if cache_path.exists():
        print("\n[Step 1] Loading cached features …")
        with open(cache_path, "rb") as f:
            all_normals, all_anomalies = pickle.load(f)
    else:
        print("\n[Step 1] Extracting max-pooled DINOv2 features …")
        all_normals: list[dict] = []
        all_anomalies: list[dict] = []

        for cat in STRUCTURAL_CATS:
            normals, anomalies = collect_paths(cat)
            print(f"  [{cat}] {len(normals)} normal, {len(anomalies)} anomalous")
            print(f"    Extracting normals …")
            extract_maxpool(normals, dino, batch_size=_BATCH)
            print(f"    Extracting anomalies …")
            extract_maxpool(anomalies, dino, batch_size=_BATCH)
            all_normals.extend(normals)
            all_anomalies.extend(anomalies)

        with open(cache_path, "wb") as f:
            pickle.dump((all_normals, all_anomalies), f)
        print(f"  Cached → {cache_path}")

    X_norm = np.stack([r["max_feat"] for r in all_normals])     # (N_norm, 1024)
    X_anom = np.stack([r["max_feat"] for r in all_anomalies])   # (N_anom, 1024)
    print(f"\n  Normal vectors  : {X_norm.shape}")
    print(f"  Anomaly vectors : {X_anom.shape}")

    # ── STEP 2: Discriminative dimensions ─────────────────────────────────────
    print(f"\n[Step 2] Finding top {_TOP_DIM} discriminative dimensions …")
    disc = X_anom.mean(axis=0) - X_norm.mean(axis=0)   # (1024,)
    top_dims = np.argsort(np.abs(disc))[::-1][:_TOP_DIM]
    print(f"  Max disc score: {np.abs(disc).max():.4f}  "
          f"Top dims: {top_dims[:5].tolist()} …")

    X_anom_sub  = X_anom[:, top_dims]   # (N_anom, 100)
    X_norm_sub  = X_norm[:, top_dims]   # (N_norm, 100)

    # Standardise for K-means
    scaler = StandardScaler().fit(X_anom_sub)
    X_anom_sc = scaler.transform(X_anom_sub)

    # ── STEP 3: K-means clustering ────────────────────────────────────────────
    print(f"\n[Step 3] K-means (K={_K}) on anomalous images …")
    km = KMeans(n_clusters=_K, random_state=42, n_init=10)
    labels = km.fit_predict(X_anom_sc)   # (N_anom,)

    for k in range(_K):
        mask = labels == k
        cats_in = sorted(set(all_anomalies[i]["category"] for i in np.where(mask)[0]))
        print(f"  Cluster {k}: {mask.sum():3d} images  categories: {cats_in}")

    # Find 9 closest images to each centroid
    cluster_rep_paths: list[list[str]] = []
    for k in range(_K):
        mask = labels == k
        idxs = np.where(mask)[0]
        center = km.cluster_centers_[k]
        dists  = np.linalg.norm(X_anom_sc[idxs] - center, axis=1)
        top9   = idxs[np.argsort(dists)[:9]]
        cluster_rep_paths.append([all_anomalies[i]["path"] for i in top9])

    # ── STEP 4: Build grids + VLM naming ─────────────────────────────────────
    print(f"\n[Step 4] Building grids and querying VLM …")
    vlm_results_path = _OUT / "structural_vlm_results.json"

    # Load prior results if crash-resuming
    if vlm_results_path.exists():
        with open(vlm_results_path) as f:
            vlm_results: list[dict] = json.load(f)
        done_ids = {r["cluster_id"] for r in vlm_results}
        print(f"  Resuming — already done: {sorted(done_ids)}")
    else:
        vlm_results, done_ids = [], set()

    for k in range(_K):
        if k in done_ids:
            print(f"  Cluster {k}: skipped (already done)")
            continue

        paths9 = cluster_rep_paths[k]
        grid   = build_grid(paths9)
        grid_path = _GRIDS / f"cluster_{k}.png"
        grid.save(grid_path)

        print(f"  Cluster {k}: querying VLM … ", end="", flush=True)
        try:
            result = query_vlm(grid, k)
            result["cluster_size"] = int((labels == k).sum())
            result["rep_paths"]    = paths9
            result["categories"]   = sorted(set(
                all_anomalies[i]["category"]
                for i in np.where(labels == k)[0]
            ))
            result["defect_types"] = sorted(set(
                all_anomalies[i]["defect_type"]
                for i in np.where(labels == k)[0]
            ))
            print(f"→ '{result['concept_name']}'  [{result['confidence']}]")
        except Exception as exc:
            print(f"FAILED: {exc}")
            result = {
                "cluster_id": k, "concept_name": "VLM_ERROR",
                "common_pattern": str(exc), "confidence": "none",
                "raw": str(exc),
                "cluster_size": int((labels == k).sum()),
                "rep_paths": paths9, "categories": [], "defect_types": [],
            }

        vlm_results.append(result)
        # Incremental save — safe against crashes
        with open(vlm_results_path, "w") as f:
            json.dump(vlm_results, f, indent=2)

    # Sort by cluster_id for consistent ordering
    vlm_results.sort(key=lambda x: x["cluster_id"])

    # ── STEP 5: Print vocabulary ──────────────────────────────────────────────
    print(f"\n[Step 5] Discovered structural vocabulary:")
    print(f"{'─'*66}")
    print(f"  {'Clust':>5} {'Concept name':<30} {'Size':>5} {'Conf':<8} {'Categories'}")
    print(f"  {'─'*62}")
    for r in vlm_results:
        cats = ", ".join(r.get("categories", []))
        print(f"  {r['cluster_id']:>5}  {r['concept_name']:<30} "
              f"{r['cluster_size']:>5}  {r['confidence']:<8}  {cats}")
    print(f"{'─'*66}")

    # ── STEP 6: Train structural concept heads ────────────────────────────────
    print(f"\n[Step 6] Training structural concept heads (1024-dim input, no SAE) …\n")

    head_results: list[dict] = []
    valid = [r for r in vlm_results if r["confidence"] in ("high", "medium")]
    invalid = [r for r in vlm_results if r["confidence"] not in ("high", "medium")]
    if invalid:
        print(f"  Skipping {len(invalid)} low-confidence / error clusters: "
              f"{[r['cluster_id'] for r in invalid]}")

    X_all = np.vstack([X_anom, X_norm])        # (N_anom + N_norm, 1024)
    # y: 1=positive-for-this-concept, 0=negative
    for r in valid:
        k    = r["cluster_id"]
        name = r["concept_name"]

        pos_mask = (labels == k)                       # anomaly indices in cluster k
        n_pos    = pos_mask.sum()
        n_neg_anom = (~pos_mask).sum()

        if n_pos < 5:
            print(f"  Cluster {k} ({name}): SKIP — only {n_pos} positives")
            continue

        # Positives: anomaly images in cluster k
        X_pos = X_anom[pos_mask]
        # Negatives: all normals + anomaly images from other clusters
        X_neg = np.vstack([X_norm, X_anom[~pos_mask]])

        X = np.vstack([X_pos, X_neg]).astype(np.float32)
        y = np.array([1] * len(X_pos) + [0] * len(X_neg), dtype=np.int32)

        m = train_head(X, y)
        print(f"  Cluster {k} ({name:<30})  "
              f"n+={n_pos:<4}  f1={m['f1']:.3f}  "
              f"acc={m['accuracy']:.3f}  auc={m['auc']:.3f}")
        head_results.append({
            "cluster_id": k, "concept_name": name,
            "cluster_size": int(n_pos),
            "confidence": r["confidence"],
            "categories": r.get("categories", []),
            **m,
        })

    # ── STEP 7: Final report ──────────────────────────────────────────────────
    print(f"\n{'='*68}")
    print(f"  FINAL REPORT — Structural Concept Heads (1024-dim DINOv2 max-pool)")
    print(f"{'='*68}")
    print(f"\n  {'Concept':<32} {'Size':>5} {'F1':>7} {'AUC':>7}  Categories")
    print(f"  {'─'*64}")
    for h in sorted(head_results, key=lambda x: -x["auc"]):
        cats = ", ".join(h["categories"])
        print(f"  {h['concept_name']:<32} {h['cluster_size']:>5} "
              f"{h['f1']:>7.3f} {h['auc']:>7.3f}  {cats}")

    if head_results:
        mean_f1  = np.mean([h["f1"]  for h in head_results])
        mean_auc = np.mean([h["auc"] for h in head_results])
        print(f"  {'─'*64}")
        print(f"  {'MEAN':<32} {'':>5} {mean_f1:>7.3f} {mean_auc:>7.3f}")
        print(f"\n  Mean F1 structural concept heads : {mean_f1:.3f}")
        print(f"  Mean F1 surface  concept heads   : 0.909  (reference)")
        print(f"\n  Structural head mean I-AUC       : {mean_auc:.3f}")

    # Save final summary
    summary = {
        "vlm_results":   vlm_results,
        "head_results":  head_results,
        "n_clusters":    _K,
        "top_dims_used": _TOP_DIM,
        "feature_type":  "max_pool_1024_dinov2_no_sae",
    }
    with open(_OUT / "structural_concept_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Grids        → {_GRIDS}/cluster_{{0..{_K-1}}}.png")
    print(f"  VLM results  → {_OUT}/structural_vlm_results.json")
    print(f"  Summary      → {_OUT}/structural_concept_summary.json")
    print(f"{'='*68}")


if __name__ == "__main__":
    main()
