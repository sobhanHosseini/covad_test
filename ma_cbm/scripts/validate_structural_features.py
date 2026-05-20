"""
validate_structural_features.py — Diagnostic experiment.

Question: Can DINOv2 global features (CLS token or pooled patches)
separate structural anomalies from normal images — specifically for
cable (concept-only I-AUC = 0.562)?

DINOv2 API notes (read from features/dinov2_extractor.py):
  - _run(x) returns (cls_token, patch_tokens) via forward_features()
    cls_token    : (B, 1024)        ← x_norm_clstoken
    patch_tokens : (B, 256, 1024)   ← x_norm_patchtokens
  - No extractor modification needed; _run() is called directly.
  - extract_pooled() returns concat(CLS, mean_patches) = 2048-dim,
    so we must call _run() manually to get CLS alone (1024-dim).

4 feature types extracted in ONE DINOv2 forward pass:
  A) CLS token          (1024-dim) — whole-image structural summary
  B) Mean-pooled patches (1024-dim) — global average of local features
  C) Max-pooled patches  (1024-dim) — strongest local activation per dim
  D) Mean-pooled SAE codes (4096-dim) — global average in sparse atom space

READ-ONLY: does not modify any file under covad_test/.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

# ── Paths ─────────────────────────────────────────────────────────────────────

_COVAD = Path(__file__).resolve().parents[2]   # covad_test/
sys.path.insert(0, str(_COVAD))

from features.dinov2_extractor import DINOv2Extractor
from features.sae import SparseAutoencoder

_OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
_OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────

_MVTEC = Path("/home/sobhan_hosseini/datasets/mvtec")
_SAE_WEIGHTS = _COVAD / "sae_training" / "sae_vitl14reg_C4096_k64.pt"
_DEVICE = "cuda"
_BATCH  = 8

CATEGORIES = [
    ("cable",      "structural", 0.562),
    ("transistor", "structural", 0.608),
    ("toothbrush", "structural", 0.694),
    ("bottle",     "structural", 0.669),
    ("hazelnut",   "surface",    0.932),
]

CONCEPT_ONLY_MEAN_STRUCTURAL = 0.662   # from previous experiment


# ── Data loading ──────────────────────────────────────────────────────────────

def collect_image_paths(cat: str) -> tuple[list[Path], list[int]]:
    """Return (paths, labels) for all normal + all defect images."""
    root  = _MVTEC / cat
    paths, labels = [], []

    # All normal images (train + test)
    for split in ("train", "test"):
        good = root / split / "good"
        if good.exists():
            for p in sorted(good.glob("*.png")):
                paths.append(p);  labels.append(0)

    # All defect images (test only — MVTec protocol)
    test_dir = root / "test"
    for defect_dir in sorted(test_dir.iterdir()):
        if defect_dir.name == "good" or not defect_dir.is_dir():
            continue
        for p in sorted(defect_dir.glob("*.png")):
            paths.append(p);  labels.append(1)

    return paths, labels


# ── Feature extraction ────────────────────────────────────────────────────────

@torch.no_grad()
def extract_all_features(
    paths: list[Path],
    dino: DINOv2Extractor,
    sae: SparseAutoencoder,
    batch_size: int = 8,
) -> dict[str, np.ndarray]:
    """Extract A/B/C/D features for all images in one DINOv2 pass per batch.

    Returns dict with keys "cls", "mean_patch", "max_patch", "sae_mean",
    each an (N, D) float32 numpy array.

    DINOv2 API used:
        x = dino._prepare(images)            → normalised (B,3,224,224) tensor
        cls, patches = dino._run(x)          → (B,1024) and (B,256,1024)
    SAE API:
        codes = sae.encode(patches_flat)     → (B*256, 4096)
    """
    feat_cls, feat_mean, feat_max, feat_sae = [], [], [], []
    n = len(paths)

    for start in range(0, n, batch_size):
        batch_paths = paths[start:start + batch_size]
        images = [Image.open(p).convert("RGB") for p in batch_paths]

        # Single DINOv2 forward pass
        x           = dino._prepare(images)
        cls_token, patch_tokens = dino._run(x)
        # cls_token   : (b, 1024)
        # patch_tokens: (b, 256, 1024)

        b = cls_token.shape[0]

        # A — CLS token
        feat_cls.append(cls_token.cpu().float().numpy())

        # B — mean-pooled patches
        feat_mean.append(patch_tokens.mean(dim=1).cpu().float().numpy())

        # C — max-pooled patches (element-wise max over 256 patches)
        feat_max.append(patch_tokens.max(dim=1).values.cpu().float().numpy())

        # D — mean-pooled SAE codes
        flat = patch_tokens.reshape(b * 256, 1024)          # (b*256, 1024)
        codes = sae.encode(flat.to(sae.b_dec.device))       # (b*256, 4096)
        codes = codes.detach().cpu().float()
        feat_sae.append(codes.reshape(b, 256, 4096).mean(dim=1).numpy())

        done = min(start + batch_size, n)
        print(f"    {done}/{n}", end="\r", flush=True)

    print()
    return {
        "cls":        np.concatenate(feat_cls,  axis=0),
        "mean_patch": np.concatenate(feat_mean, axis=0),
        "max_patch":  np.concatenate(feat_max,  axis=0),
        "sae_mean":   np.concatenate(feat_sae,  axis=0),
    }


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_feature(
    X: np.ndarray,
    y: np.ndarray,
) -> dict[str, float]:
    """Train LogReg on 80% of data and evaluate on 20%. Returns metrics dict."""
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    clf = LogisticRegression(
        C=1.0, class_weight="balanced",
        max_iter=1000, solver="lbfgs",
        random_state=42,
    )
    clf.fit(X_tr, y_tr)
    y_pred  = clf.predict(X_te)
    y_prob  = clf.predict_proba(X_te)[:, 1]

    return {
        "accuracy": float(accuracy_score(y_te, y_pred)),
        "f1":       float(f1_score(y_te, y_pred, zero_division=0)),
        "auc":      float(roc_auc_score(y_te, y_prob)),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 72)
    print("  Structural Feature Validation — DINOv2 global features vs concept")
    print("=" * 72)

    print("\n[init] Loading DINOv2 ViT-L/14-reg …")
    dino = DINOv2Extractor("dinov2_vitl14_reg", device=torch.device(_DEVICE))
    print(f"  embed_dim={dino.EMBED_DIM}  (CLS token will be {dino.EMBED_DIM}-dim)")

    print("[init] Loading SAE …")
    sae = SparseAutoencoder.load(str(_SAE_WEIGHTS), device=_DEVICE)
    sae = sae.to(_DEVICE).eval()
    print(f"  d_input={sae.config.d_input}  d_hidden={sae.config.d_hidden}")

    # Results table: cat → feature_name → metrics
    results: dict[str, dict[str, dict]] = {}
    feature_names = ["cls", "mean_patch", "max_patch", "sae_mean"]

    for cat, cat_type, concept_auc in CATEGORIES:
        print(f"\n{'─'*60}")
        print(f"  {cat.upper()}  [{cat_type}]  "
              f"(concept-only AUC = {concept_auc:.3f})")
        print(f"{'─'*60}")

        paths, labels = collect_image_paths(cat)
        y = np.array(labels, dtype=np.int32)
        n_normal = int((y == 0).sum())
        n_anom   = int((y == 1).sum())
        print(f"  Images: {n_normal} normal + {n_anom} anomalous = {len(y)} total")

        print(f"  Extracting features (batch={_BATCH}) …")
        feats = extract_all_features(paths, dino, sae, batch_size=_BATCH)

        cat_results: dict[str, dict] = {}
        for feat_name in feature_names:
            X = feats[feat_name]
            m = evaluate_feature(X, y)
            cat_results[feat_name] = m
            beats = "✓ beats concept" if m["auc"] > concept_auc else "✗"
            print(f"  {feat_name:<15}  acc={m['accuracy']:.3f}  "
                  f"f1={m['f1']:.3f}  auc={m['auc']:.3f}  {beats}")

        results[cat] = cat_results

    # ── Comparison table ──────────────────────────────────────────────────────
    print()
    print("=" * 72)
    print("  COMPARISON TABLE — I-AUC (ROC AUC)")
    print("=" * 72)
    print(f"\n  {'Category':<12} {'Type':<11} {'CLS':>7} {'Mean-ptch':>10} "
          f"{'Max-ptch':>9} {'SAE-mean':>9} {'Concept':>8}")
    print("  " + "─" * 62)

    for cat, cat_type, concept_auc in CATEGORIES:
        r = results[cat]
        print(f"  {cat:<12} {cat_type:<11} "
              f"{r['cls']['auc']:>7.3f} "
              f"{r['mean_patch']['auc']:>10.3f} "
              f"{r['max_patch']['auc']:>9.3f} "
              f"{r['sae_mean']['auc']:>9.3f} "
              f"{concept_auc:>8.3f}")

    print("  " + "─" * 62)

    # Mean for structural categories
    struct_cats = [c for c, t, _ in CATEGORIES if t == "structural"]
    for feat in feature_names:
        aucs = [results[c][feat]["auc"] for c in struct_cats]
        col = {"cls": 7, "mean_patch": 10, "max_patch": 9, "sae_mean": 9}[feat]
        _ = aucs   # used below

    print(f"  {'MEAN (struct)':<12} {'':<11}", end="")
    for feat in feature_names:
        aucs = [results[c][feat]["auc"] for c in struct_cats]
        w = {"cls": 7, "mean_patch": 10, "max_patch": 9, "sae_mean": 9}[feat]
        print(f"  {np.mean(aucs):{w}.3f}", end="")
    print(f"  {CONCEPT_ONLY_MEAN_STRUCTURAL:>8.3f}")

    # ── Verdict ───────────────────────────────────────────────────────────────
    # Find best feature for structural categories
    best_feat, best_mean = None, -1.0
    for feat in feature_names:
        aucs = [results[c][feat]["auc"] for c in struct_cats]
        mean = float(np.mean(aucs))
        if mean > best_mean:
            best_mean, best_feat = mean, feat

    feat_label = {
        "cls": "CLS token (1024-dim)",
        "mean_patch": "Mean-pooled patches (1024-dim)",
        "max_patch":  "Max-pooled patches (1024-dim)",
        "sae_mean":   "Mean-pooled SAE codes (4096-dim)",
    }[best_feat]

    if best_mean > 0.80:
        interpretation = (
            "CLS/global approach IS VIABLE\n"
            "  → proceed with structural concept heads using global DINOv2 features"
        )
    elif best_mean >= 0.70:
        interpretation = (
            "MARGINAL — structural concepts may work but will be weak\n"
            "  → consider richer features (multi-scale, attention maps)"
        )
    else:
        interpretation = (
            "Global DINOv2 features do NOT discriminate structural anomalies\n"
            "  → need fundamentally different approach\n"
            "  → candidates: spatial attention, object-level features, pose-aware detection"
        )

    print(f"\n{'─'*72}")
    print("  VERDICT:")
    print(f"  Best feature type for structural categories: {feat_label}")
    print(f"  Mean I-AUC on structural categories: {best_mean:.3f}")
    print(f"  vs concept-only mean:               {CONCEPT_ONLY_MEAN_STRUCTURAL:.3f}")
    print(f"\n  {interpretation}")
    print(f"{'─'*72}")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    csv_path = _OUT_DIR / "structural_validation_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "category", "type", "concept_only_auc",
            "cls_auc", "cls_acc", "cls_f1",
            "mean_patch_auc", "mean_patch_acc", "mean_patch_f1",
            "max_patch_auc", "max_patch_acc", "max_patch_f1",
            "sae_mean_auc", "sae_mean_acc", "sae_mean_f1",
        ])
        for cat, cat_type, concept_auc in CATEGORIES:
            r = results[cat]
            writer.writerow([
                cat, cat_type, concept_auc,
                round(r["cls"]["auc"], 4), round(r["cls"]["accuracy"], 4), round(r["cls"]["f1"], 4),
                round(r["mean_patch"]["auc"], 4), round(r["mean_patch"]["accuracy"], 4), round(r["mean_patch"]["f1"], 4),
                round(r["max_patch"]["auc"], 4), round(r["max_patch"]["accuracy"], 4), round(r["max_patch"]["f1"], 4),
                round(r["sae_mean"]["auc"], 4), round(r["sae_mean"]["accuracy"], 4), round(r["sae_mean"]["f1"], 4),
            ])
    print(f"\n  Results saved → {csv_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
