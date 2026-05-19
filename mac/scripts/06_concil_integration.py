"""
06_concil_integration.py — MA-CBM + CONCIL: sequential anomaly detection on
all 15 MVTec categories.

Architecture
────────────
  Per image:
    DINOv2 ViT-L/14-reg (frozen) → 256 patch tokens (1024-dim each)
    SAE (frozen, 4096 atoms, k=64) → 256 × 4096 sparse codes
    6 pre-trained sklearn LogReg heads → 256 × 6 probability maps
    MAX pool over 256 patches → 6-dim concept vector per image

  CONCIL (zero-forgetting ridge regression):
    Input: 6-dim concept vector  (fixed across all 15 tasks)
    Target: binary anomaly label
    Accumulated Gram matrix is 7×7 (K+1=7) — trivially small

Key properties:
  • DINOv2 and SAE are NEVER updated
  • Concept heads are NEVER updated after training in step 5
  • CONCIL anomaly head has zero-forgetting guarantee (K fixed)
  • Concept vectors are cached after extraction → BWT is pure matrix ops

Metrics reported:
  • I-AUC per category after each task (15×15 matrix + diagonal = initial)
  • BWT(I-AUC) over 15 tasks
  • Spatial precision per concept (mean activation inside / outside mask)
  • C-AUC for categories with CONVAD annotations (bottle, capsule, hazelnut,
    metal_nut, screw) — skipped for others

Usage:
    cd /mnt/nvme1/sobhan_hosseini/covad_test
    uv run mac/scripts/06_concil_integration.py
"""

from __future__ import annotations

import json
import pickle
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image
from sklearn.metrics import roc_auc_score
from torchvision import transforms

_ROOT = Path(__file__).resolve().parent.parent   # covad_test/mac/
sys.path.insert(0, str(_ROOT.parent))            # covad_test/
sys.path.insert(0, str(_ROOT))                   # covad_test/mac/

from features.dinov2_extractor import DINOv2Extractor
from features.sae import SparseAutoencoder
from solvers.concil import ConcilSolver

# ── Constants ─────────────────────────────────────────────────────────────────

MVTEC_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper",
]

# Categories with CONVAD concept annotations available
ANNOTATED_CATS = {"bottle", "capsule", "hazelnut", "metal_nut", "screw"}

CONCEPT_NAMES = [
    "surface_discontinuity",
    "surface_discoloration",
    "surface_crack",
    "surface_abrasion",
    "surface_void",
    "normality",
]
K = len(CONCEPT_NAMES)

_PATCHES_PER_SIDE = 16   # 224 px / 14 px patch = 16
_N_PATCHES        = _PATCHES_PER_SIDE ** 2   # 256

_MASK_RESIZE = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.NEAREST),
    transforms.CenterCrop(224),
])


# ── Feature extraction ────────────────────────────────────────────────────────

def _encode_batch(
    images: list[Image.Image],
    dino: DINOv2Extractor,
    sae: SparseAutoencoder,
    heads: dict[str, Any],
    device: str,
) -> np.ndarray:
    """Extract concept vectors for a batch of PIL images.

    Each image → DINOv2 (256 patches × 1024) → SAE (256 × 4096) →
    6 heads (256 × 6 probabilities) → MAX pool → 6-dim vector.

    Args:
        images: List of PIL RGB images.
        dino: Frozen DINOv2 extractor.
        sae: Frozen SparseAutoencoder.
        heads: Dict mapping concept name → fitted sklearn LogReg.
        device: Torch device string.

    Returns:
        (N, 6) float32 numpy array — one concept vector per image.
    """
    with torch.no_grad():
        patch_tokens = dino.extract_patch_tokens(images)   # (N, 256, 1024)

    out = np.zeros((len(images), K), dtype=np.float32)
    for i in range(len(images)):
        with torch.no_grad():
            codes = sae.encode(patch_tokens[i].to(device))  # (256, 4096)
        codes_np = codes.detach().cpu().numpy().astype(np.float32)
        patch_probs = np.zeros((_N_PATCHES, K), dtype=np.float32)
        for k, name in enumerate(CONCEPT_NAMES):
            if name in heads:
                patch_probs[:, k] = heads[name].predict_proba(codes_np)[:, 1]
        out[i] = patch_probs.max(axis=0)   # max pool over patches
    return out


def extract_all_vectors(
    image_paths: list[Path],
    dino: DINOv2Extractor,
    sae: SparseAutoencoder,
    heads: dict[str, Any],
    device: str,
    batch_size: int = 4,
) -> np.ndarray:
    """Run _encode_batch over all image_paths in mini-batches.

    Args:
        image_paths: Ordered list of image file paths.
        dino, sae, heads, device: Model components.
        batch_size: Images per DINOv2 forward pass (default 4).

    Returns:
        (N, 6) float32 numpy array.
    """
    all_vecs: list[np.ndarray] = []
    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start:start + batch_size]
        imgs = [Image.open(p).convert("RGB") for p in batch_paths]
        vecs = _encode_batch(imgs, dino, sae, heads, device)
        all_vecs.append(vecs)
        done = min(start + batch_size, len(image_paths))
        print(f"    {done}/{len(image_paths)}", end="\r", flush=True)
    print()
    return np.concatenate(all_vecs, axis=0)


# ── Spatial precision ─────────────────────────────────────────────────────────

def extract_spatial_map(
    image: Image.Image,
    dino: DINOv2Extractor,
    sae: SparseAutoencoder,
    heads: dict[str, Any],
    device: str,
) -> np.ndarray:
    """Return per-patch concept probability map for one image.

    Args:
        image: PIL RGB image.

    Returns:
        (16, 16, 6) float32 array — head probabilities on the patch grid.
    """
    with torch.no_grad():
        patch_tokens = dino.extract_patch_tokens([image])  # (1, 256, 1024)
    with torch.no_grad():
        codes = sae.encode(patch_tokens[0].to(device))     # (256, 4096)
    codes_np = codes.detach().cpu().numpy().astype(np.float32)

    spatial = np.zeros((_N_PATCHES, K), dtype=np.float32)
    for k, name in enumerate(CONCEPT_NAMES):
        if name in heads:
            spatial[:, k] = heads[name].predict_proba(codes_np)[:, 1]

    return spatial.reshape(_PATCHES_PER_SIDE, _PATCHES_PER_SIDE, K)


def compute_spatial_precision(
    spatial_map: np.ndarray,       # (16, 16, K)
    mask_path: Path,
) -> np.ndarray:
    """Mean activation inside defect mask / mean activation outside mask.

    Args:
        spatial_map: (16, 16, K) per-concept activation grid.
        mask_path: Path to the binary pixel mask PNG.

    Returns:
        (K,) precision ratio per concept. NaN if mask is empty.
    """
    mask_img = Image.open(mask_path).convert("L")
    mask_224  = _MASK_RESIZE(mask_img)
    mask_arr  = np.array(mask_224, dtype=np.float32) / 255.0    # (224, 224)

    # Downsample to 16×16 patch grid (mean pool)
    mask_16 = mask_arr.reshape(
        _PATCHES_PER_SIDE, 14, _PATCHES_PER_SIDE, 14
    ).mean(axis=(1, 3))           # (16, 16)

    inside  = mask_16 > 0.3       # patches at least 30% inside mask
    outside = mask_16 < 0.1       # patches clearly outside

    if inside.sum() == 0 or outside.sum() == 0:
        return np.full(K, float("nan"))

    precision = np.zeros(K, dtype=np.float32)
    for k in range(K):
        m_in  = spatial_map[:, :, k][inside].mean()
        m_out = spatial_map[:, :, k][outside].mean() + 1e-8
        precision[k] = float(m_in / m_out)
    return precision


# ── MVTec dataset helpers ─────────────────────────────────────────────────────

def _mvtec_train_normal(category_root: Path) -> list[Path]:
    return sorted((category_root / "train" / "good").glob("*.png"))


def _mvtec_test_normal(category_root: Path) -> list[Path]:
    return sorted((category_root / "test" / "good").glob("*.png"))


def _mvtec_defect_paths(category_root: Path) -> dict[str, list[Path]]:
    """Return {defect_type: [image_paths]} for all defect subfolders."""
    test_dir = category_root / "test"
    return {
        d.name: sorted(d.glob("*.png"))
        for d in sorted(test_dir.iterdir())
        if d.is_dir() and d.name != "good"
    }


def _mvtec_mask_path(category_root: Path, defect: str, img: Path) -> Path | None:
    """Infer mask path from image path."""
    mask_name = img.stem + "_mask" + img.suffix
    mask = category_root / "ground_truth" / defect / mask_name
    if mask.exists():
        return mask
    mask2 = category_root / "ground_truth" / defect / img.name
    return mask2 if mask2.exists() else None


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    cfg_path = _ROOT / "configs" / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    mvtec_root  = Path(cfg["mvtec_root"])
    device      = cfg.get("device", "cuda")
    out_dir     = Path(cfg["output_dir"]) / "concil"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 68)
    print("  MA-CBM + CONCIL — 15-category sequential experiment")
    print("=" * 68)
    print(f"  MVTec root : {mvtec_root}")
    print(f"  Output     : {out_dir}")
    print(f"  Concepts   : {CONCEPT_NAMES}")
    print()

    # ── Load models ───────────────────────────────────────────────────────────
    print("[init] Loading DINOv2 ViT-L/14-reg …")
    dino = DINOv2Extractor(
        model_name=cfg.get("dino_model", "dinov2_vitl14_reg"),
        device=torch.device(device),
    )

    print("[init] Loading SAE …")
    sae = SparseAutoencoder.load(cfg["sae_weights"], device=device)
    sae = sae.to(device)
    sae.eval()

    print("[init] Loading 6 concept heads …")
    heads_path = (Path(cfg["output_dir"]) / "cross_category" /
                  "cross_category_6heads_concept_heads.pkl")
    with open(heads_path, "rb") as f:
        heads: dict[str, Any] = pickle.load(f)
    print(f"  Loaded heads: {list(heads.keys())}")

    # ── Phase 1: extract & cache concept vectors for all categories ───────────
    print("\n[Phase 1] Extracting concept vectors for all 15 categories …")
    print("  (This is the only DINOv2 pass — vectors are cached for BWT)\n")

    cache_path = out_dir / "concept_vector_cache.pkl"
    if cache_path.exists():
        print(f"  Loading cache from {cache_path.name} …")
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
    else:
        cache: dict[str, dict] = {}

    rng = random.Random(42)

    for cat in MVTEC_CATEGORIES:
        if cat in cache:
            print(f"  [{cat}] cached ({len(cache[cat]['all_paths'])} images)")
            continue

        cat_root = mvtec_root / cat
        print(f"  [{cat}] extracting …")

        # Paths
        normal_train = _mvtec_train_normal(cat_root)
        normal_test  = _mvtec_test_normal(cat_root)
        defect_dict  = _mvtec_defect_paths(cat_root)

        all_defect = [p for paths in defect_dict.values() for p in paths]
        # 80/20 defect split (fixed seed)
        shuffled = list(all_defect)
        rng_det  = random.Random(42)
        rng_det.shuffle(shuffled)
        n_train  = max(1, int(len(shuffled) * 0.8))
        defect_train = shuffled[:n_train]
        defect_test  = shuffled[n_train:]

        # Mask lookup for defect_test images
        defect_to_mask: dict[str, Path | None] = {}
        for defect, paths in defect_dict.items():
            for p in paths:
                mk = _mvtec_mask_path(cat_root, defect, p)
                defect_to_mask[str(p)] = mk

        # Ordered for extraction: train_normal + defect_train + test_normal + defect_test
        all_paths = normal_train + defect_train + normal_test + defect_test
        y_all = (
            [0] * len(normal_train) +
            [1] * len(defect_train) +
            [0] * len(normal_test) +
            [1] * len(defect_test)
        )
        is_train_mask = (
            [True]  * len(normal_train) +
            [True]  * len(defect_train) +
            [False] * len(normal_test) +
            [False] * len(defect_test)
        )

        vecs = extract_all_vectors(
            [Path(p) for p in all_paths],
            dino, sae, heads, device, batch_size=4,
        )

        cache[cat] = {
            "all_paths":    [str(p) for p in all_paths],
            "y_all":        y_all,
            "is_train_mask": is_train_mask,
            "vecs":         vecs,
            "defect_to_mask": defect_to_mask,
            "n_train_normal":  len(normal_train),
            "n_defect_train":  len(defect_train),
            "n_test_normal":   len(normal_test),
            "n_defect_test":   len(defect_test),
        }
        print(f"    {cat}: {len(normal_train)} train-normal, "
              f"{len(defect_train)} defect-train, "
              f"{len(normal_test)} test-normal, "
              f"{len(defect_test)} defect-test")

        with open(cache_path, "wb") as f:
            pickle.dump(cache, f)

    print("\n[Phase 1] Done.\n")

    # ── Phase 2: sequential CONCIL training + evaluation ─────────────────────
    print("[Phase 2] Sequential CONCIL training (anomaly head only) …\n")

    solver = ConcilSolver(
        input_dim=K,          # only anomaly head is used; D doesn't matter here
        lambda_anomaly=1e-4,
    )
    anomaly_w: np.ndarray | None = None   # (K,)
    anomaly_b: float | None      = None   # scalar

    # i_auc[cat_idx][after_task_idx] — upper triangle filled as tasks progress
    i_auc_matrix = np.full((15, 15), float("nan"))

    initial_i_auc: dict[str, float] = {}   # I-AUC right after task t
    final_i_auc:   dict[str, float] = {}   # I-AUC after task 15

    header = (f"  {'Task':<4} {'Category':<14} "
              f"{'N-tr':<6} {'N+tr':<6} "
              f"{'I-AUC':>7}")
    print(header)
    print("  " + "─" * (len(header) - 2))

    for t_idx, cat in enumerate(MVTEC_CATEGORIES):
        task_num = t_idx + 1
        c = cache[cat]

        vecs  = c["vecs"]                      # (N_all, K)
        y     = np.array(c["y_all"])
        train = np.array(c["is_train_mask"])

        X_train = vecs[train]
        y_train = y[train]
        X_test  = vecs[~train]
        y_test  = y[~train]

        # Update CONCIL anomaly head with this task's training data
        C_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_tensor = torch.tensor(y_train, dtype=torch.float32)
        anomaly_w, anomaly_b_arr = solver.update_anomaly_head(C_tensor, y_tensor)
        anomaly_b = float(anomaly_b_arr[0])

        # Evaluate on this task's test set
        scores_test = X_test @ anomaly_w + anomaly_b
        try:
            auc = float(roc_auc_score(y_test, scores_test))
        except ValueError:
            auc = float("nan")

        i_auc_matrix[t_idx, t_idx] = auc
        initial_i_auc[cat] = auc

        n_train_pos = int(y_train.sum())
        n_train_neg = int((y_train == 0).sum())

        print(f"  T{task_num:<3} {cat:<14} "
              f"{n_train_neg:<6} {n_train_pos:<6} "
              f"{auc:>7.4f}")

    # ── Re-evaluate all previous tasks after final task ───────────────────────
    print("\n  Re-evaluating all 15 tasks after final task (for BWT) …")
    for t_idx, cat in enumerate(MVTEC_CATEGORIES):
        c = cache[cat]
        vecs  = c["vecs"]
        y     = np.array(c["y_all"])
        train = np.array(c["is_train_mask"])
        X_test = vecs[~train]
        y_test  = y[~train]
        scores  = X_test @ anomaly_w + anomaly_b
        try:
            auc = float(roc_auc_score(y_test, scores))
        except ValueError:
            auc = float("nan")
        i_auc_matrix[t_idx, 14] = auc
        final_i_auc[cat] = auc
        print(f"    {cat:<14} I-AUC={auc:.4f}")

    # ── BWT ───────────────────────────────────────────────────────────────────
    bwt_values = []
    for cat in MVTEC_CATEGORIES[:-1]:   # all but last task
        bwt = final_i_auc[cat] - initial_i_auc[cat]
        bwt_values.append(bwt)
    mean_bwt = float(np.mean(bwt_values))

    # ── Phase 3: spatial precision (anomalous images only) ───────────────────
    print("\n[Phase 3] Computing spatial precision …")
    all_precisions: list[np.ndarray] = []
    cat_precisions: dict[str, list[np.ndarray]] = {cat: [] for cat in MVTEC_CATEGORIES}

    for cat in MVTEC_CATEGORIES:
        c = cache[cat]
        cat_root = mvtec_root / cat
        paths    = c["all_paths"]
        y        = c["y_all"]
        d2m      = c["defect_to_mask"]

        print(f"  [{cat}] ", end="", flush=True)
        done = 0
        for p_str, label in zip(paths, y):
            if label == 0:
                continue
            mask_path = d2m.get(p_str)
            if mask_path is None or not Path(mask_path).exists():
                continue
            img = Image.open(p_str).convert("RGB")
            smap = extract_spatial_map(img, dino, sae, heads, device)
            prec = compute_spatial_precision(smap, Path(mask_path))
            if not np.all(np.isnan(prec)):
                all_precisions.append(prec)
                cat_precisions[cat].append(prec)
            done += 1
        print(f"{done} anomalous images processed")

    # ── Compile results ───────────────────────────────────────────────────────
    all_prec_arr = np.stack(all_precisions) if all_precisions else np.full((1, K), float("nan"))
    mean_precision = np.nanmean(all_prec_arr, axis=0)

    # Save full results
    results = {
        "initial_i_auc":  initial_i_auc,
        "final_i_auc":    final_i_auc,
        "mean_bwt":       mean_bwt,
        "per_cat_bwt":    {cat: final_i_auc[cat] - initial_i_auc[cat]
                           for cat in MVTEC_CATEGORIES[:-1]},
        "spatial_precision_mean": {CONCEPT_NAMES[k]: float(mean_precision[k])
                                   for k in range(K)},
        "i_auc_matrix":   {
            "categories": MVTEC_CATEGORIES,
            "matrix": i_auc_matrix.tolist(),
        },
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # ── Print final table ─────────────────────────────────────────────────────
    print()
    print("=" * 68)
    print("  FINAL RESULTS — MA-CBM + CONCIL (15 tasks)")
    print("=" * 68)

    col_w = 8
    print(f"\n  Per-category results after Task 15:\n")
    print(f"  {'Category':<14} {'I-AUC(init)':>12} {'I-AUC(final)':>13} "
          f"{'BWT':>8}  {'Spat.Prec(disc)':>16}")
    print("  " + "─" * 68)

    for t_idx, cat in enumerate(MVTEC_CATEGORIES):
        init_auc  = initial_i_auc[cat]
        final_auc = final_i_auc[cat]
        bwt       = final_auc - init_auc if cat != MVTEC_CATEGORIES[-1] else float("nan")
        cat_prec  = cat_precisions.get(cat, [])
        prec_str  = (f"{np.nanmean([p[0] for p in cat_prec]):.2f}"
                     if cat_prec else "n/a")
        bwt_str   = f"{bwt:+.4f}" if not np.isnan(bwt) else "  (last)"
        print(f"  {cat:<14} {init_auc:>12.4f} {final_auc:>13.4f} "
              f"{bwt_str:>8}  {prec_str:>16}")

    print("  " + "─" * 68)
    mean_init  = np.nanmean(list(initial_i_auc.values()))
    mean_final = np.nanmean(list(final_i_auc.values()))
    print(f"  {'MEAN':<14} {mean_init:>12.4f} {mean_final:>13.4f} "
          f"{mean_bwt:>+8.4f}")
    print(f"\n  CONCIL BWT (I-AUC) : {mean_bwt:+.4f}  "
          f"(≈0 expected — zero-forgetting guarantee)")

    print(f"\n  Spatial precision per concept (mean inside/outside mask ratio):\n")
    print(f"  {'Concept':<35} {'Precision':>10}  {'Interpretation'}")
    print("  " + "─" * 68)
    for k, name in enumerate(CONCEPT_NAMES):
        p = mean_precision[k]
        if np.isnan(p):
            interp = "n/a"
        elif p > 2.0:
            interp = "strong spatial alignment"
        elif p > 1.3:
            interp = "moderate spatial alignment"
        elif p > 1.0:
            interp = "weak spatial alignment"
        else:
            interp = "no spatial alignment"
        print(f"  {name:<35} {p:>10.3f}  {interp}")

    print(f"\n  Results saved → {out_dir}/results.json")
    print("=" * 68)


if __name__ == "__main__":
    main()
