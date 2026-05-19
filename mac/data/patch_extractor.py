"""
patch_extractor.py — Extract SAE-encoded anomaly patches from MVTec images.

For a given category, walks every defect subfolder under test/, aligns each
image with its pixel mask, projects the mask onto the DINOv2 patch grid
(16×16 for 224-px input with 14-px patches), and keeps only patches whose
mask-overlap exceeds the configured threshold.

Normal patches are loaded from the precomputed tensor produced during SAE
training, so DINOv2 does NOT need to run again for the normal split.
"""

from __future__ import annotations

import pickle
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

# ── spatial transforms (no normalisation — we also need visual crops) ─────────

_SPATIAL = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
])

_MASK_SPATIAL = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.NEAREST),
    transforms.CenterCrop(224),
])

# DINOv2 with 14-px patches on a 224-px image → 16 patches per side (16×16=256)
_PATCHES_PER_SIDE = 16
_PATCH_PX = 14  # pixels per patch side in the 224×224 space


# ── helpers ───────────────────────────────────────────────────────────────────

def _mask_overlaps(mask_arr: np.ndarray) -> np.ndarray:
    """Compute mean mask value for every patch position.

    Args:
        mask_arr: float32 array of shape (224, 224), values in [0, 1].

    Returns:
        overlaps: float32 array of shape (256,) — one overlap per patch, in
                  row-major order (patch index = row * 16 + col).
    """
    P = _PATCH_PX
    S = _PATCHES_PER_SIDE
    overlaps = np.zeros(S * S, dtype=np.float32)
    for r in range(S):
        for c in range(S):
            region = mask_arr[r * P:(r + 1) * P, c * P:(c + 1) * P]
            overlaps[r * S + c] = float(region.mean())
    return overlaps


# ── public API ────────────────────────────────────────────────────────────────

def extract_anomaly_patches(
    category: str,
    mvtec_root: str | Path,
    sae,
    dino,
    overlap_threshold: float = 0.3,
    device: str = "cuda",
    save_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Extract SAE-encoded anomaly patches for one MVTec category.

    Walks every defect subfolder under ``mvtec_root/<category>/test/`` (all
    subdirs except ``good``), finds the corresponding mask in
    ``mvtec_root/<category>/ground_truth/``, projects the mask onto the
    16×16 DINOv2 patch grid, and keeps patches where mask overlap exceeds
    *overlap_threshold*.

    Args:
        category: MVTec category name, e.g. ``"hazelnut"``.
        mvtec_root: Path to the MVTec root directory.
        sae: Loaded ``SparseAutoencoder`` instance (eval mode, on *device*).
        dino: Loaded ``DINOv2Extractor`` instance (frozen, on *device*).
        overlap_threshold: Minimum fraction of patch pixels that must lie
            inside the defect mask (default 0.3).
        device: Torch device string for SAE encoding.
        save_path: If given, pickles the result list to this path.

    Returns:
        List of dicts, one per kept patch::

            {
                "patch_image":   PIL.Image  (14×14 crop, unscaled),
                "sae_code":      torch.Tensor (4096,) float32 on CPU,
                "mask_overlap":  float,
                "category":      str,
                "defect_type":   str,
                "image_path":    str,
            }
    """
    mvtec_root = Path(mvtec_root)
    test_root = mvtec_root / category / "test"
    gt_root = mvtec_root / category / "ground_truth"

    if not test_root.exists():
        raise FileNotFoundError(f"Test directory not found: {test_root}")

    defect_dirs = sorted([d for d in test_root.iterdir() if d.is_dir() and d.name != "good"])
    if not defect_dirs:
        raise RuntimeError(f"No defect subfolders found under {test_root}")

    print(f"[patch_extractor] Category: {category}")
    print(f"[patch_extractor] Defect types: {[d.name for d in defect_dirs]}")

    sae = sae.to(device).eval()
    patches_out: list[dict[str, Any]] = []

    for defect_dir in defect_dirs:
        defect_type = defect_dir.name
        image_paths = sorted(defect_dir.glob("*.png")) + sorted(defect_dir.glob("*.jpg"))
        print(f"  [{defect_type}] {len(image_paths)} images", end="", flush=True)

        kept_count = 0
        for img_path in image_paths:
            # ── load image and mask ──────────────────────────────────────────
            img_pil = Image.open(img_path).convert("RGB")
            mask_name = img_path.stem + "_mask" + img_path.suffix
            mask_path = gt_root / defect_type / mask_name
            if not mask_path.exists():
                # some datasets use the same stem without _mask suffix
                mask_path = gt_root / defect_type / img_path.name
            if not mask_path.exists():
                print(f"\n  [WARN] mask not found for {img_path}, skipping")
                continue

            mask_pil = Image.open(mask_path).convert("L")

            # ── apply spatial transform (224×224, no normalisation) ──────────
            img_224 = _SPATIAL(img_pil)          # PIL RGB 224×224
            mask_224 = _MASK_SPATIAL(mask_pil)   # PIL L   224×224

            mask_arr = np.array(mask_224, dtype=np.float32) / 255.0  # [0,1]

            # ── DINOv2 patch tokens ──────────────────────────────────────────
            patch_tokens = dino.extract_patch_tokens([img_pil])  # (1, 256, 1024)
            patch_tokens = patch_tokens[0]                        # (256, 1024)

            # ── SAE encode all 256 patches at once ──────────────────────────
            with torch.no_grad():
                sae_codes = sae.encode(patch_tokens.to(device))  # (256, 4096)
            sae_codes_cpu = sae_codes.cpu()

            # ── per-patch overlap filter ─────────────────────────────────────
            overlaps = _mask_overlaps(mask_arr)  # (256,)

            for patch_idx in range(_PATCHES_PER_SIDE * _PATCHES_PER_SIDE):
                if overlaps[patch_idx] < overlap_threshold:
                    continue

                r = patch_idx // _PATCHES_PER_SIDE
                c = patch_idx % _PATCHES_PER_SIDE
                left, upper = c * _PATCH_PX, r * _PATCH_PX
                right, lower = left + _PATCH_PX, upper + _PATCH_PX
                patch_img = img_224.crop((left, upper, right, lower))

                patches_out.append({
                    "patch_image":  patch_img,
                    "sae_code":     sae_codes_cpu[patch_idx],
                    "mask_overlap": float(overlaps[patch_idx]),
                    "category":     category,
                    "defect_type":  defect_type,
                    "image_path":   str(img_path),
                })
                kept_count += 1

        print(f" → {kept_count} patches kept")

    print(f"[patch_extractor] Total anomaly patches: {len(patches_out)}")

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(patches_out, f)
        print(f"[patch_extractor] Saved → {save_path}")

    return patches_out


def load_normal_patches(
    category: str,
    patches_tensor_path: str | Path,
    index_path: str | Path,
    sae,
    device: str = "cuda",
    sample_size: int = 2000,
    seed: int = 42,
) -> torch.Tensor:
    """Load and SAE-encode normal patches from the precomputed DINOv2 tensor.

    The SAE training pipeline saved all MVTec normal patch tokens in a single
    tensor (shape N×1024) alongside a category index list.  This function
    looks up the hazelnut (or any other) slice, optionally sub-samples it, and
    returns the SAE sparse codes.

    Args:
        category: MVTec category name.
        patches_tensor_path: Path to the precomputed ``(N, 1024)`` DINOv2
            patch tensor (``mvtec_normal_patches_vitl14reg.pt``).
        index_path: Path to the category index list
            (``mvtec_patch_index_reg.pt``).
        sae: Loaded ``SparseAutoencoder`` instance.
        device: Torch device for SAE encoding.
        sample_size: Maximum number of patches to sample (0 = all).
        seed: Random seed for reproducible sampling.

    Returns:
        Tensor of shape ``(M, 4096)`` SAE codes on CPU.
    """
    print(f"[patch_extractor] Loading precomputed normal patches for '{category}' …")

    all_patches = torch.load(patches_tensor_path, map_location="cpu", weights_only=False)
    index: list[dict] = torch.load(index_path, map_location="cpu", weights_only=False)

    # Find category slice
    entry = next((e for e in index if e["category"] == category), None)
    if entry is None:
        available = [e["category"] for e in index]
        raise ValueError(f"Category '{category}' not in index. Available: {available}")

    raw = all_patches[entry["row_start"]:entry["row_end"]]  # (N_cat, 1024)
    print(f"  Found {raw.shape[0]} normal patches for '{category}'")

    # Sub-sample if requested
    if sample_size > 0 and raw.shape[0] > sample_size:
        rng = random.Random(seed)
        indices = rng.sample(range(raw.shape[0]), sample_size)
        raw = raw[indices]
        print(f"  Sampled {sample_size} patches")

    # Encode with SAE in batches to avoid OOM
    sae = sae.to(device).eval()
    batch_size = 1024
    codes_list = []
    with torch.no_grad():
        for start in range(0, raw.shape[0], batch_size):
            batch = raw[start:start + batch_size].to(device)
            codes_list.append(sae.encode(batch).cpu())

    codes = torch.cat(codes_list, dim=0)  # (M, 4096)
    print(f"[patch_extractor] Normal SAE codes: {codes.shape}")
    return codes


def load_all_normal_patches(
    patches_tensor_path: str | Path,
    index_path: str | Path,
    sae,
    device: str = "cuda",
    sample_per_category: int = 500,
    seed: int = 42,
) -> torch.Tensor:
    """Load and SAE-encode normal patches from ALL MVTec categories.

    Slices the precomputed tensor once per category, sub-samples to
    *sample_per_category* rows, then encodes all slices together.

    Args:
        patches_tensor_path: Path to the precomputed ``(N, 1024)`` tensor.
        index_path: Path to the category index list.
        sae: Loaded ``SparseAutoencoder`` instance.
        device: Torch device for SAE encoding.
        sample_per_category: Patches to sample per category (default 500).
        seed: Random seed.

    Returns:
        Tensor of shape ``(M, 4096)`` SAE codes on CPU.
    """
    print("[patch_extractor] Loading normal patches for ALL categories …")

    all_patches = torch.load(patches_tensor_path, map_location="cpu", weights_only=False)
    index: list[dict] = torch.load(index_path, map_location="cpu", weights_only=False)

    rng = random.Random(seed)
    raw_parts: list[torch.Tensor] = []

    for entry in index:
        cat = entry["category"]
        raw = all_patches[entry["row_start"]:entry["row_end"]]
        if sample_per_category > 0 and raw.shape[0] > sample_per_category:
            idx = rng.sample(range(raw.shape[0]), sample_per_category)
            raw = raw[idx]
        raw_parts.append(raw)
        print(f"  {cat:<15s}: {raw.shape[0]} patches")

    combined = torch.cat(raw_parts, dim=0)
    print(f"  Total raw: {combined.shape[0]} patches — encoding …")

    sae = sae.to(device).eval()
    batch_size = 1024
    codes_list = []
    with torch.no_grad():
        for start in range(0, combined.shape[0], batch_size):
            batch = combined[start:start + batch_size].to(device)
            codes_list.append(sae.encode(batch).cpu())

    codes = torch.cat(codes_list, dim=0)
    print(f"[patch_extractor] All-category normal SAE codes: {codes.shape}")
    return codes
