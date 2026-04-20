"""
dataset.py — MVTec dataset image discovery.

Generalised: works with any category, handles both .png and .jpg images.
No category-specific logic anywhere in this file.
"""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)

# MVTec uses .png; some other industrial datasets use .jpg
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def _glob_images(directory: Path) -> list[str]:
    """Return sorted list of image paths from a directory (png + jpg)."""
    paths = []
    for suffix in _IMAGE_SUFFIXES:
        paths.extend(directory.glob(f"*{suffix}"))
    return sorted(str(p) for p in paths)


def discover_images(dataset_path: str, category: str) -> dict[str, list[str]]:
    """
    Walk the MVTec folder structure for one category.

    Returns
    -------
    dict with keys:
      "normal"       — train/good images (used for Stages 1-2)
      "normal_test"  — test/good images  (annotated in Stage 3)
      "<defect>"     — one key per defect folder under test/
    """
    root = Path(dataset_path) / category
    result: dict[str, list[str]] = {}

    # Training normal images
    normal_dir = root / "train" / "good"
    if not normal_dir.exists():
        raise FileNotFoundError(
            f"Normal training directory not found: {normal_dir}\n"
            f"Expected MVTec structure: {{dataset_path}}/{{category}}/train/good/"
        )
    result["normal"] = _glob_images(normal_dir)
    log.info("Found %d normal training images", len(result["normal"]))

    # Defect test images (every subfolder of test/ except "good")
    test_root = root / "test"
    if not test_root.exists():
        raise FileNotFoundError(f"Test directory not found: {test_root}")

    for defect_dir in sorted(test_root.iterdir()):
        if not defect_dir.is_dir():
            continue
        if defect_dir.name == "good":
            continue
        paths = _glob_images(defect_dir)
        if paths:
            result[defect_dir.name] = paths
            log.info("Found %3d images for defect '%s'", len(paths), defect_dir.name)

    # Normal test images (test/good)
    normal_test_dir = test_root / "good"
    if normal_test_dir.exists():
        result["normal_test"] = _glob_images(normal_test_dir)
        log.info("Found %d normal test images", len(result["normal_test"]))
    else:
        result["normal_test"] = []
        log.info("No normal test images found (test/good does not exist)")

    return result


def get_defect_types(image_groups: dict[str, list[str]]) -> list[str]:
    """Return defect type names from image_groups (excludes 'normal', 'normal_test')."""
    return [k for k in image_groups if k not in ("normal", "normal_test")]


def get_mask_path(
    dataset_path: str, category: str, defect_type: str, image_path: str
) -> str:
    """
    Return the ground-truth mask path for a defect image, or empty string if absent.
    MVTec naming convention: <img_stem>_mask.png
    """
    img_name = Path(image_path).stem
    mask_path = (
        Path(dataset_path) / category / "ground_truth" / defect_type / f"{img_name}_mask.png"
    )
    return str(mask_path) if mask_path.exists() else ""
