"""
01_extract_patches.py — Extract SAE-encoded anomaly patches for hazelnut.

Reads config from covad_test/mac/configs/config.yaml, loads DINOv2 + SAE,
walks the MVTec hazelnut test set, and saves the patch list to:
    <output_dir>/hazelnut_anomaly_patches.pkl

Import resolution (via sys.path):
    _ROOT            = covad_test/mac/   → resolves data/, concepts/
    _ROOT.parent     = covad_test/       → resolves features/

Usage:
    cd /mnt/nvme1/sobhan_hosseini/covad_test
    uv run mac/scripts/01_extract_patches.py
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent   # covad_test/mac/
sys.path.insert(0, str(_ROOT.parent))            # covad_test/  → features/
sys.path.insert(0, str(_ROOT))                   # covad_test/mac/ → data/, concepts/

import pickle
from collections import Counter

import torch
import yaml

from features.dinov2_extractor import DINOv2Extractor
from features.sae import SparseAutoencoder
from data.patch_extractor import extract_anomaly_patches


def main() -> None:
    """Load config, models, run extraction, print stats."""
    cfg_path = _ROOT / "configs" / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    mvtec_root = cfg["mvtec_root"]
    if mvtec_root == "/path/to/MVTec":
        print("ERROR: Please set mvtec_root in mac/configs/config.yaml before running.")
        sys.exit(1)

    device     = cfg.get("device", "cuda")
    category   = cfg["category"]
    output_dir = Path(cfg["output_dir"])
    save_path  = output_dir / f"{category}_anomaly_patches.pkl"

    print("=" * 60)
    print("  MA-CBM  |  Step 1 — Extract Anomaly Patches")
    print("=" * 60)
    print(f"  Category   : {category}")
    print(f"  MVTec root : {mvtec_root}")
    print(f"  Device     : {device}")
    print(f"  Output     : {save_path}")
    print()

    # ── Load DINOv2 ──────────────────────────────────────────────────────────
    print("[01] Loading DINOv2 extractor …")
    dino = DINOv2Extractor(
        model_name=cfg.get("dino_model", "dinov2_vitl14_reg"),
        device=torch.device(device),
    )
    print(f"     embed_dim={dino.EMBED_DIM}  patch_size={dino.PATCH_SIZE}")

    # ── Load SAE ─────────────────────────────────────────────────────────────
    print("[01] Loading SAE …")
    sae = SparseAutoencoder.load(cfg["sae_weights"], device=device)
    sae.eval()
    print(f"     d_input={sae.config.d_input}  d_hidden={sae.config.d_hidden}  k={sae.config.k}")

    # ── Extract anomaly patches ───────────────────────────────────────────────
    print("[01] Extracting anomaly patches …\n")
    patches = extract_anomaly_patches(
        category=category,
        mvtec_root=mvtec_root,
        sae=sae,
        dino=dino,
        overlap_threshold=cfg.get("patch_overlap_threshold", 0.3),
        device=device,
        save_path=save_path,
    )

    # ── Stats ─────────────────────────────────────────────────────────────────
    print()
    print("── Summary ──────────────────────────────────────────────────")
    defect_counts = Counter(p["defect_type"] for p in patches)
    for dt, n in sorted(defect_counts.items()):
        print(f"  {dt:<20s}: {n} patches")
    overlaps = [p["mask_overlap"] for p in patches]
    print(f"  Total patches    : {len(patches)}")
    print(f"  Mean mask overlap: {sum(overlaps) / len(overlaps):.3f}")
    print(f"  Saved to         : {save_path}")
    print("─────────────────────────────────────────────────────────────")


if __name__ == "__main__":
    main()
