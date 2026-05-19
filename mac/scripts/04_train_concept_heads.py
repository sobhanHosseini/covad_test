"""
04_train_concept_heads.py — Train one binary concept head per vocabulary concept.

Reads:
    <output_dir>/hazelnut_vocabulary.json
    <output_dir>/hazelnut_anomaly_patches.pkl
    Precomputed normal patches tensor (from config)

Writes:
    <output_dir>/hazelnut_concept_heads.json   — per-concept metrics
    <output_dir>/hazelnut_concept_heads.pkl    — fitted sklearn models

Import resolution (via sys.path):
    _ROOT            = covad_test/mac/   → resolves data/, concepts/
    _ROOT.parent     = covad_test/       → resolves features/

Usage:
    cd /mnt/nvme1/sobhan_hosseini/covad_test
    uv run mac/scripts/04_train_concept_heads.py
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent   # covad_test/mac/
sys.path.insert(0, str(_ROOT.parent))            # covad_test/  → features/
sys.path.insert(0, str(_ROOT))                   # covad_test/mac/ → data/, concepts/

import json
import pickle

import yaml

from features.sae import SparseAutoencoder
from data.patch_extractor import load_normal_patches
from concepts.concept_head_trainer import train_concept_heads


def main() -> None:
    """Load data, train concept heads, print per-concept accuracy."""
    cfg_path = _ROOT / "configs" / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    device    = cfg.get("device", "cuda")
    category  = cfg["category"]
    out_dir   = Path(cfg["output_dir"])

    vocab_path   = out_dir / f"{category}_vocabulary.json"
    patches_path = out_dir / f"{category}_anomaly_patches.pkl"

    print("=" * 60)
    print("  MA-CBM  |  Step 4 — Train Concept Heads")
    print("=" * 60)
    print(f"  Category : {category}")
    print()

    for p, prev in [(patches_path, "01_extract_patches.py"),
                    (vocab_path,   "03_run_vlm.py")]:
        if not p.exists():
            print(f"ERROR: {p} not found. Run {prev} first.")
            sys.exit(1)

    # ── Load data ─────────────────────────────────────────────────────────────
    print("[04] Loading vocabulary …")
    with open(vocab_path) as f:
        vocabulary = json.load(f)
    print(f"     {len(vocabulary['concepts'])} concepts")

    print("[04] Loading anomaly patches …")
    with open(patches_path, "rb") as f:
        anomaly_patches = pickle.load(f)
    print(f"     {len(anomaly_patches)} patches")

    print("[04] Loading SAE …")
    sae = SparseAutoencoder.load(cfg["sae_weights"], device=device)
    sae.eval()

    print("[04] Loading precomputed normal patches …")
    normal_codes = load_normal_patches(
        category=category,
        patches_tensor_path=cfg["normal_patches_tensor"],
        index_path=cfg["normal_patches_index"],
        sae=sae,
        device=device,
        sample_size=cfg.get("normal_patches_sample_size", 2000),
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    print("[04] Training concept heads …\n")
    results = train_concept_heads(
        vocabulary=vocabulary,
        anomaly_patches=anomaly_patches,
        normal_codes=normal_codes,
        output_dir=out_dir,
        pos_percentile=70.0,
        neg_percentile=30.0,
        normal_sample=200,
        C=0.1,
        test_size=0.2,
        seed=42,
    )


if __name__ == "__main__":
    main()
