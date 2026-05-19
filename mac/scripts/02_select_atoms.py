"""
02_select_atoms.py — Score SAE atoms and select the most anomaly-relevant ones.

Reads:
    <output_dir>/hazelnut_anomaly_patches.pkl
    Precomputed normal patches tensor (path from config)

Writes:
    <output_dir>/hazelnut_relevant_atoms.json

Import resolution (via sys.path):
    _ROOT            = covad_test/mac/   → resolves data/, concepts/
    _ROOT.parent     = covad_test/       → resolves features/

Usage:
    cd /mnt/nvme1/sobhan_hosseini/covad_test
    uv run mac/scripts/02_select_atoms.py
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent   # covad_test/mac/
sys.path.insert(0, str(_ROOT.parent))            # covad_test/  → features/
sys.path.insert(0, str(_ROOT))                   # covad_test/mac/ → data/, concepts/

import pickle

import yaml

from features.sae import SparseAutoencoder
from data.patch_extractor import load_normal_patches
from concepts.atom_selector import select_anomaly_relevant_atoms


def main() -> None:
    """Load patches, score atoms, print top 10, save results."""
    cfg_path = _ROOT / "configs" / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    device       = cfg.get("device", "cuda")
    category     = cfg["category"]
    out_dir      = Path(cfg["output_dir"])
    patches_path = out_dir / f"{category}_anomaly_patches.pkl"
    save_path    = out_dir / f"{category}_relevant_atoms.json"

    print("=" * 60)
    print("  MA-CBM  |  Step 2 — Select Anomaly-Relevant Atoms")
    print("=" * 60)
    print(f"  Category     : {category}")
    print(f"  Patches file : {patches_path}")
    print()

    if not patches_path.exists():
        print(f"ERROR: {patches_path} not found. Run 01_extract_patches.py first.")
        sys.exit(1)

    # ── Load anomaly patches ──────────────────────────────────────────────────
    print("[02] Loading anomaly patches …")
    with open(patches_path, "rb") as f:
        anomaly_patches = pickle.load(f)
    print(f"     {len(anomaly_patches)} anomaly patches loaded")

    # ── Load SAE (needed to encode normal patches) ────────────────────────────
    print("[02] Loading SAE …")
    sae = SparseAutoencoder.load(cfg["sae_weights"], device=device)
    sae.eval()

    # ── Load normal patches from precomputed tensor ───────────────────────────
    print("[02] Loading precomputed normal patches …")
    normal_codes = load_normal_patches(
        category=category,
        patches_tensor_path=cfg["normal_patches_tensor"],
        index_path=cfg["normal_patches_index"],
        sae=sae,
        device=device,
        sample_size=cfg.get("normal_patches_sample_size", 2000),
    )

    # ── Score and select atoms ────────────────────────────────────────────────
    print("[02] Scoring atoms …\n")
    relevant_atoms = select_anomaly_relevant_atoms(
        anomaly_patches=anomaly_patches,
        normal_codes=normal_codes,
        min_count=cfg.get("min_anomaly_patches_per_atom", 10),
        top_n=cfg.get("top_atoms_for_validation", 50),
        save_path=save_path,
    )

    # ── Print top 10 ─────────────────────────────────────────────────────────
    print()
    print("── Top 10 Atoms ─────────────────────────────────────────────")
    print(f"  {'Rank':<5} {'Atom ID':<9} {'Disc Score':<13} {'Anom Count':<12} {'Mean Anom':<11} {'Mean Norm'}")
    print("  " + "-" * 60)
    for rank, atom in enumerate(relevant_atoms[:10], start=1):
        print(
            f"  {rank:<5} {atom['atom_id']:<9} "
            f"{atom['discrimination_score']:<13.4f} "
            f"{atom['anomaly_count']:<12} "
            f"{atom['mean_anomaly']:<11.4f} "
            f"{atom['mean_normal']:.4f}"
        )
    print(f"\n  Total selected atoms : {len(relevant_atoms)}")
    print(f"  Saved to             : {save_path}")
    print("─────────────────────────────────────────────────────────────")


if __name__ == "__main__":
    main()
