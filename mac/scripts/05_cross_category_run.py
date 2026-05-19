"""
05_cross_category_run.py — Full MA-CBM pipeline across all 15 MVTec categories.

Pooling anomaly patches from all categories forces the atom selection and VLM
interpretation to discover concepts that generalise across object types rather
than being hazelnut-specific.  With ~15× more patches, singleton atoms should
reduce substantially.

Pipeline:
  1. For each of 15 categories: load cached anomaly patches or extract fresh.
  2. Pool all anomaly patches into one list.
  3. Load ALL normal patches from the precomputed tensor (500 per category).
  4. Select top-100 discriminative SAE atoms on the pooled anomaly set.
  5. Run VLM (gemma4:e4b) on top-100 atoms → grids + concept names.
  6. Build vocabulary (LLM grouping + nomic embedding dedup).
  7. Train concept heads on cross-category data.
  8. Print per-concept F1 and atom count.

Outputs (all under <output_dir>/cross_category/):
    {cat}_anomaly_patches.pkl       (per-category cache, shared with hazelnut run)
    cross_category_relevant_atoms.json
    cross_category_vlm_results.json
    grids/atom_<id>.png
    cross_category_vocabulary.json
    cross_category_concept_heads.json
    cross_category_concept_heads.pkl

Import resolution:
    _ROOT            = covad_test/mac/
    _ROOT.parent     = covad_test/       → features/

Usage:
    cd /mnt/nvme1/sobhan_hosseini/covad_test
    uv run mac/scripts/05_cross_category_run.py
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent   # covad_test/mac/
sys.path.insert(0, str(_ROOT.parent))            # covad_test/  → features/
sys.path.insert(0, str(_ROOT))                   # covad_test/mac/ → data/, concepts/

import json
import pickle

import torch
import yaml

from features.dinov2_extractor import DINOv2Extractor
from features.sae import SparseAutoencoder
from data.patch_extractor import extract_anomaly_patches, load_all_normal_patches
from concepts.atom_selector import select_anomaly_relevant_atoms
from concepts.vlm_interpreter import run_vlm_on_atoms
from concepts.vocabulary_builder import build_vocabulary
from concepts.concept_head_trainer import train_concept_heads

MVTEC_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper",
]


def _load_or_extract(
    category: str,
    cache_dir: Path,
    mvtec_root: str,
    sae,
    dino,
    overlap_threshold: float,
    device: str,
) -> list[dict]:
    """Load cached anomaly patches or extract them from MVTec images.

    Cache is stored at ``cache_dir/{category}_anomaly_patches.pkl``.  The
    hazelnut cache from the single-category run is reused automatically.

    Args:
        category: MVTec category name.
        cache_dir: Directory to look for / write the .pkl cache file.
        mvtec_root: Path to MVTec dataset root.
        sae: Loaded SparseAutoencoder.
        dino: Loaded DINOv2Extractor.
        overlap_threshold: Mask-overlap threshold for patch selection.
        device: Torch device string.

    Returns:
        List of patch dicts (same format as extract_anomaly_patches).
    """
    cache_path = cache_dir / f"{category}_anomaly_patches.pkl"
    if cache_path.exists():
        print(f"  [{category}] loading cache ({cache_path.name}) …", end=" ")
        with open(cache_path, "rb") as f:
            patches = pickle.load(f)
        print(f"{len(patches)} patches")
        return patches

    print(f"  [{category}] extracting …")
    try:
        patches = extract_anomaly_patches(
            category=category,
            mvtec_root=mvtec_root,
            sae=sae,
            dino=dino,
            overlap_threshold=overlap_threshold,
            device=device,
            save_path=cache_path,
        )
    except FileNotFoundError as exc:
        print(f"  [{category}] SKIP — {exc}")
        patches = []
    return patches


def main() -> None:
    """Run the full cross-category pipeline."""
    cfg_path = _ROOT / "configs" / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    mvtec_root = cfg["mvtec_root"]
    if mvtec_root == "/path/to/MVTec":
        print("ERROR: Set mvtec_root in mac/configs/config.yaml first.")
        sys.exit(1)

    device         = cfg.get("device", "cuda")
    base_out       = Path(cfg["output_dir"])
    out_dir        = base_out / "cross_category"
    out_dir.mkdir(parents=True, exist_ok=True)

    overlap_thr    = cfg.get("patch_overlap_threshold", 0.3)
    vlm_model      = cfg.get("vlm_model", "gemma4:e4b")
    ollama_host    = cfg.get("ollama_host", "http://localhost:6000")
    top_patches_k  = cfg.get("top_patches_per_atom", 9)
    top_atoms      = 100   # more atoms for larger patch pool

    print("=" * 65)
    print("  MA-CBM  |  Step 5 — Cross-Category Run (all 15 categories)")
    print("=" * 65)
    print(f"  MVTec root : {mvtec_root}")
    print(f"  Output dir : {out_dir}")
    print(f"  Top atoms  : {top_atoms}")
    print()

    # ── Load models (shared across all categories) ────────────────────────────
    print("[05] Loading DINOv2 …")
    dino = DINOv2Extractor(
        model_name=cfg.get("dino_model", "dinov2_vitl14_reg"),
        device=torch.device(device),
    )
    print("[05] Loading SAE …")
    sae = SparseAutoencoder.load(cfg["sae_weights"], device=device)
    sae.eval()

    # ── Step 1: pool anomaly patches from all categories ─────────────────────
    print("\n[05] Collecting anomaly patches …")
    all_patches: list[dict] = []
    for cat in MVTEC_CATEGORIES:
        patches = _load_or_extract(
            cat, base_out, mvtec_root, sae, dino, overlap_thr, device
        )
        all_patches.extend(patches)

    total_anom = len(all_patches)
    print(f"\n  Total anomaly patches pooled: {total_anom}")
    if total_anom == 0:
        print("ERROR: No anomaly patches found. Check mvtec_root.")
        sys.exit(1)

    # Save pooled list (handy for reloading)
    pooled_path = out_dir / "cross_category_anomaly_patches.pkl"
    with open(pooled_path, "wb") as f:
        pickle.dump(all_patches, f)
    print(f"  Pooled patches saved → {pooled_path}")

    # ── Step 2: load all normal patches ──────────────────────────────────────
    print("\n[05] Loading normal patches (all categories) …")
    normal_codes = load_all_normal_patches(
        patches_tensor_path=cfg["normal_patches_tensor"],
        index_path=cfg["normal_patches_index"],
        sae=sae,
        device=device,
        sample_per_category=500,
    )

    # ── Step 3: atom selection ────────────────────────────────────────────────
    atoms_path = out_dir / "cross_category_relevant_atoms.json"
    if atoms_path.exists():
        print(f"\n[05] Loading cached atom scores from {atoms_path.name} …")
        with open(atoms_path) as f:
            relevant_atoms = json.load(f)
    else:
        print("\n[05] Selecting top atoms …")
        relevant_atoms = select_anomaly_relevant_atoms(
            anomaly_patches=all_patches,
            normal_codes=normal_codes,
            min_count=cfg.get("min_anomaly_patches_per_atom", 10),
            top_n=top_atoms,
            save_path=atoms_path,
        )

    print(f"  Selected {len(relevant_atoms)} atoms")

    # ── Step 4: VLM naming ────────────────────────────────────────────────────
    vlm_results_path = out_dir / "cross_category_vlm_results.json"
    if vlm_results_path.exists():
        print(f"\n[05] Loading cached VLM results from {vlm_results_path.name} …")
        with open(vlm_results_path) as f:
            vlm_results = json.load(f)
    else:
        print(f"\n[05] Running VLM on {len(relevant_atoms)} atoms …")
        vlm_results = run_vlm_on_atoms(
            relevant_atoms=relevant_atoms,
            anomaly_patches=all_patches,
            output_dir=out_dir,
            vlm_model=vlm_model,
            ollama_host=ollama_host,
            top_k=top_patches_k,
            run_name="cross_category",
        )

    # ── Step 5: vocabulary consolidation ─────────────────────────────────────
    vocab_path = out_dir / "cross_category_vocabulary.json"
    print("\n[05] Building vocabulary …")
    vocabulary = build_vocabulary(
        vlm_results=vlm_results,
        ollama_host=ollama_host,
        clip_threshold=0.75,
        save_path=vocab_path,
    )
    print(f"  Vocabulary: {vocabulary['total_unique_concepts']} concepts")

    # ── Step 6: concept head training ────────────────────────────────────────
    print("\n[05] Training concept heads …\n")
    results = train_concept_heads(
        vocabulary=vocabulary,
        anomaly_patches=all_patches,
        normal_codes=normal_codes,
        output_dir=out_dir,
        pos_percentile=70.0,
        neg_percentile=30.0,
        normal_sample=500,
        C=0.1,
        test_size=0.2,
        seed=42,
    )

    # Rename output files to cross_category prefix
    for stem in ("hazelnut_concept_heads.json", "hazelnut_concept_heads.pkl"):
        src = out_dir / stem
        if src.exists():
            src.rename(out_dir / stem.replace("hazelnut", "cross_category"))

    print("\n[05] Done.")
    print(f"  All outputs → {out_dir}")


if __name__ == "__main__":
    main()
