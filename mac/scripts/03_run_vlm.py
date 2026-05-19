"""
03_run_vlm.py — Query Qwen2.5-VL to generate concept names for each atom.

Reads:
    <output_dir>/hazelnut_anomaly_patches.pkl
    <output_dir>/hazelnut_relevant_atoms.json

Writes:
    <output_dir>/grids/atom_<id>.png          (one per atom)
    <output_dir>/hazelnut_vlm_results.json    (incremental, safe against crash)
    <output_dir>/hazelnut_vocabulary.json     (final de-duplicated vocabulary)

Import resolution (via sys.path):
    _ROOT            = covad_test/mac/   → resolves data/, concepts/
    _ROOT.parent     = covad_test/       → resolves features/

Usage:
    cd /mnt/nvme1/sobhan_hosseini/covad_test
    uv run mac/scripts/03_run_vlm.py
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent   # covad_test/mac/
sys.path.insert(0, str(_ROOT.parent))            # covad_test/  → features/
sys.path.insert(0, str(_ROOT))                   # covad_test/mac/ → data/, concepts/

import json
import pickle

import yaml

from concepts.vlm_interpreter import run_vlm_on_atoms
from concepts.vocabulary_builder import build_vocabulary


def main() -> None:
    """Load patches + atoms, run VLM loop, print concept name per atom."""
    cfg_path = _ROOT / "configs" / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    category     = cfg["category"]
    out_dir      = Path(cfg["output_dir"])
    vlm_model    = cfg.get("vlm_model", "gemma4:e4b")
    ollama_host  = cfg.get("ollama_host", "http://localhost:6000")
    top_k        = cfg.get("top_patches_per_atom", 9)

    patches_path = out_dir / f"{category}_anomaly_patches.pkl"
    atoms_path   = out_dir / f"{category}_relevant_atoms.json"
    vocab_path   = out_dir / f"{category}_vocabulary.json"

    print("=" * 60)
    print("  MA-CBM  |  Step 3 — VLM Concept Naming")
    print("=" * 60)
    print(f"  Category  : {category}")
    print(f"  VLM model : {vlm_model}")
    print(f"  Grid size : {top_k} patches")
    print()

    for p, prev in [(patches_path, "01_extract_patches.py"),
                    (atoms_path,   "02_select_atoms.py")]:
        if not p.exists():
            print(f"ERROR: {p} not found. Run {prev} first.")
            sys.exit(1)

    # ── Load data ─────────────────────────────────────────────────────────────
    print("[03] Loading anomaly patches …")
    with open(patches_path, "rb") as f:
        anomaly_patches = pickle.load(f)
    print(f"     {len(anomaly_patches)} patches")

    print("[03] Loading relevant atoms …")
    with open(atoms_path) as f:
        relevant_atoms = json.load(f)
    print(f"     {len(relevant_atoms)} atoms to interpret\n")

    # ── Run VLM ───────────────────────────────────────────────────────────────
    vlm_results = run_vlm_on_atoms(
        relevant_atoms=relevant_atoms,
        anomaly_patches=anomaly_patches,
        output_dir=out_dir,
        vlm_model=vlm_model,
        ollama_host=ollama_host,
        top_k=top_k,
    )

    # ── Build vocabulary ──────────────────────────────────────────────────────
    print("\n[03] Building vocabulary …")
    vocab = build_vocabulary(
        vlm_results=vlm_results,
        ollama_host=ollama_host,
        save_path=vocab_path,
    )

    # ── Final summary ─────────────────────────────────────────────────────────
    print()
    print("── Concept Vocabulary ───────────────────────────────────────")
    for c in vocab["concepts"]:
        atom_ids_str = ", ".join(str(a) for a in c["atom_ids"][:5])
        if len(c["atom_ids"]) > 5:
            atom_ids_str += f" … (+{len(c['atom_ids']) - 5})"
        print(f"  {c['name']:<35s} ×{c['frequency']}  atoms: [{atom_ids_str}]")
    print(f"\n  Total unique concepts : {vocab['total_unique_concepts']}")
    print(f"  Vocabulary saved to   : {vocab_path}")
    print("─────────────────────────────────────────────────────────────")


if __name__ == "__main__":
    main()
