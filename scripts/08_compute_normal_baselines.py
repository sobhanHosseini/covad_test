"""
Compute per-category mean SAE atom activation on all normal training patches.

Uses pre-extracted tokens — no DINOv2 inference needed.

Input:
  sae_training/mvtec_normal_patches_vitl14reg.pt   (N_total, 1024)
  sae_training/mvtec_patch_index_reg.pt
  sae_training/sae_vitl14reg_C4096_k64.pt

Output:
  sae_training/baselines/{category}_normal_mean.pt   (4096,) float32 per category

Run from project root:
    python scripts/08_compute_normal_baselines.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from features.sae import SparseAutoencoder

# ── config ────────────────────────────────────────────────────────────────────

TOKENS_PATH   = Path("sae_training/mvtec_normal_patches_vitl14reg.pt")
INDEX_PATH    = Path("sae_training/mvtec_patch_index_reg.pt")
SAE_PATH      = Path("sae_training/sae_vitl14reg_C4096_k64.pt")
BASELINES_DIR = Path("sae_training/baselines")
CHUNK         = 10_000
DEVICE        = torch.device("cuda:0")

# ── preflight ─────────────────────────────────────────────────────────────────

def preflight():
    missing = [p for p in [TOKENS_PATH, INDEX_PATH, SAE_PATH] if not p.exists()]
    if missing:
        print("ERROR — required files not found:")
        for p in missing:
            print(f"  {p}")
        sys.exit(1)

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    preflight()
    BASELINES_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading SAE …")
    sae = SparseAutoencoder.load(str(SAE_PATH), device="cpu")
    sae.to(DEVICE).eval()
    C = sae.config.d_hidden   # 4096

    print("Loading pre-extracted tokens …")
    all_tokens  = torch.load(TOKENS_PATH, map_location="cpu", weights_only=True)
    patch_index = torch.load(INDEX_PATH,  map_location="cpu", weights_only=False)
    print(f"  Token tensor: {all_tokens.shape}  ({all_tokens.nbytes/1e9:.2f} GB)\n")

    for entry in patch_index:
        category  = entry["category"]
        n_patches = entry["n_patches"]
        row_start = entry["row_start"]
        row_end   = entry["row_end"]

        cat_tokens = all_tokens[row_start:row_end]   # (N_patches, 1024) CPU

        z_sum = torch.zeros(C, dtype=torch.float32)

        with torch.no_grad():
            for i in range(0, n_patches, CHUNK):
                chunk = cat_tokens[i : i + CHUNK].to(DEVICE)
                z     = sae.encode(chunk)              # (chunk, C) float32
                z_sum += z.sum(0).cpu()
                del chunk, z

        z_mean = z_sum / n_patches                     # (C,) float32

        out_path = BASELINES_DIR / f"{category}_normal_mean.pt"
        torch.save(z_mean, out_path)

        # Top-3 atoms by mean activation
        top3_vals, top3_idx = z_mean.topk(3)
        top3_str = "  |  ".join(
            f"atom {int(idx)}: {float(val):.5f}"
            for idx, val in zip(top3_idx.tolist(), top3_vals.tolist())
        )
        print(f"{category:<14}  N={n_patches:>7,}  top-3: {top3_str}")

    print(f"\nBaselines saved → {BASELINES_DIR}/")
    print(f"Files: {len(list(BASELINES_DIR.glob('*_normal_mean.pt')))} categories")


if __name__ == "__main__":
    main()
