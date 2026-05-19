"""
SAE Atom Visualization — Sanity Check.

For a sample of atoms, finds the patches they activate most
strongly and displays them as image crops. Visually coherent
patches per atom = atom is monosemantic (good).

Output:
  sae_training/atom_viz_midfreq.png   — concept-level atoms
  sae_training/atom_viz_highfreq.png  — generic/texture atoms

Run from project root:
    uv run python scripts/03_visualize_atoms.py
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from features.sae import SparseAutoencoder

# ── paths ────────────────────────────────────────────────────────────
MVTEC_ROOT  = Path("/home/sobhan_hosseini/datasets/mvtec")
SAE_PATH    = Path("sae_training/sae_vitl14_C4096_k64.pt")
STATS_PATH  = Path("sae_training/sae_vitl14_atom_stats_k64.pt")
TOKENS_PATH = Path("sae_training/mvtec_normal_patches_vitl14.pt")
OUT_MID     = Path("sae_training/atom_viz_midfreq.png")
OUT_HIGH    = Path("sae_training/atom_viz_highfreq.png")

CATEGORIES = [
    "bottle","cable","capsule","carpet","grid","hazelnut",
    "leather","metal_nut","pill","screw","tile","toothbrush",
    "transistor","wood","zipper",
]

# ── image transform (same as extraction) ────────────────────────────
_MEAN = torch.tensor([0.485, 0.456, 0.406])
_STD  = torch.tensor([0.229, 0.224, 0.225])

transform = T.Compose([
    T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=_MEAN.tolist(), std=_STD.tolist()),
])

PATCH_SIZE   = 14
PATCHES_PER_SIDE = 16   # 224 / 14 = 16
PATCHES_PER_IMG  = 256  # 16 × 16


# ── helpers ──────────────────────────────────────────────────────────

def build_image_list():
    """Rebuild the ordered path list in the same order as extraction."""
    paths = []
    for cat in CATEGORIES:
        good_dir = MVTEC_ROOT / cat / "train" / "good"
        cat_paths = sorted(good_dir.glob("*.png")) + \
                    sorted(good_dir.glob("*.jpg"))
        paths.extend(cat_paths)
    return paths


def token_idx_to_patch(token_idx, image_paths):
    """
    Map a flat token index → (image_path, patch_row, patch_col).
    patch_row, patch_col are in [0, 15].
    """
    img_idx   = token_idx // PATCHES_PER_IMG
    patch_idx = token_idx  % PATCHES_PER_IMG
    row = patch_idx // PATCHES_PER_SIDE
    col = patch_idx  % PATCHES_PER_SIDE
    return image_paths[img_idx], row, col


def load_patch_crop(image_path, patch_row, patch_col, context=2):
    """
    Load image → apply transform → crop a context window
    around (patch_row, patch_col). Returns an RGB numpy array.
    context: number of surrounding patches to include on each side.
    """
    img_tensor = transform(Image.open(image_path).convert("RGB"))
    # Un-normalise for display
    img_display = (img_tensor * _STD[:, None, None] + _MEAN[:, None, None])
    img_display = img_display.clamp(0, 1).permute(1, 2, 0).numpy()  # H W C

    r0 = max(0, patch_row - context) * PATCH_SIZE
    r1 = min(PATCHES_PER_SIDE, patch_row + context + 1) * PATCH_SIZE
    c0 = max(0, patch_col - context) * PATCH_SIZE
    c1 = min(PATCHES_PER_SIDE, patch_col + context + 1) * PATCH_SIZE

    crop = img_display[r0:r1, c0:c1]

    # Highlight the actual activated patch with a border
    border = 2
    hr0 = (patch_row - max(0, patch_row - context)) * PATCH_SIZE
    hc0 = (patch_col - max(0, patch_col - context)) * PATCH_SIZE
    hr1 = hr0 + PATCH_SIZE
    hc1 = hc0 + PATCH_SIZE
    crop_marked = crop.copy()
    crop_marked[hr0:hr0+border, hc0:hc1] = [1, 0.2, 0.2]
    crop_marked[hr1-border:hr1, hc0:hc1] = [1, 0.2, 0.2]
    crop_marked[hr0:hr1, hc0:hc0+border] = [1, 0.2, 0.2]
    crop_marked[hr0:hr1, hc1-border:hc1] = [1, 0.2, 0.2]
    return crop_marked


def compute_atom_activations(tokens, sae, atom_indices):
    """
    Efficiently compute activation strength of selected atoms
    on all tokens WITHOUT running full SAE encode.
    Returns: (N, len(atom_indices)) float32 tensor.
    """
    print(f"Computing activations for {len(atom_indices)} atoms "
          f"on {len(tokens):,} tokens...")
    tokens_norm    = tokens / tokens.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    tokens_centered = tokens_norm - sae.b_dec.cpu()   # (N, d)
    W_enc_sel = sae.W_enc[:, atom_indices].cpu()       # (d, n_atoms)
    b_enc_sel = sae.b_enc[atom_indices].cpu()          # (n_atoms,)

    # Process in chunks to avoid huge RAM spike
    CHUNK = 50_000
    results = []
    for i in range(0, len(tokens_centered), CHUNK):
        chunk = tokens_centered[i:i+CHUNK]             # (chunk, d)
        pre   = chunk @ W_enc_sel + b_enc_sel          # (chunk, n_atoms)
        results.append(pre.relu())
    return torch.cat(results, dim=0)                   # (N, n_atoms)


def make_atom_figure(atom_indices, freq, activations, image_paths,
                     top_k=9, n_cols=4, title_prefix=""):
    """
    Create a figure showing top-activating patches for each atom.
    Each atom gets a cell with top_k patches in a 3×3 grid.
    """
    n_atoms = len(atom_indices)
    n_rows  = (n_atoms + n_cols - 1) // n_cols

    fig_w = n_cols * 5
    fig_h = n_rows * 5.5
    fig, axes = plt.subplots(
        n_rows * 3, n_cols,
        figsize=(fig_w, fig_h),
        gridspec_kw={"hspace": 0.05, "wspace": 0.05},
    )
    if axes.ndim == 1:
        axes = axes[:, None]

    # Flatten to iterate
    atom_ax_rows = [axes[r*3:(r+1)*3, :] for r in range(n_rows)]

    for atom_pos, atom_c in enumerate(atom_indices):
        grid_row = atom_pos // n_cols
        grid_col = atom_pos  % n_cols
        row_axes = atom_ax_rows[grid_row]   # shape (3, n_cols)

        # Top-K token indices for this atom
        acts_c   = activations[:, atom_pos]
        topk_idx = acts_c.topk(top_k)[1].numpy()

        # Title in top-left cell of the 3×3 sub-grid
        title_ax = row_axes[0, grid_col]
        title_ax.set_facecolor("#1a1a2e")
        title_ax.text(0.5, 0.7,
                      f"Atom {atom_c}",
                      ha="center", va="center",
                      color="white", fontsize=10, fontweight="bold",
                      transform=title_ax.transAxes)
        title_ax.text(0.5, 0.3,
                      f"freq = {freq[atom_c]:.4f}",
                      ha="center", va="center",
                      color="#aaaaff", fontsize=8,
                      transform=title_ax.transAxes)
        title_ax.axis("off")

        # Fill 3×3 grid with patch crops (skip first cell = title)
        patch_slots = [(r, c)
                       for r in range(3) for c in range(3)
                       if not (r == 0 and c == 0)][:top_k-1]

        for slot_idx, (pr, pc) in enumerate(patch_slots):
            if slot_idx >= len(topk_idx) - 1:
                break
            tok_idx   = topk_idx[slot_idx + 1]
            img_path, p_row, p_col = token_idx_to_patch(
                int(tok_idx), image_paths)
            try:
                crop = load_patch_crop(img_path, p_row, p_col, context=2)
                ax   = row_axes[pr, grid_col] if pr > 0 else \
                       row_axes[0, grid_col]
                # Use correct axes slot
                target_ax = row_axes[pr, grid_col]
                if pr == 0 and pc == 0:
                    pass  # title cell
                else:
                    target_ax = row_axes[pr, grid_col]
                row_axes[pr, grid_col].imshow(crop)
                row_axes[pr, grid_col].axis("off")
            except Exception as e:
                row_axes[pr, grid_col].text(
                    0.5, 0.5, "err", ha="center",
                    transform=row_axes[pr, grid_col].transAxes)
                row_axes[pr, grid_col].axis("off")

        # Top patch in title row (col 1,2)
        for extra_slot, extra_pc in enumerate([1, 2]):
            if extra_slot < len(topk_idx):
                tok_idx = topk_idx[extra_slot]
                img_path, p_row, p_col = token_idx_to_patch(
                    int(tok_idx), image_paths)
                try:
                    crop = load_patch_crop(img_path, p_row, p_col, context=2)
                    row_axes[0, grid_col].imshow(crop)
                except Exception:
                    pass

    # Hide any unused axes
    for atom_pos in range(n_atoms, n_rows * n_cols):
        grid_row = atom_pos // n_cols
        grid_col = atom_pos  % n_cols
        for r in range(3):
            atom_ax_rows[grid_row][r, grid_col].axis("off")

    fig.suptitle(f"{title_prefix}  |  top-9 activating patches per atom  "
                 f"|  red border = activated patch  |  "
                 f"context = ±2 patches",
                 fontsize=11, y=1.002)
    return fig


# ── main ─────────────────────────────────────────────────────────────

def main():
    print("Loading SAE and stats...")
    sae   = SparseAutoencoder.load(str(SAE_PATH), device="cpu")
    stats = torch.load(STATS_PATH, map_location="cpu", weights_only=False)
    freq  = stats["activation_freq"]          # (C,)
    tokens = torch.load(TOKENS_PATH,
                        map_location="cpu", weights_only=True)

    image_paths = build_image_list()
    print(f"Image list: {len(image_paths)} images")

    # ── select atoms ───────────────────────────────────────────────
    # Mid-freq: 0.003 < freq < 0.10  →  concept-level atoms
    mid_mask = (freq > 0.003) & (freq < 0.10)
    mid_idx  = mid_mask.nonzero(as_tuple=True)[0]
    torch.manual_seed(42)
    mid_sel  = mid_idx[torch.randperm(len(mid_idx))[:16]].tolist()

    # High-freq: 0.10 < freq < 0.50  →  generic/texture atoms
    high_mask = (freq > 0.10) & (freq < 0.50)
    high_idx  = high_mask.nonzero(as_tuple=True)[0]
    high_sel  = high_idx[torch.randperm(len(high_idx))[:8]].tolist()

    print(f"\nMid-freq atoms  ({len(mid_idx):4d} total) → showing 16")
    print(f"High-freq atoms ({len(high_idx):4d} total) → showing 8")
    print(f"Super-atom excluded (freq={freq.max():.4f})")

    all_sel = sorted(set(mid_sel + high_sel))

    # ── compute activations once for all selected atoms ────────────
    atom_indices_tensor = torch.tensor(all_sel)
    activations = compute_atom_activations(tokens, sae,
                                           atom_indices_tensor)
    # activations: (N, n_all_sel)

    # Split back
    mid_positions  = [all_sel.index(a) for a in mid_sel]
    high_positions = [all_sel.index(a) for a in high_sel]

    acts_mid  = activations[:, mid_positions]
    acts_high = activations[:, high_positions]

    # ── figure 1: mid-freq ─────────────────────────────────────────
    print("\nGenerating mid-freq figure (16 atoms)...")
    fig1 = make_atom_figure(
        mid_sel, freq, acts_mid, image_paths,
        top_k=9, n_cols=4,
        title_prefix="Mid-frequency atoms  (0.003 < freq < 0.10)",
    )
    fig1.savefig(OUT_MID, dpi=120, bbox_inches="tight",
                 facecolor="#f8f8f8")
    plt.close(fig1)
    print(f"Saved → {OUT_MID}")

    # ── figure 2: high-freq ────────────────────────────────────────
    print("Generating high-freq figure (8 atoms)...")
    fig2 = make_atom_figure(
        high_sel, freq, acts_high, image_paths,
        top_k=9, n_cols=4,
        title_prefix="High-frequency atoms  (0.10 < freq < 0.50)  — expect generic textures",
    )
    fig2.savefig(OUT_HIGH, dpi=120, bbox_inches="tight",
                 facecolor="#f8f8f8")
    plt.close(fig2)
    print(f"Saved → {OUT_HIGH}")

    # ── text summary ───────────────────────────────────────────────
    print("\n── Frequency distribution of usable atoms ──")
    bands = [
        ("dead        (freq = 0)",        (freq == 0).sum().item()),
        ("too rare    (0 < freq < 0.001)",(((freq > 0) & (freq < 0.001))).sum().item()),
        ("low         (0.001–0.003)",     (((freq >= 0.001) & (freq < 0.003))).sum().item()),
        ("mid-low     (0.003–0.01)",      (((freq >= 0.003) & (freq < 0.01))).sum().item()),
        ("mid         (0.01–0.10)",       (((freq >= 0.01)  & (freq < 0.10))).sum().item()),
        ("high        (0.10–0.50)",       (((freq >= 0.10)  & (freq < 0.50))).sum().item()),
        ("super       (freq ≥ 0.50)",     ((freq >= 0.50)).sum().item()),
    ]
    total = len(freq)
    for label, count in bands:
        bar = "█" * int(count / total * 40)
        print(f"  {label:35s} {count:5d}  {bar}")
    print(f"  {'TOTAL':35s} {total:5d}")
    usable = sum(c for l, c in bands
                 if "dead" not in l and "too rare" not in l
                 and "super" not in l)
    print(f"\n  Usable atoms (0.001 < freq < 0.50): {usable}")


if __name__ == "__main__":
    main()
