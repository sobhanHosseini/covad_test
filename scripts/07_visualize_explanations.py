"""
Phase 4: Paper-quality explanation figures for SAE-CBM.

For each of the 5 categories generates figures for:
  - hardest defect (lowest final I-AUC from Phase 3)
  - best defect   (highest final I-AUC from Phase 3)

Each figure: 5-panel (1×5), figsize=(20,4), dpi=150
  Panel 1 — Original image
  Panel 2 — PatchCore heatmap (Branch 1)
  Panel 3 — Guide score heatmap (Branch 2)
  Panel 4 — Top concept atom (highest contribution = mean_z[c] × (g⁺[c]−g⁻[c]))
  Panel 5 — Text explanation (top-3 concepts by same contribution score)

Outputs:
  sae_training/figures/{category}_{defect}.png
  Console summary of top firing atoms per figure.

Run from project root:
    python scripts/07_visualize_explanations.py
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from features.dinov2_extractor import DINOv2Extractor
from features.sae              import SparseAutoencoder
from features.guide_trainer    import GuideCoeffTrainer
from features.patchcore_memory import PatchCoreMemory

# ── config ────────────────────────────────────────────────────────────────────

MVTEC_ROOT  = Path("/home/sobhan_hosseini/datasets/mvtec")
ANN_ROOT    = Path("annotations")
GUIDES_DIR  = Path("sae_training/guides")
TOKENS_PATH = Path("sae_training/mvtec_normal_patches_vitl14reg.pt")
INDEX_PATH  = Path("sae_training/mvtec_patch_index_reg.pt")
SAE_PATH    = Path("sae_training/sae_vitl14reg_C4096_k64.pt")
NAMES_PATH  = Path("sae_training/atom_names.json")
FIGURES_DIR   = Path("sae_training/figures")
BASELINES_DIR = Path("sae_training/baselines")

MODEL_NAME         = "dinov2_vitl14_reg"
EMBED_DIM          = 1024
CORESET_SIZE       = 10_000
DEVICE             = torch.device("cuda:0")
DEFECT_TRAIN_RATIO = 0.80
SEED               = 42
GUIDE_SCORE_THRESH = 0.5    # min guide score to select image; fallback: highest scorer

# Hardest / best defect per category from Phase 3 results_summary.json
# (T-final I-AUC): hardest = lowest, best = highest
BEST_WORST = {
    "bottle":    {"worst": "broken_small",   "best": "broken_large"},
    "capsule":   {"worst": "faulty_imprint", "best": "scratch"},
    "hazelnut":  {"worst": "cut",            "best": "crack"},
    "metal_nut": {"worst": "bent",           "best": "color"},
    "screw":     {"worst": "scratch_head",   "best": "scratch_neck"},
}

_HEATMAP_ZOOM = 14   # 16 × 14 = 224   (matching visualizer.py)
_IMG_SIZE     = 224

plt.rcParams.update({
    "font.family":    "DejaVu Sans",
    "font.size":      9,
    "figure.dpi":     150,
    "savefig.dpi":    150,
    "savefig.bbox":   "tight",
})

# ── helpers ───────────────────────────────────────────────────────────────────

def load_defect_split(task_csv_path: str):
    import pandas as pd
    df        = pd.read_csv(task_csv_path)
    defect_df = df[df["label_index"] == 1].reset_index(drop=True)
    n_defect  = len(defect_df)
    n_train   = max(1, int(n_defect * DEFECT_TRAIN_RATIO))
    rng          = np.random.RandomState(SEED)
    shuffled_idx = rng.permutation(n_defect)
    held_idx     = shuffled_idx[n_train:]
    return defect_df.iloc[held_idx]["image_path"].tolist()


def find_task_csv(category: str, defect: str) -> str:
    task_seq_path = ANN_ROOT / category / "cl_tasks" / "task_sequence.json"
    with open(task_seq_path) as f:
        tasks = json.load(f)
    for t in tasks:
        if t["defect"] == defect:
            return t["csv_path"]
    raise ValueError(f"Defect '{defect}' not found in {task_seq_path}")


def minmax_norm(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo + 1e-8)


def zoom_to_224(map_16x16: np.ndarray) -> np.ndarray:
    """Upsample (16,16) spatial map to (224,224) matching visualizer.py."""
    return scipy.ndimage.zoom(map_16x16, _HEATMAP_ZOOM, order=3)


def overlay_heatmap(ax, img_arr: np.ndarray, heatmap_224: np.ndarray,
                    cmap: str = "hot", alpha: float = 0.55, title: str = ""):
    """Overlay a (224,224) heatmap on the image (per-image min-max, matching visualizer)."""
    hm_norm = minmax_norm(heatmap_224)
    ax.imshow(img_arr)
    im = ax.imshow(hm_norm, cmap=cmap, alpha=alpha, vmin=0, vmax=1)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=8, pad=3)
    return im


# ── per-image figure ──────────────────────────────────────────────────────────

@torch.no_grad()
def make_figure(
    img_path: str,
    category: str,
    defect: str,
    memory: PatchCoreMemory,
    trainer: GuideCoeffTrainer,
    sae: SparseAutoencoder,
    extractor: DINOv2Extractor,
    atom_names: dict,
    usable_mask: torch.Tensor,
    normal_mean: torch.Tensor,
) -> plt.Figure:
    # ── load image ────────────────────────────────────────────────────────────
    img_pil     = Image.open(img_path).convert("RGB")
    img_display = img_pil.resize((_IMG_SIZE, _IMG_SIZE), Image.BICUBIC)
    img_arr     = np.array(img_display)

    # ── extract patch tokens ──────────────────────────────────────────────────
    tokens_3d   = extractor.extract_patch_tokens([img_pil])    # (1, 256, 1024)
    tokens_flat = tokens_3d.squeeze(0)                         # (256, 1024)

    # ── Branch 1: PatchCore patch distances ───────────────────────────────────
    _, anomaly_maps = memory.score(tokens_3d)                  # maps: (1, 16, 16)
    amap_16         = anomaly_maps[0].cpu().numpy()            # (16, 16)
    amap_224        = zoom_to_224(amap_16)                     # (224, 224)
    s_novel         = float(anomaly_maps[0].mean().item())

    # ── Branch 2: per-patch guide scores ─────────────────────────────────────
    patch_scores    = trainer.score_image_patches(tokens_flat) # (256,) CPU
    guide_16        = patch_scores.reshape(16, 16).numpy()
    guide_224       = zoom_to_224(guide_16)
    s_guide         = float(patch_scores.max().item())

    # ── SAE encoding ─────────────────────────────────────────────────────────
    z = sae.encode(tokens_flat.to(DEVICE)).cpu()               # (256, 4096) float32

    # Panel 4: atom that contributes most to anomaly score (causal explanation)
    # contrib(c) = mean_z[c] × (g⁺[c] − g⁻[c])  — signed contribution to score
    g_diff      = (trainer.g_pos - trainer.g_neg).cpu()        # (C,)
    mean_acts   = z.mean(0)                                     # (C,) mean over 256 patches
    contrib     = mean_acts * g_diff                            # (C,) signed contribution

    contrib_usable             = contrib.clone()
    contrib_usable[~usable_mask] = -1.0
    c_star      = int(contrib_usable.argmax().item())

    # Heatmap: per-patch contribution of c_star to the guide score
    patch_contrib = z[:, c_star] * g_diff[c_star].item()       # (256,)
    atom_map_16   = patch_contrib.reshape(16, 16).numpy()
    atom_map_224  = zoom_to_224(atom_map_16)
    c_star_name   = atom_names.get(str(c_star), {}).get("names", ["unknown"])[0]

    # Panel 5: contrastive explanation — absolute diff vs normal baseline
    diff     = mean_acts - normal_mean                      # (C,) signed, no division
    min_act  = 0.005
    meaningful = usable_mask & ((mean_acts > min_act) | (normal_mean > min_act))

    diff_masked = diff.clone()
    diff_masked[~meaningful] = 0.0

    # Top INCREASED: highest positive diff, deduplicated by name
    seen_inc: set[str] = set()
    top_increased: list[tuple[int, float, str]] = []
    for i in diff_masked.argsort(descending=True).tolist():
        if diff[i] < 0.002:
            break
        name = atom_names.get(str(i), {}).get("names", ["?"])[0]
        if name not in seen_inc:
            seen_inc.add(name)
            top_increased.append((i, float(diff[i]), name))
        if len(top_increased) == 3:
            break

    # Top DECREASED: most negative diff, deduplicated by name
    seen_dec: set[str] = set()
    top_decreased: list[tuple[int, float, str]] = []
    for i in diff_masked.argsort(descending=False).tolist():
        if diff[i] > -0.002:
            break
        name = atom_names.get(str(i), {}).get("names", ["?"])[0]
        if name not in seen_dec:
            seen_dec.add(name)
            top_decreased.append((i, float(diff[i]), name))
        if len(top_decreased) == 2:
            break

    # Cross-direction dedup: same name in both = contradictory, remove from both
    increased_names = {nm for _, _, nm in top_increased}
    decreased_names = {nm for _, _, nm in top_decreased}
    shared_names    = increased_names & decreased_names
    if shared_names:
        top_increased = [(i, d, n) for i, d, n in top_increased
                         if n not in shared_names]
        top_decreased = [(i, d, n) for i, d, n in top_decreased
                         if n not in shared_names]

    # ── build figure ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    fig.subplots_adjust(wspace=0.08)

    # Panel 1: original
    axes[0].imshow(img_arr)
    axes[0].axis("off")
    axes[0].set_title("Original", fontsize=9, fontweight="bold")

    # Panel 2: PatchCore heatmap (B1)
    overlay_heatmap(axes[1], img_arr, amap_224, cmap="hot",
                    title=f"PatchCore (B1)\ns_novel={s_novel:.3f}")

    # Panel 3: guide score heatmap (B2)
    overlay_heatmap(axes[2], img_arr, guide_224, cmap="hot",
                    title=f"Guide Score (B2)\ns_guide={s_guide:.3f}")

    # Panel 4: top concept atom
    overlay_heatmap(axes[3], img_arr, atom_map_224, cmap="plasma",
                    title=f"Top Concept\n\"{c_star_name}\"")

    # Panel 5: contrastive text explanation
    ax5 = axes[4]
    ax5.set_facecolor("#fafafa")
    ax5.axis("off")
    ax5.set_title("Explanation", fontsize=9, fontweight="bold")

    lines = [
        "Decision: Level 2 – Known Anomaly",
        "─" * 32,
        "Concept changes vs normal:",
    ]
    if top_increased:
        lines.append("↑ More active than usual:")
        for _, dv, nm in top_increased:
            short = nm if len(nm) <= 24 else nm[:22] + "…"
            lines.append(f"  {short}")
            lines.append(f"    Δ+{dv:.4f}")
    if top_decreased:
        lines.append("↓ Less active than usual:")
        for _, dv, nm in top_decreased:
            short = nm if len(nm) <= 24 else nm[:22] + "…"
            lines.append(f"  {short}")
            lines.append(f"    Δ{dv:.4f}")
    if not top_increased and not top_decreased:
        lines += ["No significant concept", "shift detected."]
    lines += [
        "─" * 32,
        f"B1: {s_novel:.3f}  |  B2: {s_guide:.3f}",
    ]

    ax5.text(
        0.05, 0.95, "\n".join(lines),
        transform=ax5.transAxes,
        va="top", ha="left",
        fontsize=7.5,
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#cccccc", lw=0.8),
    )

    fig.suptitle(
        f"{category.upper()} — {defect}  |  {Path(img_path).name}",
        fontsize=10, y=1.02,
    )
    return fig, c_star, c_star_name, top_increased, top_decreased


# ── per-category runner ───────────────────────────────────────────────────────

@torch.no_grad()
def run_category(
    category: str,
    sae: SparseAutoencoder,
    all_tokens: torch.Tensor,
    patch_index: list[dict],
    extractor: DINOv2Extractor,
    atom_names: dict,
    usable_mask: torch.Tensor,
    baselines: dict,
) -> None:
    print(f"\n{'='*60}")
    print(f"  {category.upper()}")
    print(f"{'='*60}")

    # Build PatchCore from pre-extracted normal tokens
    cat_info           = next(e for e in patch_index if e["category"] == category)
    n_images           = cat_info["n_images"]
    normal_tokens_flat = all_tokens[cat_info["row_start"]:cat_info["row_end"]]
    normal_tokens_3d   = normal_tokens_flat.reshape(n_images, 256, EMBED_DIM)

    memory = PatchCoreMemory(coreset_size=CORESET_SIZE, device=DEVICE, seed=42)
    memory.build(normal_tokens_3d)

    # Load last-task guide trainer
    last_ckpt   = sorted((GUIDES_DIR / category).glob("task_*.pt"))[-1]
    trainer     = GuideCoeffTrainer.load(last_ckpt, sae, device=DEVICE)
    normal_mean = baselines[category]   # (4096,) mean SAE activation on normals

    defects_to_viz = [
        ("worst", BEST_WORST[category]["worst"]),
        ("best",  BEST_WORST[category]["best"]),
    ]

    for kind, defect in defects_to_viz:
        print(f"\n  [{kind}] defect: {defect}")

        # Get held-out images
        csv_path   = find_task_csv(category, defect)
        held_paths = load_defect_split(csv_path)

        if not held_paths:
            print(f"    No held-out images — skipping.")
            continue

        # Score all held-out images; pick one above threshold
        scores = []
        for p in held_paths:
            try:
                img    = Image.open(p).convert("RGB")
                tokens = extractor.extract_patch_tokens([img]).squeeze(0)
                sc     = trainer.score_image(tokens)
                scores.append((sc, p))
            except Exception as e:
                print(f"    Warning: could not score {p}: {e}")

        if not scores:
            print("    All images failed to score — skipping.")
            continue

        scores.sort(key=lambda x: -x[0])
        chosen_score, chosen_path = scores[0]
        print(f"    Selected: {Path(chosen_path).name}  guide_score={chosen_score:.3f}")

        # Generate figure
        try:
            fig, c_star, c_name, top_increased, top_decreased = make_figure(
                chosen_path, category, defect,
                memory, trainer, sae, extractor,
                atom_names, usable_mask, normal_mean,
            )
        except Exception as e:
            print(f"    ERROR generating figure: {e}")
            continue

        # Save
        out_path = FIGURES_DIR / f"{category}_{defect}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"    Saved → {out_path}")

        # Summary printout
        print(f"    Top atom: #{c_star}  \"{c_name}\"")
        if top_increased:
            print(f"    Top increased: {top_increased[0][2]} (Δ+{top_increased[0][1]:.4f})")
        else:
            print(f"    Top increased: none")
        if top_decreased:
            print(f"    Top decreased: {top_decreased[0][2]} (Δ{top_decreased[0][1]:.4f})")
        else:
            print(f"    Top decreased: none")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    # Preflight
    required = [TOKENS_PATH, INDEX_PATH, SAE_PATH, NAMES_PATH]
    missing  = [p for p in required if not p.exists()]
    if missing:
        print("ERROR — missing required files:")
        for p in missing:
            print(f"  {p}")
        sys.exit(1)

    missing_baselines = [
        cat for cat in BEST_WORST
        if not (BASELINES_DIR / f"{cat}_normal_mean.pt").exists()
    ]
    if missing_baselines:
        print(f"ERROR — baselines not found for: {missing_baselines}")
        print("Run first: python scripts/08_compute_normal_baselines.py")
        sys.exit(1)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Phase 4: Explanation Visualisation")
    print("=" * 60)

    # Load shared resources
    print("\nLoading SAE …")
    sae = SparseAutoencoder.load(str(SAE_PATH), device="cpu")
    sae.to(DEVICE).eval()

    print("Loading atom names …")
    with open(NAMES_PATH) as f:
        atom_names = json.load(f)    # keys are strings

    # Build usable-atom boolean mask (4096,)
    usable_mask = torch.zeros(sae.config.d_hidden, dtype=torch.bool)
    for k in atom_names:
        usable_mask[int(k)] = True
    print(f"  Usable atoms: {usable_mask.sum().item()} / {sae.config.d_hidden}")

    print("Loading pre-extracted tokens …")
    all_tokens  = torch.load(TOKENS_PATH, map_location="cpu", weights_only=True)
    patch_index = torch.load(INDEX_PATH,  map_location="cpu", weights_only=False)

    print("Loading DINOv2 extractor …")
    extractor = DINOv2Extractor(MODEL_NAME, device=DEVICE)
    extractor.eval()

    print("Loading normal baselines …")
    baselines = {
        cat: torch.load(BASELINES_DIR / f"{cat}_normal_mean.pt", weights_only=True)
        for cat in BEST_WORST
    }

    for category in BEST_WORST:
        run_category(
            category, sae, all_tokens, patch_index,
            extractor, atom_names, usable_mask, baselines,
        )

    print("\n" + "=" * 60)
    print(f"  Done. Figures saved to {FIGURES_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
