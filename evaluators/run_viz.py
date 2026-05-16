"""Clean CLI for generating all thesis figures for one or more categories.

Pass just the category name — everything else is auto-discovered.

═══════════════════════════════════════════════
GENERATE EVERYTHING FOR ONE CATEGORY
═══════════════════════════════════════════════
    python -m evaluators.run_viz --category hazelnut

═══════════════════════════════════════════════
SPECIFIC PLOTS ONLY
═══════════════════════════════════════════════
    python -m evaluators.run_viz --category hazelnut --plots evolution
    python -m evaluators.run_viz --category hazelnut --plots full failures
    python -m evaluators.run_viz --category capsule  --plots patch_space full

Available plot names:
    full          — 5-perspective figure (image / heatmap / t-SNE / concepts / decision)
    patch_space   — patch feature space t-SNE (3 panels)
    evolution     — C-AUC over tasks: CONCIL vs naive BWT comparison
    failures      — failure case gallery

═══════════════════════════════════════════════
MULTIPLE CATEGORIES
═══════════════════════════════════════════════
    python -m evaluators.run_viz --categories hazelnut capsule bottle

═══════════════════════════════════════════════
CUSTOM PATHS (if not using default structure)
═══════════════════════════════════════════════
    python -m evaluators.run_viz \\
        --category    hazelnut \\
        --checkpoint  ./checkpoints/hazelnut-final \\
        --figures_dir ./my_figures
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


_ALL_PLOTS = ["full", "patch_space", "evolution", "failures"]
_MVTEC_DEFAULT = "/home/sobhan_hosseini/datasets/mvtec"


# ── argument parsing ──────────────────────────────────────────────────────────

def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate thesis figures for CONVAD-CL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # which categories
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--category",   help="single category, e.g. hazelnut")
    grp.add_argument("--categories", nargs="+", help="multiple categories")

    # which plots (default = all)
    p.add_argument("--plots", nargs="+", choices=_ALL_PLOTS, default=_ALL_PLOTS,
                   metavar="PLOT",
                   help=f"which figures to generate (default: all). "
                        f"choices: {_ALL_PLOTS}")

    # paths
    p.add_argument("--mvtec_root",      default=_MVTEC_DEFAULT)
    p.add_argument("--annotations_root", default="./annotations")
    p.add_argument("--checkpoints_root", default="./checkpoints")
    p.add_argument("--figures_root",     default="./thesis_figures",
                   help="figures saved to {figures_root}/{category}/")

    # overrides for single-category runs
    p.add_argument("--checkpoint",  default=None,
                   help="override checkpoint dir (single --category only)")
    p.add_argument("--defect",      default=None,
                   help="use this specific defect image for full/patch_space "
                        "(default: first defect type, first image)")
    p.add_argument("--n_memory_samples", type=int, default=200,
                   help="memory patches to show in t-SNE (default 200)")

    return p.parse_args()


# ── loader helpers ────────────────────────────────────────────────────────────

def _load_log(path: Path):
    from evaluators.evaluator_cl import TaskEvalResult, ContinualLog
    if not path.exists():
        return None
    with open(path) as f:
        d = json.load(f)
    return ContinualLog(results=[TaskEvalResult(**r) for r in d["results"]])


def _build_config(cat: str, args: argparse.Namespace):
    from evaluators.visualizer import VisualizerConfig
    ckpt = Path(args.checkpoint) if (args.checkpoint and args.category) \
           else Path(args.checkpoints_root) / f"{cat}-final"
    return VisualizerConfig(
        category        = cat,
        mvtec_root      = Path(args.mvtec_root),
        annotations_dir = Path(args.annotations_root) / cat,
        checkpoint_dir  = ckpt,
    )


# ── per-plot generators ───────────────────────────────────────────────────────

def _plot_full(viz, args, fig_dir: Path) -> None:
    """5-perspective figure for one defect image and one normal image."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    mvtec_test = viz.config.mvtec_root / viz.config.category / "test"

    # Pick defect
    defect = args.defect or viz.defect_types[0]
    defect_imgs = sorted((mvtec_test / defect).glob("*.png"))
    normal_imgs = sorted((mvtec_test / "good").glob("*.png"))

    if not defect_imgs:
        print(f"  [full] No images found for defect={defect}")
        return

    viz.visualize_full(
        image_path = str(defect_imgs[0]),
        true_label = defect,
        save_path  = str(fig_dir / f"full_{defect}_000.png"),
        n_memory_samples = args.n_memory_samples,
    )
    print(f"  Saved: full_{defect}_000.png")

    viz.visualize_full(
        image_path = str(normal_imgs[0]),
        true_label = "good",
        save_path  = str(fig_dir / "full_normal_000.png"),
        n_memory_samples = args.n_memory_samples,
    )
    print(f"  Saved: full_normal_000.png")


def _plot_patch_space(viz, args, fig_dir: Path) -> None:
    """3-panel patch t-SNE for one defect and one normal image."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    mvtec_test = viz.config.mvtec_root / viz.config.category / "test"

    defect = args.defect or viz.defect_types[0]
    defect_imgs = sorted((mvtec_test / defect).glob("*.png"))
    normal_imgs = sorted((mvtec_test / "good").glob("*.png"))

    viz.visualize_patch_space(
        test_image_path = str(defect_imgs[0]),
        true_label      = defect,
        save_path       = str(fig_dir / f"patch_space_{defect}_000.png"),
        n_memory_samples = args.n_memory_samples,
    )
    print(f"  Saved: patch_space_{defect}_000.png")

    viz.visualize_patch_space(
        test_image_path = str(normal_imgs[0]),
        true_label      = "good",
        save_path       = str(fig_dir / "patch_space_normal_000.png"),
        n_memory_samples = args.n_memory_samples,
    )
    print(f"  Saved: patch_space_normal_000.png")


def _plot_evolution(viz, args, fig_dir: Path) -> None:
    """C-AUC over tasks: CONCIL vs naive BWT comparison line plot."""
    from evaluators.visualizer import visualize_concept_evolution
    fig_dir.mkdir(parents=True, exist_ok=True)
    cat = viz.config.category

    log_concil = _load_log(Path(args.checkpoints_root) / f"{cat}-final" / "log_final.json")
    log_naive  = _load_log(Path(args.checkpoints_root) / f"{cat}-baseline-final" / "log_final.json")

    if log_concil is None:
        print(f"  [evolution] No CONCIL log found — run the experiment first.")
        return

    save = str(fig_dir / "concept_evolution.png")
    visualize_concept_evolution(
        log_concil   = log_concil,
        log_naive    = log_naive,
        defect_names = viz.defect_types,
        save_path    = save,
    )
    print(f"  Saved: concept_evolution.png")

    bwt_c = log_concil.mean_concept_bwt()
    bwt_n = log_naive.mean_concept_bwt() if log_naive else float("nan")
    print(f"  BWT — CONCIL: {bwt_c:+.4f}   Naive: {bwt_n:+.4f}")


def _plot_failures(viz, args, fig_dir: Path) -> None:
    """Failure case gallery (auto-scans all defect types)."""
    from evaluators.visualizer import (
        find_failure_cases_generic, visualize_prediction
    )
    fail_dir = fig_dir / "failures"
    fail_dir.mkdir(parents=True, exist_ok=True)

    tier_map = str(viz.config.annotations_dir / "cl_tasks" / "concept_tier_map.json")
    failures = find_failure_cases_generic(
        viz.config, viz, max_per_type=3, tier_map_path=tier_map
    )

    if not failures:
        print("  [failures] No failure cases found on this test set.")
        return

    import matplotlib.pyplot as plt, matplotlib.patches as mpatches
    from PIL import Image
    import numpy as np

    thumbs, labels = [], []
    for fc in failures:
        img  = Image.open(fc.image_path).convert("RGB")
        r    = viz._infer(img, fc.true_label)
        stem = Path(fc.image_path).stem
        fn   = fail_dir / f"{fc.failure_type}_{fc.true_label}_{stem}.png"
        visualize_prediction(img, r, save_path=str(fn),
                             title_prefix=f"[{fc.failure_type.upper()}]",
                             normal_baseline=viz.normal_baseline)
        thumbs.append(np.array(img.resize((112,112))))
        labels.append(f"{fc.failure_type}\n{fc.true_label}\ns={fc.s_novel:.2f}")

    # summary grid
    n = len(thumbs); ncols = min(n, 6); nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*2.5, nrows*3))
    flat = np.array(axes).flatten() if n > 1 else [axes]
    for i, (th, lb) in enumerate(zip(thumbs, labels)):
        flat[i].imshow(th); flat[i].set_title(lb, fontsize=7); flat[i].axis("off")
    for ax in flat[len(thumbs):]: ax.axis("off")
    fig.suptitle(f"{viz.config.category} — Failure Gallery", fontsize=11, fontweight="bold")
    fig.savefig(fail_dir / "failure_gallery_summary.png", bbox_inches="tight")
    plt.close(fig)

    by_type = {}
    for fc in failures:
        by_type.setdefault(fc.failure_type, []).append(fc)
    for ft, cases in sorted(by_type.items()):
        print(f"  {ft:<22} {len(cases)} case(s)")
    print(f"  Saved: failures/ ({len(failures)} figures + summary)")


# ── main ──────────────────────────────────────────────────────────────────────

_PLOT_FNS = {
    "full":        _plot_full,
    "patch_space": _plot_patch_space,
    "evolution":   _plot_evolution,
    "failures":    _plot_failures,
}


def run_category(cat: str, args: argparse.Namespace) -> None:
    from evaluators.visualizer import CategoryVisualizer

    print(f"\n{'═'*56}")
    print(f"  {cat.upper()}  —  plots: {args.plots}")
    print(f"{'═'*56}")

    config  = _build_config(cat, args)
    fig_dir = Path(args.figures_root) / cat

    # Load model
    print("  Loading model ...")
    viz = CategoryVisualizer(config)
    print(f"  τ={viz.tau:.4f}  K={len(viz.concept_names)}  "
          f"defects={viz.defect_types}")

    # Run requested plots
    for plot_name in args.plots:
        print(f"\n  ── {plot_name} ──────────────────────────────")
        _PLOT_FNS[plot_name](viz, args, fig_dir)

    # List all saved files
    saved = sorted(fig_dir.rglob("*.png"))
    print(f"\n  {len(saved)} figure(s) saved to {fig_dir}/")
    for p in saved:
        print(f"    {p.relative_to(fig_dir)}")


def main() -> None:
    args = _parse()

    categories = args.categories if args.categories else [args.category]

    print(f"CONVAD-CL Visualizer")
    print(f"Categories : {categories}")
    print(f"Plots      : {args.plots}")
    print(f"Figures    : {args.figures_root}/{{category}}/")

    for cat in categories:
        run_category(cat, args)

    print("\nDone.")


if __name__ == "__main__":
    main()
