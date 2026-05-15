"""Generic visualization entry point — runs for hazelnut AND capsule.

Usage:
    python -m evaluators.generic_viz
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from evaluators.visualizer import (
    VisualizerConfig, CategoryVisualizer, find_failure_cases_generic,
    visualize_failure_gallery, _run_inference, _LEVEL_LABEL,
)


def _run_category(config: VisualizerConfig, fig_dir: Path) -> None:
    cat = config.category
    fig_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"  Category: {cat.upper()}")
    print(f"{'='*60}")

    viz = CategoryVisualizer(config)

    mvtec_test = config.mvtec_root / cat / "test"
    tier_map   = str(config.annotations_dir / "cl_tasks" / "concept_tier_map.json")

    # ── 1. patch space: one defect, one normal ────────────────────────────────
    defect0  = viz.defect_types[0]
    defect_imgs = sorted((mvtec_test / defect0).glob("*.png"))
    normal_imgs = sorted((mvtec_test / "good").glob("*.png"))

    print(f"\n  Generating patch space figures ({defect0} + good) ...")
    viz.visualize_patch_space(
        str(defect_imgs[0]), defect0,
        str(fig_dir / f"patch_space_{defect0}_000.png"),
        n_memory_samples=300,
    )
    print(f"    Saved: patch_space_{defect0}_000.png")

    viz.visualize_patch_space(
        str(normal_imgs[0]), "good",
        str(fig_dir / f"patch_space_normal_000.png"),
        n_memory_samples=300,
    )
    print(f"    Saved: patch_space_normal_000.png")

    # ── 2. full 5-perspective figures ─────────────────────────────────────────
    print(f"\n  Generating 5-perspective figures ...")
    viz.visualize_full(
        str(defect_imgs[0]), defect0,
        str(fig_dir / f"full_{defect0}_000.png"),
        n_memory_samples=200,
    )
    print(f"    Saved: full_{defect0}_000.png")

    # Find a false positive normal for second figure (or use first normal)
    full_fp = str(normal_imgs[min(2, len(normal_imgs)-1)])
    viz.visualize_full(
        full_fp, "good",
        str(fig_dir / "full_normal_000.png"),
        n_memory_samples=200,
    )
    print(f"    Saved: full_normal_000.png")

    # ── 3. failure gallery ────────────────────────────────────────────────────
    print(f"\n  Scanning for failure cases ...")
    failures = find_failure_cases_generic(config, viz, max_per_type=2,
                                          tier_map_path=tier_map)
    by_type = {}
    for fc in failures:
        by_type.setdefault(fc.failure_type, []).append(fc)

    print(f"  Failure summary:")
    for ft, cases in sorted(by_type.items()):
        print(f"    {ft:<22} {len(cases)} case(s)")
        for c in cases[:1]:
            print(f"      {Path(c.image_path).name:<18} {c.description}")

    if failures:
        fail_dir = fig_dir / "failures"
        for fc in failures:
            img  = Image.open(fc.image_path).convert("RGB")
            r    = viz._infer(img, fc.true_label)
            stem = Path(fc.image_path).stem
            fn   = fail_dir / f"{fc.failure_type}_{fc.true_label}_{stem}.png"
            from evaluators.visualizer import visualize_prediction
            visualize_prediction(
                img, r, save_path=str(fn),
                title_prefix=f"[{fc.failure_type.upper()}]",
                normal_baseline=viz.normal_baseline,
            )
        # summary grid
        thumbs, labels = [], []
        for fc in failures:
            img = Image.open(fc.image_path).convert("RGB")
            thumbs.append(np.array(img.resize((112,112))))
            labels.append(f"{fc.failure_type}\n{fc.true_label}\ns={fc.s_novel:.2f}")

        import matplotlib.pyplot as plt, matplotlib.patches as mpatches
        n = len(thumbs); ncols = min(n,6); nrows = (n+ncols-1)//ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*2.5, nrows*3))
        flat = np.array(axes).flatten() if n > 1 else [axes]
        for i,(th,lb) in enumerate(zip(thumbs,labels)):
            flat[i].imshow(th); flat[i].set_title(lb,fontsize=7); flat[i].axis("off")
        for ax in flat[len(thumbs):]: ax.axis("off")
        fig.suptitle(f"{cat} — Failure Summary", fontsize=11, fontweight="bold")
        fail_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(fail_dir / "failure_gallery_summary.png", bbox_inches="tight")
        plt.close(fig)
        print(f"  Failure gallery saved to {fail_dir}/")

    # ── print saved files ─────────────────────────────────────────────────────
    saved = sorted(fig_dir.rglob("*.png"))
    print(f"\n  Files saved to {fig_dir}/")
    for p in saved:
        print(f"    {p.relative_to(fig_dir)}")


def main():
    BASE = Path(__file__).resolve().parent.parent

    # Derive mvtec_root from hazelnut CSV
    _first = pd.read_csv(BASE / "annotations/hazelnut/hazelnut.csv",
                          nrows=1)["image_path"].iloc[0]
    mvtec_root = Path(_first).parents[3]

    configs = [
        VisualizerConfig(
            category       = "hazelnut",
            mvtec_root     = mvtec_root,
            annotations_dir= BASE / "annotations/hazelnut",
            checkpoint_dir = BASE / "checkpoints/hazelnut-final",
        ),
        VisualizerConfig(
            category       = "capsule",
            mvtec_root     = mvtec_root,
            annotations_dir= BASE / "annotations/capsule",
            checkpoint_dir = BASE / "checkpoints/capsule",
        ),
    ]

    for cfg in configs:
        fig_dir = BASE / "thesis_figures" / cfg.category
        _run_category(cfg, fig_dir)

    print(f"\nAll figures complete.")


if __name__ == "__main__":
    main()
