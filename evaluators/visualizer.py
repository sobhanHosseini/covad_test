"""Publication-quality visualization for CONVAD-CL thesis figures.

Functions:
    visualize_prediction()       — 2×2 panel: image / heatmap / concepts / decision
    find_failure_cases()         — scan test set for failure examples
    visualize_failure_gallery()  — save per-failure figures + summary grid
    visualize_concept_evolution()— C-AUC over tasks: CONCIL vs naive baseline

cv2 is replaced by scipy.ndimage.zoom for heatmap upsampling.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import scipy.ndimage
import torch
from PIL import Image

from evaluators.evaluator_cl import ContinualLog

# ── global style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "font.size":          10,
    "axes.linewidth":     0.8,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "figure.dpi":         150,
    "savefig.dpi":        150,
    "savefig.bbox":       "tight",
})

_LEVEL_COLOUR = {1: "#2ecc71", 2: "#f39c12", 3: "#e74c3c"}
_LEVEL_LABEL  = {1: "NORMAL",  2: "KNOWN ANOMALY", 3: "NOVEL ANOMALY"}
_IMG_SIZE     = 224
_HEATMAP_ZOOM = _IMG_SIZE // 16   # 14


# ── dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class PredictionResult:
    s_novel:          float
    tau:              float
    anomaly_map:      np.ndarray    # (16, 16)
    c:                np.ndarray    # (K,) concept activations [0,1]
    y_pred:           float
    concept_names:    list[str]
    level:            int           # 1=Normal, 2=Known, 3=Novel
    true_label:       str
    predicted_label:  str
    correct:          bool


@dataclass
class FailureCase:
    image_path:    str
    true_label:    str
    failure_type:  str             # "false_negative","false_positive","explainability","level3"
    s_novel:       float
    c:             np.ndarray
    anomaly_map:   np.ndarray
    y_pred:        float
    description:   str


# ── inference helper ──────────────────────────────────────────────────────────

@torch.no_grad()
def _run_inference(
    extractor, memory, concept_heads, anomaly_head,
    image, tau: float, theta_concept: float, true_label: str = "",
) -> PredictionResult:
    if isinstance(image, (str, Path)):
        image = Image.open(image).convert("RGB")

    patch_tokens, pooled_z = extractor.extract_both([image])     # (1,256,768), (1,1536)
    s_novel_t, amap_t      = memory.score(patch_tokens)          # (1,), (1,16,16)
    c_t                    = concept_heads(pooled_z)             # (1,K)
    y_pred_t               = anomaly_head(c_t)                   # (1,)

    s_novel    = float(s_novel_t[0].cpu())
    anomaly_map = amap_t[0].cpu().numpy()
    c          = c_t[0].cpu().numpy()
    y_pred     = float(y_pred_t[0].cpu())

    if s_novel < tau:
        level, predicted = 1, "normal"
    elif float(c.max()) > theta_concept:
        level, predicted = 2, "known_anomaly"
    else:
        level, predicted = 3, "novel_anomaly"

    is_normal = true_label.lower() in ("normal", "good", "")
    correct   = (predicted == "normal") if is_normal else (predicted != "normal")

    return PredictionResult(
        s_novel=s_novel, tau=tau, anomaly_map=anomaly_map,
        c=c, y_pred=y_pred, concept_names=concept_heads.concept_names,
        level=level, true_label=true_label,
        predicted_label=predicted, correct=correct,
    )


# ── helpers for visualize_prediction ─────────────────────────────────────────

def _bar(val: float, width: int = 10, max_val: float = 1.0) -> str:
    filled = int(min(max(val / max_val, 0.0), 1.0) * width)
    return "█" * filled + "░" * (width - filled)


def _add_border(ax, colour: str, lw: float = 5.0) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(lw)
        spine.set_color(colour)


# ── FUNCTION 0 ────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_normal_baseline(
    concept_heads,
    extractor,
    normal_images: list,          # list of PIL.Image or path strings
    concept_names: list[str],
    save_path: Optional[str] = None,
    batch_size: int = 16,
) -> dict[str, float]:
    """Compute mean (and std) concept activation across all normal training images.

    Call once after Task 1 and cache the result.  Used by visualize_prediction()
    to show deviation-from-normal rather than raw activation values.

    Args:
        concept_heads: trained ConceptHeads instance
        extractor:     DINOv2Extractor instance
        normal_images: all normal training images (train/good/)
        concept_names: ordered concept names (must match concept_heads order)
        save_path:     if given, saves JSON with 'means' and 'stds' dicts
        batch_size:    extraction batch size

    Returns:
        {concept_name: mean_activation_on_normals}  — also saved to JSON.
    """
    all_activations: list[np.ndarray] = []

    for i in range(0, len(normal_images), batch_size):
        batch = normal_images[i : i + batch_size]
        batch_pil = [
            Image.open(p).convert("RGB") if isinstance(p, (str, Path)) else p
            for p in batch
        ]
        _, pooled_z = extractor.extract_both(batch_pil)
        c = concept_heads(pooled_z).cpu().numpy()        # (B, K)
        all_activations.append(c)

    A     = np.vstack(all_activations)                   # (N, K)
    means = A.mean(axis=0)                               # (K,)
    stds  = A.std(axis=0)                                # (K,)

    means_dict = {concept_names[i]: float(means[i]) for i in range(len(concept_names))}
    stds_dict  = {concept_names[i]: float(stds[i])  for i in range(len(concept_names))}

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump({"means": means_dict, "stds": stds_dict}, f, indent=2)

    return means_dict


# ── FUNCTION 1 ────────────────────────────────────────────────────────────────

def visualize_prediction(
    image: Image.Image,
    result: PredictionResult,
    save_path: Optional[str] = None,
    title_prefix: str = "",
    normal_baseline: Optional[dict[str, float]] = None,
) -> plt.Figure:
    """2×2 panel figure: original | heatmap | top-10 concepts | decision.

    Args:
        normal_baseline: if provided (from compute_normal_baseline), Panel 3
                         shows deviation from normal instead of raw activation,
                         and Panel 4 gains an EXPLANATION section.
    """

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.subplots_adjust(hspace=0.35, wspace=0.3)

    img_arr = np.array(image.resize((_IMG_SIZE, _IMG_SIZE)))

    # ── Panel 1: original image ───────────────────────────────────────────────
    ax1 = axes[0, 0]
    ax1.imshow(img_arr)
    ax1.set_title(f'"{result.true_label}"', fontsize=12, fontweight="bold")
    ax1.axis("off")
    border_col = "#2ecc71" if result.correct else "#e74c3c"
    rect = mpatches.FancyBboxPatch(
        (0, 0), 1, 1, boxstyle="square,pad=0",
        linewidth=6, edgecolor=border_col, facecolor="none",
        transform=ax1.transAxes, clip_on=False,
    )
    ax1.add_patch(rect)
    ax1.text(0.02, 0.02, "✓ CORRECT" if result.correct else "✗ WRONG",
             transform=ax1.transAxes, fontsize=9, fontweight="bold",
             color=border_col, va="bottom", bbox=dict(fc="white", alpha=0.7, pad=2))

    # ── Panel 2: anomaly heatmap overlay ─────────────────────────────────────
    ax2 = axes[0, 1]
    heatmap_224 = scipy.ndimage.zoom(result.anomaly_map, _HEATMAP_ZOOM, order=3)
    heatmap_norm = (heatmap_224 - heatmap_224.min()) / (heatmap_224.max() - heatmap_224.min() + 1e-8)

    ax2.imshow(img_arr)
    im2 = ax2.imshow(heatmap_norm, cmap="jet", alpha=0.5, vmin=0, vmax=1)
    cb  = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cb.set_label("normalised patch distance", fontsize=8)
    ax2.set_title(
        f"Anomaly map   s_novel={result.s_novel:.3f}   τ={result.tau:.3f}",
        fontsize=10,
    )
    ax2.axis("off")
    # Mark whether s_novel exceeds τ
    exceed_label = "s_novel > τ  ⚠" if result.s_novel >= result.tau else "s_novel < τ  ✓"
    exceed_col   = "#e74c3c" if result.s_novel >= result.tau else "#2ecc71"
    ax2.text(0.98, 0.02, exceed_label, transform=ax2.transAxes, fontsize=9,
             fontweight="bold", color=exceed_col, ha="right", va="bottom",
             bbox=dict(fc="white", alpha=0.7, pad=2))

    # ── Panel 3: concepts ──────────────────────────────────────────────────────
    ax3 = axes[1, 0]

    if normal_baseline is not None:
        # ── deviation-from-normal mode ────────────────────────────────────────
        deviations = np.array([
            result.c[i] - normal_baseline.get(result.concept_names[i], 0.5)
            for i in range(len(result.c))
        ])
        top10_idx = np.argsort(np.abs(deviations))[::-1][:10]
        top10_dev = deviations[top10_idx]
        top10_nms = [
            result.concept_names[i][:22] + "…" if len(result.concept_names[i]) > 23
            else result.concept_names[i]
            for i in top10_idx
        ]
        bar_colours = [
            "#e74c3c" if d > 0.2 else "#3498db" if d < -0.2 else "#bdc3c7"
            for d in top10_dev
        ]
        ax3.barh(range(10), top10_dev[::-1], color=bar_colours[::-1],
                 edgecolor="white", linewidth=0.5)
        ax3.set_yticks(range(10))
        ax3.set_yticklabels(top10_nms[::-1], fontsize=8)
        ax3.set_xlim(-1.0, 1.0)
        ax3.axvline(0, color="#333", linestyle="--", linewidth=1.0)
        ax3.set_xlabel("Deviation from normal baseline", fontsize=9)
        ax3.set_title("Top-10 concepts by deviation from normal", fontsize=10)
        # Deviation labels on bars
        for j, (dev, nm) in enumerate(zip(top10_dev[::-1], top10_nms[::-1])):
            x_off = 0.03 if dev >= 0 else -0.03
            ha    = "left"  if dev >= 0 else "right"
            ax3.text(dev + x_off, j, f"{dev:+.2f}", va="center",
                     ha=ha, fontsize=7, color="#222")
        ax3.legend(handles=[
            mpatches.Patch(facecolor="#e74c3c", label="activated  (>+0.2)"),
            mpatches.Patch(facecolor="#3498db", label="suppressed (<−0.2)"),
            mpatches.Patch(facecolor="#bdc3c7", label="neutral"),
        ], fontsize=7, loc="lower right")
        ax3.spines["left"].set_visible(False)
        ax3.tick_params(axis="y", length=0)
    else:
        # ── raw activation mode (no baseline available) ───────────────────────
        top10_idx = np.argsort(result.c)[::-1][:10]
        top10_val = result.c[top10_idx]
        top10_nms = [
            result.concept_names[i][:22] + "…" if len(result.concept_names[i]) > 23
            else result.concept_names[i]
            for i in top10_idx
        ]
        colours = plt.cm.RdYlBu_r(top10_val)
        ax3.barh(range(10), top10_val[::-1], color=colours[::-1],
                 edgecolor="white", linewidth=0.5)
        ax3.set_yticks(range(10))
        ax3.set_yticklabels(top10_nms[::-1], fontsize=8)
        ax3.set_xlim(0, 1)
        ax3.axvline(0.5, color="#888", linestyle="--", linewidth=0.8, label="θ=0.5")
        ax3.set_xlabel("Activation", fontsize=9)
        ax3.set_title("Top-10 concept activations", fontsize=10)
        ax3.legend(fontsize=8, loc="lower right")
        ax3.spines["left"].set_visible(False)
        ax3.tick_params(axis="y", length=0)

    # ── Panel 4: decision summary ──────────────────────────────────────────────
    ax4 = axes[1, 1]
    ax4.axis("off")

    lc    = _LEVEL_COLOUR[result.level]
    ll    = _LEVEL_LABEL[result.level]
    top_c = result.concept_names[int(result.c.argmax())]
    top_v = float(result.c.max())

    snov_bar  = _bar(result.s_novel, max_val=max(result.s_novel * 1.5, result.tau * 2, 0.01))
    snov_tag  = "HIGH" if result.s_novel >= result.tau else "low"
    ypred_bar = _bar(result.y_pred)

    lines = [
        ("DECISION: " + ll,   14, "bold",   lc),
        ("─" * 36,            10, "normal", "#555"),
        ("Detection branch:",  10, "bold",   "#222"),
        (f"  s_novel : {result.s_novel:.3f}  [{snov_bar}]  {snov_tag}", 9, "normal", "#333"),
        (f"  τ       : {result.tau:.3f}", 9, "normal", "#333"),
        ("",                   6,  "normal", "#fff"),
        ("Concept branch:",    10, "bold",   "#222"),
        (f"  y_pred  : {result.y_pred:.3f}  [{ypred_bar}]", 9, "normal", "#333"),
        (f"  top concept: {top_c[:26]}",  9, "normal", "#333"),
        (f"  activation  : {top_v:.3f}", 9, "normal", "#333"),
        ("",                   6,  "normal", "#fff"),
    ]

    # ── EXPLANATION (only when baseline available) ────────────────────────────
    if normal_baseline is not None:
        deviations = np.array([
            result.c[i] - normal_baseline.get(result.concept_names[i], 0.5)
            for i in range(len(result.c))
        ])
        top3_idx = np.argsort(np.abs(deviations))[::-1][:3]
        lines.append(("EXPLANATION:",  10, "bold",   "#222"))
        for idx in top3_idx:
            name   = result.concept_names[idx][:26]
            dev    = deviations[idx]
            arrow  = "↑" if dev > 0 else "↓"
            status = "activated" if dev > 0 else "suppressed"
            col    = "#c0392b" if dev > 0 else "#2980b9"
            lines.append((f"  {arrow} {name}: {dev:+.2f}", 9, "normal", col))
        lines.append(("",  6, "normal", "#fff"))

    lines += [
        (f"Level  {result.level} / 3", 11, "bold", lc),
        ("─" * 36,            10, "normal", "#555"),
        ("✓ CORRECT" if result.correct else "✗ WRONG", 12, "bold",
         "#2ecc71" if result.correct else "#e74c3c"),
    ]

    y = 0.97
    for text, size, weight, colour in lines:
        ax4.text(0.05, y, text, transform=ax4.transAxes,
                 fontsize=size, fontweight=weight, color=colour,
                 va="top", fontfamily="monospace" if "─" in text or "[" in text else "sans-serif")
        y -= size * 0.012 + 0.01

    # Light background
    ax4.set_facecolor("#f8f9fa")
    ax4.patch.set_visible(True)

    # ── figure title ──────────────────────────────────────────────────────────
    fig.suptitle(
        f"{title_prefix}  {result.true_label} → {ll}",
        fontsize=13, fontweight="bold", y=1.01,
    )

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
        plt.close(fig)

    return fig


# ── FUNCTION 2 ────────────────────────────────────────────────────────────────

def find_failure_cases(
    extractor, memory, concept_heads, anomaly_head,
    tau: float, theta_concept: float,
    mvtec_root: str, category: str, defect_names: list[str],
    concept_names: list[str],
    max_per_type: int = 5,
    tier_map_path: Optional[str] = None,
) -> list[FailureCase]:
    """Scan MVTec test images and return failure cases of four types."""

    mvtec_cat = Path(mvtec_root) / category

    # Load tier map for explainability failure detection
    tier3_for_defect: dict[str, set[str]] = {}
    if tier_map_path and Path(tier_map_path).exists():
        with open(tier_map_path) as f:
            tm = json.load(f)
        for d in defect_names:
            tier3_for_defect[d] = set(tm.get(f"tier3_{d}", []))

    buckets: dict[str, list[FailureCase]] = {
        "false_negative": [], "false_positive": [],
        "explainability": [], "level3": [],
    }

    def _process(img_path: Path, true_label: str) -> None:
        r = _run_inference(extractor, memory, concept_heads, anomaly_head,
                           img_path, tau, theta_concept, true_label)
        amap = r.anomaly_map
        top_c = concept_heads.concept_names[int(r.c.argmax())]

        if true_label in ("good", "normal"):
            if r.s_novel >= tau:
                buckets["false_positive"].append(FailureCase(
                    image_path=str(img_path), true_label=true_label,
                    failure_type="false_positive", s_novel=r.s_novel,
                    c=r.c, anomaly_map=amap, y_pred=r.y_pred,
                    description=f"False alarm: s_novel={r.s_novel:.3f} > τ={tau:.3f}",
                ))
        else:
            if r.s_novel < tau:
                buckets["false_negative"].append(FailureCase(
                    image_path=str(img_path), true_label=true_label,
                    failure_type="false_negative", s_novel=r.s_novel,
                    c=r.c, anomaly_map=amap, y_pred=r.y_pred,
                    description=f"Missed: s_novel={r.s_novel:.3f} < τ={tau:.3f}",
                ))
            elif r.c.max() > theta_concept:
                # Detected + concept fires — check if right concept
                expected = tier3_for_defect.get(true_label, set())
                if expected and top_c not in expected:
                    buckets["explainability"].append(FailureCase(
                        image_path=str(img_path), true_label=true_label,
                        failure_type="explainability", s_novel=r.s_novel,
                        c=r.c, anomaly_map=amap, y_pred=r.y_pred,
                        description=f"Wrong concept: '{top_c}' on {true_label}",
                    ))
            else:
                buckets["level3"].append(FailureCase(
                    image_path=str(img_path), true_label=true_label,
                    failure_type="level3", s_novel=r.s_novel,
                    c=r.c, anomaly_map=amap, y_pred=r.y_pred,
                    description=f"Level 3: detected but unexplained (max_c={r.c.max():.3f})",
                ))

    # Scan normal test images
    for p in sorted((mvtec_cat / "test" / "good").glob("*.png")):
        _process(p, "good")

    # Scan defect test images
    for d in defect_names:
        d_dir = mvtec_cat / "test" / d
        if d_dir.exists():
            for p in sorted(d_dir.glob("*.png")):
                _process(p, d)

    # Sort and truncate
    buckets["false_negative"].sort(key=lambda x: x.s_novel)          # closest to τ
    buckets["false_positive"].sort(key=lambda x: x.s_novel, reverse=True)  # highest alarm
    buckets["explainability"].sort(key=lambda x: x.s_novel, reverse=True)
    buckets["level3"].sort(key=lambda x: x.s_novel, reverse=True)

    result: list[FailureCase] = []
    for key in buckets:
        result.extend(buckets[key][:max_per_type])

    return result


# ── FUNCTION 3 ────────────────────────────────────────────────────────────────

def visualize_failure_gallery(
    failure_cases: list[FailureCase],
    extractor, memory, concept_heads, anomaly_head,
    tau: float, save_dir: str, concept_names: list[str],
    theta_concept: float = 0.5,
    normal_baseline: Optional[dict[str, float]] = None,
) -> None:
    """Save individual failure figures and one summary thumbnail grid."""

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    thumbs: list[np.ndarray] = []
    labels: list[str] = []

    for fc in failure_cases:
        img  = Image.open(fc.image_path).convert("RGB")
        r    = _run_inference(extractor, memory, concept_heads, anomaly_head,
                              img, tau, theta_concept, fc.true_label)
        stem = Path(fc.image_path).stem
        fn   = save_dir / f"{fc.failure_type}_{fc.true_label}_{stem}.png"

        visualize_prediction(
            img, r, save_path=str(fn),
            title_prefix=f"[{fc.failure_type.upper()}]",
            normal_baseline=normal_baseline,
        )
        thumbs.append(np.array(img.resize((112, 112))))
        labels.append(f"{fc.failure_type}\n{fc.true_label}\ns={fc.s_novel:.2f}")

    if not thumbs:
        return

    # ── summary grid ─────────────────────────────────────────────────────────
    n      = len(thumbs)
    ncols  = min(n, 6)
    nrows  = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.5, nrows * 3.2))
    axes_flat = np.array(axes).flatten() if n > 1 else [axes]

    for i, (thumb, lbl) in enumerate(zip(thumbs, labels)):
        ax = axes_flat[i]
        ax.imshow(thumb)
        ax.set_title(lbl, fontsize=7, pad=3)
        ax.axis("off")

    for ax in axes_flat[len(thumbs):]:
        ax.axis("off")

    fig.suptitle("Failure Case Summary", fontsize=12, fontweight="bold")
    fig.savefig(save_dir / "failure_gallery_summary.png")
    plt.close(fig)
    print(f"  Failure gallery saved to {save_dir}/")


# ── FUNCTION 4 ────────────────────────────────────────────────────────────────

def visualize_concept_evolution(
    log_concil: ContinualLog,
    log_naive:  Optional[ContinualLog] = None,
    defect_names: Optional[list[str]] = None,
    save_path: Optional[str] = None,
    supervised_baseline: Optional[float] = None,
) -> plt.Figure:
    """Line plot: C-AUC over tasks for CONCIL vs naive baseline.

    This is the key thesis figure showing CONCIL's zero-forgetting property
    compared to catastrophic forgetting in the naive approach.
    """
    has_naive = log_naive is not None and len(log_naive.results) > 0

    ncols = 2 if has_naive else 1
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5), sharey=True)
    if ncols == 1:
        axes = [axes]

    colours = plt.cm.Set1(np.linspace(0, 0.8, 6))

    def _plot_log(ax, log: ContinualLog, title: str) -> None:
        mat   = log.c_auc_matrix()
        tasks = list(mat.columns)
        defs  = list(mat.index)

        if defect_names:
            defs = [d for d in defect_names if d in mat.index]

        for i, d in enumerate(defs):
            row    = mat.loc[d]
            x_vals = [t for t in tasks if not np.isnan(row[t])]
            y_vals = [row[t] for t in x_vals]
            if not x_vals:
                continue
            ax.plot(x_vals, y_vals, marker="o", linewidth=2.2,
                    color=colours[i], label=d, markersize=6)

        if supervised_baseline is not None:
            ax.axhline(supervised_baseline, color="#555", linestyle="--",
                       linewidth=1.2, label=f"supervised ({supervised_baseline:.2f})")

        bwt = log.mean_concept_bwt()
        bwt_str = f"{bwt:+.3f}" if not np.isnan(bwt) else "N/A"
        ax.set_title(f"{title}\nMean Concept BWT = {bwt_str}", fontsize=11, fontweight="bold")
        ax.set_xlabel("After task", fontsize=10)
        ax.set_ylabel("C-AUC (mean)", fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.set_xticks(tasks)
        ax.legend(fontsize=8, loc="lower left")
        ax.grid(axis="y", alpha=0.3, linewidth=0.6)

    _plot_log(axes[0], log_concil, "CONCIL (ours)")
    if has_naive:
        _plot_log(axes[1], log_naive, "Naive sequential")

    fig.suptitle(
        "Concept-level Forgetting: CONCIL vs Naive Sequential",
        fontsize=13, fontweight="bold", y=1.02,
    )
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
        plt.close(fig)

    return fig


# ── FUNCTION 4b: branch disagreement ────────────────────────────────────────

def visualize_branch_disagreement(
    image: Image.Image,
    result: PredictionResult,
    save_path: Optional[str] = None,
    normal_baseline: Optional[dict[str, float]] = None,
) -> plt.Figure:
    """3-panel figure for images where s_novel and y_pred disagree.

    Highlights the two-branch architecture: one branch may flag while the
    other does not, requiring a decision policy (conservative = OR, strict = AND).
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.subplots_adjust(wspace=0.38)

    img_arr = np.array(image.resize((_IMG_SIZE, _IMG_SIZE)))

    # ── Panel 1: image with yellow border ─────────────────────────────────────
    axes[0].imshow(img_arr)
    axes[0].axis("off")
    axes[0].set_title("UNCERTAIN\nbranches disagree", fontsize=12,
                      fontweight="bold", color="#d4a017")
    rect = mpatches.FancyBboxPatch(
        (0, 0), 1, 1, boxstyle="square,pad=0",
        linewidth=7, edgecolor="#f1c40f", facecolor="none",
        transform=axes[0].transAxes, clip_on=False,
    )
    axes[0].add_patch(rect)

    # ── Panel 2: two branch gauges ────────────────────────────────────────────
    ax2 = axes[1]
    ax2.axis("off")

    # Draw two vertical gauge bars manually using patches
    max_scale = max(result.s_novel * 1.15, 1.1)

    def _draw_gauge(ax, x_left, width, value, tau, label, max_val):
        """Draw a single vertical gauge bar with label."""
        bar_h = value / max_val
        tau_y = tau   / max_val
        col   = "#e74c3c" if value >= tau else "#2ecc71"

        # Background rectangle
        ax.add_patch(mpatches.FancyBboxPatch(
            (x_left, 0), width, 1.0, boxstyle="square,pad=0",
            facecolor="#f0f0f0", edgecolor="#ccc", linewidth=0.8,
            transform=ax.transAxes,
        ))
        # Value fill
        ax.add_patch(mpatches.FancyBboxPatch(
            (x_left, 0), width, bar_h, boxstyle="square,pad=0",
            facecolor=col, edgecolor="none",
            transform=ax.transAxes,
        ))
        # τ dashed line (plot accepts transform; axhline does not)
        ax.plot([x_left, x_left + width], [tau_y, tau_y],
                color="#555", linestyle="--", linewidth=1.5,
                transform=ax.transAxes)
        # Labels
        ax.text(x_left + width / 2, -0.05, label, ha="center", va="top",
                fontsize=9, fontweight="bold", transform=ax.transAxes)
        ax.text(x_left + width / 2, bar_h + 0.03, f"{value:.3f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold",
                color=col, transform=ax.transAxes)
        ax.text(x_left + width + 0.02, tau_y, f"τ={tau:.3f}", ha="left",
                va="center", fontsize=8, color="#555", transform=ax.transAxes)

    _draw_gauge(ax2, 0.05, 0.38, result.s_novel, result.tau, "Detection\n(s_novel)", max_scale)
    _draw_gauge(ax2, 0.57, 0.38, result.y_pred,  result.tau, "Concept\n(y_pred)",   1.0)

    # Disagreement label
    if result.s_novel >= result.tau and result.y_pred < result.tau:
        msg = "⚠ PatchCore alarm\n   Concept says normal\n→ recommend human review"
    elif result.s_novel < result.tau and result.y_pred >= result.tau:
        msg = "⚠ Concept alarm\n   PatchCore says normal\n→ recommend human review"
    else:
        msg = "Branches agree"

    ax2.text(0.5, 1.08, msg, ha="center", va="bottom", fontsize=9,
             color="#c0392b", transform=ax2.transAxes,
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#fef9e7", edgecolor="#f39c12"))
    ax2.set_title("Branch scores vs threshold", fontsize=10, fontweight="bold")
    ax2.set_ylim(-0.15, 1.25)
    ax2.set_xlim(0, 1)

    # ── Panel 3: concept deviation (same as Panel 3 in visualize_prediction) ──
    ax3 = axes[2]
    if normal_baseline is not None:
        deviations = np.array([
            result.c[i] - normal_baseline.get(result.concept_names[i], 0.5)
            for i in range(len(result.c))
        ])
        top10_idx = np.argsort(np.abs(deviations))[::-1][:10]
        top10_dev = deviations[top10_idx]
        top10_nms = [result.concept_names[i][:22] for i in top10_idx]
        bar_cols  = ["#e74c3c" if d > 0.2 else "#3498db" if d < -0.2 else "#bdc3c7"
                     for d in top10_dev]
        ax3.barh(range(10), top10_dev[::-1], color=bar_cols[::-1], edgecolor="white")
        ax3.set_yticks(range(10))
        ax3.set_yticklabels(top10_nms[::-1], fontsize=8)
        ax3.set_xlim(-1, 1)
        ax3.axvline(0, color="#333", linestyle="--", linewidth=1.0)
        ax3.set_xlabel("Deviation from normal baseline", fontsize=9)
    else:
        top10_idx = np.argsort(result.c)[::-1][:10]
        ax3.barh(range(10), result.c[top10_idx][::-1], color="#bdc3c7")
        ax3.set_yticks(range(10))
        ax3.set_yticklabels([result.concept_names[i][:22] for i in top10_idx][::-1], fontsize=8)
    ax3.set_title("Concept activations\n(concept branch correctly saw no defect pattern)", fontsize=9)
    ax3.spines["left"].set_visible(False)
    ax3.tick_params(axis="y", length=0)

    fig.suptitle(
        f"Two-Branch Disagreement — true label: '{result.true_label}'",
        fontsize=12, fontweight="bold", y=1.02,
    )
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    return fig


# ── FUNCTION 4c: concept weight bar chart ────────────────────────────────────

def visualize_concept_weights(
    anomaly_head,
    concept_names: list[str],
    save_path: Optional[str] = None,
    top_n: int = 10,
) -> plt.Figure:
    """Standalone thesis figure: horizontal bar chart of linear head weights."""
    weights = anomaly_head.concept_weights                   # (K,)
    paired  = sorted(zip(concept_names, weights), key=lambda x: abs(x[1]), reverse=True)[:top_n]
    names   = [p[0] for p in paired]
    vals    = [p[1] for p in paired]

    fig, ax = plt.subplots(figsize=(9, 5))
    y_pos   = np.arange(top_n)
    cols    = ["#e74c3c" if v > 0 else "#3498db" for v in vals]

    # Plot in descending |weight| order, top at top
    ax.barh(y_pos[::-1], vals, color=cols, edgecolor="white", linewidth=0.5, height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.axvline(0, color="#333", linewidth=1.0)
    ax.set_xlabel("Weight (contribution to anomaly score)", fontsize=10)
    ax.set_title(
        "Linear head concept weights — learned by CONCIL\n"
        "Top 10 by absolute magnitude",
        fontsize=11, fontweight="bold",
    )
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", alpha=0.3, linewidth=0.6)

    # Annotations
    ax.text(max(vals) * 1.02, top_n * 0.95,
            "+ weight: activated by defect →\n  concept presence = anomaly signal",
            fontsize=8, color="#e74c3c", va="top",
            bbox=dict(fc="#fdf2f2", ec="#e74c3c", pad=4, boxstyle="round,pad=0.4"))
    ax.text(min(vals) * 1.02, top_n * 0.95,
            "← − weight: suppressed by defect\n   concept absence = anomaly signal",
            fontsize=8, color="#3498db", va="top", ha="right",
            bbox=dict(fc="#f2f6fd", ec="#3498db", pad=4, boxstyle="round,pad=0.4"))

    # Weight value labels on bars
    for i, (v, name) in enumerate(zip(vals, names)):
        x_off = 0.005 if v >= 0 else -0.005
        ha    = "left" if v >= 0 else "right"
        ax.text(v + x_off, top_n - 1 - i, f"{v:+.3f}", va="center",
                ha=ha, fontsize=8, color="#222")

    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    return fig


# ── intervention dataclass ────────────────────────────────────────────────────

@dataclass
class InterventionResult:
    c_orig:              np.ndarray             # (K,) original concept activations
    c_modified:          np.ndarray             # (K,) after intervention
    y_pred_orig:         float
    y_pred_modified:     float
    s_novel:             float
    delta_y:             float                  # y_pred_modified - y_pred_orig
    intervened_concepts: dict[str, tuple[float, float]]  # {name: (orig, new)}
    tau:                 float = 0.0            # stored for plotting


# ── FUNCTION 5: run_intervention ──────────────────────────────────────────────

@torch.no_grad()
def run_intervention(
    image,
    concept_heads,
    anomaly_head,
    memory,
    extractor,
    tau: float,
    concept_names: list[str],
    interventions: dict[str, float],
) -> InterventionResult:
    """Apply concept-level interventions and measure change in anomaly score.

    The detection branch (s_novel) is not affected — only the concept branch
    (y_pred from the linear anomaly head) changes.

    Args:
        interventions: {concept_name: new_activation_value}
    """
    if isinstance(image, (str, Path)):
        image = Image.open(image).convert("RGB")

    patch_tokens, pooled_z = extractor.extract_both([image])

    s_novel_t, _ = memory.score(patch_tokens)
    c_orig_t      = concept_heads(pooled_z)           # (1, K)
    y_orig_t      = anomaly_head(c_orig_t)            # (1,)

    s_novel     = float(s_novel_t[0].cpu())
    c_orig      = c_orig_t[0].cpu().numpy()
    y_pred_orig = float(y_orig_t[0].cpu())

    # Apply interventions
    c_mod = c_orig_t.clone()
    intervened: dict[str, tuple[float, float]] = {}
    for name, new_val in interventions.items():
        if name in concept_names:
            idx = concept_names.index(name)
            intervened[name] = (float(c_orig[idx]), float(new_val))
            c_mod[0, idx] = float(new_val)

    y_mod_t      = anomaly_head(c_mod)
    y_pred_mod   = float(y_mod_t[0].cpu())

    return InterventionResult(
        c_orig              = c_orig,
        c_modified          = c_mod[0].cpu().numpy(),
        y_pred_orig         = y_pred_orig,
        y_pred_modified     = y_pred_mod,
        s_novel             = s_novel,
        delta_y             = y_pred_mod - y_pred_orig,
        intervened_concepts = intervened,
        tau                 = tau,
    )


# ── FUNCTION 6: visualize_intervention ───────────────────────────────────────

def visualize_intervention(
    image: Image.Image,
    result: InterventionResult,
    save_path: Optional[str],
    true_label: str,
    concept_names: list[str],
    anomaly_head,
    normal_baseline: Optional[dict[str, float]] = None,
    top_n_by_weight: int = 5,
) -> plt.Figure:
    """3-panel figure: image | concept bars before/after | score comparison."""

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.subplots_adjust(wspace=0.38)

    img_arr = np.array(image.resize((_IMG_SIZE, _IMG_SIZE)))

    # ── Panel 1: image ────────────────────────────────────────────────────────
    axes[0].imshow(img_arr)
    axes[0].set_title(f'"{true_label}"', fontsize=12, fontweight="bold")
    axes[0].axis("off")

    # ── Panel 2: grouped bars (intervened + top-N by weight) ─────────────────
    ax2 = axes[1]
    weights_np = anomaly_head.concept_weights            # (K,)

    # Build list of concepts to show
    top_w_idx  = np.argsort(np.abs(weights_np))[::-1][:top_n_by_weight]
    top_w_names = [concept_names[i] for i in top_w_idx]
    show_names  = list(dict.fromkeys(list(result.intervened_concepts.keys()) + top_w_names))

    n_show = len(show_names)
    y_pos  = np.arange(n_show)
    h      = 0.35

    orig_vals = [result.c_orig[concept_names.index(c)] for c in show_names]
    mod_vals  = [result.c_modified[concept_names.index(c)] for c in show_names]

    ax2.barh(y_pos - h / 2, orig_vals, h, color="#bdc3c7", label="original", edgecolor="white")
    for i, (v_mod, v_orig) in enumerate(zip(mod_vals, orig_vals)):
        col = "#e74c3c" if v_mod > v_orig + 0.01 else \
              "#3498db" if v_mod < v_orig - 0.01 else "#bdc3c7"
        ax2.barh(i + h / 2, v_mod, h, color=col, edgecolor="white")

    ax2.set_yticks(y_pos)
    ylabels = []
    for c in show_names:
        short = c[:22] + "…" if len(c) > 23 else c
        ylabels.append(("→ " + short) if c in result.intervened_concepts else short)
    ax2.set_yticklabels(ylabels, fontsize=8)
    # Bold for intervened concepts
    for tick, c in zip(ax2.get_yticklabels(), show_names):
        if c in result.intervened_concepts:
            tick.set_fontweight("bold")
            tick.set_color("#c0392b")

    ax2.set_xlim(0, 1)
    ax2.set_xlabel("Activation", fontsize=9)
    ax2.set_title("Concept activations\nbefore (grey) vs after (coloured)", fontsize=10)
    ax2.legend(
        handles=[
            mpatches.Patch(facecolor="#bdc3c7", label="original"),
            mpatches.Patch(facecolor="#e74c3c", label="increased"),
            mpatches.Patch(facecolor="#3498db", label="decreased"),
        ],
        fontsize=8, loc="lower right",
    )
    ax2.axvline(0.5, color="#aaa", linestyle=":", linewidth=0.8)
    ax2.spines["left"].set_visible(False)

    # ── Panel 3: anomaly score before/after ───────────────────────────────────
    ax3 = axes[2]
    bars_x = [0, 1]
    y_vals = [result.y_pred_orig, result.y_pred_modified]
    bar_cols = []
    for v in y_vals:
        bar_cols.append("#e74c3c" if v >= result.tau else "#2ecc71")

    bars = ax3.bar(bars_x, y_vals, width=0.55, color=bar_cols, edgecolor="white", linewidth=1.2)
    ax3.axhline(result.tau, color="#555", linestyle="--", linewidth=1.2,
                label=f"τ = {result.tau:.3f}")
    ax3.set_xticks(bars_x)
    ax3.set_xticklabels(["Original", "Intervened"], fontsize=10)
    ax3.set_ylim(0, 1.05)
    ax3.set_ylabel("y_pred (anomaly score)", fontsize=9)

    # Value labels on bars
    for bar, v in zip(bars, y_vals):
        ax3.text(bar.get_x() + bar.get_width() / 2, v + 0.02,
                 f"{v:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Arrow between bars
    ax3.annotate(
        "",
        xy=(1, result.y_pred_modified),
        xytext=(0, result.y_pred_orig),
        arrowprops=dict(arrowstyle="->", color="#333", lw=2.0),
    )

    direction = "▲" if result.delta_y > 0 else "▼"
    ax3.set_title(
        f"Anomaly score change\n{direction} Δy = {result.delta_y:+.3f}",
        fontsize=11, fontweight="bold",
    )
    ax3.text(0.5, -0.12, "s_novel unchanged (detection branch)",
             transform=ax3.transAxes, ha="center", fontsize=8, color="#888")
    ax3.legend(fontsize=9, loc="upper right")

    fig.suptitle(
        f"Concept Intervention — {true_label}  "
        f"({'✓ drops below τ' if result.y_pred_modified < result.tau and result.y_pred_orig >= result.tau else '✗ stays above τ' if result.y_pred_modified >= result.tau else 'below τ'})",
        fontsize=12, fontweight="bold", y=1.01,
    )

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
        plt.close(fig)

    return fig


# ── FUNCTION 7: visualize_sensitivity_sweep ───────────────────────────────────

def visualize_sensitivity_sweep(
    image,
    concept_heads,
    anomaly_head,
    extractor,
    tau: float,
    concept_names: list[str],
    top_n: int = 5,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Line chart: y_pred vs swept concept value for the top-N weighted concepts."""
    if isinstance(image, (str, Path)):
        image = Image.open(image).convert("RGB")

    with torch.no_grad():
        _, pooled_z = extractor.extract_both([image])
        c_base = concept_heads(pooled_z)[0].cpu().numpy()   # (K,)

    weights_np  = anomaly_head.concept_weights
    top_n_idx   = np.argsort(np.abs(weights_np))[::-1][:top_n]
    sweep_vals  = np.linspace(0.0, 1.0, 11)

    fig, ax = plt.subplots(figsize=(10, 5))
    colours = plt.cm.Set1(np.linspace(0, 0.85, top_n))

    slopes: dict[str, float] = {}
    for colour, idx in zip(colours, top_n_idx):
        name   = concept_names[idx]
        y_preds = []
        for v in sweep_vals:
            c_test    = c_base.copy()
            c_test[idx] = v
            c_t       = torch.tensor(c_test, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                yp = float(anomaly_head(c_t)[0].cpu())
            y_preds.append(yp)

        ax.plot(sweep_vals, y_preds, marker="o", markersize=5,
                linewidth=2.0, color=colour,
                label=f"{name[:28]}  (w={weights_np[idx]:+.3f})")
        # Slope estimate (linear regression over sweep)
        slopes[name] = float(np.polyfit(sweep_vals, y_preds, 1)[0])

    ax.axhline(tau, color="#555", linestyle="--", linewidth=1.2, label=f"τ = {tau:.3f}")
    ax.set_xlabel("Concept activation (0 → 1)", fontsize=10)
    ax.set_ylabel("y_pred (anomaly score)", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, 1)
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1))
    ax.grid(axis="y", alpha=0.3, linewidth=0.6)
    ax.set_title(
        "Anomaly score sensitivity to individual concept interventions\n"
        "(each line: one concept swept 0→1, all others held at model output)",
        fontsize=11,
    )
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)

    steepest = max(slopes, key=lambda k: abs(slopes[k]))
    return fig, slopes, steepest


# ── __main__ ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, dataclasses

    from features.dinov2_extractor import DINOv2Extractor
    from features.patchcore_memory import PatchCoreMemory
    from models.concept_heads       import ConceptHeads
    from models.linear_head         import LinearAnomalyHead

    CATEGORY    = "hazelnut"
    CKPT_DIR    = Path("checkpoints/hazelnut")
    MVTEC_ROOT  = "/home/sobhan_hosseini/datasets/mvtec"
    ANN_DIR     = Path("annotations/hazelnut")
    FIGURE_DIR  = Path("thesis_figures")
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    THETA       = 0.5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── 1. load Task 4 checkpoint ─────────────────────────────────────────────
    print("\nLoading Task-4 checkpoint ...")
    concept_heads = ConceptHeads.load(CKPT_DIR / "task_4" / "concept_heads.pt")
    anomaly_head  = LinearAnomalyHead.load(CKPT_DIR / "task_4" / "anomaly_head.pt")
    memory        = PatchCoreMemory.load(CKPT_DIR / "task_1" / "memory.pt", device=device)
    with open(CKPT_DIR / "task_4" / "tau.json") as f:
        tau = json.load(f)["tau"]
    extractor = DINOv2Extractor(device=device)

    print(f"  K={concept_heads.n_concepts} concepts  τ={tau:.4f}  "
          f"memory={memory.coreset_size:,} patches")

    mvtec_cat    = Path(MVTEC_ROOT) / CATEGORY
    defect_names = ["crack", "hole", "cut", "print"]

    # ── 2. compute / load normal baseline ─────────────────────────────────────
    baseline_path = CKPT_DIR / "concept_normal_baseline.json"
    if baseline_path.exists():
        print(f"\nLoading cached baseline from {baseline_path}")
        with open(baseline_path) as f:
            bl_data = json.load(f)
        normal_baseline = bl_data["means"]
        stds_dict       = bl_data["stds"]
    else:
        print("\nComputing concept normal baseline (train/good images) ...")
        train_good_paths = sorted((mvtec_cat / "train" / "good").glob("*.png"))
        train_good_pil   = [Image.open(p).convert("RGB") for p in train_good_paths]
        normal_baseline  = compute_normal_baseline(
            concept_heads, extractor, train_good_pil,
            concept_names=concept_heads.concept_names,
            save_path=str(baseline_path),
        )
        with open(baseline_path) as f:
            stds_dict = json.load(f)["stds"]
        print(f"  Saved → {baseline_path}  ({len(normal_baseline)} concepts)")

    # ── Sanity check: stable vs discriminative concepts ───────────────────────
    print("\n  5 most STABLE normal concepts (smallest σ on normal images):")
    for name, std in sorted(stds_dict.items(), key=lambda x: x[1])[:5]:
        print(f"    {name:<42} σ={std:.4f}  mean={normal_baseline[name]:.3f}")

    print("\n  Computing defect discriminability (mean |dev| across defect types) ...")
    disc: dict[str, float] = {n: 0.0 for n in concept_heads.concept_names}
    for d in defect_names:
        d_paths = sorted((mvtec_cat / "test" / d).glob("*.png"))
        d_pil   = [Image.open(p).convert("RGB") for p in d_paths]
        acts: list[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(d_pil), 16):
                _, z = extractor.extract_both(d_pil[i : i + 16])
                acts.append(concept_heads(z).cpu().numpy())
        dmeans = np.vstack(acts).mean(axis=0)
        for j, n in enumerate(concept_heads.concept_names):
            disc[n] += abs(dmeans[j] - normal_baseline.get(n, 0.5))
    for n in disc:
        disc[n] /= len(defect_names)

    print("\n  5 most DISCRIMINATIVE concepts (largest mean |dev| across defects):")
    for name, score in sorted(disc.items(), key=lambda x: -x[1])[:5]:
        print(f"    {name:<42} avg|dev|={score:.4f}  baseline={normal_baseline[name]:.3f}")

    # ── 3. visualize one crack + one normal image ─────────────────────────────
    print("\nGenerating prediction figures ...")
    crack_imgs  = sorted((mvtec_cat / "test" / "crack").glob("*.png"))
    test_normals = sorted((mvtec_cat / "test" / "good").glob("*.png"))

    for img_path, label in [(crack_imgs[0], "crack"), (test_normals[0], "good")]:
        img = Image.open(img_path).convert("RGB")
        r   = _run_inference(extractor, memory, concept_heads, anomaly_head,
                             img, tau, THETA, true_label=label)
        save = FIGURE_DIR / f"prediction_{label}_{img_path.stem}.png"
        visualize_prediction(img, r, save_path=str(save),
                             title_prefix=f"Hazelnut / {label}",
                             normal_baseline=normal_baseline)
        print(f"  Saved: {save}")
        print(f"    → level={r.level} ({_LEVEL_LABEL[r.level]})  "
              f"s_novel={r.s_novel:.3f}  y_pred={r.y_pred:.3f}  correct={r.correct}")

    # ── 4. find failure cases ─────────────────────────────────────────────────
    print("\nScanning test images for failures ...")
    tier_map_p = str(ANN_DIR / "cl_tasks" / "concept_tier_map.json")

    failures = find_failure_cases(
        extractor, memory, concept_heads, anomaly_head,
        tau=tau, theta_concept=THETA,
        mvtec_root=MVTEC_ROOT, category=CATEGORY,
        defect_names=defect_names,
        concept_names=concept_heads.concept_names,
        max_per_type=5,
        tier_map_path=tier_map_p,
    )

    by_type: dict[str, list] = {}
    for fc in failures:
        by_type.setdefault(fc.failure_type, []).append(fc)

    print(f"\n  Failure case summary:")
    for ft, cases in sorted(by_type.items()):
        print(f"    {ft:<20} {len(cases)} case(s)")
        for c in cases[:2]:
            print(f"      {Path(c.image_path).name:<18} {c.description}")

    if failures:
        visualize_failure_gallery(
            failures, extractor, memory, concept_heads, anomaly_head,
            tau=tau, save_dir=str(FIGURE_DIR / "failures"),
            concept_names=concept_heads.concept_names, theta_concept=THETA,
            normal_baseline=normal_baseline,
        )
    else:
        print("  No failures found — model is clean on this test set.")

    # ── 5. concept evolution plot ─────────────────────────────────────────────
    print("\nGenerating concept evolution plot ...")

    def _load_log(path: Path) -> Optional[ContinualLog]:
        from evaluators.evaluator_cl import TaskEvalResult
        if not path.exists():
            return None
        with open(path) as f:
            d = json.load(f)
        return ContinualLog(results=[TaskEvalResult(**r) for r in d["results"]])

    log_concil = _load_log(CKPT_DIR / "log_final.json")
    log_naive  = _load_log(Path("checkpoints/hazelnut-baseline/log_final.json"))
    if log_naive is None:
        log_naive = _load_log(Path("checkpoints/hazelnut-baseline/log.json"))

    if log_concil:
        save_evo = str(FIGURE_DIR / "concept_evolution.png")
        visualize_concept_evolution(
            log_concil, log_naive,
            defect_names=defect_names,
            save_path=save_evo,
        )
        print(f"  Saved: {save_evo}")
        if log_naive:
            print(f"  CONCIL BWT  : {log_concil.mean_concept_bwt():+.4f}")
            print(f"  Naive BWT   : {log_naive.mean_concept_bwt():+.4f}")
        else:
            print(f"  CONCIL BWT  : {log_concil.mean_concept_bwt():+.4f}")
            print("  Naive log not found — run --baseline to completion first.")
    else:
        print("  CONCIL log not found.")

    # ══════════════════════════════════════════════════════════════════════════
    # INTERVENTION EXPERIMENTS
    # ══════════════════════════════════════════════════════════════════════════

    cnames  = concept_heads.concept_names
    weights = anomaly_head.concept_weights          # (K,) numpy

    # ── Step 1: print linear head weights ────────────────────────────────────
    print("\n" + "=" * 55)
    print("LINEAR HEAD WEIGHTS — top 15 (sorted by |weight|)")
    print("=" * 55)
    paired    = sorted(zip(cnames, weights), key=lambda x: abs(x[1]), reverse=True)
    top5_w_names = [n for n, _ in paired[:5]]
    print(f"  {'Rank':<5} {'Concept':<42} {'Weight':>8}")
    print(f"  {'-'*4:<5} {'-'*41:<42} {'-'*7:>8}")
    for rank, (name, w) in enumerate(paired[:15], 1):
        print(f"  {rank:<5} {name:<42} {w:>+8.4f}")

    # Top-3 concepts by weight for Experiment A
    top3_names = [n for n, _ in paired[:3]]

    # ── Concept weight bar chart ──────────────────────────────────────────────
    print("\nGenerating concept weights figure ...")
    visualize_concept_weights(
        anomaly_head, cnames,
        save_path=str(FIGURE_DIR / "concept_weights.png"),
        top_n=10,
    )
    print(f"  Saved: thesis_figures/concept_weights.png")

    # ── Experiment A (corrected): crack, set negative-weight concepts → 0.0 ──
    # Top-3 by |weight| with NEGATIVE weights:
    neg_top3 = [(n, w) for n, w in paired if w < 0][:3]
    neg_top3_names = [n for n, _ in neg_top3]
    print("\n" + "=" * 55)
    print("EXPERIMENT A (corrected) — crack image, semantic intervention")
    print(f"  Image: test/crack/002.png")
    print(f"  Set top-3 NEGATIVE-weight concepts → 0.0:")
    for n, w in neg_top3:
        print(f"    {n:<42} w={w:+.4f}")
    print("  Logic: defect removes these normal features → setting absent")
    print("=" * 55)

    crack_img_path = mvtec_cat / "test" / "crack" / "002.png"
    crack_img      = Image.open(crack_img_path).convert("RGB")

    iv_A = run_intervention(
        crack_img, concept_heads, anomaly_head, memory, extractor,
        tau=tau, concept_names=cnames,
        interventions={n: 0.0 for n in neg_top3_names},
    )
    print(f"  y_pred before : {iv_A.y_pred_orig:.4f}  "
          f"({'above' if iv_A.y_pred_orig >= tau else 'below'} τ={tau:.3f})")
    print(f"  y_pred after  : {iv_A.y_pred_modified:.4f}  "
          f"({'above' if iv_A.y_pred_modified >= tau else 'below'} τ={tau:.3f})")
    print(f"  Delta         : {iv_A.delta_y:+.4f}")
    if iv_A.delta_y > 0:
        print("  ✓ y_pred increased — bottleneck encodes physically meaningful signals")
    else:
        print("  (unexpected: intervention did not increase y_pred)")

    visualize_intervention(
        crack_img, iv_A,
        save_path=str(FIGURE_DIR / "intervention_crack_positive.png"),
        true_label="crack",
        concept_names=cnames, anomaly_head=anomaly_head,
        normal_baseline=normal_baseline,
    )
    print(f"  Saved: thesis_figures/intervention_crack_positive.png")

    # ── Experiment B: false positive — branch disagreement ────────────────────
    print("\n" + "=" * 55)
    print("EXPERIMENT B — normal (false positive), branch disagreement")
    print("  Image: test/good/002.png  (s_novel > τ but y_pred < τ)")
    print("=" * 55)

    good_fp_path = mvtec_cat / "test" / "good" / "002.png"
    good_fp_img  = Image.open(good_fp_path).convert("RGB")

    r_fp = _run_inference(extractor, memory, concept_heads, anomaly_head,
                          good_fp_img, tau, THETA, true_label="good")
    print(f"  s_novel : {r_fp.s_novel:.4f}  "
          f"({'above' if r_fp.s_novel >= tau else 'below'} τ={tau:.3f})  ← detection branch")
    print(f"  y_pred  : {r_fp.y_pred:.4f}  "
          f"({'above' if r_fp.y_pred >= tau else 'below'} τ={tau:.3f})  ← concept branch")
    if r_fp.s_novel >= tau and r_fp.y_pred < tau:
        print("  ⚠ Branches disagree: PatchCore fired, concept branch correctly said normal")

    visualize_branch_disagreement(
        good_fp_img, r_fp,
        save_path=str(FIGURE_DIR / "branch_disagreement_good_002.png"),
        normal_baseline=normal_baseline,
    )
    print(f"  Saved: thesis_figures/branch_disagreement_good_002.png")

    iv_B = run_intervention(
        good_fp_img, concept_heads, anomaly_head, memory, extractor,
        tau=tau, concept_names=cnames,
        interventions={"unexpected_surface_pattern": 0.0},
    )
    visualize_intervention(
        good_fp_img, iv_B,
        save_path=str(FIGURE_DIR / "intervention_normal_negative.png"),
        true_label="good (false positive)",
        concept_names=cnames, anomaly_head=anomaly_head,
        normal_baseline=normal_baseline,
    )
    print(f"  Saved: thesis_figures/intervention_normal_negative.png")

    # ── Experiment C: sensitivity sweep ───────────────────────────────────────
    print("\n" + "=" * 55)
    print("EXPERIMENT C — sensitivity sweep (top-5 weight concepts)")
    print("=" * 55)

    _, slopes, steepest = visualize_sensitivity_sweep(
        crack_img, concept_heads, anomaly_head, extractor,
        tau=tau, concept_names=cnames, top_n=5,
        save_path=str(FIGURE_DIR / "intervention_sensitivity.png"),
    )
    print(f"  Slopes (Δy per unit concept change):")
    for name, slope in sorted(slopes.items(), key=lambda x: -abs(x[1])):
        print(f"    {name:<42}  slope={slope:+.4f}")
    print(f"\n  Steepest concept: '{steepest}'")
    print(f"  Saved: thesis_figures/intervention_sensitivity.png")

    # ── final file listing ─────────────────────────────────────────────────────
    print(f"\nAll figures saved to {FIGURE_DIR}/")
    saved = list(FIGURE_DIR.rglob("*.png"))
    for p in sorted(saved):
        print(f"  {p.relative_to(FIGURE_DIR)}")


# ══════════════════════════════════════════════════════════════════════════════
# GENERIC VISUALIZATION SYSTEM — CategoryVisualizer
# Improvements 1–4: category-agnostic, patch feature space, 5-perspective figure
# ══════════════════════════════════════════════════════════════════════════════

from dataclasses import dataclass as _dc, field as _field
import pandas as _pd
from sklearn.manifold import TSNE as _TSNE
from sklearn.decomposition import PCA as _PCA


@_dc
class VisualizerConfig:
    """All paths and hyperparameters needed for one MVTec category."""
    category:           str
    mvtec_root:         Path
    annotations_dir:    Path
    checkpoint_dir:     Path
    defect_train_ratio: float = 0.8
    tau_percentile:     float = 95.0
    seed:               int   = 42


class CategoryVisualizer:
    """Loads all model components from a checkpoint and exposes visualization methods.

    Works for ANY MVTec category — no hardcoded paths.
    """

    def __init__(self, config: VisualizerConfig):
        from features.dinov2_extractor import DINOv2Extractor
        from features.patchcore_memory import PatchCoreMemory
        from models.concept_heads       import ConceptHeads
        from models.linear_head         import LinearAnomalyHead

        self.config   = config
        self._device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._cat_dir = config.mvtec_root / config.category

        # ── find latest task checkpoint ───────────────────────────────────────
        task_dirs = sorted(
            [d for d in config.checkpoint_dir.iterdir()
             if d.is_dir() and d.name.startswith("task_")],
            key=lambda d: int(d.name.split("_")[1]),
        )
        if not task_dirs:
            raise FileNotFoundError(f"No task_N dirs in {config.checkpoint_dir}")
        latest = task_dirs[-1]

        # ── load model components ─────────────────────────────────────────────
        print(f"  [{config.category}] Loading from {latest} ...")
        self.extractor     = DINOv2Extractor(device=self._device)
        self.memory        = PatchCoreMemory.load(
            config.checkpoint_dir / "task_1" / "memory.pt", device=self._device
        )
        self.concept_heads = ConceptHeads.load(latest / "concept_heads.pt")
        self.anomaly_head  = LinearAnomalyHead.load(latest / "anomaly_head.pt")

        with open(latest / "tau.json") as f:
            self.tau = json.load(f)["tau"]

        self.concept_names = self.concept_heads.concept_names

        # ── concept normal baseline (compute if missing) ──────────────────────
        baseline_path = config.checkpoint_dir / "concept_normal_baseline.json"
        if baseline_path.exists():
            with open(baseline_path) as f:
                bl = json.load(f)
            self.normal_baseline: dict[str, float] = bl["means"]
        else:
            print(f"  [{config.category}] Computing normal baseline ...")
            from evaluators.visualizer import compute_normal_baseline
            train_paths = sorted((self._cat_dir / "train" / "good").glob("*.png"))
            train_imgs  = [Image.open(p).convert("RGB") for p in train_paths]
            self.normal_baseline = compute_normal_baseline(
                self.concept_heads, self.extractor, train_imgs,
                concept_names=self.concept_names,
                save_path=str(baseline_path),
            )

        # ── annotation CSV for concept labels ─────────────────────────────────
        self._full_csv = _pd.read_csv(
            config.annotations_dir / f"{config.category}.csv"
        )
        _META = {"image_path","label_index","mask_path","anomaly_type","split","view"}
        self._concept_cols = [c for c in self._full_csv.columns if c not in _META]

        # ── discover defect types ─────────────────────────────────────────────
        self.defect_types = sorted(
            d.name for d in (self._cat_dir / "test").iterdir()
            if d.is_dir() and d.name != "good"
        )
        print(f"  [{config.category}] τ={self.tau:.4f}  "
              f"K={len(self.concept_names)}  defects={self.defect_types}")

    # ── internal inference ────────────────────────────────────────────────────

    @torch.no_grad()
    def _infer(self, image: Image.Image, true_label: str = "",
               theta: float = 0.5) -> PredictionResult:
        return _run_inference(
            self.extractor, self.memory, self.concept_heads, self.anomaly_head,
            image, self.tau, theta, true_label,
        )

    # ── panel helpers (shared by visualize_prediction and visualize_full) ─────

    def _panel_image(self, ax, img_arr, result: PredictionResult):
        ax.imshow(img_arr)
        ax.set_title(f'"{result.true_label}"', fontsize=11, fontweight="bold")
        ax.axis("off")
        col = "#2ecc71" if result.correct else "#e74c3c"
        rect = mpatches.FancyBboxPatch(
            (0,0),1,1, boxstyle="square,pad=0",
            linewidth=5, edgecolor=col, facecolor="none",
            transform=ax.transAxes, clip_on=False,
        )
        ax.add_patch(rect)
        ax.text(0.02,0.02,"✓" if result.correct else "✗",
                transform=ax.transAxes, fontsize=10, fontweight="bold",
                color=col, va="bottom")

    def _panel_heatmap(self, ax, fig, img_arr, result: PredictionResult):
        hm = scipy.ndimage.zoom(result.anomaly_map, _HEATMAP_ZOOM, order=3)
        hm_norm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-8)
        ax.imshow(img_arr)
        im2 = ax.imshow(hm_norm, cmap="jet", alpha=0.5, vmin=0, vmax=1)
        fig.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)
        col = "#e74c3c" if result.s_novel >= result.tau else "#2ecc71"
        ax.set_title(
            f"Anomaly map  s={result.s_novel:.3f}  τ={result.tau:.3f}",
            fontsize=9,
        )
        ax.axis("off")

    def _panel_concepts(self, ax, result: PredictionResult):
        devs = np.array([
            result.c[i] - self.normal_baseline.get(result.concept_names[i], 0.5)
            for i in range(len(result.c))
        ])
        top10 = np.argsort(np.abs(devs))[::-1][:10]
        top_dev = devs[top10]
        top_nms = [
            result.concept_names[i][:20] + "…" if len(result.concept_names[i]) > 21
            else result.concept_names[i]
            for i in top10
        ]
        bar_colours = ["#e74c3c" if d > 0.2 else "#3498db" if d < -0.2 else "#bdc3c7"
                       for d in top_dev]
        ax.barh(range(10), top_dev[::-1], color=bar_colours[::-1],
                edgecolor="white", linewidth=0.4)
        ax.set_yticks(range(10))
        ax.set_yticklabels(top_nms[::-1], fontsize=7)
        ax.set_xlim(-1,1); ax.axvline(0, color="#333", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Δ from normal", fontsize=8)
        ax.set_title("Top-10 concept deviations", fontsize=9)
        ax.legend(handles=[
            mpatches.Patch(facecolor="#e74c3c", label="activated"),
            mpatches.Patch(facecolor="#3498db", label="suppressed"),
        ], fontsize=6, loc="lower right")
        ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)

    def _panel_decision(self, ax, result: PredictionResult):
        ax.axis("off"); ax.set_facecolor("#f8f9fa"); ax.patch.set_visible(True)
        lc = _LEVEL_COLOUR[result.level]; ll = _LEVEL_LABEL[result.level]
        devs = np.array([
            result.c[i] - self.normal_baseline.get(result.concept_names[i], 0.5)
            for i in range(len(result.c))
        ])
        top3 = np.argsort(np.abs(devs))[::-1][:3]
        lines = [
            ("DECISION: " + ll, 12, "bold", lc),
            ("─"*32, 9, "normal", "#555"),
            (f"s_novel: {result.s_novel:.3f}  [{_bar(result.s_novel,max_val=max(result.s_novel*1.5,result.tau*2,0.01))}]",
             8, "normal", "#333"),
            (f"τ: {result.tau:.3f}", 8, "normal", "#333"),
            ("", 5, "normal", "#fff"),
            (f"y_pred: {result.y_pred:.3f}  [{_bar(result.y_pred)}]", 8, "normal", "#333"),
            ("", 5, "normal", "#fff"),
            ("EXPLANATION:", 9, "bold", "#222"),
        ]
        for idx in top3:
            n = result.concept_names[idx][:24]; d = devs[idx]
            a = "↑" if d > 0 else "↓"; col = "#c0392b" if d > 0 else "#2980b9"
            lines.append((f"  {a} {n}: {d:+.2f}", 8, "normal", col))
        lines += [
            ("", 5, "normal", "#fff"),
            (f"Level {result.level}/3", 10, "bold", lc),
            ("✓ CORRECT" if result.correct else "✗ WRONG", 10, "bold",
             "#2ecc71" if result.correct else "#e74c3c"),
        ]
        y = 0.97
        for text, size, wt, col in lines:
            ax.text(0.04, y, text, transform=ax.transAxes,
                    fontsize=size, fontweight=wt, color=col, va="top",
                    fontfamily="monospace" if any(c in text for c in "─█░↑↓") else "sans-serif")
            y -= size * 0.013 + 0.008

    def _panel_patch_tsne(self, ax, img_arr, result: PredictionResult,
                          patch_tokens: torch.Tensor,
                          patch_distances: np.ndarray,
                          n_memory_samples: int = 200):
        """t-SNE scatter: memory bank vs test image patches."""
        import torch.nn.functional as _F

        # Sample from memory bank
        mem = self.memory.memory.cpu()                    # (N_stored, 768)
        torch.manual_seed(self.config.seed)
        idx = torch.randperm(len(mem))[:n_memory_samples]
        mem_sample = mem[idx].numpy()                     # (n_mem, 768)

        # Test patches (already L2-normalised by PatchCore scoring path)
        test_p = _F.normalize(patch_tokens[0].cpu().float(), p=2, dim=1).numpy()  # (256, 768)

        all_p = np.vstack([mem_sample, test_p])           # (n_mem+256, 768)

        # PCA 50 → t-SNE 2
        pca = _PCA(n_components=min(50, all_p.shape[0]-1), random_state=42)
        all_r = pca.fit_transform(all_p)
        emb   = _TSNE(n_components=2, perplexity=30, random_state=42,
                      max_iter=500, learning_rate="auto", init="pca").fit_transform(all_r)

        mem_emb  = emb[:n_memory_samples]
        test_emb = emb[n_memory_samples:]

        most_anom = int(patch_distances.argmax())
        is_defect = result.true_label.lower() not in ("good","normal","")
        test_col  = "#e74c3c" if is_defect else "#3498db"

        ax.scatter(mem_emb[:,0], mem_emb[:,1],
                   c="#2ecc71", s=8, alpha=0.35, label=f"Memory ({n_memory_samples})")
        ax.scatter(test_emb[:,0], test_emb[:,1],
                   c=test_col, s=18, alpha=0.7, label="Test patches")
        ax.scatter(test_emb[most_anom,0], test_emb[most_anom,1],
                   c="#f1c40f", s=200, marker="*", zorder=5,
                   label=f"★ worst patch (d={patch_distances[most_anom]:.3f})")

        det_txt = "ANOMALY: patches outside cluster" if result.s_novel >= result.tau \
                  else "NORMAL: patches inside cluster"
        det_col = "#e74c3c" if result.s_novel >= result.tau else "#2ecc71"
        ax.text(0.02, 0.98, det_txt, transform=ax.transAxes, fontsize=7,
                color=det_col, va="top", fontweight="bold")
        ax.set_title(f"Patch feature space (t-SNE)\ns={result.s_novel:.3f}  τ={result.tau:.3f}",
                     fontsize=9)
        ax.legend(fontsize=6, loc="lower right", markerscale=0.8)
        ax.set_xticks([]); ax.set_yticks([])

    def _panel_patch_grid(self, ax, img_arr, patch_distances: np.ndarray):
        """16×16 patch grid coloured by anomaly distance."""
        grid = patch_distances.reshape(16, 16)
        norm = (grid - grid.min()) / (grid.max() - grid.min() + 1e-8)
        ax.imshow(img_arr)
        ax.imshow(norm, cmap="jet", alpha=0.6, vmin=0, vmax=1,
                  extent=[0, img_arr.shape[1], img_arr.shape[0], 0],
                  interpolation="nearest")
        # Draw grid lines
        h, w = img_arr.shape[:2]
        for i in range(17):
            ax.axhline(i * h/16, color="white", lw=0.3, alpha=0.4)
            ax.axvline(i * w/16, color="white", lw=0.3, alpha=0.4)
        # Highlight worst patch
        worst = int(patch_distances.argmax())
        row, col = divmod(worst, 16)
        rect = mpatches.Rectangle(
            (col * w/16, row * h/16), w/16, h/16,
            linewidth=2.5, edgecolor="#f1c40f", facecolor="none",
        )
        ax.add_patch(rect)
        ax.set_title("Per-patch distances\n(★ = worst, matches t-SNE)", fontsize=9)
        ax.axis("off")

    # ── PUBLIC: visualize_patch_space ─────────────────────────────────────────

    @torch.no_grad()
    def visualize_patch_space(
        self,
        test_image_path: str,
        true_label: str,
        save_path: str,
        n_memory_samples: int = 300,
    ) -> plt.Figure:
        """3-panel: image with patch grid | t-SNE | patch distance grid."""
        img  = Image.open(test_image_path).convert("RGB")
        result = self._infer(img, true_label)

        patch_tokens, _ = self.extractor.extract_both([img])    # (1, 256, 768)
        _, amap = self.memory.score(patch_tokens)
        patch_distances = amap[0].cpu().numpy().flatten()        # (256,)

        img_arr = np.array(img.resize((_IMG_SIZE, _IMG_SIZE)))

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.subplots_adjust(wspace=0.12)

        # Panel 1 — image with patch grid overlay
        axes[0].imshow(img_arr)
        h, w = img_arr.shape[:2]
        for i in range(17):
            axes[0].axhline(i*h/16, color="white", lw=0.4, alpha=0.5)
            axes[0].axvline(i*w/16, color="white", lw=0.4, alpha=0.5)
        axes[0].set_title(f'"{true_label}"\n16×16 patch grid', fontsize=11, fontweight="bold")
        axes[0].axis("off")
        col = "#2ecc71" if result.correct else "#e74c3c"
        axes[0].add_patch(mpatches.FancyBboxPatch(
            (0,0),1,1, boxstyle="square,pad=0",
            linewidth=5, edgecolor=col, facecolor="none",
            transform=axes[0].transAxes, clip_on=False,
        ))

        # Panel 2 — t-SNE
        self._panel_patch_tsne(axes[1], img_arr, result, patch_tokens,
                               patch_distances, n_memory_samples)

        # Panel 3 — patch grid
        self._panel_patch_grid(axes[2], img_arr, patch_distances)

        fig.suptitle(
            f"{self.config.category.capitalize()} / {true_label}  →  "
            f"{_LEVEL_LABEL[result.level]}",
            fontsize=12, fontweight="bold", y=1.02,
        )
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        return fig

    # ── PUBLIC: visualize_full ────────────────────────────────────────────────

    @torch.no_grad()
    def visualize_full(
        self,
        image_path: str,
        true_label: str,
        save_path: str,
        n_memory_samples: int = 200,
    ) -> plt.Figure:
        """5-perspective figure: image | heatmap | t-SNE | concepts | patch grid | decision."""
        img    = Image.open(image_path).convert("RGB")
        result = self._infer(img, true_label)

        patch_tokens, _ = self.extractor.extract_both([img])
        _, amap = self.memory.score(patch_tokens)
        patch_distances = amap[0].cpu().numpy().flatten()

        img_arr = np.array(img.resize((_IMG_SIZE, _IMG_SIZE)))

        fig = plt.figure(figsize=(20, 12))
        fig.subplots_adjust(hspace=0.35, wspace=0.3)

        ax1 = fig.add_subplot(2, 3, 1)
        ax2 = fig.add_subplot(2, 3, 2)
        ax3 = fig.add_subplot(2, 3, 3)
        ax4 = fig.add_subplot(2, 3, 4)
        ax5 = fig.add_subplot(2, 3, 5)
        ax6 = fig.add_subplot(2, 3, 6)

        self._panel_image(ax1, img_arr, result)
        self._panel_heatmap(ax2, fig, img_arr, result)
        self._panel_patch_tsne(ax3, img_arr, result, patch_tokens,
                               patch_distances, n_memory_samples)
        self._panel_concepts(ax4, result)
        self._panel_patch_grid(ax5, img_arr, patch_distances)
        self._panel_decision(ax6, result)

        fig.suptitle(
            f"{self.config.category.capitalize()} / {true_label}  "
            f"→  {_LEVEL_LABEL[result.level]}",
            fontsize=14, fontweight="bold",
        )
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        return fig


# ── IMPROVEMENT 4 — Generic failure cases (accepts VisualizerConfig) ─────────

def find_failure_cases_generic(
    config: VisualizerConfig,
    viz: CategoryVisualizer,
    max_per_type: int = 3,
    theta: float = 0.5,
    tier_map_path: Optional[str] = None,
) -> list[FailureCase]:
    """Scan all available defect types for failures. No hardcoded paths."""
    full_df     = _pd.read_csv(config.annotations_dir / f"{config.category}.csv")
    tier3: dict[str, set[str]] = {}
    if tier_map_path and Path(tier_map_path).exists():
        with open(tier_map_path) as f:
            tm = json.load(f)
        for d in viz.defect_types:
            tier3[d] = set(tm.get(f"tier3_{d}", []))

    buckets: dict[str, list[FailureCase]] = {k: [] for k in
        ["false_negative","false_positive","explainability","level3"]}

    def _proc(img_path: Path, label: str):
        img = Image.open(img_path).convert("RGB")
        r   = viz._infer(img, label, theta)
        top_c = viz.concept_names[int(r.c.argmax())]

        if label in ("good","normal"):
            if r.s_novel >= viz.tau:
                buckets["false_positive"].append(FailureCase(
                    str(img_path), label, "false_positive",
                    r.s_novel, r.c, r.anomaly_map, r.y_pred,
                    f"False alarm: s={r.s_novel:.3f}>τ={viz.tau:.3f}",
                ))
        else:
            if r.s_novel < viz.tau:
                buckets["false_negative"].append(FailureCase(
                    str(img_path), label, "false_negative",
                    r.s_novel, r.c, r.anomaly_map, r.y_pred,
                    f"Missed: s={r.s_novel:.3f}<τ={viz.tau:.3f}",
                ))
            elif r.c.max() > theta:
                expected = tier3.get(label, set())
                if expected and top_c not in expected:
                    buckets["explainability"].append(FailureCase(
                        str(img_path), label, "explainability",
                        r.s_novel, r.c, r.anomaly_map, r.y_pred,
                        f"Wrong concept: '{top_c}' on {label}",
                    ))
            else:
                buckets["level3"].append(FailureCase(
                    str(img_path), label, "level3",
                    r.s_novel, r.c, r.anomaly_map, r.y_pred,
                    f"Level3: detected, unexplained (max_c={r.c.max():.3f})",
                ))

    # Held-out 20% defects (same split as training)
    for defect in viz.defect_types:
        defect_df = full_df[full_df["anomaly_type"] == defect].reset_index(drop=True)
        n = len(defect_df)
        n_train = max(1, int(n * config.defect_train_ratio))
        rng  = np.random.RandomState(config.seed)
        shuf = rng.permutation(n)
        held = defect_df.iloc[shuf[n_train:]]["image_path"].tolist()
        for p in held:
            _proc(Path(p), defect)

    # Test normals (false positives)
    for p in sorted((config.mvtec_root / config.category / "test" / "good").glob("*.png")):
        _proc(p, "good")

    result: list[FailureCase] = []
    for key in buckets:
        sub = sorted(buckets[key],
            key=lambda x: x.s_novel if key == "false_positive"
                         else (-x.s_novel if key == "false_negative" else -x.s_novel))
        result.extend(sub[:max_per_type])
    return result


# ── __main__ for generic system ────────────────────────────────────────────────

if __name__ == "__main__" and False:   # disabled: run as python -m evaluators.generic_viz
    pass
