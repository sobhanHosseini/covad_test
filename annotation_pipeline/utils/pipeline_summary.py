"""
pipeline_summary.py — Generate a clean summary report from annotation pipeline
                      checkpoints, without rerunning anything.

Reads the saved JSON checkpoints for each category and produces:
  1. Per-category vocabulary summary (concept counts per tier)
  2. Dataset statistics (images per split, per defect type)
  3. Concept quality metrics (frequency, visual dimension distribution)
  4. Per-defect concept breakdown

Usage:
    python pipeline_summary.py \
        --annotations_dir ./annotations \
        --categories hazelnut capsule bottle transistor \
        --output_dir ./annotations/summary

Output files:
    summary/pipeline_summary.csv       ← main table (one row per category)
    summary/concept_details.csv        ← per-concept details across categories
    summary/dataset_stats.csv          ← image counts per category/defect
    summary/pipeline_summary.md        ← markdown table for thesis/presentation
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from collections import Counter

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

META_COLS = {'image_path', 'label_index', 'mask_path',
             'anomaly_type', 'split', 'view'}


# ── Checkpoint loaders ────────────────────────────────────────────────────────

def load_json(path: Path):
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def find_checkpoint_dir(annotations_dir: str, category: str) -> Path:
    """Find the category checkpoint folder."""
    base = Path(annotations_dir)
    # Check category subfolder first
    cat_dir = base / category
    if cat_dir.exists():
        return cat_dir
    # Fallback: flat structure
    return base


def find_csv(cat_dir: Path, category: str, suffix: str = 'final') -> Path | None:
    """Find the main (non-holdout) CSV for a category."""
    candidates = [
        cat_dir / f"{category}_{suffix}.csv",
        cat_dir / f"{category}_v6_{suffix}.csv",
        cat_dir / f"{category}_v6_freq15.csv",
        cat_dir / f"{category}.csv",
    ]
    # Also search for any non-holdout CSV
    for p in sorted(cat_dir.glob(f"{category}*.csv")):
        if 'holdout' not in p.name:
            candidates.append(p)

    for p in candidates:
        if p.exists():
            return p
    return None


# ── Analysis functions ────────────────────────────────────────────────────────

def analyze_normal_concepts(stage2: list[dict]) -> dict:
    """Extract stats from Stage 2 normal concept dictionary."""
    if not stage2:
        return {}

    dims = Counter(c.get('visual_dimension', 'unknown') for c in stage2)
    freqs = [c.get('frequency', 0) for c in stage2]

    return {
        'n_normal_concepts': len(stage2),
        'normal_dim_texture':   dims.get('texture', 0),
        'normal_dim_color':     dims.get('color', 0) + dims.get('surface color', 0),
        'normal_dim_shape':     dims.get('shape', 0) + dims.get('shape/geometry', 0),
        'normal_dim_finish':    dims.get('finish', 0) + dims.get('material finish', 0),
        'normal_dim_structure': dims.get('structure', 0) + dims.get('structural integrity', 0),
        'normal_dim_marking':   dims.get('marking', 0) + dims.get('visible surface markings', 0),
        'normal_freq_mean':     round(sum(freqs) / len(freqs), 3) if freqs else 0,
        'normal_freq_min':      round(min(freqs), 3) if freqs else 0,
        'normal_concepts_list': [c['name'] for c in stage2],
    }


def analyze_stage3(stage3: list[dict]) -> dict:
    """Extract dataset and defect concept stats from Stage 3 annotations."""
    if not stage3:
        return {}

    defect_counts = Counter()
    defect_concepts: dict[str, set] = {}
    n_normal = 0

    for ann in stage3:
        dt = ann.get('defect_category', ann.get('anomaly_type', 'unknown'))
        if dt == 'good':
            n_normal += 1
        else:
            defect_counts[dt] += 1
            for c in ann.get('new_defect_concepts', []):
                defect_concepts.setdefault(dt, set()).add(c['name'])

    return {
        'n_normal_images':   n_normal,
        'n_defect_images':   sum(defect_counts.values()),
        'n_total_images':    len(stage3),
        'defect_types':      sorted(defect_counts.keys()),
        'defect_counts':     dict(defect_counts),
        'defect_concepts':   {k: sorted(v) for k, v in defect_concepts.items()},
        'n_defect_types':    len(defect_counts),
    }


def analyze_csv(csv_path: Path) -> dict:
    """Extract vocabulary stats from the final output CSV."""
    if not csv_path or not csv_path.exists():
        return {}

    df = pd.read_csv(csv_path)
    concept_cols = [c for c in df.columns if c not in META_COLS]

    defect_df = df[df['label_index'] == 1]
    normal_df = df[df['label_index'] == 0]

    # Identify generic concepts (5 fixed ones)
    generic = {
        'surface_irregularity', 'color_deviation', 'structural_discontinuity',
        'texture_inconsistency', 'unexpected_surface_pattern'
    }

    # Classify concepts into tiers
    # (approximation: concepts active in defects but not clearly normal = Tier 3)
    tier1, tier2, tier3 = [], [], []
    for c in concept_cols:
        if c in generic:
            tier2.append(c)
        elif c in df.columns:
            # If concept has high activation in normal → Tier 1
            normal_activation = normal_df[c].mean() if len(normal_df) > 0 else 0
            defect_activation = defect_df[c].mean() if len(defect_df) > 0 else 0
            if normal_activation > 0.3:
                tier1.append(c)
            else:
                tier3.append(c)

    return {
        'total_concepts':  len(concept_cols),
        'n_tier1_approx':  len(tier1),
        'n_tier2':         len(tier2),
        'n_tier3_approx':  len(tier3),
        'all_concepts':    concept_cols,
        'csv_path':        str(csv_path),
        'csv_rows':        len(df),
    }


# ── Main report builder ───────────────────────────────────────────────────────

def build_report(annotations_dir: str, categories: list[str]) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    summary_rows = []
    concept_rows = []
    dataset_rows = []

    for category in categories:
        log.info("Processing: %s", category)
        cat_dir = find_checkpoint_dir(annotations_dir, category)

        # Load checkpoints — try category-prefixed name first, then unprefixed fallback
        stage2 = (
            load_json(cat_dir / f"{category}_stage2_normal_dict.json") or
            load_json(cat_dir / "stage2_normal_dict.json")
        )
        stage3 = (
            load_json(cat_dir / f"{category}_stage3_annotations.json") or
            load_json(cat_dir / "stage3_annotations.json")
        )
        csv_path = find_csv(cat_dir, category)

        if not stage2 and not stage3:
            log.warning("  No checkpoints found in %s — skipping", cat_dir)
            continue

        # Analyze
        s2 = analyze_normal_concepts(stage2 or [])
        s3 = analyze_stage3(stage3 or [])
        cv = analyze_csv(csv_path)

        # ── Summary row ───────────────────────────────────────────────────────
        row = {
            'category':           category,
            'n_normal_concepts':  s2.get('n_normal_concepts', '?'),
            'n_generic_concepts': 5,  # always 5 fixed
            'n_defect_concepts':  cv.get('n_tier3_approx', '?'),
            'total_vocab':        cv.get('total_concepts', '?'),
            'n_defect_types':     s3.get('n_defect_types', '?'),
            'n_normal_images':    s3.get('n_normal_images', '?'),
            'n_defect_images':    s3.get('n_defect_images', '?'),
            'n_total_images':     s3.get('n_total_images', '?'),
            'normal_freq_mean':   s2.get('normal_freq_mean', '?'),
            'dim_texture':        s2.get('normal_dim_texture', '?'),
            'dim_color':          s2.get('normal_dim_color', '?'),
            'dim_shape':          s2.get('normal_dim_shape', '?'),
            'dim_finish':         s2.get('normal_dim_finish', '?'),
            'dim_structure':      s2.get('normal_dim_structure', '?'),
            'dim_marking':        s2.get('normal_dim_marking', '?'),
        }
        summary_rows.append(row)
        log.info("  Vocab: %s normal + 5 generic + %s defect = %s total",
                 s2.get('n_normal_concepts', '?'),
                 cv.get('n_tier3_approx', '?'),
                 cv.get('total_concepts', '?'))

        # ── Per-concept rows ──────────────────────────────────────────────────
        for c in (stage2 or []):
            concept_rows.append({
                'category':         category,
                'concept':          c['name'],
                'tier':             'Tier 1 (normal)',
                'visual_dimension': c.get('visual_dimension', '?'),
                'frequency':        c.get('frequency', '?'),
                'description':      c.get('description', ''),
            })

        # Generic concepts
        generic_names = [
            'surface_irregularity', 'color_deviation', 'structural_discontinuity',
            'texture_inconsistency', 'unexpected_surface_pattern'
        ]
        for name in generic_names:
            concept_rows.append({
                'category':         category,
                'concept':          name,
                'tier':             'Tier 2 (generic)',
                'visual_dimension': 'mixed',
                'frequency':        'N/A',
                'description':      'Fixed generic anomaly concept',
            })

        # Defect-specific concepts
        defect_concepts = s3.get('defect_concepts', {})
        for defect_type, concepts in defect_concepts.items():
            for c_name in concepts:
                concept_rows.append({
                    'category':         category,
                    'concept':          c_name,
                    'tier':             f'Tier 3 ({defect_type})',
                    'visual_dimension': '?',
                    'frequency':        '?',
                    'description':      '',
                })

        # ── Dataset rows ──────────────────────────────────────────────────────
        defect_counts = s3.get('defect_counts', {})
        dataset_rows.append({
            'category':       category,
            'split':          'normal (train)',
            'defect_type':    'good',
            'count':          s3.get('n_normal_images', '?'),
        })
        for defect_type, count in sorted(defect_counts.items()):
            dataset_rows.append({
                'category':    category,
                'split':       'test (defect)',
                'defect_type': defect_type,
                'count':       count,
            })

    summary_df = pd.DataFrame(summary_rows)
    concept_df = pd.DataFrame(concept_rows)
    dataset_df = pd.DataFrame(dataset_rows)

    return summary_df, concept_df, dataset_df


def build_markdown(summary_df: pd.DataFrame) -> str:
    """Build a thesis-ready markdown table."""
    lines = [
        "# Annotation Pipeline Summary\n",
        "## Vocabulary per Category\n",
        "| Category | Normal | Generic | Defect-specific | Total | Defect types |",
        "|----------|--------|---------|-----------------|-------|--------------|",
    ]
    for _, r in summary_df.iterrows():
        lines.append(
            f"| {r['category']} | {r['n_normal_concepts']} | "
            f"{r['n_generic_concepts']} | {r['n_defect_concepts']} | "
            f"{r['total_vocab']} | {r['n_defect_types']} |"
        )

    lines += [
        "\n## Dataset Statistics\n",
        "| Category | Normal images | Defect images | Total |",
        "|----------|--------------|---------------|-------|",
    ]
    for _, r in summary_df.iterrows():
        lines.append(
            f"| {r['category']} | {r['n_normal_images']} | "
            f"{r['n_defect_images']} | {r['n_total_images']} |"
        )

    lines += [
        "\n## Normal Concept Dimension Coverage\n",
        "| Category | Texture | Color | Shape | Finish | Structure | Marking |",
        "|----------|---------|-------|-------|--------|-----------|---------|",
    ]
    for _, r in summary_df.iterrows():
        lines.append(
            f"| {r['category']} | {r['dim_texture']} | {r['dim_color']} | "
            f"{r['dim_shape']} | {r['dim_finish']} | {r['dim_structure']} | "
            f"{r['dim_marking']} |"
        )

    return "\n".join(lines)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Generate pipeline summary report")
    p.add_argument("--annotations_dir", required=True,
                   help="Root annotations directory (contains category subfolders)")
    p.add_argument("--categories", nargs="+",
                   default=["hazelnut", "capsule", "bottle", "transistor"])
    p.add_argument("--output_dir", default=None,
                   help="Where to save reports (default: annotations_dir/summary)")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir or Path(args.annotations_dir) / "summary")
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_df, concept_df, dataset_df = build_report(
        args.annotations_dir, args.categories
    )

    # Save CSVs
    summary_path = out_dir / "pipeline_summary.csv"
    concept_path = out_dir / "concept_details.csv"
    dataset_path = out_dir / "dataset_stats.csv"
    md_path      = out_dir / "pipeline_summary.md"

    summary_df.to_csv(summary_path, index=False)
    concept_df.to_csv(concept_path, index=False)
    dataset_df.to_csv(dataset_path, index=False)

    md = build_markdown(summary_df)
    with open(md_path, "w") as f:
        f.write(md)

    # Print to terminal
    log.info("\n%s", "=" * 60)
    log.info("PIPELINE SUMMARY")
    log.info("%s", "=" * 60)
    display_cols = ['category', 'n_normal_concepts', 'n_generic_concepts',
                    'n_defect_concepts', 'total_vocab', 'n_defect_types',
                    'n_total_images']
    avail = [c for c in display_cols if c in summary_df.columns]
    print(summary_df[avail].to_string(index=False))

    log.info("\nSaved:")
    log.info("  %s", summary_path)
    log.info("  %s", concept_path)
    log.info("  %s", dataset_path)
    log.info("  %s  ← paste into thesis", md_path)


if __name__ == "__main__":
    main()