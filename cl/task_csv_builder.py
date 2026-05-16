"""Build per-task DataFrames from the full annotated CSV for CONVAD-CL Scenario B.

Usage:
    python -m cl.task_csv_builder \
        --full_csv    annotations/hazelnut/hazelnut.csv \
        --output_dir  annotations/hazelnut/cl_tasks \
        --task_sequence crack hole cut print

Outputs (all written to output_dir):
    task_1_crack.csv       — normal (431) + crack (18) rows, all 42 concept columns
    task_2_hole.csv        — hole (18) rows only
    task_3_cut.csv         — cut (17) rows only
    task_4_print.csv       — print (17) rows only
    concept_tier_map.json  — tier1/tier2/tier3 concept classification
    task_sequence.json     — ordered task metadata
"""

from __future__ import annotations

import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# ── column names that are metadata, not concepts ──────────────────────────────
_META_COLS = frozenset(
    ["image_path", "label_index", "mask_path", "anomaly_type", "split", "view"]
)

# ── tier detection thresholds ─────────────────────────────────────────────────
# Tier 3 uses absolute mean-difference to catch both signal directions:
#   - fires HIGH on defect, LOW on normal  (e.g. structural_discontinuity)
#   - fires LOW on defect, HIGH on normal  (e.g. deep_shell_fissure)
# The VLM annotation pipeline (Stage 3) labels all concepts on all images,
# so "normal" surface concepts like deep_shell_fissure end up with mean≈1.00
# on normal images and low mean on defect images — the opposite of the naive
# expectation, but equally discriminative.
TIER3_DIFF_THRESH = 0.30    # |mean_defect - mean_normal| must exceed this
TIER1_NORMAL_THRESH = 0.30  # concept mean on normal rows to count as baseline normal

# NOTE — pre-generated CSVs vs real CL deployment:
# The hazelnut.csv was annotated by the pipeline across ALL defect types at once,
# so all 42 concept columns exist from Task 1. In real CL deployment via
# `annotation_pipeline --append_defect`, genuinely new concept columns appear
# in the CSV when a new defect arrives, and CONCIL adds new FC head columns.
# For this experiment CONCIL updates existing weights only (no vocabulary expansion).
# Both cases are valid demonstrations of concept-level continual learning.
#
# THESIS NOTE — new concept heads per task (hazelnut experiment):
#   T1 crack  → 32 concept heads established (all discriminative concepts)
#   T2 hole   → 0 new heads, CONCIL updates existing weights only
#   T3 cut    → 0 new heads, CONCIL updates existing weights only
#   T4 print  → 3 new heads (matte_surface_finish, non_reflective_sheen,
#                             non_reflective_surface — print-specific reflectivity)
# The CL experiment still exercises CONCIL's forgetting prevention and the
# concept-BWT metric at every task, even when no vocabulary expansion occurs.


def concept_cols(df: pd.DataFrame) -> list[str]:
    """Return concept column names (all columns except metadata)."""
    return [c for c in df.columns if c not in _META_COLS]


def build_tier_map(df: pd.DataFrame, defect_types: list[str]) -> dict[str, list[str]]:
    """Classify every concept column into tier1 / tier2 / tier3_<defect>.

    Assignment priority:
      1. tier3_<defect>: |mean_defect - mean_normal| > TIER3_DIFF_THRESH
         Captures both signal directions (goes up OR down on defect images).
         A concept can be tier3 for multiple defects simultaneously.
      2. tier1_normal:   not tier3 for any defect
                         AND mean_on_normal >= TIER1_NORMAL_THRESH
      3. tier2_generic:  everything else (low signal on normal, low diff)
    """
    concepts = concept_cols(df)
    normal_df = df[df["anomaly_type"] == "good"]
    normal_means = normal_df[concepts].mean()

    tier3: dict[str, list[str]] = {d: [] for d in defect_types}
    assigned_to_tier3: set[str] = set()

    for defect in defect_types:
        defect_df = df[df["anomaly_type"] == defect]
        if defect_df.empty:
            continue
        defect_means = defect_df[concepts].mean()
        for c in concepts:
            if abs(defect_means[c] - normal_means[c]) > TIER3_DIFF_THRESH:
                tier3[defect].append(c)
                assigned_to_tier3.add(c)

    tier1_normal: list[str] = []
    tier2_generic: list[str] = []
    for c in concepts:
        if c in assigned_to_tier3:
            continue
        if normal_means[c] >= TIER1_NORMAL_THRESH:
            tier1_normal.append(c)
        else:
            tier2_generic.append(c)

    return {
        "tier1_normal": sorted(tier1_normal),
        "tier2_generic": sorted(tier2_generic),
        **{f"tier3_{d}": sorted(v) for d, v in tier3.items()},
    }


def build_task_dataframes(
    full_csv: Path,
    task_sequence: list[str],
    output_dir: Path,
) -> tuple[list[dict], dict]:
    """Derive per-task DataFrames and save all output files.

    Task 1: normal rows + first defect rows  (both classes needed for initial CONCIL fit)
    Task t>1: defect rows only               (normals already encoded in A_1)

    Returns (task_metadata_list, tier_map).
    """
    df = pd.read_csv(full_csv)
    concepts = concept_cols(df)
    tier_map = build_tier_map(df, task_sequence)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Remove any stale task_*.csv files from previous runs with different sequences
    for stale in output_dir.glob("task_*.csv"):
        stale.unlink()

    # Only include train/good/ normals — test/good/ is held out for evaluation
    all_normal_df  = df[df["anomaly_type"] == "good"]
    normal_df = all_normal_df[
        all_normal_df["image_path"].str.contains("/train/good/", regex=False)
    ].copy()
    n_normal_before = len(all_normal_df)
    n_normal = len(normal_df)
    if n_normal < n_normal_before:
        print(f"  n_normal: {n_normal_before} → {n_normal}  (test/good/ excluded from task CSVs)")

    # Track which tier3 concepts have already appeared in prior tasks
    seen_tier3: set[str] = set()
    tasks: list[dict] = []

    for i, defect in enumerate(task_sequence):
        task_id = i + 1
        defect_df = df[df["anomaly_type"] == defect].copy()
        n_defect = len(defect_df)

        if task_id == 1:
            task_df = pd.concat([normal_df, defect_df], ignore_index=True)
            n_total = n_normal + n_defect
        else:
            task_df = defect_df.copy()
            n_total = n_defect

        # New concepts = tier3 for this defect not yet seen in any prior task
        this_tier3 = set(tier_map.get(f"tier3_{defect}", []))
        new_concepts = sorted(this_tier3 - seen_tier3)
        seen_tier3.update(this_tier3)

        csv_path = output_dir / f"task_{task_id}_{defect}.csv"
        task_df.to_csv(csv_path, index=False)

        tasks.append(
            {
                "task_id": task_id,
                "defect": defect,
                "n_images": n_total,
                "n_normal": n_normal if task_id == 1 else 0,
                "n_defect": n_defect,
                "n_concepts_total": len(concepts),
                "new_concepts": new_concepts,
                "csv_path": str(csv_path),
            }
        )

    # ── save JSON outputs ─────────────────────────────────────────────────────
    with open(output_dir / "concept_tier_map.json", "w") as f:
        json.dump(tier_map, f, indent=2)

    with open(output_dir / "task_sequence.json", "w") as f:
        json.dump(tasks, f, indent=2)

    return tasks, tier_map


# ── printing ──────────────────────────────────────────────────────────────────

def print_summary(tasks: list[dict], tier_map: dict) -> None:
    W = 72
    print("\n" + "=" * W)
    print("CONVAD-CL — Task CSV Builder (Scenario B, hazelnut)")
    print("=" * W)
    print(
        f"{'Task':<5} {'Defect':<8} {'Total':>6} {'Normal':>7} "
        f"{'Defect':>7} {'New concepts':>13}"
    )
    print("-" * W)
    for t in tasks:
        print(
            f"  T{t['task_id']}  {t['defect']:<8} {t['n_images']:>6} "
            f"{t['n_normal']:>7} {t['n_defect']:>7} {len(t['new_concepts']):>13}"
        )
    print("=" * W)

    total_concepts = tasks[0]["n_concepts_total"]
    n_tier1 = len(tier_map["tier1_normal"])
    n_tier2 = len(tier_map["tier2_generic"])
    tier3_counts = {
        k.replace("tier3_", ""): len(v)
        for k, v in tier_map.items()
        if k.startswith("tier3_")
    }
    n_tier3_unique = len(
        {c for k, v in tier_map.items() if k.startswith("tier3_") for c in v}
    )

    print(f"\nConcept vocabulary: {total_concepts} total")
    print(f"  Tier 1 — normal baseline :  {n_tier1:>3} concepts")
    print(f"  Tier 2 — generic/mixed   :  {n_tier2:>3} concepts")
    for defect, count in tier3_counts.items():
        print(f"  Tier 3 — {defect:<8}      :  {count:>3} concepts")
    print(f"           (unique tier3)  :  {n_tier3_unique:>3}")

    print("\nTier 3 concept names by defect:")
    for key, names in tier_map.items():
        if not key.startswith("tier3_") or not names:
            continue
        defect = key.replace("tier3_", "")
        print(f"\n  [{defect}]")
        for name in names:
            print(f"    · {name}")

    if tier_map["tier2_generic"]:
        print("\nTier 2 — generic/mixed concepts:")
        for name in tier_map["tier2_generic"]:
            print(f"    · {name}")


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build per-task CSVs for CONVAD-CL Scenario B"
    )
    parser.add_argument(
        "--full_csv",
        default="annotations/hazelnut/hazelnut.csv",
        help="Path to the full annotated CSV",
    )
    parser.add_argument(
        "--output_dir",
        default="annotations/hazelnut/cl_tasks",
        help="Directory to write per-task CSVs and JSON metadata",
    )
    parser.add_argument(
        "--task_sequence",
        nargs="+",
        default=["crack", "hole", "cut", "print"],
        help="Defect types in CL order",
    )
    args = parser.parse_args()

    tasks, tier_map = build_task_dataframes(
        Path(args.full_csv),
        args.task_sequence,
        Path(args.output_dir),
    )

    print_summary(tasks, tier_map)
    print(f"\nOutputs written to: {args.output_dir}/")
    print(f"  concept_tier_map.json")
    print(f"  task_sequence.json")
    for t in tasks:
        print(f"  {Path(t['csv_path']).name}")


def discover_defect_sequence(
    annotations_dir: Path,
    category: str,
    mvtec_root: Path,
    order: str = "alphabetical",
    defect_train_ratio: float = 0.8,
    seed: int = 42,
    explicit_sequence: list[str] | None = None,
) -> list[str]:
    """Auto-discover defect types and build per-task CSVs.

    Reads {category}.csv, finds all defect types, cross-references with
    MVTec test/ directories, sorts by 'order', builds task CSVs, and saves
    the updated task_sequence.json.

    Args:
        annotations_dir:    path to annotations/{category}/
        category:           e.g. "hazelnut"
        mvtec_root:         MVTec dataset root
        order:              "alphabetical" | "size_asc" | "size_desc"
        defect_train_ratio: fraction for training (rest held out)
        seed:               random seed for split
        explicit_sequence:  if provided, overrides order (comma-separated or list)

    Returns:
        Ordered list of defect type strings.
    """
    import numpy as _np

    annotations_dir = Path(annotations_dir)
    mvtec_root      = Path(mvtec_root)
    full_csv        = annotations_dir / f"{category}.csv"
    output_dir      = annotations_dir / "cl_tasks"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(full_csv)

    # All defect types in CSV
    csv_defects = [d for d in df["anomaly_type"].unique() if d != "good"]

    # Cross-reference with MVTec test/ directories
    test_root = mvtec_root / category / "test"
    mvtec_defects = {d.name for d in test_root.iterdir() if d.is_dir() and d.name != "good"}
    valid_defects = [d for d in csv_defects if d in mvtec_defects]

    if not valid_defects:
        raise RuntimeError(
            f"No matching defect types between {full_csv} and {test_root}"
        )

    # Count images per defect (for size-based ordering)
    defect_counts = {d: len(df[df["anomaly_type"] == d]) for d in valid_defects}

    # Determine sequence order
    if explicit_sequence is not None:
        if isinstance(explicit_sequence, str):
            explicit_sequence = [s.strip() for s in explicit_sequence.split(",")]
        missing = set(explicit_sequence) - set(valid_defects)
        if missing:
            raise ValueError(f"Explicit sequence contains unknown defects: {missing}")
        ordered = explicit_sequence
    elif order == "alphabetical":
        ordered = sorted(valid_defects)
    elif order == "size_asc":
        ordered = sorted(valid_defects, key=lambda d: defect_counts[d])
    elif order == "size_desc":
        ordered = sorted(valid_defects, key=lambda d: defect_counts[d], reverse=True)
    else:
        raise ValueError(f"Unknown order: {order!r}")

    print(f"Auto-discovered {len(ordered)} defect types for {category}:")

    # Build per-task CSVs
    tasks, tier_map = build_task_dataframes(full_csv, ordered, output_dir)

    # Print sequence with train/eval counts
    for t in tasks:
        n      = defect_counts[t["defect"]]
        n_tr   = max(1, int(n * defect_train_ratio))
        n_eval = n - n_tr
        print(f"  T{t['task_id']}: {t['defect']:<20} "
              f"({n} images, {n_tr} train / {n_eval} eval)")

    print(f"Sequence saved → {output_dir}/task_sequence.json")
    return ordered


if __name__ == "__main__":
    main()
