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
TIER3_DEFECT_THRESH = 0.30  # concept mean on defect rows must exceed this
TIER3_NORMAL_THRESH = 0.10  # concept mean on normal rows must be below this
TIER1_NORMAL_THRESH = 0.30  # concept mean on normal rows to count as "normal concept"


def concept_cols(df: pd.DataFrame) -> list[str]:
    """Return concept column names (all columns except metadata)."""
    return [c for c in df.columns if c not in _META_COLS]


def build_tier_map(df: pd.DataFrame, defect_types: list[str]) -> dict[str, list[str]]:
    """Classify every concept column into tier1 / tier2 / tier3_<defect>.

    Assignment priority:
      1. tier3_<defect>: mean_on_defect > TIER3_DEFECT_THRESH
                         AND mean_on_normal < TIER3_NORMAL_THRESH
         A concept can be tier3 for multiple defects simultaneously.
      2. tier1_normal:   not tier3 for any defect
                         AND mean_on_normal >= TIER1_NORMAL_THRESH
      3. tier2_generic:  everything else
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
            if (
                defect_means[c] > TIER3_DEFECT_THRESH
                and normal_means[c] < TIER3_NORMAL_THRESH
            ):
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

    normal_df = df[df["anomaly_type"] == "good"].copy()
    n_normal = len(normal_df)

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


if __name__ == "__main__":
    main()
