#!/usr/bin/env python3
"""
run_holdouts.py — Auto holdout runner.

Waits for a Stage 3 checkpoint to appear (or uses an existing one),
then automatically generates all holdout CSV variants by running
Stage 4 + CSV only (--load_from_stage 3) for each defect type.

Usage:
    # Wait for a running Stage 3 to finish, then run holdouts:
    python run_holdouts.py --category hazelnut --dataset_path /path/to/mvtec \\
        --base_save_path ./annotations/hazelnut_v6_bge.csv

    # Stage 3 already done, run holdouts immediately:
    python run_holdouts.py --category hazelnut --dataset_path /path/to/mvtec \\
        --base_save_path ./annotations/hazelnut_v6_bge.csv --no_wait

    # Custom defect list (default: auto-discovered from dataset folder):
    python run_holdouts.py --category hazelnut --dataset_path /path/to/mvtec \\
        --base_save_path ./annotations/hazelnut_v6_bge.csv \\
        --defects crack cut hole print

The script polls every 60 seconds for the Stage 3 checkpoint file.
When found, it fires holdout runs sequentially (or in parallel with --parallel).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def discover_defects(dataset_path: str, category: str) -> list[str]:
    """Auto-discover defect types from the MVTec folder structure."""
    test_dir = Path(dataset_path) / category / "test"
    if not test_dir.exists():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")
    defects = sorted(
        d.name for d in test_dir.iterdir()
        if d.is_dir() and d.name != "good"
    )
    return defects


def stage3_checkpoint_path(base_save_path: str, category: str) -> Path:
    """Return the expected Stage 3 checkpoint path."""
    return Path(base_save_path).parent / f"stage3_annotations.json"


def wait_for_stage3(checkpoint: Path, poll_interval: int = 60) -> None:
    """Block until the Stage 3 checkpoint file exists."""
    log.info("Waiting for Stage 3 checkpoint: %s", checkpoint)
    log.info("Polling every %d seconds... (Ctrl+C to abort)", poll_interval)
    while not checkpoint.exists():
        time.sleep(poll_interval)
        log.info("  Still waiting... (%s not yet found)", checkpoint.name)
    log.info("Stage 3 checkpoint found! Starting holdout runs.")


def run_holdout(
    defect: str,
    dataset_path: str,
    category: str,
    base_save_path: str,
    extra_args: list[str],
) -> bool:
    """
    Run a single holdout variant using --load_from_stage 3.
    Returns True on success, False on failure.
    """
    # Output CSV path: insert holdout name before .csv
    base = Path(base_save_path)
    save_path = base.parent / f"{base.stem}_holdout_{defect}.csv"

    cmd = [
        sys.executable, "-m", "annotation_pipeline",
        "--dataset_path", dataset_path,
        "--category",     category,
        "--save_path",    str(save_path),
        "--load_from_stage", "3",
        "--holdout_defect",  defect,
    ] + extra_args

    log.info("=" * 60)
    log.info("Running holdout: %s", defect)
    log.info("Output: %s", save_path)
    log.info("Command: %s", " ".join(cmd))
    log.info("=" * 60)

    result = subprocess.run(cmd, cwd=Path.cwd())
    if result.returncode == 0:
        log.info("✓ Holdout '%s' complete → %s", defect, save_path)
        return True
    else:
        log.error("✗ Holdout '%s' FAILED (return code %d)", defect, result.returncode)
        return False


def main() -> None:
    p = argparse.ArgumentParser(description="Auto holdout runner for annotation pipeline")

    p.add_argument("--dataset_path",   required=True,
                   help="Root MVTec dataset directory")
    p.add_argument("--category",       required=True,
                   help="Category name (e.g. hazelnut)")
    p.add_argument("--base_save_path", required=True,
                   help="Full run CSV path (e.g. ./annotations/hazelnut_v6_bge.csv). "
                        "Holdout CSVs are saved next to it with _holdout_<defect> suffix.")
    p.add_argument("--defects",        nargs="+", default=None,
                   help="Defect types to hold out. Default: auto-discover from dataset.")
    p.add_argument("--no_wait",        action="store_true",
                   help="Don't wait for Stage 3 — assume checkpoint already exists.")
    p.add_argument("--poll_interval",  type=int, default=60,
                   help="Polling interval in seconds when waiting (default: 60).")

    # Pass-through args to the pipeline (Stage 4 parameters)
    p.add_argument("--min_defect_freq",          type=float, default=0.12)
    p.add_argument("--stage4_cluster_threshold", type=float, default=0.65)
    p.add_argument("--skip_stage4",              action="store_true")
    p.add_argument("--model_name",               default="gemma4:e4b")
    p.add_argument("--ollama_host",              default="http://localhost:6000")
    p.add_argument("--skip_clip_filter",         action="store_true")
    p.add_argument("--min_cohen_d",              type=float, default=0.10)

    args = p.parse_args()

    # Discover defects
    defects = args.defects or discover_defects(args.dataset_path, args.category)
    log.info("Holdout targets: %s", defects)

    # Wait for Stage 3 checkpoint if needed
    checkpoint = stage3_checkpoint_path(args.base_save_path, args.category)
    if not args.no_wait:
        if checkpoint.exists():
            log.info("Stage 3 checkpoint already exists — starting immediately.")
        else:
            wait_for_stage3(checkpoint, args.poll_interval)
    else:
        if not checkpoint.exists():
            log.error("--no_wait set but checkpoint not found: %s", checkpoint)
            sys.exit(1)

    # Build pass-through args for pipeline
    extra_args = [
        "--min_defect_freq",          str(args.min_defect_freq),
        "--stage4_cluster_threshold", str(args.stage4_cluster_threshold),
        "--model_name",               args.model_name,
        "--ollama_host",              args.ollama_host,
        "--min_cohen_d",              str(args.min_cohen_d),
    ]
    if args.skip_stage4:
        extra_args.append("--skip_stage4")
    if args.skip_clip_filter:
        extra_args.append("--skip_clip_filter")

    # Run all holdouts sequentially
    results: dict[str, bool] = {}
    for defect in defects:
        success = run_holdout(
            defect=defect,
            dataset_path=args.dataset_path,
            category=args.category,
            base_save_path=args.base_save_path,
            extra_args=extra_args,
        )
        results[defect] = success

    # Summary
    log.info("\n" + "=" * 60)
    log.info("HOLDOUT RUN SUMMARY — %s", args.category)
    log.info("=" * 60)
    for defect, ok in results.items():
        status = "✓ OK  " if ok else "✗ FAIL"
        out = Path(args.base_save_path)
        csv = out.parent / f"{out.stem}_holdout_{defect}.csv"
        log.info("  %s  %-15s  → %s", status, defect, csv)
    log.info("=" * 60)

    failed = [d for d, ok in results.items() if not ok]
    if failed:
        log.error("Failed holdouts: %s", failed)
        sys.exit(1)
    else:
        log.info("All %d holdout variants completed successfully.", len(defects))


if __name__ == "__main__":
    main()