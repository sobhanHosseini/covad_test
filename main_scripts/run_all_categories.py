"""Run CONCIL + naive + joint for multiple MVTec categories sequentially.

All output is logged automatically to ./results/run_all_log.txt.
Intermediate per-category results saved to ./results/{category}_results.json.
No pipes or tee needed — just run:

    python -m main_scripts.run_all_categories \\
        --mvtec_root /home/sobhan_hosseini/datasets/mvtec \\
        --categories bottle capsule hazelnut metal_nut screw
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np


# ── logging setup ─────────────────────────────────────────────────────────────

def _setup_logging(results_dir: Path) -> logging.Logger:
    results_dir.mkdir(parents=True, exist_ok=True)
    log_path = results_dir / "run_all_log.txt"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.FileHandler(log_path, mode="a"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logger = logging.getLogger("run_all")
    logger.info(f"Log file: {log_path.resolve()}")
    return logger


# ── argument parsing ──────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CONVAD-CL: Run all categories (CONCIL + naive + joint)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mvtec_root",  required=True)
    p.add_argument("--categories",  nargs="+", required=True)
    p.add_argument("--annotations_root",   default="./annotations")
    p.add_argument("--checkpoints_root",   default="./checkpoints")
    p.add_argument("--results_dir",        default="./results")
    p.add_argument("--defect_train_ratio", type=float, default=0.8)
    p.add_argument("--sequence_order",     default="alphabetical",
                   choices=["alphabetical","size_asc","size_desc"])

    # ── mode flags ────────────────────────────────────────────────────────────
    p.add_argument("--skip_concil",  action="store_true", help="skip CONCIL runs")
    p.add_argument("--skip_naive",   action="store_true", help="skip naive baseline runs")
    p.add_argument("--skip_joint",   action="store_true", help="skip joint training runs")
    p.add_argument("--resume",       action="store_true",
                   help="skip categories whose results already exist in results_dir")
    p.add_argument("--dry_run",      action="store_true",
                   help="verify all data and print plan, then exit without training")
    return p.parse_args()


# ── helpers ───────────────────────────────────────────────────────────────────

def _ensure_dirs(categories: list[str], checkpoints_root: str,
                 results_dir: Path, logger: logging.Logger) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    for cat in categories:
        for suffix in ["-final", "-baseline-final", "-joint-final"]:
            d = Path(checkpoints_root) / f"{cat}{suffix}"
            d.mkdir(parents=True, exist_ok=True)
    logger.info("All output directories created ✓")


def _run_one(cmd: list[str], label: str, logger: logging.Logger) -> int:
    logger.info(f"  ▶ {label}")
    result = subprocess.run(cmd, cwd=str(Path(__file__).parent.parent))
    rc = result.returncode
    if rc == 0:
        logger.info(f"  ✓ {label} done")
    elif rc == 130:
        logger.info(f"  ⚠ {label} interrupted (Ctrl-C)")
    else:
        logger.info(f"  ✗ {label} failed (exit {rc})")
    return rc


def _load_cauc(log_path: Path) -> dict[str, float]:
    if not log_path.exists():
        return {}
    with open(log_path) as f:
        results = json.load(f)["results"]
    best: dict[str, dict] = {}
    for r in results:
        d = r["defect_name"]
        if d not in best or r["evaluated_after_task"] > best[d]["evaluated_after_task"]:
            best[d] = r
    return {d: r["c_auc_mean"] for d, r in best.items()}


def _load_bwt(log_path: Path) -> float:
    if not log_path.exists():
        return float("nan")
    from evaluators.evaluator_cl import TaskEvalResult, ContinualLog
    with open(log_path) as f:
        d = json.load(f)
    log = ContinualLog(results=[TaskEvalResult(**r) for r in d["results"]])
    return log.mean_concept_bwt()


def _load_joint_cauc(joint_path: Path) -> float:
    if not joint_path.exists():
        return float("nan")
    with open(joint_path) as f:
        d = json.load(f)
    vals = list(d.get("joint_c_auc", {}).values())
    return float(np.nanmean(vals)) if vals else float("nan")


def _discover_sequence(cat: str, mvtec_root: str, ann_root: str,
                       order: str, ratio: float, logger: logging.Logger) -> list[str]:
    from cl.task_csv_builder import discover_defect_sequence
    seq = discover_defect_sequence(
        annotations_dir    = Path(ann_root) / cat,
        category           = cat,
        mvtec_root         = Path(mvtec_root),
        order              = order,
        defect_train_ratio = ratio,
    )
    return seq


# ── dry run verification ──────────────────────────────────────────────────────

def _dry_run_category(cat: str, args: argparse.Namespace,
                      logger: logging.Logger) -> bool:
    """Verify data files and print task plan. Returns True if all OK."""
    from cl.task_csv_builder import discover_defect_sequence
    import pandas as pd

    ann_dir   = Path(args.annotations_root) / cat
    full_csv  = ann_dir / f"{cat}.csv"
    mvtec_cat = Path(args.mvtec_root) / cat

    issues: list[str] = []

    if not full_csv.exists():
        issues.append(f"    MISSING annotation CSV: {full_csv}")
    if not mvtec_cat.exists():
        issues.append(f"    MISSING MVTec dir: {mvtec_cat}")

    if issues:
        for s in issues: logger.info(s)
        return False

    seq = _discover_sequence(cat, args.mvtec_root, args.annotations_root,
                             args.sequence_order, args.defect_train_ratio, logger)
    df = pd.read_csv(full_csv)
    n_total = len(df[df["anomaly_type"] == "good"])
    n_train_normals = len(df[
        (df["anomaly_type"] == "good") &
        df["image_path"].str.contains("/train/good/", regex=False)
    ])

    logger.info(f"  Category:  {cat}  ({len(seq)} tasks)")
    logger.info(f"  Normals:   {n_total} total → {n_train_normals} train/good (eval: test/good)")
    for i, d in enumerate(seq, 1):
        n = len(df[df["anomaly_type"] == d])
        nt = max(1, int(n * args.defect_train_ratio))
        ne = n - nt
        logger.info(f"    T{i}: {d:<22} ({n} images, {nt} train / {ne} eval)")

    # Check MVTec test dirs
    for d in seq:
        p = mvtec_cat / "test" / d
        if not p.exists():
            issues.append(f"    MISSING: {p}")

    if issues:
        for s in issues: logger.info(s)
        return False

    ckpt_r = args.checkpoints_root
    for label, suf in [("CONCIL",  "-final"),
                       ("Naive",   "-baseline-final"),
                       ("Joint",   "-joint-final")]:
        d = Path(ckpt_r) / f"{cat}{suf}"
        logger.info(f"  {label:<8} → {d}/")

    return True


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args       = _parse_args()
    results_dir = Path(args.results_dir)
    logger     = _setup_logging(results_dir)

    logger.info("=" * 62)
    logger.info("CONVAD-CL — Run all categories")
    logger.info(f"Categories : {args.categories}")
    logger.info(f"Results    : {results_dir.resolve()}")
    logger.info(f"Log        : {results_dir / 'run_all_log.txt'}")
    logger.info("=" * 62)

    # ── dry run ───────────────────────────────────────────────────────────────
    if args.dry_run:
        logger.info("\nDRY RUN — verifying plan only\n")
        all_ok = True
        for cat in args.categories:
            logger.info(f"\n{'─'*50}")
            logger.info(f"Category: {cat.upper()}")
            ok = _dry_run_category(cat, args, logger)
            if not ok:
                logger.info(f"  ✗ {cat}: issues found")
                all_ok = False
            else:
                logger.info(f"  All data files found ✓")

        logger.info("\n" + "=" * 62)
        if all_ok:
            logger.info("All checks passed ✓  Ready — remove --dry_run to start.")
        else:
            logger.info("Some categories have issues — fix before running.")
        return

    # ── full run ──────────────────────────────────────────────────────────────
    _ensure_dirs(args.categories, args.checkpoints_root, results_dir, logger)

    base_cmd = [
        sys.executable, "-m", "main_scripts.run_cl",
        "--mvtec_root",         args.mvtec_root,
        "--annotations_root",   args.annotations_root,
        "--checkpoints_root",   args.checkpoints_root,
        "--defect_train_ratio", str(args.defect_train_ratio),
        "--sequence_order",     args.sequence_order,
    ]

    summary: list[dict] = []

    for cat in args.categories:
        logger.info(f"\n{'═'*62}")
        logger.info(f"CATEGORY: {cat.upper()}")
        logger.info(f"{'═'*62}")

        intermediate = results_dir / f"{cat}_results.json"

        # Resume check
        if args.resume and intermediate.exists():
            logger.info(f"  --resume: loading existing results for {cat}")
            with open(intermediate) as f:
                summary.append(json.load(f))
            continue

        log_concil = Path(args.checkpoints_root) / f"{cat}-final"       / "log_final.json"
        log_naive  = Path(args.checkpoints_root) / f"{cat}-baseline-final" / "log_final.json"
        log_joint  = Path(args.checkpoints_root) / f"{cat}-joint-final"  / "joint_training_results.json"

        interrupted = False

        if not args.skip_concil:
            rc = _run_one(base_cmd + ["--category", cat],
                          f"{cat} — CONCIL", logger)
            if rc == 130: interrupted = True

        if not interrupted and not args.skip_naive:
            rc = _run_one(base_cmd + ["--category", cat, "--baseline"],
                          f"{cat} — Naive baseline", logger)
            if rc == 130: interrupted = True

        if not interrupted and not args.skip_joint:
            rc = _run_one(base_cmd + ["--category", cat, "--joint"],
                          f"{cat} — Joint training", logger)
            if rc == 130: interrupted = True

        # Collect metrics
        concil_cauc = _load_cauc(log_concil)
        naive_cauc  = _load_cauc(log_naive)
        n_tasks     = len(concil_cauc)

        cat_result = {
            "category":      cat,
            "n_tasks":       n_tasks,
            "concil_c_auc":  float(np.nanmean(list(concil_cauc.values()))) if concil_cauc else float("nan"),
            "naive_c_auc":   float(np.nanmean(list(naive_cauc.values())))  if naive_cauc  else float("nan"),
            "joint_c_auc":   _load_joint_cauc(log_joint),
            "bwt_concil":    _load_bwt(log_concil),
            "bwt_naive":     _load_bwt(log_naive),
            "concil_per_defect": concil_cauc,
            "naive_per_defect":  naive_cauc,
        }
        summary.append(cat_result)

        # Save intermediate results immediately
        with open(intermediate, "w") as f:
            json.dump(cat_result, f, indent=2)
        logger.info(f"  Intermediate results → {intermediate}")

        if interrupted:
            logger.info("Interrupted — stopping run_all_categories.")
            break

    # ── final summary table ───────────────────────────────────────────────────
    W = 84
    logger.info("\n" + "═"*W)
    logger.info("  SUMMARY — All categories  (Standard BWT, Lopez-Paz 2017)")
    logger.info("═"*W)
    hdr = (f"  {'Category':<12} {'Tasks':>5}  {'Joint':>7}  {'CONCIL':>7}  "
           f"{'Naive':>7}  {'BWT(C)':>8}  {'BWT(N)':>8}")
    logger.info(hdr)
    logger.info(f"  {'-'*11:<12} {'-'*4:>5}  {'-'*6:>7}  {'-'*6:>7}  "
                f"{'-'*6:>7}  {'-'*7:>8}  {'-'*7:>8}")
    for s in summary:
        logger.info(
            f"  {s['category']:<12} {s['n_tasks']:>5}  "
            f"{s['joint_c_auc']:>7.4f}  {s['concil_c_auc']:>7.4f}  "
            f"{s['naive_c_auc']:>7.4f}  {s['bwt_concil']:>+8.4f}  "
            f"{s['bwt_naive']:>+8.4f}"
        )
    if summary:
        jm = np.nanmean([s["joint_c_auc"]  for s in summary])
        cm = np.nanmean([s["concil_c_auc"] for s in summary])
        nm = np.nanmean([s["naive_c_auc"]  for s in summary])
        bc = np.nanmean([s["bwt_concil"]   for s in summary])
        bn = np.nanmean([s["bwt_naive"]    for s in summary])
        logger.info(f"  {'─'*11:<12} {'─'*4:>5}  {'─'*6:>7}  {'─'*6:>7}  "
                    f"{'─'*6:>7}  {'─'*7:>8}  {'─'*7:>8}")
        logger.info(f"  {'Average':<12} {'─':>5}  {jm:>7.4f}  {cm:>7.4f}  "
                    f"{nm:>7.4f}  {bc:>+8.4f}  {bn:>+8.4f}")
    logger.info("═"*W)

    # ── save results ──────────────────────────────────────────────────────────
    summary_json = results_dir / "all_categories_summary.json"
    summary_csv  = results_dir / "all_categories_summary.csv"

    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"\nFull summary → {summary_json}")

    import csv
    with open(summary_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "category","n_tasks","joint_c_auc","concil_c_auc",
            "naive_c_auc","bwt_concil","bwt_naive",
        ])
        w.writeheader()
        for s in summary:
            w.writerow({k: s[k] for k in w.fieldnames})
    logger.info(f"CSV summary  → {summary_csv}")


if __name__ == "__main__":
    main()
