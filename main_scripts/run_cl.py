"""CONVAD-CL Scenario B — CLI entry point.

Usage — minimal (everything auto-derived):
  python -m main_scripts.run_cl \\
      --mvtec_root /data/mvtec \\
      --category   hazelnut

Usage — with options:
  python -m main_scripts.run_cl \\
      --mvtec_root /data/mvtec \\
      --category   capsule \\
      --baseline

  python -m main_scripts.run_cl \\
      --mvtec_root /data/mvtec \\
      --category   hazelnut \\
      --sequence   "print,crack,hole,cut"

  python -m main_scripts.run_cl \\
      --mvtec_root /data/mvtec \\
      --category   hazelnut \\
      --joint      --dry_run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


class _Tee:
    """Write all stdout/stderr to both the terminal and a log file."""

    def __init__(self, log_path: Path, stream):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._file   = open(log_path, "w", buffering=1)
        self._stream = stream

    def write(self, data):
        self._stream.write(data)
        self._file.write(data)

    def flush(self):
        self._stream.flush()
        self._file.flush()

    def close(self):
        self._file.close()

    # Proxy everything else to the original stream
    def __getattr__(self, name):
        return getattr(self._stream, name)

from cl.cl_trainer              import CLTrainer
from cl.naive_sequential_trainer import NaiveSequentialTrainer
from cl.joint_training           import JointTrainer
from cl.task_csv_builder         import discover_defect_sequence


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CONVAD-CL: Continual Learning for Visual Anomaly Detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # ── required ──────────────────────────────────────────────────────────────
    p.add_argument("--mvtec_root",  required=True,
                   help="MVTec AD dataset root directory")
    p.add_argument("--category",    required=True,
                   help="MVTec category name, e.g. hazelnut, capsule, bottle")

    # ── path overrides (auto-derived from category if omitted) ────────────────
    p.add_argument("--annotations_root", default="./annotations",
                   help="parent of per-category annotation dirs")
    p.add_argument("--checkpoints_root", default="./checkpoints",
                   help="parent of per-category checkpoint dirs")
    p.add_argument("--annotations_dir",  default=None,
                   help="override: full path to annotations/{category}/ dir")
    p.add_argument("--checkpoint_dir",   default=None,
                   help="override: full checkpoint output dir")

    # ── sequence control ──────────────────────────────────────────────────────
    p.add_argument("--sequence_order",   default="alphabetical",
                   choices=["alphabetical","size_asc","size_desc"],
                   help="ordering for auto-discovered defect types")
    p.add_argument("--sequence", default=None,
                   help='explicit comma-separated sequence, e.g. "print,crack,hole,cut"')

    # ── training hyperparameters ──────────────────────────────────────────────
    p.add_argument("--defect_train_ratio", type=float, default=0.8,
                   help="fraction of defect images used for training (0.8=paper)")
    p.add_argument("--lambda_concept",  type=float, default=1e-4)
    p.add_argument("--lambda_anomaly",  type=float, default=1e-4)
    p.add_argument("--coreset_size",    type=int,   default=10_000)
    p.add_argument("--tau_percentile",  type=float, default=95.0)
    p.add_argument("--batch_size",      type=int,   default=16)
    p.add_argument("--naive_epochs",    type=int,   default=50,
                   help="[--baseline] Adam epochs per task")
    p.add_argument("--naive_lr",        type=float, default=1e-3,
                   help="[--baseline] Adam learning rate")

    # ── mode flags ────────────────────────────────────────────────────────────
    p.add_argument("--baseline", action="store_true",
                   help="NaiveSequentialTrainer (catastrophic forgetting baseline)")
    p.add_argument("--joint",    action="store_true",
                   help="JointTrainer (upper bound — all data at once)")
    p.add_argument("--resume_from", type=int, default=None, metavar="N",
                   help="resume CONCIL run from task N checkpoint")
    p.add_argument("--dry_run",  action="store_true",
                   help="verify data + components, no training")

    return p.parse_args()


def build_config(args: argparse.Namespace) -> dict:
    # Derive paths from category if not explicitly overridden
    ann_dir  = Path(args.annotations_dir) if args.annotations_dir \
               else Path(args.annotations_root) / args.category
    if args.joint:
        suffix = "-joint-final"
    elif args.baseline:
        suffix = "-baseline-final"
    else:
        suffix = "-final"
    ckpt_dir = Path(args.checkpoint_dir) if args.checkpoint_dir \
               else Path(args.checkpoints_root) / f"{args.category}{suffix}"

    return {
        "mvtec_root":          args.mvtec_root,
        "annotations_dir":     str(ann_dir),
        "category":            args.category,
        "checkpoint_dir":      str(ckpt_dir),
        "lambda_concept":      args.lambda_concept,
        "lambda_anomaly":      args.lambda_anomaly,
        "coreset_size":        args.coreset_size,
        "tau_percentile":      args.tau_percentile,
        "batch_size":          args.batch_size,
        "defect_train_ratio":  args.defect_train_ratio,
        "naive_epochs":        args.naive_epochs,
        "naive_lr":            args.naive_lr,
    }


def main() -> None:
    args   = _parse_args()
    config = build_config(args)

    # ── auto-tee stdout+stderr to run_log.txt inside checkpoint dir ───────────
    if not args.dry_run:
        ckpt_dir = Path(config["checkpoint_dir"])
        log_path = ckpt_dir / "run_log.txt"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        _tee_out = _Tee(log_path, sys.stdout)
        _tee_err = _Tee(log_path, sys.stderr)
        sys.stdout, sys.stderr = _tee_out, _tee_err
        # Print after redirecting so the message also goes to the log
        print(f"Logging to: {log_path}")

    # ── mode label ────────────────────────────────────────────────────────────
    if args.joint:
        mode = "JOINT TRAINING (upper bound)"
    elif args.baseline:
        mode = "NAIVE BASELINE (gradient descent, no replay)"
    else:
        mode = "CONCIL sequential"

    pct = int(args.defect_train_ratio * 100)
    print(f"CONVAD-CL — {mode}")
    print(f"Category    : {args.category}")
    print(f"Defect split: {pct}% train / {100-pct}% held-out (seed=42)")
    print(f"Annotations : {config['annotations_dir']}")
    print(f"Checkpoints : {config['checkpoint_dir']}")
    print()

    # ── auto-discover defect sequence (builds task CSVs if needed) ────────────
    ann_dir = Path(config["annotations_dir"])
    explicit = [s.strip() for s in args.sequence.split(",")] if args.sequence else None
    discover_defect_sequence(
        annotations_dir    = ann_dir,
        category           = args.category,
        mvtec_root         = Path(args.mvtec_root),
        order              = args.sequence_order,
        defect_train_ratio = args.defect_train_ratio,
        explicit_sequence  = explicit,
    )

    task_seq_path = ann_dir / "cl_tasks" / "task_sequence.json"

    # ── instantiate trainer ───────────────────────────────────────────────────
    if args.joint:
        trainer = JointTrainer(config)
    elif args.baseline:
        trainer = NaiveSequentialTrainer(config)
    else:
        trainer = CLTrainer(config)

    # ── dry run ───────────────────────────────────────────────────────────────
    if args.dry_run and not args.joint:
        ok = trainer.dry_run(str(task_seq_path))
        print("\nDry run complete." if ok else "\nDry run found issues — see above.")
        sys.exit(0 if ok else 1)

    # ── resume check ─────────────────────────────────────────────────────────
    if args.resume_from is not None and not args.baseline and not args.joint:
        ckpt = Path(config["checkpoint_dir"]) / f"task_{args.resume_from}"
        if not ckpt.exists():
            print(f"Error: checkpoint not found: {ckpt}")
            ckpt_root = Path(config["checkpoint_dir"])
            if ckpt_root.exists():
                existing = [d.name for d in sorted(ckpt_root.iterdir()) if d.is_dir()]
                print("Available:", existing or "(none)")
            sys.exit(1)

    # ── run ───────────────────────────────────────────────────────────────────
    try:
        if args.resume_from is not None and not args.baseline and not args.joint:
            result = trainer.resume(args.resume_from, str(task_seq_path))
        else:
            result = trainer.run(str(task_seq_path))
        if hasattr(result, "summary_table"):
            print(result.summary_table())
    except KeyboardInterrupt:
        print("\nInterrupted. Checkpoint saved up to last completed task.")
        sys.exit(130)

    sys.exit(0)


if __name__ == "__main__":
    main()
