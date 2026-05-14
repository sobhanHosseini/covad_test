"""CONVAD-CL Scenario B — CLI entry point.

Usage:
  # Full CONCIL run (default)
  python -m main_scripts.run_cl \\
      --mvtec_root      /data/mvtec \\
      --annotations_dir ./annotations/hazelnut

  # Catastrophic-forgetting baseline (gradient descent, no replay)
  python -m main_scripts.run_cl \\
      --mvtec_root      /data/mvtec \\
      --annotations_dir ./annotations/hazelnut \\
      --baseline

  # Dry run (verify wiring, no inference)
  python -m main_scripts.run_cl \\
      --mvtec_root      /data/mvtec \\
      --annotations_dir ./annotations/hazelnut \\
      --dry_run

  # Resume CONCIL run from task N checkpoint
  python -m main_scripts.run_cl \\
      --mvtec_root      /data/mvtec \\
      --annotations_dir ./annotations/hazelnut \\
      --resume_from     2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from cl.cl_trainer              import CLTrainer
from cl.naive_sequential_trainer import NaiveSequentialTrainer
from cl.joint_training           import JointTrainer


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CONVAD-CL: Continual Learning for Visual Anomaly Detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mvtec_root",      required=True,
                   help="MVTec AD dataset root directory")
    p.add_argument("--annotations_dir", required=True,
                   help="annotations/<category>/ directory")
    p.add_argument("--category",        default="hazelnut")
    p.add_argument("--checkpoint_dir",  default=None,
                   help="checkpoint output directory (default: ./checkpoints/<category>[-baseline])")
    p.add_argument("--lambda_concept",  type=float, default=1e-4,
                   help="CONCIL ridge regularisation for concept heads")
    p.add_argument("--lambda_anomaly",  type=float, default=1e-4,
                   help="CONCIL ridge regularisation for anomaly head")
    p.add_argument("--coreset_size",    type=int,   default=10_000,
                   help="PatchCore memory bank coreset size")
    p.add_argument("--tau_percentile",  type=float, default=95.0,
                   help="percentile of normal s_novel scores used to set τ")
    p.add_argument("--batch_size",      type=int,   default=16,
                   help="images per DINOv2 extraction batch")
    p.add_argument("--naive_epochs",    type=int,   default=50,
                   help="[--baseline] gradient descent epochs per task")
    p.add_argument("--naive_lr",        type=float, default=1e-3,
                   help="[--baseline] Adam learning rate")
    p.add_argument("--baseline",        action="store_true",
                   help="run NaiveSequentialTrainer (catastrophic forgetting baseline)")
    p.add_argument("--joint",           action="store_true",
                   help="run JointTrainer (upper bound: all data at once)")
    p.add_argument("--resume_from",     type=int,   default=None, metavar="N",
                   help="resume CONCIL run from task N checkpoint")
    p.add_argument("--dry_run",         action="store_true",
                   help="verify data files and components, then exit")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if args.joint:
        default_ckpt = f"./checkpoints/{args.category}/joint"
    elif args.baseline:
        default_ckpt = f"./checkpoints/{args.category}-baseline"
    else:
        default_ckpt = f"./checkpoints/{args.category}"

    config = {
        "mvtec_root":      args.mvtec_root,
        "annotations_dir": args.annotations_dir,
        "category":        args.category,
        "checkpoint_dir":  args.checkpoint_dir or default_ckpt,
        "lambda_concept":  args.lambda_concept,
        "lambda_anomaly":  args.lambda_anomaly,
        "coreset_size":    args.coreset_size,
        "tau_percentile":  args.tau_percentile,
        "batch_size":      args.batch_size,
        "naive_epochs":    args.naive_epochs,
        "naive_lr":        args.naive_lr,
    }

    if args.joint:
        mode = "JOINT TRAINING (upper bound — all data at once)"
    elif args.baseline:
        mode = "NAIVE BASELINE (gradient descent, no replay)"
    else:
        mode = "CONCIL sequential"
    print(f"CONVAD-CL — {mode}")
    print("-" * 54)
    for k, v in config.items():
        print(f"  {k:<20} {v}")
    print()

    task_seq_path = Path(args.annotations_dir) / "cl_tasks" / "task_sequence.json"
    if not task_seq_path.exists():
        print(f"Error: task_sequence.json not found at {task_seq_path}")
        print("Run:  python -m cl.task_csv_builder  to generate task CSVs first.")
        sys.exit(1)

    # Instantiate the requested trainer
    if args.joint:
        trainer = JointTrainer(config)
    elif args.baseline:
        trainer = NaiveSequentialTrainer(config)
    else:
        trainer = CLTrainer(config)

    if args.dry_run and not args.joint:
        ok = trainer.dry_run(str(task_seq_path))
        print("\nDry run complete." if ok else "\nDry run found issues — see above.")
        sys.exit(0 if ok else 1)

    if args.resume_from is not None and not args.baseline:
        ckpt_dir = Path(config["checkpoint_dir"]) / f"task_{args.resume_from}"
        if not ckpt_dir.exists():
            print(f"Error: checkpoint not found: {ckpt_dir}")
            ckpt_root = Path(config["checkpoint_dir"])
            if ckpt_root.exists():
                existing = [p.name for p in sorted(ckpt_root.iterdir()) if p.is_dir()]
                print("Available:", existing or "(none)")
            sys.exit(1)

    try:
        if args.resume_from is not None and not args.baseline and not args.joint:
            log = trainer.resume(args.resume_from, str(task_seq_path))
        else:
            result = trainer.run(str(task_seq_path))
            # JointTrainer returns a dict, others return ContinualLog
            if hasattr(result, "summary_table"):
                print(result.summary_table())
    except KeyboardInterrupt:
        print("\nInterrupted. Checkpoint saved up to last completed task.")
        sys.exit(130)

    sys.exit(0)


if __name__ == "__main__":
    main()
