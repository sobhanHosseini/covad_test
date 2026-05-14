"""CONVAD-CL Scenario B — CLI entry point.

Usage:
  # Full run from Task 1
  python -m main_scripts.run_cl \\
      --mvtec_root      /data/mvtec \\
      --annotations_dir ./annotations/hazelnut

  # Dry run (verify wiring, no inference)
  python -m main_scripts.run_cl \\
      --mvtec_root      /data/mvtec \\
      --annotations_dir ./annotations/hazelnut \\
      --dry_run

  # Resume from a completed task checkpoint
  python -m main_scripts.run_cl \\
      --mvtec_root      /data/mvtec \\
      --annotations_dir ./annotations/hazelnut \\
      --resume_from     2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from cl.cl_trainer import CLTrainer


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
                   help="checkpoint output directory (default: ./checkpoints/<category>)")
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
    p.add_argument("--resume_from",     type=int,   default=None, metavar="N",
                   help="resume from task N checkpoint (task N must be completed)")
    p.add_argument("--dry_run",         action="store_true",
                   help="verify data files and components, then exit")
    return p.parse_args()


def main() -> None:
    args = p.parse_args() if False else _parse_args()   # keeps linters quiet

    config = {
        "mvtec_root":      args.mvtec_root,
        "annotations_dir": args.annotations_dir,
        "category":        args.category,
        "checkpoint_dir":  args.checkpoint_dir or f"./checkpoints/{args.category}",
        "lambda_concept":  args.lambda_concept,
        "lambda_anomaly":  args.lambda_anomaly,
        "coreset_size":    args.coreset_size,
        "tau_percentile":  args.tau_percentile,
        "batch_size":      args.batch_size,
    }

    print("CONVAD-CL — Configuration")
    print("-" * 44)
    for k, v in config.items():
        print(f"  {k:<20} {v}")
    print()

    task_seq_path = Path(args.annotations_dir) / "cl_tasks" / "task_sequence.json"
    if not task_seq_path.exists():
        print(f"Error: task_sequence.json not found at {task_seq_path}")
        print("Run:  python -m cl.task_csv_builder  to generate task CSVs first.")
        sys.exit(1)

    trainer = CLTrainer(config)

    if args.dry_run:
        ok = trainer.dry_run(str(task_seq_path))
        print("\nDry run complete." if ok else "\nDry run found issues — see above.")
        sys.exit(0 if ok else 1)

    if args.resume_from is not None:
        ckpt_dir = Path(config["checkpoint_dir"]) / f"task_{args.resume_from}"
        if not ckpt_dir.exists():
            print(f"Error: checkpoint not found: {ckpt_dir}")
            ckpt_root = Path(config["checkpoint_dir"])
            if ckpt_root.exists():
                existing = sorted(ckpt_root.iterdir())
                if existing:
                    print("Available checkpoints:")
                    for p in existing:
                        print(f"  {p.name}")
                else:
                    print("No checkpoints found in checkpoint_dir.")
            sys.exit(1)

    try:
        if args.resume_from is not None:
            log = trainer.resume(args.resume_from, str(task_seq_path))
        else:
            log = trainer.run(str(task_seq_path))
    except KeyboardInterrupt:
        print("\nInterrupted. Checkpoint saved up to last completed task.")
        sys.exit(130)

    print(log.summary_table())
    sys.exit(0)


if __name__ == "__main__":
    main()
