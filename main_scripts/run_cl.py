"""Entry point for the CONVAD-CL sequential experiment (Scenario B).

Placeholder — implement after CLTrainer and CLEvaluator are complete.

Usage (target):
    python -m main_scripts.run_cl \
        --full_csv     annotations/hazelnut/hazelnut.csv \
        --mvtec_root   /path/to/mvtec \
        --category     hazelnut \
        --task_sequence crack hole cut print \
        --checkpoint_dir checkpoints/hazelnut \
        --device       cuda

TODO — implement:
    - parse args
    - build per-task DataFrames via task_csv_builder
    - instantiate DINOv2Extractor, PatchCoreMemory, ConceptHeads, LinearAnomalyHead
    - run CLTrainer.run_all()
    - run CLEvaluator.summarise()
"""

import argparse


def main():
    raise NotImplementedError("run_cl.py not yet implemented")


if __name__ == "__main__":
    main()
