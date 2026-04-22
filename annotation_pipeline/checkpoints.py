"""
checkpoints.py — Save and load intermediate stage outputs to disk.

This is the key enabler for:
  1. Fast holdout generation: run Stages 1-3 once, regenerate Stage 4 + CSV
     for each holdout variant in minutes instead of hours.
  2. Continual learning: append new defect annotations to Stage 3 checkpoint,
     then rerun Stage 4 only.
  3. Crash recovery: resume from last completed stage.

Checkpoint files per category (all saved next to the output CSV):
  {category}_stage1_raw.json          — raw per-image concept extractions
  {category}_stage2_normal_dict.json  — refined normal concept dictionary
  {category}_stage3_annotations.json  — all per-image annotations (normal + defect)
  {category}_clip_normal.npz          — CLIP embedding cache for normal images
  
  Folder structure (auto-created):
  annotations/
    hazelnut/
      stage1_raw.json
      stage2_normal_dict.json
      stage3_annotations.json
      clip_normal.npz
      hazelnut_final.csv
    capsule/
      stage1_raw.json
      ...

Each category gets its own subfolder — parallel runs never conflict.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


def _checkpoint_dir(save_path: str, category: str) -> Path:
    """
    Return the category subfolder for checkpoints, creating it if needed.

    Handles both:
      ./annotations/hazelnut_final.csv     → ./annotations/hazelnut/
      ./annotations/hazelnut/final.csv     → ./annotations/hazelnut/
    """
    p = Path(save_path).parent
    if p.name != category:
        p = p / category
    p.mkdir(parents=True, exist_ok=True)
    return p


# ── Path helpers ──────────────────────────────────────────────────────────────

def stage1_path(save_path: str, category: str) -> Path:
    return _checkpoint_dir(save_path, category) / "stage1_raw.json"


def stage2_path(save_path: str, category: str) -> Path:
    return _checkpoint_dir(save_path, category) / "stage2_normal_dict.json"


def stage3_path(save_path: str, category: str) -> Path:
    return _checkpoint_dir(save_path, category) / "stage3_annotations.json"


def clip_cache_path(save_path: str, category: str) -> Path:
    return _checkpoint_dir(save_path, category) / "clip_normal.npz"


# ── Save helpers ──────────────────────────────────────────────────────────────

def save_stage1(raw_concepts: list[dict], save_path: str, category: str) -> None:
    p = stage1_path(save_path, category)
    with open(p, "w") as f:
        json.dump(raw_concepts, f, indent=2)
    log.info("Stage 1 checkpoint saved → %s", p)


def save_stage2(normal_dict: list[dict], save_path: str, category: str) -> None:
    p = stage2_path(save_path, category)
    with open(p, "w") as f:
        json.dump(normal_dict, f, indent=2)
    log.info("Stage 2 checkpoint saved → %s", p)


def save_stage3(annotations: list[dict], save_path: str, category: str) -> None:
    """Most valuable checkpoint — represents 8-10h of VLM work."""
    p = stage3_path(save_path, category)
    with open(p, "w") as f:
        json.dump(annotations, f, indent=2)
    log.info("Stage 3 checkpoint saved → %s  (%d annotations)", p, len(annotations))


# ── Load helpers ──────────────────────────────────────────────────────────────

def load_stage1(save_path: str, category: str) -> list[dict] | None:
    p = stage1_path(save_path, category)
    if not p.exists():
        return None
    with open(p) as f:
        data = json.load(f)
    log.info("Stage 1 loaded from checkpoint (%d raw concepts)", len(data))
    return data


def load_stage2(save_path: str, category: str) -> list[dict] | None:
    p = stage2_path(save_path, category)
    if not p.exists():
        return None
    with open(p) as f:
        data = json.load(f)
    log.info("Stage 2 loaded from checkpoint (%d normal concepts)", len(data))
    return data


def load_stage3(save_path: str, category: str) -> list[dict] | None:
    p = stage3_path(save_path, category)
    if not p.exists():
        return None
    with open(p) as f:
        data = json.load(f)
    log.info("Stage 3 loaded from checkpoint (%d annotations)", len(data))
    return data


# ── CL helper ─────────────────────────────────────────────────────────────────

def append_to_stage3(
    new_annotations: list[dict],
    save_path: str,
    category: str,
) -> list[dict]:
    existing = load_stage3(save_path, category) or []
    existing_paths = {a["image_path"] for a in existing}
    added = [a for a in new_annotations if a["image_path"] not in existing_paths]
    merged = existing + added
    save_stage3(merged, save_path, category)
    log.info(
        "CL: appended %d new annotations (%d already existed, %d total)",
        len(added), len(existing), len(merged),
    )
    return merged


# ── Status report ─────────────────────────────────────────────────────────────

def checkpoint_status(save_path: str, category: str) -> dict[str, bool]:
    return {
        "stage1":     stage1_path(save_path, category).exists(),
        "stage2":     stage2_path(save_path, category).exists(),
        "stage3":     stage3_path(save_path, category).exists(),
        "clip_cache": clip_cache_path(save_path, category).exists(),
    }


def log_checkpoint_status(save_path: str, category: str) -> None:
    d = _checkpoint_dir(save_path, category)
    log.info("Checkpoint directory: %s", d)
    status = checkpoint_status(save_path, category)
    for name, exists in status.items():
        log.info("  %-15s %s", name, "✓" if exists else "✗ (will be computed)")