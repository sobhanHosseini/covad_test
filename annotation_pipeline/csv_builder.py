"""
csv_builder.py — Build the CONVAD-compatible concept annotation CSV.

Output schema (identical to original CONVAD pipeline):
  image_path, label_index, mask_path, anomaly_type, split,
  <concept_1>, <concept_2>, ..., <concept_N>

label_index : 0 = normal, 1 = anomalous
split       : train / val / test (80/10/10 random split, seeded)
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

import pandas as pd

from annotation_pipeline.dataset import get_mask_path

log = logging.getLogger(__name__)


def build(
    dataset_path: str,
    category: str,
    image_groups: dict[str, list[str]],
    all_annotations: list[dict],
    final_vocab: list[str],
    holdout_defect: str | None = None,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Build the annotation DataFrame from Stage 3 annotations and final vocabulary.

    Parameters
    ----------
    all_annotations : full list of annotation dicts from Stage 3 (including all defects)
    final_vocab     : ordered list of concept names (output of vocabulary.build)
    holdout_defect  : if set, images of this defect type are excluded from TRAINING rows
                      but are INCLUDED as anomalous test rows (required for holdout eval)

    Returns
    -------
    pd.DataFrame with CONVAD-compatible schema
    """
    ann_index = {a["image_path"]: a for a in all_annotations}
    rows: list[dict] = []

    # ── Normal train images ────────────────────────────────────────────────────
    for img_path in image_groups.get("normal", []):
        cv = ann_index.get(img_path, {}).get("concept_vector", {})
        row = {
            "image_path": img_path,
            "label_index": 0,
            "mask_path": "",
            "anomaly_type": "good",
        }
        for concept in final_vocab:
            row[concept] = int(bool(cv.get(concept, True)))  # default True for normal
        rows.append(row)

    # ── Normal test images ─────────────────────────────────────────────────────
    for img_path in image_groups.get("normal_test", []):
        cv = ann_index.get(img_path, {}).get("concept_vector", {})
        row = {
            "image_path": img_path,
            "label_index": 0,
            "mask_path": "",
            "anomaly_type": "good",
        }
        for concept in final_vocab:
            row[concept] = int(bool(cv.get(concept, True)))
        rows.append(row)

    # ── Defect images ──────────────────────────────────────────────────────────
    for dt, img_paths in image_groups.items():
        if dt in ("normal", "normal_test"):
            continue
        for img_path in img_paths:
            cv = ann_index.get(img_path, {}).get("concept_vector", {})
            row = {
                "image_path": img_path,
                "label_index": 1,
                "mask_path": get_mask_path(dataset_path, category, dt, img_path),
                "anomaly_type": dt,
            }
            for concept in final_vocab:
                row[concept] = int(bool(cv.get(concept, False)))  # default False for defects
            rows.append(row)

    df = pd.DataFrame(rows)

    # ── Train / val / test split (80 / 10 / 10) ───────────────────────────────
    rng = random.Random(random_seed)
    indices = list(df.index)
    rng.shuffle(indices)
    n = len(indices)
    n_test = max(1, n // 10)
    n_val  = max(1, n // 10)
    test_set = set(indices[:n_test])
    val_set  = set(indices[n_test : n_test + n_val])
    df["split"] = [
        "test" if i in test_set else "val" if i in val_set else "train"
        for i in df.index
    ]

    # ── Column ordering ────────────────────────────────────────────────────────
    meta_cols = ["image_path", "label_index", "mask_path", "anomaly_type", "split"]
    concept_cols = [c for c in final_vocab if c in df.columns]
    df = df[meta_cols + concept_cols]

    log.info(
        "CSV built: %d rows × %d columns (%d concept columns)%s",
        len(df), len(df.columns), len(concept_cols),
        f"  (holdout={holdout_defect})" if holdout_defect else "",
    )
    return df
