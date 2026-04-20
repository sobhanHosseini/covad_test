"""
stages/stage3.py — Per-image concept annotation for all images (Stage 3).

Normal images:   VLM annotates True/False for each concept in the shared vocabulary.
Defect images:   VLM annotates True/False for normal concepts AND extracts new
                 defect-specific concepts via comparative inspection with normal refs.

P1-A: CLIP selects the k most visually similar normal references per defect image.
P2-B: ThreadPoolExecutor for parallel defect annotation.
"""

from __future__ import annotations

import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from annotation_pipeline.config import (
    PROMPT_DEFECT_ANNOTATION,
    PROMPT_NORMAL_IMAGE_ANNOTATION,
)
from annotation_pipeline.utils import call_vlm, extract_json, to_snake_case
from annotation_pipeline.clip_utils import select_normal_references

log = logging.getLogger(__name__)


# ── Single-image helpers ──────────────────────────────────────────────────────

def _annotate_normal(
    client,
    model_name: str,
    image_path: str,
    all_concepts: list[dict],
    category: str,
    debug: bool = False,
) -> dict:
    """Annotate one normal image: True/False for each concept in vocabulary."""
    concept_list = json.dumps([c["name"] for c in all_concepts])
    prompt = PROMPT_NORMAL_IMAGE_ANNOTATION.format(
        category=category, concept_list=concept_list
    )
    raw = call_vlm(client, model_name, prompt, image_paths=[image_path], debug=debug)
    parsed = extract_json(raw)
    if not isinstance(parsed, dict):
        # Fallback: assume all concepts present (normal image)
        return {c["name"]: True for c in all_concepts}
    return {c["name"]: bool(parsed.get(c["name"], True)) for c in all_concepts}


def _annotate_defect(
    client,
    model_name: str,
    defect_image_path: str,
    normal_ref_paths: list[str],
    all_concepts: list[dict],
    defect_type: str,
    category: str,
    max_new_concepts: int = 6,
    debug: bool = False,
) -> dict:
    """
    Annotate one defect image.

    Returns a dict with:
      "normal_concept_annotations" : {concept_name: bool}
      "new_defect_concepts"        : [{name, description, visual_dimension}]
      "defect_category"            : str
    """
    fallback = {
        "normal_concept_annotations": {c["name"]: False for c in all_concepts},
        "new_defect_concepts": [],
        "defect_category": defect_type,
    }
    if not normal_ref_paths:
        return fallback

    n_refs = len(normal_ref_paths)
    normal_dict_json = json.dumps(
        [{"name": c["name"], "description": c["description"]} for c in all_concepts],
        indent=2,
    )
    prompt = PROMPT_DEFECT_ANNOTATION.format(
        n_images=n_refs + 1,
        n_refs=n_refs,
        query_idx=n_refs + 1,
        category=category,
        defect_type=defect_type,
        normal_dict_json=normal_dict_json,
        n_concepts=len(all_concepts),
        max_new=max_new_concepts,
    )

    raw = call_vlm(
        client, model_name, prompt,
        image_paths=list(normal_ref_paths) + [defect_image_path],
        debug=debug,
    )
    parsed = extract_json(raw)

    if not isinstance(parsed, dict):
        log.warning("[Stage 3] Parse failed for %s", Path(defect_image_path).name)
        return fallback

    annotations = parsed.get("normal_concept_annotations", {})
    for c in all_concepts:
        annotations.setdefault(c["name"], False)

    new_concepts = []
    for item in parsed.get("new_defect_concepts", []):
        if isinstance(item, dict) and item.get("name"):
            new_concepts.append({
                "name": to_snake_case(str(item["name"])),
                "description": str(item.get("description", "")),
                "visual_dimension": str(item.get("visual_dimension", "unknown")),
            })

    return {
        "normal_concept_annotations": annotations,
        "new_defect_concepts": new_concepts[:max_new_concepts],
        "defect_category": defect_type,
    }


# ── Stage 3 orchestrator ──────────────────────────────────────────────────────

def run(
    client,
    model_name: str,
    image_groups: dict[str, list[str]],
    defect_types: list[str],
    all_concepts: list[dict],
    category: str,
    normal_embeddings: dict,
    defect_embeddings: dict,
    n_normal_refs: int = 3,
    max_new_defect_concepts: int = 6,
    n_workers: int = 1,
    n_annotate_sample: int | None = None,
    debug: bool = False,
) -> list[dict]:
    """
    Annotate all images and return a flat list of annotation dicts.

    Each annotation dict:
      image_path, anomaly_type, concept_vector (dict of name→bool),
      new_defect_concepts (list of dicts), defect_category

    Parameters
    ----------
    normal_embeddings : {path: embedding} for normal training images  (P1-A)
    defect_embeddings : {path: embedding} for defect images           (P1-A)
        Both are computed in pipeline.py and cached to disk.
        If CLIP is unavailable, both will be empty dicts and the fallback
        (fixed middle normal image) is used automatically.
    """
    all_annotations: list[dict] = []
    lock = threading.Lock()

    def _append(ann: dict) -> None:
        with lock:
            all_annotations.append(ann)

    # ── P1-A: reference selector ───────────────────────────────────────────────
    all_normal = image_groups.get("normal", [])

    def _get_refs(img_path: str) -> list[str]:
        # Use CLIP to find the k most visually similar normal images for this defect image
        if normal_embeddings and defect_embeddings and img_path in defect_embeddings:
            return select_normal_references(
                defect_embeddings[img_path], normal_embeddings, k=n_normal_refs
            )
        # Fallback: stable middle image (used when CLIP is unavailable or n_normal_refs=1)
        return [all_normal[len(all_normal) // 2]] if all_normal else []

    # ── 3a. Normal training images ─────────────────────────────────────────────
    normal_sample = all_normal
    if n_annotate_sample:
        normal_sample = all_normal[:n_annotate_sample]

    log.info("Annotating %d normal (train) images...", len(normal_sample))
    for i, img_path in enumerate(normal_sample, 1):
        if i % 20 == 0 or i == len(normal_sample):
            log.info("  [%d/%d] %s", i, len(normal_sample), Path(img_path).name)
        cv = _annotate_normal(client, model_name, img_path, all_concepts, category, debug)
        _append({
            "image_path": img_path, "anomaly_type": "good",
            "concept_vector": cv, "new_defect_concepts": [], "defect_category": "good",
        })

    # ── 3b. Normal test images ─────────────────────────────────────────────────
    normal_test = image_groups.get("normal_test", [])
    if n_annotate_sample:
        normal_test = normal_test[:n_annotate_sample]

    log.info("Annotating %d normal (test) images...", len(normal_test))
    for i, img_path in enumerate(normal_test, 1):
        if i % 20 == 0 or i == len(normal_test):
            log.info("  [%d/%d] %s", i, len(normal_test), Path(img_path).name)
        cv = _annotate_normal(client, model_name, img_path, all_concepts, category, debug)
        _append({
            "image_path": img_path, "anomaly_type": "good",
            "concept_vector": cv, "new_defect_concepts": [], "defect_category": "good",
        })

    # ── 3c. Defect images (parallel, P2-B) ────────────────────────────────────
    def _worker(img_path: str, defect_type: str) -> dict:
        refs = _get_refs(img_path)
        ann = _annotate_defect(
            client, model_name, img_path, refs, all_concepts,
            defect_type, category,
            max_new_concepts=max_new_defect_concepts,
            debug=debug,
        )
        cv = dict(ann["normal_concept_annotations"])
        for c in ann["new_defect_concepts"]:
            cv[c["name"]] = True
        return {
            "image_path": img_path,
            "anomaly_type": defect_type,
            "concept_vector": cv,
            "new_defect_concepts": ann["new_defect_concepts"],
            "defect_category": defect_type,
        }

    defect_tasks: list[tuple[str, str]] = []
    for dt in defect_types:
        imgs = image_groups[dt]
        if n_annotate_sample:
            imgs = imgs[:n_annotate_sample]
        defect_tasks.extend((p, dt) for p in imgs)

    log.info(
        "Annotating %d defect images with %d worker(s)...",
        len(defect_tasks), n_workers,
    )
    completed = 0
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(_worker, p, dt): (p, dt) for p, dt in defect_tasks}
        for fut in as_completed(futures):
            _append(fut.result())
            completed += 1
            if completed % 10 == 0 or completed == len(defect_tasks):
                log.info("  Defect annotation: %d/%d", completed, len(defect_tasks))

    log.info("Stage 3 complete: %d total annotations", len(all_annotations))
    return all_annotations


# ── Helper for Stage 4 ────────────────────────────────────────────────────────

def collect_defect_concepts(
    annotations: list[dict],
) -> dict[str, list[dict]]:
    """
    Aggregate new defect concepts per defect type from Stage 3 output.
    Used as input to Stage 4.
    Returns {defect_type: [concept_dict, ...]}  (unique concepts per type).
    """
    defect_concepts: dict[str, dict] = {}
    for ann in annotations:
        dt = ann.get("defect_category", "unknown")
        if dt == "good":
            continue
        for c in ann.get("new_defect_concepts", []):
            defect_concepts.setdefault(dt, {})[c["name"]] = c
    return {k: list(v.values()) for k, v in defect_concepts.items()}
