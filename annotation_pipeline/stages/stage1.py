"""
stages/stage1.py — Per-image normal concept extraction from normal (train) images.

Extracts 12 visual attribute concepts per image using the VLM.
Output is a flat list of concept dicts, each tagged with its source image path.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

from annotation_pipeline.config import PROMPT_NORMAL_EXTRACTION
from annotation_pipeline.utils import call_vlm, extract_json, to_snake_case

log = logging.getLogger(__name__)


def _extract_concepts_single(
    client,
    model_name: str,
    image_path: str,
    category: str,
    debug: bool = False,
) -> list[dict]:
    """
    Call the VLM on one normal image and parse the 12 concepts it returns.
    Returns an empty list on parse failure (logged as a warning).
    """
    prompt = PROMPT_NORMAL_EXTRACTION.format(category=category)
    raw = call_vlm(client, model_name, prompt, image_paths=[image_path], debug=debug)
    parsed = extract_json(raw)

    if not isinstance(parsed, list):
        log.warning(
            "[Stage 1] Parse failed for %s — response: %.80s",
            Path(image_path).name, raw,
        )
        return []

    valid = []
    for item in parsed:
        if not isinstance(item, dict) or not item.get("name"):
            continue
        valid.append({
            "name": to_snake_case(str(item["name"])),
            "description": str(item.get("description", "")),
            "visual_dimension": str(item.get("visual_dimension", "unknown")),
        })
    return valid


def run(
    client,
    model_name: str,
    normal_images: list[str],
    category: str,
    n_sample: int | None = None,
    random_seed: int = 42,
    debug: bool = False,
) -> list[dict]:
    """
    Run Stage 1 on normal_images.

    Parameters
    ----------
    n_sample : int | None
        If set, randomly sample this many images from normal_images.
        None = use all images.

    Returns
    -------
    list of concept dicts, each with an extra "source_image" key.
    """
    images = list(normal_images)
    if n_sample and n_sample < len(images):
        rng = random.Random(random_seed)
        images = rng.sample(images, n_sample)
        log.info("Stage 1: sampling %d of %d normal images", n_sample, len(normal_images))
    else:
        log.info("Stage 1: processing all %d normal images", len(normal_images))

    all_concepts: list[dict] = []
    for i, img_path in enumerate(images, 1):
        log.info("  [%3d/%d] %s", i, len(images), Path(img_path).name)
        concepts = _extract_concepts_single(client, model_name, img_path, category, debug)
        for c in concepts:
            c["source_image"] = img_path
        all_concepts.extend(concepts)
        log.info("           → %d concepts extracted", len(concepts))

    log.info(
        "Stage 1 complete: %d total raw concepts from %d images",
        len(all_concepts), len(images),
    )
    return all_concepts
