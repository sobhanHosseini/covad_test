"""
Two-Stage Concept Annotation Pipeline for CONVAD
=================================================
Author: Sobhan Hosseini — MSc Thesis, University of Padova, 2025–2026
Supervisor: Francesco Borsatti, Davide Dalle Pezze

Replaces the original folder-level VLM annotation in CONVAD with:
  Stage 1 — Per-image normal concept extraction (10-15 concepts/image)
  Stage 2 — Post-processing: deduplicate, cluster, frequency-filter → shared normal dictionary
  Stage 3 — Per-image defect annotation with comparison to a normal reference image
  Stage 4 — Build final CONVAD-compatible CSV (same schema as original pipeline)

VLM backend: Ollama at http://localhost:6000 (same as original CONVAD)
Default model: gemma3:27b  (set --model_name to override)

Usage:
  python annotation_pipeline.py \
      --dataset_path /path/to/mvtec \
      --category hazelnut \
      --model_name gemma3:27b \
      --save_path /path/to/output/hazelnut_concepts_new.csv

Optional flags:
  --holdout_defect crack     # exclude this defect type's concepts from the final vocab
  --n_normal_sample 50       # how many normal images to use for Stage 1 (default: all)
  --n_annotate_sample 20     # limit Stage 3 annotation to N images per group (default: all)
  --min_concept_freq 0.20    # minimum fraction of normal images a concept must appear in
  --max_concept_freq 0.95    # maximum fraction (avoids always-true concepts)
  --cluster_threshold 0.80   # cosine similarity threshold for concept clustering
  --debug                    # print all VLM prompts and raw responses
"""

import os
import re
import json
import random
import argparse
import logging
from pathlib import Path
from collections import Counter
from typing import Optional

import numpy as np
import pandas as pd
from ollama import Client

# ── Optional: sentence-transformers for embedding-based clustering ─────────────
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.cluster import AgglomerativeClustering
    CLUSTERING_AVAILABLE = True
except ImportError:
    CLUSTERING_AVAILABLE = False
    logging.warning(
        "sentence-transformers or sklearn not found. "
        "Falling back to VLM-only deduplication for Stage 2."
    )

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# 0.  UTILITY HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def extract_json(text: str):
    """
    Strip markdown fences and parse JSON — mirrors original CONVAD extract_json().
    Returns parsed object on success, None on failure.
    """
    text = text.strip()
    # Remove ```json ... ``` or ``` ... ``` fences
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to extract the first [ ... ] or { ... } block
        for pattern in (r"(\[.*\])", r"(\{.*\})"):
            match = re.search(pattern, text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1))
                except json.JSONDecodeError:
                    pass
    return None


def to_snake_case(name: str) -> str:
    """Convert a concept name to snake_case column identifier."""
    name = name.lower().strip()
    name = re.sub(r"[^a-z0-9\s_]", "", name)
    name = re.sub(r"\s+", "_", name)
    return name


def call_vlm(client: Client, model_name: str, prompt: str,
             image_paths: list[str] | None = None,
             debug: bool = False) -> str:
    """
    Unified wrapper around client.chat() — matches CONVAD's exact calling convention.
    image_paths: list of absolute file path strings (Ollama reads from disk).
    Returns the raw content string from the assistant message.
    """
    message: dict = {"role": "user", "content": prompt}
    if image_paths:
        message["images"] = image_paths

    if debug:
        log.debug("── VLM CALL ──────────────────────────────────")
        log.debug(f"  model      : {model_name}")
        log.debug(f"  images     : {image_paths}")
        log.debug(f"  prompt     :\n{prompt}")

    response = client.chat(model=model_name, messages=[message])
    content = response["message"]["content"]

    if debug:
        log.debug(f"  response   :\n{content}")
        log.debug("──────────────────────────────────────────────")

    return content


# ══════════════════════════════════════════════════════════════════════════════
# 1.  DATASET DISCOVERY
# ══════════════════════════════════════════════════════════════════════════════

def discover_images(dataset_path: str, category: str) -> dict[str, list[str]]:
    """
    Walk the MVTec directory structure and return a dict:
      {
        "normal":        [list of abs paths from train/good/],
        "crack":         [list of abs paths from test/crack/],
        "hole":          [...],
        ...
      }
    """
    root = Path(dataset_path) / category
    result: dict[str, list[str]] = {}

    # Normal images: train/good/
    normal_dir = root / "train" / "good"
    if normal_dir.exists():
        paths = sorted(str(p) for p in normal_dir.glob("*.png"))
        result["normal"] = paths
        log.info(f"Found {len(paths)} normal training images")
    else:
        raise FileNotFoundError(f"Normal train directory not found: {normal_dir}")

    # Defective images: test/<defect_type>/
    test_dir = root / "test"
    for defect_dir in sorted(test_dir.iterdir()):
        if defect_dir.is_dir() and defect_dir.name != "good":
            paths = sorted(str(p) for p in defect_dir.glob("*.png"))
            result[defect_dir.name] = paths
            log.info(f"Found {len(paths):3d} images for defect '{defect_dir.name}'")

    # Normal test images (for val/test split)
    normal_test_dir = root / "test" / "good"
    if normal_test_dir.exists():
        paths = sorted(str(p) for p in normal_test_dir.glob("*.png"))
        result["normal_test"] = paths
        log.info(f"Found {len(paths)} normal test images")

    return result


def get_mask_path(dataset_path: str, category: str,
                  defect_type: str, image_path: str) -> str:
    """
    Reconstruct the ground truth mask path from an image path.
    MVTec structure: ground_truth/<defect_type>/<filename>_mask.png
    Returns empty string if mask does not exist.
    """
    img_name = Path(image_path).stem       # e.g. "000"
    mask_name = f"{img_name}_mask.png"
    mask_path = Path(dataset_path) / category / "ground_truth" / defect_type / mask_name
    return str(mask_path) if mask_path.exists() else ""


# ══════════════════════════════════════════════════════════════════════════════
# 2.  STAGE 1 — PER-IMAGE NORMAL CONCEPT EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

PROMPT_NORMAL_EXTRACTION = """You are an industrial quality control inspector specializing in visual inspection of manufactured parts.

You are examining a defect-free, normal-quality {category} from a production line.

Your task: extract exactly 12 visual attribute concepts that characterize the appearance of THIS specific specimen.

Rules:
- Each concept is a 2-4 word noun phrase describing a specific, concrete visual attribute
- Cover these dimensions: surface texture, surface color, shape/geometry, material finish, structural integrity, and visible surface markings
- Concepts must describe what IS visually present — not what is absent
- Concepts must be visually grounded: a camera could detect them in an image region
- Do NOT include the object name or category name in any concept
- Do NOT use vague terms like "good quality", "normal appearance", "standard condition"
- Use snake_case for all concept names (words joined by underscores, all lowercase)

Return ONLY a valid JSON array of objects. No preamble, no explanation, no markdown.

Format:
[
  {{
    "name": "smooth_shell_surface",
    "description": "The outer shell is uniformly smooth with no visible grooves or roughness",
    "visual_dimension": "texture"
  }},
  ...
]

Provide exactly 12 concepts covering all 6 visual dimensions listed above."""


def extract_normal_concepts_single(client: Client, model_name: str,
                                    image_path: str, category: str,
                                    debug: bool = False) -> list[dict]:
    """
    Run Stage 1 VLM call on a single normal image.
    Returns a list of concept dicts: [{name, description, visual_dimension}, ...]
    Returns [] on parse failure.
    """
    prompt = PROMPT_NORMAL_EXTRACTION.format(category=category)
    raw = call_vlm(client, model_name, prompt, image_paths=[image_path], debug=debug)
    parsed = extract_json(raw)

    if not isinstance(parsed, list):
        log.warning(f"  [Stage 1] Parse failed for {Path(image_path).name}: {raw[:80]}")
        return []

    # Validate and normalise each concept
    valid = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        name = item.get("name", "")
        if not name:
            continue
        valid.append({
            "name": to_snake_case(str(name)),
            "description": str(item.get("description", "")),
            "visual_dimension": str(item.get("visual_dimension", "unknown")),
        })
    return valid


def stage1_extract_normal_concepts(client: Client, model_name: str,
                                    normal_images: list[str], category: str,
                                    n_sample: int | None = None,
                                    debug: bool = False) -> list[dict]:
    """
    Run Stage 1 on all (or a sample of) normal images.
    Returns a flat list of all extracted concept dicts with an added 'source_image' field.
    """
    images = normal_images
    if n_sample and n_sample < len(images):
        images = random.sample(images, n_sample)
        log.info(f"Stage 1: sampling {n_sample} of {len(normal_images)} normal images")
    else:
        log.info(f"Stage 1: processing all {len(normal_images)} normal images")

    all_concepts: list[dict] = []
    for i, img_path in enumerate(images, 1):
        log.info(f"  [{i:3d}/{len(images)}] {Path(img_path).name}")
        concepts = extract_normal_concepts_single(
            client, model_name, img_path, category, debug=debug
        )
        for c in concepts:
            c["source_image"] = img_path
        all_concepts.extend(concepts)
        log.info(f"           → {len(concepts)} concepts extracted")

    log.info(f"Stage 1 complete: {len(all_concepts)} total concepts from {len(images)} images")
    return all_concepts


# ══════════════════════════════════════════════════════════════════════════════
# 3.  STAGE 2 — POST-PROCESSING: BUILD SHARED NORMAL DICTIONARY
# ══════════════════════════════════════════════════════════════════════════════

PROMPT_VLM_MERGE = """You are a vocabulary curator for an industrial inspection system.

Below are groups of visual attribute concepts that were automatically clustered as semantically similar.
Each group is a list of concept names extracted from normal product images.

For each group, choose ONE canonical concept name that:
1. Is the most general and precise representative of the group
2. Uses exactly 2-4 words in snake_case (lowercase, underscores between words)
3. Is visually concrete — a camera sensor could detect it in an image

Groups to merge:
{json_groups}

Return ONLY a valid JSON object mapping each original concept name to its chosen canonical name.
Example: {{"original_name_1": "canonical_name", "original_name_2": "canonical_name", ...}}
No preamble, no explanation."""


def stage2_build_normal_dictionary(
    client: Client,
    model_name: str,
    all_raw_concepts: list[dict],
    normal_images: list[str],
    category: str,
    min_freq: float = 0.20,
    max_freq: float = 0.95,
    cluster_threshold: float = 0.80,
    debug: bool = False,
) -> list[dict]:
    """
    Post-processing pipeline:
      2a. Count per-image concept frequencies
      2b. Embedding-based clustering (or VLM merge fallback)
      2c. Frequency filtering on the normal image set
      2d. Return final shared normal dictionary

    Returns a list of concept dicts: [{name, description, visual_dimension, frequency}, ...]
    """
    log.info("Stage 2: Building shared normal concept dictionary...")

    # 2a. Build per-image concept occurrence
    # image → set of concept names it contributed
    image_to_concepts: dict[str, set] = {}
    for c in all_raw_concepts:
        img = c["source_image"]
        image_to_concepts.setdefault(img, set()).add(c["name"])

    n_images = len(image_to_concepts)
    all_unique_names = list({c["name"] for c in all_raw_concepts})
    log.info(f"  Unique concept names before clustering: {len(all_unique_names)}")

    # 2b. Cluster semantically similar concepts
    canonical_map: dict[str, str] = {}  # original_name → canonical_name

    if CLUSTERING_AVAILABLE and len(all_unique_names) > 1:
        log.info("  Running embedding-based clustering (sentence-transformers)...")
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = embedder.encode(all_unique_names, normalize_embeddings=True)
        # AgglomerativeClustering with cosine distance
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=1.0 - cluster_threshold,
            metric="cosine",
            linkage="average",
        )
        labels = clustering.fit_predict(embeddings)

        # For each cluster: pick the most frequent concept as canonical
        name_freq = Counter(c["name"] for c in all_raw_concepts)
        clusters: dict[int, list[str]] = {}
        for name, label in zip(all_unique_names, labels):
            clusters.setdefault(int(label), []).append(name)

        for cluster_members in clusters.values():
            # canonical = most frequent in corpus; if tie, shortest name
            canonical = max(cluster_members, key=lambda n: (name_freq[n], -len(n)))
            for member in cluster_members:
                canonical_map[member] = canonical

        n_clusters = len(set(labels))
        log.info(f"  Clustering reduced {len(all_unique_names)} → {n_clusters} canonical concepts")

        # Optional VLM refinement for large clusters (>3 members)
        large_clusters = {
            canonical: members
            for canonical, members in {
                canonical_map[n]: [m for m in all_unique_names if canonical_map[m] == canonical_map[n]]
                for n in all_unique_names
            }.items()
            if len(members) > 3
        }
        if large_clusters:
            log.info(f"  Refining {len(large_clusters)} large clusters with VLM...")
            groups_json = json.dumps(
                [{"canonical": k, "members": v} for k, v in large_clusters.items()],
                indent=2
            )
            prompt = PROMPT_VLM_MERGE.format(json_groups=groups_json)
            raw = call_vlm(client, model_name, prompt, debug=debug)
            vlm_map = extract_json(raw)
            if isinstance(vlm_map, dict):
                for orig, canon in vlm_map.items():
                    if orig in canonical_map:
                        canonical_map[orig] = to_snake_case(str(canon))
    else:
        # Fallback: VLM-only merge (mirrors original aggregate_concepts)
        log.info("  Running VLM-only deduplication (embedding clustering unavailable)...")
        concepts_str = json.dumps(all_unique_names)
        prompt = f"""You are an industrial expert performing visual anomaly detection.

Given this list of visual concepts extracted from {category} images:
{concepts_str}

Group concepts that refer to the SAME visual attribute (same meaning, similar wording).
For each group, choose the most representative, precise name in snake_case.

Return ONLY a JSON object: {{"representative_name": ["member1", "member2", ...], ...}}"""

        raw = call_vlm(client, model_name, prompt, debug=debug)
        parsed = extract_json(raw)
        if isinstance(parsed, dict):
            for representative, members in parsed.items():
                canon = to_snake_case(str(representative))
                if isinstance(members, list):
                    for m in members:
                        canonical_map[to_snake_case(str(m))] = canon
                canonical_map[canon] = canon
        # Fill in any missed names
        for name in all_unique_names:
            if name not in canonical_map:
                canonical_map[name] = name

    # Apply canonical mapping to raw concepts
    for c in all_raw_concepts:
        c["canonical_name"] = canonical_map.get(c["name"], c["name"])

    # 2c. Frequency filtering
    # Compute: for each canonical concept, in what fraction of normal images does it appear?
    canonical_to_images: dict[str, set] = {}
    for c in all_raw_concepts:
        canon = c["canonical_name"]
        canonical_to_images.setdefault(canon, set()).add(c["source_image"])

    final_concepts = []
    for canon, image_set in canonical_to_images.items():
        freq = len(image_set) / n_images
        if min_freq <= freq <= max_freq:
            # Pick best description from the most-used variant
            variants = [c for c in all_raw_concepts if c["canonical_name"] == canon]
            # Most common visual_dimension
            dim_counter = Counter(v["visual_dimension"] for v in variants)
            best_dim = dim_counter.most_common(1)[0][0]
            # Longest description (usually most informative)
            best_desc = max((v["description"] for v in variants), key=len, default="")
            final_concepts.append({
                "name": canon,
                "description": best_desc,
                "visual_dimension": best_dim,
                "frequency": round(freq, 3),
            })

    # Sort by frequency descending
    final_concepts.sort(key=lambda x: -x["frequency"])

    # Keyword filter: remove concepts whose names contain defect-related words
    _DEFECT_KEYWORDS = {
        "crack", "fracture", "break", "void", "damage",
        "defect", "missing", "irregular", "absence", "broken",
    }
    filtered_concepts = []
    for c in final_concepts:
        name_words = set(re.split(r"[\s_]+", c["name"].lower()))
        if name_words & _DEFECT_KEYWORDS:
            log.info(f"  Removed defect-like concept from normal dict: {c['name']}")
        else:
            filtered_concepts.append(c)
    final_concepts = filtered_concepts

    log.info(f"Stage 2 complete: {len(final_concepts)} concepts in shared normal dictionary")
    log.info(f"  (after freq filter: min={min_freq}, max={max_freq})")
    for c in final_concepts:
        log.info(f"  {c['name']:35s}  freq={c['frequency']:.2f}  dim={c['visual_dimension']}")

    return final_concepts


# ══════════════════════════════════════════════════════════════════════════════
# 4.  STAGE 3 — PER-IMAGE DEFECT ANNOTATION WITH COMPARISON
# ══════════════════════════════════════════════════════════════════════════════

PROMPT_DEFECT_ANNOTATION = """You are an industrial quality control inspector performing comparative visual inspection.

You are given TWO images:
  IMAGE 1 (first image): A defect-free, normal-quality {category} reference specimen.
  IMAGE 2 (second image): A {category} specimen from the defect category "{defect_type}".

Normal concept vocabulary (established from defect-free specimens):
{normal_dict_json}

Your task has two parts:

PART A — Annotate each normal concept for IMAGE 2:
For each concept in the vocabulary above, determine whether that visual attribute is still
clearly present in the defective specimen (IMAGE 2).
  true  = the attribute is clearly visible and intact in IMAGE 2
  false = the attribute has been disrupted, damaged, or is clearly absent in IMAGE 2

PART B — Extract NEW defect-specific concepts:
List any NEW visual attributes that are present in IMAGE 2 but NOT in the normal vocabulary.
These must describe what is visually WRONG — the defect itself — not normal attributes.
Each new concept must be:
  - Visually concrete and grounded (detectable in a specific image region)
  - Different from all concepts in the normal vocabulary
  - A 2-4 word snake_case noun phrase

Return ONLY this JSON structure — no preamble, no explanation:
{{
  "normal_concept_annotations": {{
    "concept_name_1": true,
    "concept_name_2": false,
    ...
  }},
  "new_defect_concepts": [
    {{
      "name": "linear_shell_fracture",
      "description": "A visible linear crack running along the outer shell surface",
      "visual_dimension": "structure"
    }}
  ],
  "defect_category": "{defect_type}"
}}

Annotate ALL {n_concepts} concepts from the normal vocabulary in PART A."""


def annotate_defect_image(
    client: Client,
    model_name: str,
    defect_image_path: str,
    normal_ref_path: str,
    normal_concepts: list[dict],
    defect_type: str,
    category: str,
    debug: bool = False,
) -> dict:
    """
    Run Stage 3 VLM call on one defective image.
    Returns dict with 'normal_concept_annotations' and 'new_defect_concepts'.
    Falls back to all-False annotations on parse failure.
    """
    # Build a compact normal dictionary for the prompt
    normal_dict_for_prompt = [
        {"name": c["name"], "description": c["description"]}
        for c in normal_concepts
    ]
    normal_dict_json = json.dumps(normal_dict_for_prompt, indent=2)

    prompt = PROMPT_DEFECT_ANNOTATION.format(
        category=category,
        defect_type=defect_type,
        normal_dict_json=normal_dict_json,
        n_concepts=len(normal_concepts),
    )

    # Pass BOTH images: normal reference first, defective image second
    raw = call_vlm(
        client, model_name, prompt,
        image_paths=[normal_ref_path, defect_image_path],
        debug=debug,
    )
    parsed = extract_json(raw)

    # Build fallback result
    fallback = {
        "normal_concept_annotations": {c["name"]: False for c in normal_concepts},
        "new_defect_concepts": [],
        "defect_category": defect_type,
    }

    if not isinstance(parsed, dict):
        log.warning(f"  [Stage 3] Parse failed for {Path(defect_image_path).name}")
        return fallback

    # Validate annotations — fill missing concepts with False
    annotations = parsed.get("normal_concept_annotations", {})
    for c in normal_concepts:
        if c["name"] not in annotations:
            annotations[c["name"]] = False

    # Normalise new defect concepts
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
        "new_defect_concepts": new_concepts,
        "defect_category": defect_type,
    }


def annotate_normal_image(
    client: Client,
    model_name: str,
    image_path: str,
    normal_concepts: list[dict],
    category: str,
    debug: bool = False,
) -> dict:
    """
    Annotate a normal image against the shared normal dictionary.
    Uses the simpler single-image True/False prompt (mirrors original second_vlm_query).
    Normal images should score True for most normality concepts.
    """
    concept_list_str = json.dumps([c["name"] for c in normal_concepts])
    prompt = (
        f"You are an expert evaluating an industrial image to detect anomalies. "
        f"I provide an image of a {category}. "
        f"The image has been classified as normal, which implies that there is no visible defect, "
        f"anomaly or issue. Knowing this, choose which concepts you see in the image among the "
        f"following list of attributes: {concept_list_str}. "
        f"Output the result as a JSON object of this form: "
        f'{{concept_1: true, concept_2: false, ...}}. '
        f"Output ONLY the JSON object, nothing else."
    )

    raw = call_vlm(client, model_name, prompt, image_paths=[image_path], debug=debug)
    parsed = extract_json(raw)

    if not isinstance(parsed, dict):
        log.warning(f"  [Normal annotation] Parse failed for {Path(image_path).name}")
        return {c["name"]: True for c in normal_concepts}  # fallback: all True for normal

    # Fill any missing concepts as True (safe assumption for normal images)
    result = {}
    for c in normal_concepts:
        val = parsed.get(c["name"], True)
        result[c["name"]] = bool(val)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# 5.  CONCEPT VOCABULARY REFINEMENT (HOLD-OUT AWARE)
# ══════════════════════════════════════════════════════════════════════════════

def collect_defect_concepts(all_annotations: list[dict]) -> dict[str, list[dict]]:
    """
    After Stage 3, collect all new defect-specific concepts per defect type.
    Returns: {defect_type: [list of unique concept dicts]}
    """
    defect_concepts: dict[str, dict] = {}  # defect_type → {name → concept_dict}
    for ann in all_annotations:
        defect_type = ann.get("defect_category", "unknown")
        if defect_type == "good":
            continue
        for c in ann.get("new_defect_concepts", []):
            defect_concepts.setdefault(defect_type, {})[c["name"]] = c
    return {k: list(v.values()) for k, v in defect_concepts.items()}


def stage4_refine_defect_concepts(
    all_annotations: list[dict],
    defect_concept_map: dict[str, list[dict]],
    cluster_threshold: float = 0.65,
    max_concepts_per_defect: int = 5,
    min_defect_types_for_generic: int = 3,
) -> tuple[dict[str, list[dict]], list[dict]]:
    """
    Stage 4: Cluster and refine per-defect concepts, then separate generic ones.

    Steps:
      1. For each defect type, embed concept names with all-MiniLM-L6-v2 and cluster
         with AgglomerativeClustering using the given cosine similarity threshold.
      2. Keep only the most frequent concept per cluster (frequency = number of images
         in that defect type that produced it), capped at max_concepts_per_defect.
      3. Find concepts that appear in >= min_defect_types_for_generic defect types
         → move to generic_anomaly_concepts and remove from per-defect lists.

    Returns:
      - refined_defect_concept_map: {defect_type: [top concepts]}
      - generic_anomaly_concepts: [concepts shared across defects]
    """
    log.info("\n" + "═" * 60)
    log.info("STAGE 4 — Defect concept clustering and refinement")
    log.info("═" * 60)

    if not CLUSTERING_AVAILABLE:
        log.warning(
            "Stage 4: sentence-transformers not available — "
            "skipping clustering, returning raw map"
        )
        return defect_concept_map, []

    # Build per-defect image-level concept frequency from Stage 3 annotations
    concept_image_freq: dict[str, Counter] = {}
    for ann in all_annotations:
        defect_type = ann.get("defect_category", "unknown")
        if defect_type == "good":
            continue
        for c in ann.get("new_defect_concepts", []):
            concept_image_freq.setdefault(defect_type, Counter())[c["name"]] += 1

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    refined_defect_concept_map: dict[str, list[dict]] = {}

    for defect_type, concepts in defect_concept_map.items():
        if not concepts:
            refined_defect_concept_map[defect_type] = []
            log.info(f"Stage 4: '{defect_type}' had 0 concepts → 0 after clustering")
            continue

        names = [c["name"] for c in concepts]
        freq_counter = concept_image_freq.get(defect_type, Counter())

        if len(names) == 1:
            refined_defect_concept_map[defect_type] = concepts[:max_concepts_per_defect]
            log.info(f"Stage 4: '{defect_type}' had 1 concept → 1 after clustering")
            continue

        # Embed and cluster
        embeddings = embedder.encode(names, normalize_embeddings=True)
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=1.0 - cluster_threshold,
            metric="cosine",
            linkage="average",
        )
        labels = clustering.fit_predict(embeddings)

        # Per cluster: keep the most frequent concept (tie-break: shortest name)
        clusters: dict[int, list[str]] = {}
        for name, label in zip(names, labels):
            clusters.setdefault(int(label), []).append(name)

        kept_names: list[str] = []
        for cluster_members in clusters.values():
            best = max(cluster_members, key=lambda n: (freq_counter.get(n, 0), -len(n)))
            kept_names.append(best)

        # Reconstruct concept dicts, sort by frequency, cap count
        name_to_dict = {c["name"]: c for c in concepts}
        kept_concepts = [name_to_dict[n] for n in kept_names if n in name_to_dict]
        kept_concepts.sort(key=lambda c: -freq_counter.get(c["name"], 0))
        kept_concepts = kept_concepts[:max_concepts_per_defect]

        log.info(
            f"Stage 4: '{defect_type}' had {len(names)} concepts "
            f"→ {len(kept_concepts)} after clustering"
        )
        refined_defect_concept_map[defect_type] = kept_concepts

    # Identify concepts that span >= min_defect_types_for_generic defect types
    concept_defect_types: dict[str, list[str]] = {}
    for defect_type, concepts in refined_defect_concept_map.items():
        for c in concepts:
            concept_defect_types.setdefault(c["name"], []).append(defect_type)

    generic_names: set[str] = {
        name for name, types in concept_defect_types.items()
        if len(types) >= min_defect_types_for_generic
    }

    # Collect one dict per generic concept (from the first defect type that holds it)
    generic_anomaly_concepts: list[dict] = []
    seen_generic: set[str] = set()
    for concepts in refined_defect_concept_map.values():
        for c in concepts:
            if c["name"] in generic_names and c["name"] not in seen_generic:
                generic_anomaly_concepts.append(c)
                seen_generic.add(c["name"])

    # Remove generic concepts from every per-defect list
    for defect_type in refined_defect_concept_map:
        refined_defect_concept_map[defect_type] = [
            c for c in refined_defect_concept_map[defect_type]
            if c["name"] not in generic_names
        ]

    log.info(f"Stage 4: {len(generic_anomaly_concepts)} generic anomaly concepts identified")
    if generic_anomaly_concepts:
        log.info(f"  Generic concepts: {[c['name'] for c in generic_anomaly_concepts]}")

    return refined_defect_concept_map, generic_anomaly_concepts


def build_full_concept_vocabulary(
    normal_concepts: list[dict],
    defect_concept_map: dict[str, list[dict]],
    holdout_defect: str | None = None,
    generic_anomaly_concepts: list[dict] | None = None,
) -> list[str]:
    """
    Build the final concept list:
      - All normal concepts (always included)
      - Generic anomaly concepts shared across defect types (always included)
      - Defect-specific concepts from all defect types EXCEPT holdout_defect

    Returns a list of canonical concept names (column headers for the CSV).
    """
    vocab = [c["name"] for c in normal_concepts]

    # Include generic anomaly concepts (cross-defect, not tied to any single type)
    for c in (generic_anomaly_concepts or []):
        if c["name"] not in vocab:
            vocab.append(c["name"])

    excluded_defect_concepts: set = set()
    if holdout_defect:
        excluded_defect_concepts = {
            c["name"] for c in defect_concept_map.get(holdout_defect, [])
        }
        log.info(
            f"Hold-out: excluding {len(excluded_defect_concepts)} concepts "
            f"unique to '{holdout_defect}': {excluded_defect_concepts}"
        )

    for defect_type, concepts in defect_concept_map.items():
        if defect_type == holdout_defect:
            continue
        for c in concepts:
            if c["name"] not in vocab and c["name"] not in excluded_defect_concepts:
                vocab.append(c["name"])

    log.info(f"Final concept vocabulary: {len(vocab)} concepts "
             f"({'holdout=' + holdout_defect if holdout_defect else 'no holdout'})")
    return vocab


# ══════════════════════════════════════════════════════════════════════════════
# 6.  BUILD FINAL CONVAD-COMPATIBLE CSV
# ══════════════════════════════════════════════════════════════════════════════

def build_csv(
    dataset_path: str,
    category: str,
    image_groups: dict[str, list[str]],
    normal_concepts: list[dict],
    all_annotations: list[dict],
    final_vocab: list[str],
    holdout_defect: str | None = None,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Assemble the CONVAD-compatible DataFrame.

    Schema (matches original pipeline exactly):
      image_path | label_index | mask_path | anomaly_type | split | <concept_cols>

    split values: "train" / "val" / "test"
    label_index: 0 = normal, 1 = anomalous
    """
    # Index annotations by image_path for fast lookup
    ann_index: dict[str, dict] = {}
    for ann in all_annotations:
        ann_index[ann["image_path"]] = ann

    rows = []

    # ── Normal train images ────────────────────────────────────────────────
    for img_path in image_groups.get("normal", []):
        ann = ann_index.get(img_path, {})
        concept_vals = ann.get("concept_vector", {})
        row = {
            "image_path": img_path,
            "label_index": 0,
            "mask_path": "",
            "anomaly_type": "good",
        }
        for concept in final_vocab:
            row[concept] = int(bool(concept_vals.get(concept, True)))
        rows.append(row)

    # ── Normal test images ─────────────────────────────────────────────────
    for img_path in image_groups.get("normal_test", []):
        ann = ann_index.get(img_path, {})
        concept_vals = ann.get("concept_vector", {})
        row = {
            "image_path": img_path,
            "label_index": 0,
            "mask_path": "",
            "anomaly_type": "good",
        }
        for concept in final_vocab:
            row[concept] = int(bool(concept_vals.get(concept, True)))
        rows.append(row)

    # ── Defective images ───────────────────────────────────────────────────
    for defect_type, img_paths in image_groups.items():
        if defect_type in ("normal", "normal_test"):
            continue
        for img_path in img_paths:
            ann = ann_index.get(img_path, {})
            concept_vals = ann.get("concept_vector", {})
            mask = get_mask_path(dataset_path, category, defect_type, img_path)
            row = {
                "image_path": img_path,
                "label_index": 1,
                "mask_path": mask,
                "anomaly_type": defect_type,
            }
            for concept in final_vocab:
                row[concept] = int(bool(concept_vals.get(concept, False)))
            rows.append(row)

    df = pd.DataFrame(rows)

    # ── Assign splits (mirrors original split_dataframe logic) ─────────────
    rng = random.Random(random_seed)
    indices = list(df.index)
    rng.shuffle(indices)
    n = len(indices)
    n_test = max(1, n // 10)
    n_val = max(1, n // 10)

    test_idx = set(indices[:n_test])
    val_idx  = set(indices[n_test:n_test + n_val])

    def assign_split(i):
        if i in test_idx:  return "test"
        if i in val_idx:   return "val"
        return "train"

    df["split"] = [assign_split(i) for i in df.index]

    # Reorder columns: metadata first, concepts after
    meta_cols = ["image_path", "label_index", "mask_path", "anomaly_type", "split"]
    concept_cols = [c for c in final_vocab if c in df.columns]
    df = df[meta_cols + concept_cols]

    log.info(f"CSV built: {len(df)} rows × {len(df.columns)} columns "
             f"({len(concept_cols)} concept columns)")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 7.  MAIN ORCHESTRATION
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(args: argparse.Namespace):
    """Full three-stage pipeline orchestration."""
    random.seed(args.random_seed)

    # ── Setup ──────────────────────────────────────────────────────────────
    client = Client(host=args.ollama_host)
    log.info(f"Ollama host : {args.ollama_host}")
    log.info(f"Model       : {args.model_name}")
    log.info(f"Category    : {args.category}")
    log.info(f"Dataset     : {args.dataset_path}")
    if args.holdout_defect:
        log.info(f"Hold-out    : {args.holdout_defect}")

    # ── Discover images ────────────────────────────────────────────────────
    image_groups = discover_images(args.dataset_path, args.category)
    defect_types = [k for k in image_groups if k not in ("normal", "normal_test")]

    if args.holdout_defect and args.holdout_defect not in defect_types:
        raise ValueError(
            f"Holdout defect '{args.holdout_defect}' not found. "
            f"Available: {defect_types}"
        )

    # ── Stage 1: Normal concept extraction ────────────────────────────────
    log.info("\n" + "═" * 60)
    log.info("STAGE 1 — Per-image normal concept extraction")
    log.info("═" * 60)

    raw_concepts = stage1_extract_normal_concepts(
        client, args.model_name,
        image_groups["normal"],
        args.category,
        n_sample=args.n_normal_sample,
        debug=args.debug,
    )

    # Save Stage 1 output for inspection / reproducibility
    stage1_path = Path(args.save_path).parent / f"{args.category}_stage1_raw_concepts.json"
    with open(stage1_path, "w") as f:
        json.dump(raw_concepts, f, indent=2)
    log.info(f"Stage 1 raw concepts saved → {stage1_path}")

    # ── Stage 2: Build shared normal dictionary ────────────────────────────
    log.info("\n" + "═" * 60)
    log.info("STAGE 2 — Post-processing: shared normal dictionary")
    log.info("═" * 60)

    normal_concepts = stage2_build_normal_dictionary(
        client, args.model_name,
        raw_concepts,
        image_groups["normal"],
        args.category,
        min_freq=args.min_concept_freq,
        max_freq=args.max_concept_freq,
        cluster_threshold=args.cluster_threshold,
        debug=args.debug,
    )

    stage2_path = Path(args.save_path).parent / f"{args.category}_stage2_normal_dict.json"
    with open(stage2_path, "w") as f:
        json.dump(normal_concepts, f, indent=2)
    log.info(f"Normal dictionary saved → {stage2_path}")

    # ── Stage 3: Per-image annotation ─────────────────────────────────────
    log.info("\n" + "═" * 60)
    log.info("STAGE 3 — Per-image annotation")
    log.info("═" * 60)

    # Pick a stable normal reference image (middle of sorted list)
    all_normal = image_groups["normal"]
    normal_ref = all_normal[len(all_normal) // 2]
    log.info(f"Normal reference image: {Path(normal_ref).name}")

    all_annotations: list[dict] = []

    # 3a. Annotate all normal train images
    normal_train_imgs = all_normal[:args.n_annotate_sample] if args.n_annotate_sample is not None else all_normal
    log.info(f"\nAnnotating {len(normal_train_imgs)} normal (train) images...")
    for i, img_path in enumerate(normal_train_imgs, 1):
        if i % 10 == 0 or i == len(normal_train_imgs):
            log.info(f"  [{i}/{len(normal_train_imgs)}] {Path(img_path).name}")
        concept_vector = annotate_normal_image(
            client, args.model_name, img_path,
            normal_concepts, args.category, debug=args.debug,
        )
        all_annotations.append({
            "image_path": img_path,
            "anomaly_type": "good",
            "concept_vector": concept_vector,
            "new_defect_concepts": [],
            "defect_category": "good",
        })

    # 3b. Annotate normal test images
    normal_test_all = image_groups.get("normal_test", [])
    normal_test = normal_test_all[:args.n_annotate_sample] if args.n_annotate_sample is not None else normal_test_all
    log.info(f"\nAnnotating {len(normal_test)} normal (test) images...")
    for i, img_path in enumerate(normal_test, 1):
        if i % 10 == 0 or i == len(normal_test):
            log.info(f"  [{i}/{len(normal_test)}] {Path(img_path).name}")
        concept_vector = annotate_normal_image(
            client, args.model_name, img_path,
            normal_concepts, args.category, debug=args.debug,
        )
        all_annotations.append({
            "image_path": img_path,
            "anomaly_type": "good",
            "concept_vector": concept_vector,
            "new_defect_concepts": [],
            "defect_category": "good",
        })

    # 3c. Annotate defective images (comparative)
    for defect_type in defect_types:
        defect_imgs_all = image_groups[defect_type]
        defect_imgs = defect_imgs_all[:args.n_annotate_sample] if args.n_annotate_sample is not None else defect_imgs_all
        log.info(f"\nAnnotating {len(defect_imgs)} images for defect '{defect_type}'...")
        for i, img_path in enumerate(defect_imgs, 1):
            if i % 5 == 0 or i == len(defect_imgs):
                log.info(f"  [{i}/{len(defect_imgs)}] {Path(img_path).name}")
            ann = annotate_defect_image(
                client, args.model_name,
                img_path, normal_ref,
                normal_concepts, defect_type,
                args.category, debug=args.debug,
            )
            # Merge normal annotations + defect-specific concept flags
            concept_vector = dict(ann["normal_concept_annotations"])
            # Add new defect concept flags (True for this image)
            for c in ann["new_defect_concepts"]:
                concept_vector[c["name"]] = True

            all_annotations.append({
                "image_path": img_path,
                "anomaly_type": defect_type,
                "concept_vector": concept_vector,
                "new_defect_concepts": ann["new_defect_concepts"],
                "defect_category": defect_type,
            })

    # ── Build vocabulary with hold-out ─────────────────────────────────────
    defect_concept_map = collect_defect_concepts(all_annotations)

    log.info("\nNew defect-specific concepts discovered:")
    for dt, concepts in defect_concept_map.items():
        log.info(f"  {dt}: {[c['name'] for c in concepts]}")

    # ── Stage 4: Cluster and refine defect concepts ────────────────────────
    refined_defect_concept_map, generic_anomaly_concepts = stage4_refine_defect_concepts(
        all_annotations,
        defect_concept_map,
        cluster_threshold=args.stage4_cluster_threshold,
        max_concepts_per_defect=args.max_concepts_per_defect,
    )

    final_vocab = build_full_concept_vocabulary(
        normal_concepts, refined_defect_concept_map,
        holdout_defect=args.holdout_defect,
        generic_anomaly_concepts=generic_anomaly_concepts,
    )

    # ── Build and save CSV ─────────────────────────────────────────────────
    df = build_csv(
        args.dataset_path, args.category,
        image_groups, normal_concepts,
        all_annotations, final_vocab,
        holdout_defect=args.holdout_defect,
    )

    Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.save_path, index=False)
    log.info(f"\n✓ Final CSV saved → {args.save_path}")

    # ── Summary ────────────────────────────────────────────────────────────
    log.info("\n" + "═" * 60)
    log.info("PIPELINE SUMMARY")
    log.info("═" * 60)
    log.info(f"  Normal concepts in dictionary : {len(normal_concepts)}")
    log.info(f"  Total defect-specific concepts: {sum(len(v) for v in refined_defect_concept_map.values())}")
    log.info(f"  Generic anomaly concepts      : {len(generic_anomaly_concepts)}")
    log.info(f"  Final vocab size              : {len(final_vocab)}")
    log.info(f"  Total images annotated        : {len(all_annotations)}")
    log.info(f"  CSV shape                     : {df.shape}")
    log.info(f"  Hold-out defect               : {args.holdout_defect or 'none'}")
    log.info("═" * 60)

    return df, normal_concepts, refined_defect_concept_map


# ══════════════════════════════════════════════════════════════════════════════
# 8.  CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Two-stage VLM concept annotation pipeline for CONVAD"
    )
    p.add_argument("--dataset_path",     required=True,
                   help="Root path to MVTec dataset (e.g. /data/mvtec)")
    p.add_argument("--category",         required=True,
                   help="Category to annotate (e.g. hazelnut)")
    p.add_argument("--model_name",       default="gemma3:27b",
                   help="Ollama model name (default: gemma3:27b)")
    p.add_argument("--ollama_host",      default="http://localhost:6000",
                   help="Ollama server URL (default: http://localhost:6000)")
    p.add_argument("--save_path",        required=True,
                   help="Output CSV path (e.g. /data/annotations/hazelnut_new.csv)")
    p.add_argument("--holdout_defect",   default=None,
                   help="Defect type to hold out (exclude its concepts from vocab)")
    p.add_argument("--n_normal_sample",  type=int, default=None,
                   help="Sample N normal images for Stage 1 (default: all)")
    p.add_argument("--n_annotate_sample", type=int, default=None,
                   help="Limit Stage 3 annotation to at most N images per group (default: all)")
    p.add_argument("--min_concept_freq", type=float, default=0.20,
                   help="Min fraction of normal images a concept must appear in (default: 0.20)")
    p.add_argument("--max_concept_freq", type=float, default=0.95,
                   help="Max fraction of normal images a concept may appear in (default: 0.95)")
    p.add_argument("--cluster_threshold", type=float, default=0.80,
                   help="Cosine similarity threshold for Stage 2 normal concept clustering (default: 0.80)")
    p.add_argument("--stage4_cluster_threshold", type=float, default=0.55,
                   help="Cosine similarity threshold for Stage 4 defect concept clustering (default: 0.55)")
    p.add_argument("--max_concepts_per_defect", type=int, default=4,
                   help="Max concepts kept per defect type after Stage 4 clustering (default: 4)")
    p.add_argument("--random_seed",      type=int, default=42)
    p.add_argument("--debug",            action="store_true",
                   help="Print all VLM prompts and raw responses")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args)