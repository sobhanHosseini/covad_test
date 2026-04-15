"""
Two-Stage Concept Annotation Pipeline for CONVAD  —  v5 (clean rebuild)
=======================================================================
Author : Sobhan Hosseini — MSc Thesis, University of Padova, 2025-2026
Supervisors : Francesco Borsatti, Davide Dalle Pezze

DESIGN PHILOSOPHY
-----------------
Clean rebuild keeping the best features from all prior versions and removing
complexity that did not improve results.

WHAT IS KEPT vs REMOVED vs RESTORED
─────────────────────────────────────
Kept from v2   : Stage 1 prompt, Stage 2 dim-aware clustering, freq filter,
                 Stage 2b generic tier, Stage 3 comparative prompt, holdout logic
Kept from new  : P1-A multiple normal refs (CLIP), P1-B VLM audit, P2-B async,
                 P3-B CLIP embedding cache
Removed        : P3-A Stage 1b canonical normalization (extra VLM call, no gain)
Restored       : Stage 4 visual grounding (was working, then removed by mistake)
Fixed          : Stage 4 frequency counting now uses cluster-union (not name-only)

PIPELINE STAGES
───────────────
Stage 1   Per-image normal concept extraction (VLM, 12 concepts/image)
Stage 2   Normal concept post-processing
            2a  dimension-aware embedding clustering
            2b  frequency filter (min_freq ≤ freq ≤ max_freq)
            2c  VLM self-audit — remove defect-referencing concepts  [P1-B]
Stage 2b  Add 5 fixed generic anomaly concepts (AnomalyCLIP-inspired, Tier 2)
Stage 3   Per-image defect annotation with CLIP-selected normal references [P1-A]
            Async parallel workers available                          [P2-B]
Stage 4   Defect concept post-processing
            4a  dimension-aware text clustering (+ CLIP visual grounding if enabled)
            4b  cluster-union frequency filter (bug-fixed version)
            4c  cross-defect generic concept discovery
CSV build CONVAD-compatible output (same schema as original pipeline)

THREE-TIER VOCABULARY
─────────────────────
Tier 1  Normal attribute concepts    (target: 10-15)
Tier 2  Generic anomaly concepts     (5 fixed + discovered cross-defect)
Tier 3  Defect-specific concepts     (target: 5-8 per defect type)

USAGE
─────
uv run main_scripts/annotation_pipeline.py \\
    --dataset_path /path/to/mvtec \\
    --category hazelnut \\
    --model_name gemma4:e4b \\
    --save_path ./annotations/hazelnut_v5.csv \\
    --n_normal_sample 50

For blind holdout:
    --holdout_defect print

Key tuning flags (defaults validated on hazelnut):
    --min_concept_freq 0.20          normal concept minimum frequency
    --cluster_threshold 0.65         Stage 2 clustering aggressiveness
    --stage4_cluster_threshold 0.65  Stage 4 clustering (separate from Stage 2)
    --min_defect_freq 0.15           defect concept minimum frequency (cluster-union)
    --n_normal_refs 3                CLIP-selected references per defect image [P1-A]
    --use_visual_grounding           enable CLIP visual grounding in Stage 4
    --visual_weight 0.6              visual vs text weight in Stage 4
    --n_workers 1                    parallel workers for Stage 3 [P2-B]
"""

import re
import json
import random
import argparse
import logging
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import numpy as np
import pandas as pd
from ollama import Client

# ── Optional dependencies ──────────────────────────────────────────────────────
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.cluster import AgglomerativeClustering
    CLUSTERING_AVAILABLE = True
except ImportError:
    CLUSTERING_AVAILABLE = False
    logging.warning("sentence-transformers or sklearn not found — "
                    "falling back to VLM-only deduplication.")

try:
    import clip
    import torch
    from PIL import Image
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

DIMENSION_NORMALIZE: dict[str, str] = {
    "surface texture": "texture",    "surface_texture": "texture",    "texture": "texture",
    "surface color": "color",        "surface_color": "color",         "color": "color",
    "shape/geometry": "shape",       "shape_geometry": "shape",        "shape": "shape",
    "geometry": "shape",
    "material finish": "finish",     "material_finish": "finish",      "finish": "finish",
    "structural integrity": "structure", "structural_integrity": "structure",
    "structure": "structure",        "structural": "structure",
    "visible surface markings": "marking", "visible_surface_markings": "marking",
    "marking": "marking",            "markings": "marking",            "surface markings": "marking",
}

DIMENSION_THRESHOLDS: dict[str, float] = {
    "color": 0.75, "texture": 0.72, "shape": 0.70,
    "finish": 0.72, "structure": 0.60, "marking": 0.65, "unknown": 0.68,
}

# Tier 2 fixed generic concepts — always in vocabulary, never excluded by holdout
GENERIC_ANOMALY_CONCEPTS: list[dict] = [
    {"name": "surface_irregularity",
     "description": "Any visible irregularity, roughness or discontinuity on the surface "
                    "that deviates from expected normal appearance",
     "visual_dimension": "texture", "tier": "generic"},
    {"name": "color_deviation",
     "description": "Any unexpected change in color, discoloration, staining or "
                    "abnormal pigmentation compared to normal appearance",
     "visual_dimension": "color", "tier": "generic"},
    {"name": "structural_discontinuity",
     "description": "Any break, crack, hole, fracture or loss of structural continuity in the material",
     "visual_dimension": "structure", "tier": "generic"},
    {"name": "texture_inconsistency",
     "description": "Any localized region where surface texture is inconsistent "
                    "with surrounding normal texture",
     "visual_dimension": "texture", "tier": "generic"},
    {"name": "unexpected_surface_pattern",
     "description": "Any pattern, marking, deposit or foreign material on the surface "
                    "that should not be present on a normal specimen",
     "visual_dimension": "marking", "tier": "generic"},
]


# ══════════════════════════════════════════════════════════════════════════════
# UTILITY HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def extract_json(text: str):
    """Strip markdown fences and parse JSON. Returns None on failure."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        for pattern in (r"(\[.*\])", r"(\{.*\})"):
            import re as _re
            match = _re.search(pattern, text, _re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1))
                except json.JSONDecodeError:
                    pass
    return None


def to_snake_case(name: str) -> str:
    name = name.lower().strip()
    name = re.sub(r"[^a-z0-9\s_]", "", name)
    name = re.sub(r"\s+", "_", name)
    return name


def normalize_dimension(dim: str) -> str:
    if not dim:
        return "unknown"
    norm = DIMENSION_NORMALIZE.get(dim.lower().strip())
    if norm:
        return norm
    dim_lower = dim.lower()
    for key, value in DIMENSION_NORMALIZE.items():
        if key in dim_lower or dim_lower in key:
            return value
    return "unknown"


def get_dimension_threshold(dim: str, global_threshold: float) -> float:
    return DIMENSION_THRESHOLDS.get(normalize_dimension(dim), global_threshold)


def call_vlm(client: Client, model_name: str, prompt: str,
             image_paths: list[str] | None = None, debug: bool = False) -> str:
    message: dict = {"role": "user", "content": prompt}
    if image_paths:
        message["images"] = image_paths
    if debug:
        log.debug(f"VLM prompt:\n{prompt}")
    response = client.chat(model=model_name, messages=[message])
    content = response["message"]["content"]
    if debug:
        log.debug(f"VLM response:\n{content}")
    return content


# ══════════════════════════════════════════════════════════════════════════════
# DATASET DISCOVERY
# ══════════════════════════════════════════════════════════════════════════════

def discover_images(dataset_path: str, category: str) -> dict[str, list[str]]:
    root = Path(dataset_path) / category
    result: dict[str, list[str]] = {}

    normal_dir = root / "train" / "good"
    if not normal_dir.exists():
        raise FileNotFoundError(f"Normal train directory not found: {normal_dir}")
    result["normal"] = sorted(str(p) for p in normal_dir.glob("*.png"))
    log.info(f"Found {len(result['normal'])} normal training images")

    for defect_dir in sorted((root / "test").iterdir()):
        if defect_dir.is_dir() and defect_dir.name != "good":
            paths = sorted(str(p) for p in defect_dir.glob("*.png"))
            result[defect_dir.name] = paths
            log.info(f"Found {len(paths):3d} images for defect '{defect_dir.name}'")

    normal_test_dir = root / "test" / "good"
    if normal_test_dir.exists():
        result["normal_test"] = sorted(str(p) for p in normal_test_dir.glob("*.png"))
        log.info(f"Found {len(result['normal_test'])} normal test images")

    return result


def get_mask_path(dataset_path: str, category: str,
                  defect_type: str, image_path: str) -> str:
    img_name = Path(image_path).stem
    mask_path = (Path(dataset_path) / category / "ground_truth" /
                 defect_type / f"{img_name}_mask.png")
    return str(mask_path) if mask_path.exists() else ""


# ══════════════════════════════════════════════════════════════════════════════
# CLIP UTILITIES  [P1-A, P3-B]
# ══════════════════════════════════════════════════════════════════════════════

def load_or_compute_clip_embeddings(
    image_paths: list[str], clip_model, clip_preprocess, device: str,
    cache_path: str | None = None, batch_size: int = 32,
) -> dict[str, np.ndarray]:
    """Compute CLIP embeddings with optional disk cache. Returns {path: embedding}."""
    if cache_path and Path(cache_path).exists():
        data = np.load(cache_path, allow_pickle=True)
        cached = {p: e for p, e in zip(data["paths"].tolist(), data["embeddings"])}
        missing = [p for p in image_paths if p not in cached]
        if not missing:
            log.info(f"  CLIP: loaded {len(cached)} embeddings from cache")
            return {p: cached[p] for p in image_paths if p in cached}
        log.info(f"  CLIP cache incomplete ({len(missing)} missing) — recomputing")

    log.info(f"  CLIP: computing embeddings for {len(image_paths)} images...")
    embeddings: dict[str, np.ndarray] = {}

    for i in range(0, len(image_paths), batch_size):
        batch = image_paths[i: i + batch_size]
        tensors, valid = [], []
        for p in batch:
            try:
                t = clip_preprocess(Image.open(p).convert("RGB")).unsqueeze(0)
                tensors.append(t)
                valid.append(p)
            except Exception as e:
                log.warning(f"  CLIP: skipping {Path(p).name}: {e}")
        if not tensors:
            continue
        with torch.no_grad():
            emb = clip_model.encode_image(torch.cat(tensors).to(device))
            emb = emb / emb.norm(dim=-1, keepdim=True)
        for path, e in zip(valid, emb.cpu().numpy()):
            embeddings[path] = e

    if cache_path and embeddings:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path,
                 paths=np.array(list(embeddings.keys())),
                 embeddings=np.stack(list(embeddings.values())))
        log.info(f"  CLIP embeddings cached → {cache_path}")

    return embeddings


def select_normal_references(
    defect_embedding: np.ndarray,
    normal_embeddings: dict[str, np.ndarray],
    k: int = 3,
) -> list[str]:
    """Return k normal images most similar to a defect image via CLIP cosine."""
    paths = list(normal_embeddings.keys())
    embs = np.stack([normal_embeddings[p] for p in paths])
    sims = embs @ defect_embedding
    top_k = np.argsort(-sims)[: min(k, len(paths))]
    return [paths[i] for i in top_k]


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — PER-IMAGE NORMAL CONCEPT EXTRACTION
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


def extract_normal_concepts_single(client, model_name, image_path, category,
                                    debug=False) -> list[dict]:
    prompt = PROMPT_NORMAL_EXTRACTION.format(category=category)
    raw = call_vlm(client, model_name, prompt, image_paths=[image_path], debug=debug)
    parsed = extract_json(raw)
    if not isinstance(parsed, list):
        log.warning(f"  [Stage 1] Parse failed for {Path(image_path).name}: {raw[:60]}")
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


def stage1_extract_normal_concepts(client, model_name, normal_images, category,
                                    n_sample=None, debug=False) -> list[dict]:
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
            client, model_name, img_path, category, debug=debug)
        for c in concepts:
            c["source_image"] = img_path
        all_concepts.extend(concepts)
        log.info(f"           → {len(concepts)} concepts extracted")

    log.info(f"Stage 1 complete: {len(all_concepts)} total concepts from {len(images)} images")
    return all_concepts


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — NORMAL CONCEPT POST-PROCESSING
# ══════════════════════════════════════════════════════════════════════════════

PROMPT_VLM_MERGE = """You are a vocabulary curator for an industrial inspection system.

Below are groups of visual attribute concepts clustered as semantically similar.
For each group, choose ONE canonical concept name that:
1. Is the most general and precise representative of the group
2. Uses exactly 2-4 words in snake_case
3. Is visually concrete — a camera sensor could detect it in an image

Groups to merge:
{json_groups}

Return ONLY a valid JSON object: {{"original_name": "canonical_name", ...}}
No preamble, no explanation."""


PROMPT_NORMAL_CONCEPT_AUDIT = """You are auditing a normal concept vocabulary for an industrial inspection system.
Category: {category}

These concepts were extracted from defect-free images.
Remove a concept ONLY IF it explicitly references a defect or its absence:
  "absence_of_surface_cracks" → REMOVE  |  "crack_free_shell" → REMOVE
  "no_visible_damage" → REMOVE
KEEP concepts describing normal physical properties:
  "structural_integrity" → KEEP  |  "intact_shell_structure" → KEEP
  "smooth_surface" → KEEP

Concepts to audit:
{concepts_json}

Return ONLY a valid JSON object: {{"concept_name": "keep"/"remove", ...}}
No preamble, no explanation."""


def _cluster_group(names: list[str], embedder, threshold: float,
                   name_freq: Counter) -> dict[str, str]:
    """Cluster concept names and return {name: canonical}."""
    if len(names) == 1:
        return {names[0]: names[0]}
    embs = embedder.encode(names, normalize_embeddings=True)
    clustering = AgglomerativeClustering(
        n_clusters=None, distance_threshold=1.0 - threshold,
        metric="cosine", linkage="average",
    )
    labels = clustering.fit_predict(embs)
    canonical_map = {}
    clusters: dict[int, list[str]] = {}
    for name, label in zip(names, labels):
        clusters.setdefault(int(label), []).append(name)
    for members in clusters.values():
        canonical = max(members, key=lambda n: (name_freq[n], -len(n)))
        for m in members:
            canonical_map[m] = canonical
    return canonical_map


def stage2_build_normal_dictionary(
    client, model_name, all_raw_concepts, normal_images, category,
    min_freq=0.20, max_freq=0.95, cluster_threshold=0.65,
    run_vlm_audit=True, debug=False,
) -> list[dict]:
    """
    Build the shared normal concept dictionary.
    2a  Dimension-aware embedding clustering
    2b  VLM refinement for large clusters (>3 members)
    2c  Frequency filter
    2d  VLM self-audit [P1-B]
    """
    log.info("Stage 2: building shared normal concept dictionary...")

    image_to_concepts: dict[str, set] = {}
    for c in all_raw_concepts:
        image_to_concepts.setdefault(c["source_image"], set()).add(c["name"])
    n_images = len(image_to_concepts)
    all_unique_names = list({c["name"] for c in all_raw_concepts})
    log.info(f"  Unique concept names before clustering: {len(all_unique_names)}")

    canonical_map: dict[str, str] = {}

    if CLUSTERING_AVAILABLE and len(all_unique_names) > 1:
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        name_freq = Counter(c["name"] for c in all_raw_concepts)

        # Determine dominant dimension per concept name
        name_to_dims: dict[str, Counter] = {}
        for c in all_raw_concepts:
            name_to_dims.setdefault(c["name"], Counter())[
                normalize_dimension(c.get("visual_dimension", ""))] += 1
        name_to_dim = {n: dc.most_common(1)[0][0] for n, dc in name_to_dims.items()}

        # Cluster per dimension with dimension-specific threshold
        dim_groups: dict[str, list[str]] = {}
        for name in all_unique_names:
            dim_groups.setdefault(name_to_dim.get(name, "unknown"), []).append(name)

        n_clusters_total = 0
        for dim, group_names in dim_groups.items():
            threshold = get_dimension_threshold(dim, cluster_threshold)
            log.info(f"  Clustering {len(group_names):3d} '{dim}' concepts "
                     f"with threshold {threshold:.2f}")
            group_map = _cluster_group(group_names, embedder, threshold, name_freq)
            canonical_map.update(group_map)
            n_clusters_total += len(set(group_map.values()))

        log.info(f"  Clustering reduced {len(all_unique_names)} → {n_clusters_total} canonical")

        # VLM refinement for large clusters
        cluster_to_members: dict[str, list[str]] = {}
        for name in all_unique_names:
            cluster_to_members.setdefault(canonical_map[name], []).append(name)
        large = {c: m for c, m in cluster_to_members.items() if len(m) > 3}
        if large:
            log.info(f"  Refining {len(large)} large clusters with VLM...")
            groups_json = json.dumps(
                [{"canonical": k, "members": v} for k, v in large.items()], indent=2)
            raw = call_vlm(client, model_name,
                           PROMPT_VLM_MERGE.format(json_groups=groups_json), debug=debug)
            vlm_map = extract_json(raw)
            if isinstance(vlm_map, dict):
                for orig, canon in vlm_map.items():
                    if orig in canonical_map:
                        canonical_map[orig] = to_snake_case(str(canon))
    else:
        log.info("  Running VLM-only deduplication...")
        raw = call_vlm(client, model_name,
                       f"Group these concepts from {category} images by same meaning. "
                       f"Return JSON: {{representative: [members]}}.\n"
                       f"Concepts: {json.dumps(all_unique_names)}", debug=debug)
        parsed = extract_json(raw)
        if isinstance(parsed, dict):
            for rep, members in parsed.items():
                canon = to_snake_case(str(rep))
                if isinstance(members, list):
                    for m in members:
                        canonical_map[to_snake_case(str(m))] = canon
                canonical_map[canon] = canon
        for name in all_unique_names:
            canonical_map.setdefault(name, name)

    for c in all_raw_concepts:
        c["canonical_name"] = canonical_map.get(c["name"], c["name"])

    # Frequency filter
    canonical_to_images: dict[str, set] = {}
    for c in all_raw_concepts:
        canonical_to_images.setdefault(c["canonical_name"], set()).add(c["source_image"])

    final_concepts = []
    for canon, image_set in canonical_to_images.items():
        freq = len(image_set) / n_images
        if min_freq <= freq <= max_freq:
            variants = [c for c in all_raw_concepts if c["canonical_name"] == canon]
            dim_counter = Counter(v["visual_dimension"] for v in variants)
            best_dim = dim_counter.most_common(1)[0][0]
            best_desc = max((v["description"] for v in variants), key=len, default="")
            final_concepts.append({
                "name": canon, "description": best_desc,
                "visual_dimension": best_dim, "frequency": round(freq, 3),
            })

    final_concepts.sort(key=lambda x: -x["frequency"])
    log.info(f"  After frequency filter: {len(final_concepts)} concepts "
             f"(min={min_freq}, max={max_freq})")

    # VLM self-audit [P1-B]
    if run_vlm_audit and final_concepts:
        log.info("  Stage 2d: VLM self-audit...")
        audit_input = [{"name": c["name"], "description": c["description"]}
                       for c in final_concepts]
        raw_audit = call_vlm(client, model_name, PROMPT_NORMAL_CONCEPT_AUDIT.format(
            category=category, concepts_json=json.dumps(audit_input, indent=2)), debug=debug)
        audit_result = extract_json(raw_audit)
        if isinstance(audit_result, dict):
            before = len(final_concepts)
            final_concepts = [c for c in final_concepts
                              if str(audit_result.get(c["name"], "keep")).lower() != "remove"]
            log.info(f"  VLM audit: {before - len(final_concepts)} removed, "
                     f"{len(final_concepts)} kept")
        else:
            log.warning("  VLM audit parse failed — keeping all concepts")

    log.info(f"Stage 2 complete: {len(final_concepts)} concepts in shared normal dictionary")
    for c in final_concepts:
        log.info(f"  {c['name']:40s}  freq={c['frequency']:.2f}  dim={c['visual_dimension']}")
    return final_concepts


def stage2b_add_generic_concepts(normal_concepts: list[dict]) -> list[dict]:
    """Add 5 fixed Tier 2 generic concepts."""
    log.info("\n" + "═" * 60)
    log.info("STAGE 2b — Fixed generic anomaly concepts (Tier 2)")
    log.info("═" * 60)
    existing = {c["name"] for c in normal_concepts}
    all_concepts = list(normal_concepts)
    added = 0
    for gc in GENERIC_ANOMALY_CONCEPTS:
        if gc["name"] not in existing:
            all_concepts.append(gc)
            added += 1
            log.info(f"  + {gc['name']}  [{gc['visual_dimension']}]")
    log.info(f"Stage 2b: added {added} fixed generic anomaly concepts")
    return all_concepts


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3 — PER-IMAGE DEFECT ANNOTATION WITH COMPARISON
# ══════════════════════════════════════════════════════════════════════════════

PROMPT_DEFECT_ANNOTATION = """You are an industrial quality control inspector performing comparative visual inspection.

You are given {n_images} images in order:
  Images 1 to {n_refs}: Defect-free, normal-quality {category} reference specimens.
  Image {query_idx} (last): A {category} specimen from the defect category "{defect_type}".

Normal concept vocabulary (established from defect-free specimens):
{normal_dict_json}

PART A — Annotate each normal concept for the LAST image (the defective specimen).
Use the reference images for comparison.
  true  = the attribute is clearly visible and intact in the defective image
  false = the attribute has been disrupted, damaged, or is clearly absent

PART B — Extract NEW defect-specific concepts.
List at most {max_new} NEW visual attributes visible in the defective image
that are NOT in the normal vocabulary. Focus on the most visually distinctive ones.
Do NOT pad with vague or redundant concepts.
Each must be:
  - Visually concrete (detectable in a specific image region)
  - Different from all concepts in the normal vocabulary
  - A 2-4 word snake_case noun phrase

Return ONLY this JSON structure:
{{
  "normal_concept_annotations": {{"concept_name_1": true, "concept_name_2": false, ...}},
  "new_defect_concepts": [
    {{"name": "linear_shell_fracture",
      "description": "A visible linear crack along the outer shell surface",
      "visual_dimension": "structure"}}
  ],
  "defect_category": "{defect_type}"
}}

Annotate ALL {n_concepts} concepts from the normal vocabulary in PART A."""


def annotate_defect_image(client, model_name, defect_image_path,
                           normal_ref_paths, normal_concepts,
                           defect_type, category, max_new_concepts=4, debug=False) -> dict:
    """Stage 3 annotation for one defective image. P1-A: multiple references."""
    fallback = {
        "normal_concept_annotations": {c["name"]: False for c in normal_concepts},
        "new_defect_concepts": [], "defect_category": defect_type,
    }
    if not normal_ref_paths:
        return fallback

    n_refs = len(normal_ref_paths)
    normal_dict_json = json.dumps(
        [{"name": c["name"], "description": c["description"]} for c in normal_concepts], indent=2)

    prompt = PROMPT_DEFECT_ANNOTATION.format(
        n_images=n_refs + 1, n_refs=n_refs, query_idx=n_refs + 1,
        category=category, defect_type=defect_type,
        normal_dict_json=normal_dict_json,
        n_concepts=len(normal_concepts), max_new=max_new_concepts,
    )

    raw = call_vlm(client, model_name, prompt,
                   image_paths=list(normal_ref_paths) + [defect_image_path], debug=debug)
    parsed = extract_json(raw)

    if not isinstance(parsed, dict):
        log.warning(f"  [Stage 3] Parse failed for {Path(defect_image_path).name}")
        return fallback

    annotations = parsed.get("normal_concept_annotations", {})
    for c in normal_concepts:
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


def annotate_normal_image(client, model_name, image_path, normal_concepts,
                           category, debug=False) -> dict:
    """Annotate a normal image with True/False for each concept."""
    concept_list_str = json.dumps([c["name"] for c in normal_concepts])
    prompt = (
        f"You are an expert evaluating an industrial image to detect anomalies. "
        f"I provide an image of a {category}. "
        f"The image has been classified as normal — no visible defect or issue. "
        f"Choose which concepts you see among: {concept_list_str}. "
        f'Output ONLY a JSON object: {{concept_1: true, concept_2: false, ...}}'
    )
    raw = call_vlm(client, model_name, prompt, image_paths=[image_path], debug=debug)
    parsed = extract_json(raw)
    if not isinstance(parsed, dict):
        return {c["name"]: True for c in normal_concepts}
    return {c["name"]: bool(parsed.get(c["name"], True)) for c in normal_concepts}


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 4 — DEFECT CONCEPT POST-PROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def _compute_visual_prototypes(concept_names, defect_type, all_annotations,
                                clip_model, clip_preprocess, device) -> dict[str, np.ndarray]:
    """Average CLIP image embedding for each concept (its visual prototype)."""
    defect_anns = [a for a in all_annotations if a.get("anomaly_type") == defect_type]
    prototypes: dict[str, np.ndarray] = {}
    for name in concept_names:
        imgs = [a["image_path"] for a in defect_anns
                if a.get("concept_vector", {}).get(name, False)]
        if not imgs:
            continue
        embs = []
        for p in imgs:
            try:
                t = clip_preprocess(Image.open(p).convert("RGB")).unsqueeze(0).to(device)
                with torch.no_grad():
                    e = clip_model.encode_image(t)
                    e = e / e.norm(dim=-1, keepdim=True)
                embs.append(e.cpu().numpy()[0])
            except Exception:
                continue
        if embs:
            prototypes[name] = np.mean(embs, axis=0)
    return prototypes


def stage4_refine_defect_concepts(
    defect_concept_map: dict[str, list[dict]],
    all_annotations: list[dict],
    image_groups: dict[str, list[str]],
    cluster_threshold: float = 0.65,
    min_defect_freq: float = 0.15,
    min_defect_types_for_generic: int = 2,
    use_visual_grounding: bool = False,
    visual_weight: float = 0.60,
    clip_model=None, clip_preprocess=None, clip_device: str = "cpu",
) -> tuple[dict[str, list[dict]], list[dict]]:
    """
    Refine defect concepts per defect type.

    4a  Dimension-aware text clustering + optional CLIP visual grounding.
        Separate cluster_threshold from Stage 2 for independent tuning.
    4b  Cluster-UNION frequency filter (FIXED):
        freq = images where ANY cluster member is True / total defect images
    4c  Cross-defect generic concept discovery.

    Returns (refined_defect_concept_map, discovered_generic_concepts)
    """
    log.info("\n" + "═" * 60)
    log.info("STAGE 4 — Defect concept clustering and refinement")
    log.info("═" * 60)

    if not CLUSTERING_AVAILABLE:
        log.warning("Stage 4: sentence-transformers unavailable — skipping")
        return defect_concept_map, []

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    refined: dict[str, list[dict]] = {}

    for defect_type, concepts in defect_concept_map.items():
        if not concepts:
            refined[defect_type] = []
            continue

        n_defect_imgs = len(image_groups.get(defect_type, []))
        if n_defect_imgs == 0:
            refined[defect_type] = concepts
            continue

        # Group by visual dimension
        dim_to_concepts: dict[str, list[dict]] = {}
        for c in concepts:
            dim = normalize_dimension(c.get("visual_dimension", ""))
            dim_to_concepts.setdefault(dim, []).append(c)

        # cluster_members_map[canonical_name] = all original names in that cluster
        cluster_members_map: dict[str, list[str]] = {}
        all_kept: list[dict] = []

        for dim, dim_concepts in dim_to_concepts.items():
            names = [c["name"] for c in dim_concepts]
            threshold = get_dimension_threshold(dim, cluster_threshold)

            if use_visual_grounding and CLIP_AVAILABLE and clip_model is not None and len(names) > 1:
                prototypes = _compute_visual_prototypes(
                    names, defect_type, all_annotations,
                    clip_model, clip_preprocess, clip_device)
                n_proto = sum(1 for n in names if n in prototypes)
                log.info(f"  Stage 4: visual grounding for {len(names)} '{dim}' concepts "
                         f"in '{defect_type}' — {n_proto}/{len(names)} have prototypes")

                text_embs = embedder.encode(names, normalize_embeddings=True)
                text_sim = text_embs @ text_embs.T
                vis_sim = np.eye(len(names))
                for i, ni in enumerate(names):
                    for j, nj in enumerate(names):
                        if i != j and ni in prototypes and nj in prototypes:
                            vis_sim[i, j] = float(prototypes[ni] @ prototypes[nj])

                combined = visual_weight * vis_sim + (1 - visual_weight) * text_sim
                dist = np.clip(1.0 - combined, 0.0, 2.0)
                np.fill_diagonal(dist, 0.0)

                clustering = AgglomerativeClustering(
                    n_clusters=None, distance_threshold=1.0 - threshold,
                    metric="precomputed", linkage="average",
                )
                labels = clustering.fit_predict(dist)
            else:
                log.info(f"  Stage 4: clustering {len(names)} '{dim}' concepts "
                         f"for '{defect_type}' with threshold {threshold:.2f}")
                if len(names) == 1:
                    labels = [0]
                else:
                    embs = embedder.encode(names, normalize_embeddings=True)
                    clustering = AgglomerativeClustering(
                        n_clusters=None, distance_threshold=1.0 - threshold,
                        metric="cosine", linkage="average",
                    )
                    labels = clustering.fit_predict(embs)

            # Build clusters and pick canonical by frequency
            name_freq = Counter(names)
            clusters: dict[int, list[dict]] = {}
            for c, label in zip(dim_concepts, labels):
                clusters.setdefault(int(label), []).append(c)

            for members in clusters.values():
                member_names = [m["name"] for m in members]
                canonical = max(members, key=lambda c: (name_freq[c["name"]], -len(c["name"])))
                all_kept.append(canonical)
                cluster_members_map[canonical["name"]] = member_names

        log.info(f"Stage 4: '{defect_type}' had {len(concepts)} → {len(all_kept)} after clustering")

        # 4b. Cluster-union frequency filter (FIXED)
        final_for_type: list[dict] = []
        for c in all_kept:
            members = cluster_members_map.get(c["name"], [c["name"]])
            count = sum(
                1 for ann in all_annotations
                if ann.get("anomaly_type") == defect_type
                and any(ann.get("concept_vector", {}).get(m, False) for m in members)
            )
            freq = count / n_defect_imgs
            log.info(f"  Stage 4: '{c['name']}' freq={freq:.2f} "
                     f"(cluster members: {members})")
            if freq >= min_defect_freq:
                final_for_type.append(c)
            else:
                log.info(f"  Stage 4: removed '{c['name']}' from '{defect_type}' "
                         f"(freq={freq:.2f} < {min_defect_freq})")

        log.info(f"  Stage 4: '{defect_type}' final concepts: "
                 f"{[c['name'] for c in final_for_type]}")
        refined[defect_type] = final_for_type

    # 4c. Discover cross-defect generic concepts
    concept_to_defect_types: dict[str, set[str]] = {}
    for dt, concepts in refined.items():
        for c in concepts:
            concept_to_defect_types.setdefault(c["name"], set()).add(dt)

    discovered_generic: list[dict] = []
    for dt in list(refined.keys()):
        keep, move = [], []
        for c in refined[dt]:
            if len(concept_to_defect_types.get(c["name"], set())) >= min_defect_types_for_generic:
                move.append(c)
            else:
                keep.append(c)
        if move:
            refined[dt] = keep
        for c in move:
            if not any(g["name"] == c["name"] for g in discovered_generic):
                discovered_generic.append(c)
                log.info(f"  Stage 4: generic '{c['name']}' spans "
                         f"{list(concept_to_defect_types[c['name']])}")

    log.info(f"Stage 4: {len(discovered_generic)} cross-defect generic concepts found")
    return refined, discovered_generic


def collect_defect_concepts(all_annotations: list[dict]) -> dict[str, list[dict]]:
    """Collect raw new defect concepts per defect type from Stage 3 annotations."""
    defect_concepts: dict[str, dict] = {}
    for ann in all_annotations:
        dt = ann.get("defect_category", "unknown")
        if dt == "good":
            continue
        for c in ann.get("new_defect_concepts", []):
            defect_concepts.setdefault(dt, {})[c["name"]] = c
    return {k: list(v.values()) for k, v in defect_concepts.items()}


# ══════════════════════════════════════════════════════════════════════════════
# VOCABULARY BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_full_concept_vocabulary(
    normal_concepts: list[dict],
    defect_concept_map: dict[str, list[dict]],
    holdout_defect: str | None = None,
    generic_concepts: list[dict] | None = None,
    discovered_generic: list[dict] | None = None,
) -> list[str]:
    """Build three-tier vocabulary. Tier 2 is never excluded by holdout."""
    vocab: list[str] = [c["name"] for c in normal_concepts]
    n_normal = len(vocab)

    for c in (generic_concepts or []):
        if c["name"] not in vocab:
            vocab.append(c["name"])
    for c in (discovered_generic or []):
        if c["name"] not in vocab:
            vocab.append(c["name"])
    n_generic = len(vocab) - n_normal

    excluded: set[str] = set()
    if holdout_defect:
        excluded = {c["name"] for c in defect_concept_map.get(holdout_defect, [])}
        log.info(f"Hold-out: excluding {len(excluded)} concepts "
                 f"unique to '{holdout_defect}': {excluded}")

    n_before = len(vocab)
    for dt, concepts in defect_concept_map.items():
        if dt == holdout_defect:
            continue
        for c in concepts:
            if c["name"] not in vocab and c["name"] not in excluded:
                vocab.append(c["name"])
    n_defect = len(vocab) - n_before

    log.info(f"Vocabulary tiers: {n_normal} normal + {n_generic} generic "
             f"+ {n_defect} defect-specific = {len(vocab)} total"
             + (f"  (holdout={holdout_defect})" if holdout_defect else ""))
    return vocab


# ══════════════════════════════════════════════════════════════════════════════
# CSV BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_csv(dataset_path, category, image_groups, normal_concepts,
               all_annotations, final_vocab, holdout_defect=None, random_seed=42):
    """Build CONVAD-compatible CSV. Schema matches original pipeline exactly."""
    ann_index = {a["image_path"]: a for a in all_annotations}
    rows = []

    for img_path in image_groups.get("normal", []):
        cv = ann_index.get(img_path, {}).get("concept_vector", {})
        row = {"image_path": img_path, "label_index": 0,
               "mask_path": "", "anomaly_type": "good"}
        for concept in final_vocab:
            row[concept] = int(bool(cv.get(concept, True)))
        rows.append(row)

    for img_path in image_groups.get("normal_test", []):
        cv = ann_index.get(img_path, {}).get("concept_vector", {})
        row = {"image_path": img_path, "label_index": 0,
               "mask_path": "", "anomaly_type": "good"}
        for concept in final_vocab:
            row[concept] = int(bool(cv.get(concept, True)))
        rows.append(row)

    for dt, img_paths in image_groups.items():
        if dt in ("normal", "normal_test"):
            continue
        for img_path in img_paths:
            cv = ann_index.get(img_path, {}).get("concept_vector", {})
            row = {"image_path": img_path, "label_index": 1,
                   "mask_path": get_mask_path(dataset_path, category, dt, img_path),
                   "anomaly_type": dt}
            for concept in final_vocab:
                row[concept] = int(bool(cv.get(concept, False)))
            rows.append(row)

    df = pd.DataFrame(rows)
    rng = random.Random(random_seed)
    indices = list(df.index)
    rng.shuffle(indices)
    n = len(indices)
    n_test = max(1, n // 10)
    n_val  = max(1, n // 10)
    test_idx = set(indices[:n_test])
    val_idx  = set(indices[n_test:n_test + n_val])
    df["split"] = ["test" if i in test_idx else "val" if i in val_idx else "train"
                   for i in df.index]
    meta = ["image_path", "label_index", "mask_path", "anomaly_type", "split"]
    concept_cols = [c for c in final_vocab if c in df.columns]
    df = df[meta + concept_cols]
    log.info(f"CSV built: {len(df)} rows × {len(df.columns)} columns "
             f"({len(concept_cols)} concept columns)")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ORCHESTRATION
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(args: argparse.Namespace):
    random.seed(args.random_seed)
    client = Client(host=args.ollama_host)

    log.info(f"Ollama host  : {args.ollama_host}")
    log.info(f"Model        : {args.model_name}")
    log.info(f"Category     : {args.category}")
    log.info(f"Dataset      : {args.dataset_path}")
    log.info(f"Normal refs  : {args.n_normal_refs} [P1-A]")
    log.info(f"Async workers: {args.n_workers} [P2-B]")
    log.info(f"Visual gnd.  : {'on' if args.use_visual_grounding else 'off'}")
    if args.holdout_defect:
        log.info(f"Hold-out     : {args.holdout_defect}")

    # Discover images
    image_groups = discover_images(args.dataset_path, args.category)
    defect_types = [k for k in image_groups if k not in ("normal", "normal_test")]
    if args.holdout_defect and args.holdout_defect not in defect_types:
        raise ValueError(f"Holdout '{args.holdout_defect}' not in {defect_types}")
    all_normal = image_groups["normal"]

    # CLIP setup [P1-A, P3-B]
    clip_model = clip_preprocess = None
    clip_device = "cpu"
    normal_embeddings: dict[str, np.ndarray] = {}

    if CLIP_AVAILABLE:
        log.info("\n" + "═" * 60)
        log.info("CLIP SETUP  [P1-A + P3-B]")
        log.info("═" * 60)
        clip_device = "cuda" if torch.cuda.is_available() else "cpu"
        clip_model, clip_preprocess = clip.load("ViT-B/32", device=clip_device)
        clip_model.eval()
        log.info(f"  CLIP ViT-B/32 on {clip_device}")
        cache_path = str(Path(args.save_path).parent /
                         f"{args.category}_clip_normal_embeddings.npz")
        normal_embeddings = load_or_compute_clip_embeddings(
            all_normal, clip_model, clip_preprocess, clip_device, cache_path=cache_path)

    # Stage 1
    log.info("\n" + "═" * 60)
    log.info("STAGE 1 — Per-image normal concept extraction")
    log.info("═" * 60)
    raw_concepts = stage1_extract_normal_concepts(
        client, args.model_name, all_normal, args.category,
        n_sample=args.n_normal_sample, debug=args.debug)

    stage1_path = Path(args.save_path).parent / f"{args.category}_stage1_raw_concepts.json"
    with open(stage1_path, "w") as f:
        json.dump(raw_concepts, f, indent=2)
    log.info(f"Stage 1 raw concepts saved → {stage1_path}")

    # Stage 2
    log.info("\n" + "═" * 60)
    log.info("STAGE 2 — Normal concept post-processing")
    log.info("═" * 60)
    normal_concepts = stage2_build_normal_dictionary(
        client, args.model_name, raw_concepts, all_normal, args.category,
        min_freq=args.min_concept_freq, max_freq=args.max_concept_freq,
        cluster_threshold=args.cluster_threshold,
        run_vlm_audit=not args.skip_vlm_audit, debug=args.debug)

    stage2_path = Path(args.save_path).parent / f"{args.category}_stage2_normal_dict.json"
    with open(stage2_path, "w") as f:
        json.dump(normal_concepts, f, indent=2)
    log.info(f"Normal dictionary saved → {stage2_path}")

    # Stage 2b
    all_concepts = stage2b_add_generic_concepts(normal_concepts)

    # Stage 3
    log.info("\n" + "═" * 60)
    log.info("STAGE 3 — Per-image annotation")
    log.info("═" * 60)
    log.info(f"Annotating with {len(all_concepts)} concepts "
             f"({len(normal_concepts)} normal + "
             f"{len(all_concepts) - len(normal_concepts)} generic Tier 2)")

    all_annotations: list[dict] = []
    annotation_lock = threading.Lock()

    def _append(ann):
        with annotation_lock:
            all_annotations.append(ann)

    # Compute defect CLIP embeddings for P1-A reference selection
    defect_embeddings: dict[str, np.ndarray] = {}
    if normal_embeddings and args.n_normal_refs > 1:
        all_defect_paths = [p for k, v in image_groups.items()
                            if k not in ("normal", "normal_test") for p in v]
        log.info(f"  Computing CLIP embeddings for {len(all_defect_paths)} defect images...")
        defect_embeddings = load_or_compute_clip_embeddings(
            all_defect_paths, clip_model, clip_preprocess, clip_device)

    def _get_refs(img_path: str) -> list[str]:
        if normal_embeddings and args.n_normal_refs > 1 and img_path in defect_embeddings:
            return select_normal_references(
                defect_embeddings[img_path], normal_embeddings, k=args.n_normal_refs)
        return [all_normal[len(all_normal) // 2]]

    # 3a. Normal train
    normal_sample = all_normal[:args.n_annotate_sample] if args.n_annotate_sample else all_normal
    log.info(f"\nAnnotating {len(normal_sample)} normal (train) images...")
    for i, img_path in enumerate(normal_sample, 1):
        if i % 20 == 0 or i == len(normal_sample):
            log.info(f"  [{i}/{len(normal_sample)}] {Path(img_path).name}")
        cv = annotate_normal_image(
            client, args.model_name, img_path, all_concepts, args.category, debug=args.debug)
        _append({"image_path": img_path, "anomaly_type": "good",
                 "concept_vector": cv, "new_defect_concepts": [], "defect_category": "good"})

    # 3b. Normal test
    normal_test = image_groups.get("normal_test", [])
    if args.n_annotate_sample:
        normal_test = normal_test[:args.n_annotate_sample]
    log.info(f"\nAnnotating {len(normal_test)} normal (test) images...")
    for i, img_path in enumerate(normal_test, 1):
        if i % 20 == 0 or i == len(normal_test):
            log.info(f"  [{i}/{len(normal_test)}] {Path(img_path).name}")
        cv = annotate_normal_image(
            client, args.model_name, img_path, all_concepts, args.category, debug=args.debug)
        _append({"image_path": img_path, "anomaly_type": "good",
                 "concept_vector": cv, "new_defect_concepts": [], "defect_category": "good"})

    # 3c. Defect images — async [P2-B]
    def _annotate_defect(img_path: str, defect_type: str) -> dict:
        refs = _get_refs(img_path)
        ann = annotate_defect_image(
            client, args.model_name, img_path, refs, all_concepts,
            defect_type, args.category,
            max_new_concepts=args.max_new_defect_concepts, debug=args.debug)
        cv = dict(ann["normal_concept_annotations"])
        for c in ann["new_defect_concepts"]:
            cv[c["name"]] = True
        return {"image_path": img_path, "anomaly_type": defect_type,
                "concept_vector": cv, "new_defect_concepts": ann["new_defect_concepts"],
                "defect_category": defect_type}

    defect_tasks: list[tuple[str, str]] = []
    for dt in defect_types:
        imgs = image_groups[dt]
        if args.n_annotate_sample:
            imgs = imgs[:args.n_annotate_sample]
        defect_tasks.extend((p, dt) for p in imgs)

    log.info(f"\nAnnotating {len(defect_tasks)} defective images "
             f"with {args.n_workers} worker(s)...")
    completed = 0
    with ThreadPoolExecutor(max_workers=args.n_workers) as ex:
        futures = {ex.submit(_annotate_defect, p, dt): (p, dt) for p, dt in defect_tasks}
        for fut in as_completed(futures):
            _append(fut.result())
            completed += 1
            if completed % 10 == 0 or completed == len(defect_tasks):
                log.info(f"  Defect annotation: {completed}/{len(defect_tasks)}")

    # Stage 4
    defect_concept_map = collect_defect_concepts(all_annotations)
    log.info("\nDefect-specific concepts discovered (raw):")
    for dt, concepts in defect_concept_map.items():
        log.info(f"  {dt}: {[c['name'] for c in concepts]}")

    if args.skip_stage4:
        log.info("\nStage 4 skipped — using raw defect concepts (v2 mode)")
        refined_defect_map = defect_concept_map
        discovered_generic: list[dict] = []
    else:
        refined_defect_map, discovered_generic = stage4_refine_defect_concepts(
            defect_concept_map, all_annotations, image_groups,
            cluster_threshold=args.stage4_cluster_threshold,
            min_defect_freq=args.min_defect_freq,
            min_defect_types_for_generic=args.min_defect_types_for_generic,
            use_visual_grounding=args.use_visual_grounding,
            visual_weight=args.visual_weight,
            clip_model=clip_model, clip_preprocess=clip_preprocess, clip_device=clip_device,
        )

    # Build vocabulary
    final_vocab = build_full_concept_vocabulary(
        normal_concepts=normal_concepts,
        defect_concept_map=refined_defect_map,
        holdout_defect=args.holdout_defect,
        generic_concepts=all_concepts[len(normal_concepts):],
        discovered_generic=discovered_generic,
    )

    # Build and save CSV
    df = build_csv(args.dataset_path, args.category, image_groups,
                   normal_concepts, all_annotations, final_vocab,
                   holdout_defect=args.holdout_defect, random_seed=args.random_seed)

    Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.save_path, index=False)
    log.info(f"\n✓ Final CSV saved → {args.save_path}")

    # Summary
    n_tier3 = sum(len(v) for v in refined_defect_map.values())
    n_generic_fixed = len(GENERIC_ANOMALY_CONCEPTS)
    n_generic_disc  = len(discovered_generic)
    log.info("\n" + "═" * 60)
    log.info("PIPELINE SUMMARY")
    log.info("═" * 60)
    log.info(f"  Tier 1 normal concepts    : {len(normal_concepts)}")
    log.info(f"  Tier 2 generic concepts   : {n_generic_fixed} fixed + "
             f"{n_generic_disc} discovered = {n_generic_fixed + n_generic_disc}")
    log.info(f"  Tier 3 defect concepts    : {n_tier3}")
    log.info(f"  Final vocab size          : {len(final_vocab)}")
    log.info(f"  Total images annotated    : {len(all_annotations)}")
    log.info(f"  CSV shape                 : {df.shape}")
    log.info(f"  Hold-out defect           : {args.holdout_defect or 'none'}")
    log.info(f"  Stage 4                   : {'skipped (v2 mode)' if args.skip_stage4 else 'on'}")
    log.info(f"  VLM audit P1-B            : {'on' if not args.skip_vlm_audit else 'off'}")
    log.info(f"  CLIP refs P1-A            : {args.n_normal_refs}")
    log.info(f"  Visual grounding Stage 4  : {'on' if args.use_visual_grounding else 'off'}")
    log.info(f"  Async workers P2-B        : {args.n_workers}")
    log.info("═" * 60)
    return df, normal_concepts, refined_defect_map


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Two-stage VLM concept annotation pipeline for CONVAD (v5)")

    p.add_argument("--dataset_path",    required=True)
    p.add_argument("--category",        required=True)
    p.add_argument("--model_name",      default="gemma4:e4b")
    p.add_argument("--ollama_host",     default="http://localhost:6000")
    p.add_argument("--save_path",       required=True)
    p.add_argument("--holdout_defect",  default=None)
    p.add_argument("--random_seed",     type=int, default=42)
    p.add_argument("--debug",           action="store_true")

    # Sampling
    p.add_argument("--n_normal_sample",    type=int, default=None,
                   help="Sample N normal images for Stage 1 (default: all)")
    p.add_argument("--n_annotate_sample",  type=int, default=None,
                   help="Limit Stage 3 to N images per group (default: all)")

    # Stage 2 — Normal concept clustering
    p.add_argument("--min_concept_freq",   type=float, default=0.20,
                   help="Min fraction of normal images a concept must appear in (default: 0.20)")
    p.add_argument("--max_concept_freq",   type=float, default=0.95)
    p.add_argument("--cluster_threshold",  type=float, default=0.65,
                   help="Stage 2 dim-aware clustering threshold (default: 0.65)")
    p.add_argument("--skip_vlm_audit",     action="store_true",
                   help="Skip Stage 2d VLM self-audit [P1-B]")
    p.add_argument("--skip_stage4",        action="store_true",
                   help="Skip Stage 4 defect concept clustering — use raw defect concepts "
                        "as in v2 (recommended for baseline reproduction)")

    # Stage 3 — Defect annotation
    p.add_argument("--n_normal_refs",      type=int, default=3,
                   help="CLIP-selected normal references per defect image [P1-A] (default: 3)")
    p.add_argument("--max_new_defect_concepts", type=int, default=4,
                   help="Max new defect concepts the VLM extracts per image (default: 4)")

    # Stage 4 — Defect concept refinement
    p.add_argument("--stage4_cluster_threshold", type=float, default=0.65,
                   help="Stage 4 clustering threshold, separate from Stage 2 (default: 0.65)")
    p.add_argument("--min_defect_freq",    type=float, default=0.15,
                   help="Min cluster-union frequency for defect concepts (default: 0.15)")
    p.add_argument("--min_defect_types_for_generic", type=int, default=2,
                   help="Concepts spanning N+ defect types → generic tier (default: 2)")
    p.add_argument("--use_visual_grounding", action="store_true",
                   help="Enable CLIP visual grounding in Stage 4 clustering")
    p.add_argument("--visual_weight",      type=float, default=0.60,
                   help="Visual vs text weight in Stage 4 combined similarity (default: 0.60)")

    # Performance
    p.add_argument("--n_workers",          type=int, default=1,
                   help="Parallel workers for Stage 3 [P2-B] (default: 1)")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args)