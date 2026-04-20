"""
stages/stage2.py — Normal concept post-processing (Stage 2) and
                   fixed generic concept injection (Stage 2b).

Steps:
  2a  Dimension-aware embedding clustering  (groups semantically identical concepts)
  2b  VLM refinement for large clusters     (picks the best canonical name)
  2c  Frequency filter                      (keeps concepts seen in min-max% of images)
  2d  VLM self-audit [P1-B]                (removes defect-referencing concepts)
  2e  (in stage2b fn) Add 5 fixed Tier 2 generic concepts
"""

from __future__ import annotations

import json
import logging
from collections import Counter

from annotation_pipeline.config import (
    GENERIC_ANOMALY_CONCEPTS,
    PROMPT_VLM_MERGE,
    PROMPT_NORMAL_CONCEPT_AUDIT,
)
from annotation_pipeline.utils import (
    call_vlm, extract_json, normalize_dimension,
    get_dimension_threshold, to_snake_case,
)

log = logging.getLogger(__name__)

# ── Optional clustering dependencies ─────────────────────────────────────────

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.cluster import AgglomerativeClustering
    CLUSTERING_AVAILABLE = True
except ImportError:
    CLUSTERING_AVAILABLE = False
    log.warning(
        "sentence-transformers or sklearn not available — "
        "falling back to VLM-only deduplication for Stage 2."
    )


# ── Internal helpers ──────────────────────────────────────────────────────────

def _cluster_by_embedding(
    names: list[str],
    embedder,
    threshold: float,
    name_freq: Counter,
) -> dict[str, str]:
    """
    Cluster concept names using embedding similarity.

    Parameters
    ----------
    names       : unique concept names to cluster
    embedder    : SentenceTransformer instance
    threshold   : cosine similarity threshold (higher = harder to merge)
    name_freq   : Counter of name → total occurrences across all raw concepts
                  Used to pick the most-seen name as canonical.

    Returns
    -------
    dict mapping every name → canonical name
    """
    if len(names) == 1:
        return {names[0]: names[0]}

    embs = embedder.encode(names, normalize_embeddings=True)
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=1.0 - threshold,
        metric="cosine",
        linkage="average",
    )
    labels = clustering.fit_predict(embs)

    clusters: dict[int, list[str]] = {}
    for name, label in zip(names, labels):
        clusters.setdefault(int(label), []).append(name)

    canonical_map: dict[str, str] = {}
    for members in clusters.values():
        # Pick the most frequent name; break ties by preferring shorter names
        canonical = max(members, key=lambda n: (name_freq[n], -len(n)))
        for m in members:
            canonical_map[m] = canonical
    return canonical_map


# ── Stage 2 main function ─────────────────────────────────────────────────────

def run(
    client,
    model_name: str,
    raw_concepts: list[dict],
    normal_images: list[str],
    category: str,
    min_freq: float = 0.20,
    max_freq: float = 0.95,
    cluster_threshold: float = 0.65,
    run_vlm_audit: bool = True,
    debug: bool = False,
) -> list[dict]:
    """
    Build the shared normal concept dictionary from Stage 1 raw output.

    Returns a list of concept dicts (name, description, visual_dimension, frequency).
    """
    log.info("Stage 2: building shared normal concept dictionary...")

    n_images = len({c["source_image"] for c in raw_concepts})
    all_unique_names = list({c["name"] for c in raw_concepts})
    log.info("  Unique concept names before clustering: %d", len(all_unique_names))

    # ── 2a. Dimension-aware embedding clustering ───────────────────────────────
    canonical_map: dict[str, str] = {}

    if CLUSTERING_AVAILABLE and len(all_unique_names) > 1:
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        # name_freq: how many times each name appeared across ALL images (image-level count)
        name_freq: Counter = Counter()
        for c in raw_concepts:
            name_freq[c["name"]] += 1

        # Determine dominant visual dimension per concept name
        name_to_dims: dict[str, Counter] = {}
        for c in raw_concepts:
            name_to_dims.setdefault(c["name"], Counter())[
                normalize_dimension(c.get("visual_dimension", ""))
            ] += 1
        name_to_dim = {n: dc.most_common(1)[0][0] for n, dc in name_to_dims.items()}

        # Group concepts by dimension, cluster each group independently
        dim_groups: dict[str, list[str]] = {}
        for name in all_unique_names:
            dim_groups.setdefault(name_to_dim.get(name, "unknown"), []).append(name)

        n_canonical = 0
        for dim, group_names in dim_groups.items():
            thresh = get_dimension_threshold(dim, cluster_threshold)
            log.info(
                "  Clustering %3d '%s' concepts with threshold %.2f",
                len(group_names), dim, thresh,
            )
            group_map = _cluster_by_embedding(group_names, embedder, thresh, name_freq)
            canonical_map.update(group_map)
            n_canonical += len(set(group_map.values()))

        log.info("  Clustering: %d unique names → %d canonical", len(all_unique_names), n_canonical)

        # ── 2b. VLM refinement for large clusters ─────────────────────────────
        cluster_to_members: dict[str, list[str]] = {}
        for name in all_unique_names:
            cluster_to_members.setdefault(canonical_map[name], []).append(name)
        large = {c: m for c, m in cluster_to_members.items() if len(m) > 3}

        if large:
            log.info("  Refining %d large clusters with VLM...", len(large))
            groups_json = json.dumps(
                [{"canonical": k, "members": v} for k, v in large.items()], indent=2
            )
            raw_vlm = call_vlm(
                client, model_name,
                PROMPT_VLM_MERGE.format(json_groups=groups_json),
                debug=debug,
            )
            vlm_map = extract_json(raw_vlm)
            if isinstance(vlm_map, dict):
                for orig, canon in vlm_map.items():
                    if orig in canonical_map:
                        canonical_map[orig] = to_snake_case(str(canon))

    else:
        # Fallback: VLM-only deduplication (no sentence-transformers)
        log.info("  Running VLM-only deduplication...")
        raw_vlm = call_vlm(
            client, model_name,
            f"Group these concepts from {category} images by same meaning. "
            f"Return JSON: {{representative: [members]}}.\n"
            f"Concepts: {json.dumps(all_unique_names)}",
            debug=debug,
        )
        parsed = extract_json(raw_vlm)
        if isinstance(parsed, dict):
            for rep, members in parsed.items():
                canon = to_snake_case(str(rep))
                if isinstance(members, list):
                    for m in members:
                        canonical_map[to_snake_case(str(m))] = canon
                canonical_map[canon] = canon
        for name in all_unique_names:
            canonical_map.setdefault(name, name)

    # Apply canonical mapping to all raw concepts
    for c in raw_concepts:
        c["canonical_name"] = canonical_map.get(c["name"], c["name"])

    # ── 2c. Frequency filter ───────────────────────────────────────────────────
    # Count how many distinct images contain each canonical concept (union of all members)
    canonical_to_images: dict[str, set] = {}
    for c in raw_concepts:
        canonical_to_images.setdefault(c["canonical_name"], set()).add(c["source_image"])

    final_concepts: list[dict] = []
    for canon, image_set in canonical_to_images.items():
        freq = len(image_set) / n_images
        if min_freq <= freq <= max_freq:
            variants = [c for c in raw_concepts if c["canonical_name"] == canon]
            dim_counter = Counter(v["visual_dimension"] for v in variants)
            best_dim = dim_counter.most_common(1)[0][0]
            best_desc = max((v["description"] for v in variants), key=len, default="")
            final_concepts.append({
                "name": canon,
                "description": best_desc,
                "visual_dimension": best_dim,
                "frequency": round(freq, 3),
            })

    final_concepts.sort(key=lambda x: -x["frequency"])
    log.info(
        "  After frequency filter: %d concepts (min=%.2f, max=%.2f)",
        len(final_concepts), min_freq, max_freq,
    )

    # ── 2d. VLM self-audit (P1-B) ─────────────────────────────────────────────
    if run_vlm_audit and final_concepts:
        log.info("  Stage 2d: VLM self-audit...")
        audit_input = [
            {"name": c["name"], "description": c["description"]} for c in final_concepts
        ]
        raw_audit = call_vlm(
            client, model_name,
            PROMPT_NORMAL_CONCEPT_AUDIT.format(
                category=category, concepts_json=json.dumps(audit_input, indent=2)
            ),
            debug=debug,
        )
        audit_result = extract_json(raw_audit)
        if isinstance(audit_result, dict):
            before = len(final_concepts)
            final_concepts = [
                c for c in final_concepts
                if str(audit_result.get(c["name"], "keep")).lower() != "remove"
            ]
            log.info(
                "  VLM audit: %d removed, %d kept",
                before - len(final_concepts), len(final_concepts),
            )
        else:
            log.warning("  VLM audit parse failed — keeping all concepts")

    log.info("Stage 2 complete: %d concepts in shared normal dictionary", len(final_concepts))
    for c in final_concepts:
        log.info(
            "  %-42s freq=%.2f  dim=%s",
            c["name"], c["frequency"], c["visual_dimension"],
        )
    return final_concepts


# ── Stage 2b — inject fixed generic concepts ──────────────────────────────────

def add_generic_concepts(normal_concepts: list[dict]) -> list[dict]:
    """
    Append the 5 fixed Tier 2 generic anomaly concepts.
    These are always in the vocabulary and are never excluded by the holdout.
    """
    existing = {c["name"] for c in normal_concepts}
    all_concepts = list(normal_concepts)
    added = 0
    for gc in GENERIC_ANOMALY_CONCEPTS:
        if gc["name"] not in existing:
            all_concepts.append(gc)
            added += 1
    log.info("Stage 2b: added %d fixed generic anomaly concepts (Tier 2)", added)
    return all_concepts
