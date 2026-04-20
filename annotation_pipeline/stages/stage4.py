"""
stages/stage4.py — Defect concept post-processing (Stage 4).

Steps:
  4a  Dimension-aware text clustering per defect type
      Optional: CLIP visual grounding (blends visual + text similarity)
  4b  Cluster-union frequency filter  ← BUG FIXED: counts image-level occurrences
      (freq = images where ANY cluster member is True / total defect images)
  4c  Cross-defect generic concept discovery
      Concepts appearing in N+ defect types → promoted to Tier 2 generic

Design goal — explainability vs performance trade-off:
  The ideal vocabulary is ~5-8 semantically DISTINCT concepts per defect type.
  This avoids the two failure modes:
    - Too few (v3/v4b: 24 total) → CBM starved, C-AUC drops
    - Too many / redundant (v2: 97 total) → noisy explanations
  The cluster-union freq fix means we now correctly measure concept frequency,
  so a looser min_defect_freq (default 0.20) keeps more good concepts while
  clustering still removes true synonyms.
"""

from __future__ import annotations

import logging
from collections import Counter

import numpy as np

from annotation_pipeline.utils import normalize_dimension, get_dimension_threshold

log = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.cluster import AgglomerativeClustering
    CLUSTERING_AVAILABLE = True
except ImportError:
    CLUSTERING_AVAILABLE = False

try:
    import torch
    from PIL import Image as _PILImage
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False


# ── Visual prototype computation (for CLIP visual grounding) ──────────────────

def _compute_visual_prototypes(
    concept_names: list[str],
    defect_type: str,
    all_annotations: list[dict],
    clip_model,
    clip_preprocess,
    device: str,
) -> dict[str, np.ndarray]:
    """
    For each concept name, compute the average CLIP embedding over all defect
    images of that type where the concept is annotated True.
    Returns {concept_name: mean_embedding}.
    """
    defect_anns = [a for a in all_annotations if a.get("anomaly_type") == defect_type]
    prototypes: dict[str, np.ndarray] = {}
    for name in concept_names:
        imgs = [
            a["image_path"] for a in defect_anns
            if a.get("concept_vector", {}).get(name, False)
        ]
        if not imgs:
            continue
        embs = []
        for p in imgs:
            try:
                t = clip_preprocess(_PILImage.open(p).convert("RGB")).unsqueeze(0).to(device)
                with torch.no_grad():
                    e = clip_model.encode_image(t)
                    e = e / e.norm(dim=-1, keepdim=True)
                embs.append(e.cpu().numpy()[0])
            except Exception:
                continue
        if embs:
            prototypes[name] = np.mean(embs, axis=0)
    return prototypes


# ── Stage 4 main function ─────────────────────────────────────────────────────

def run(
    defect_concept_map: dict[str, list[dict]],
    all_annotations: list[dict],
    image_groups: dict[str, list[str]],
    cluster_threshold: float = 0.65,
    min_defect_freq: float = 0.20,
    min_defect_types_for_generic: int = 2,
    use_visual_grounding: bool = False,
    visual_weight: float = 0.60,
    clip_model=None,
    clip_preprocess=None,
    clip_device: str = "cpu",
) -> tuple[dict[str, list[dict]], list[dict]]:
    """
    Refine defect concepts per defect type.

    Returns
    -------
    (refined_defect_map, discovered_generic_concepts)
      refined_defect_map : {defect_type: [concept_dict, ...]}
      discovered_generic : concepts promoted to Tier 2 (appear in multiple defect types)
    """
    if not CLUSTERING_AVAILABLE:
        log.warning("Stage 4: sentence-transformers unavailable — returning raw concepts")
        return defect_concept_map, []

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    refined: dict[str, list[dict]] = {}

    for defect_type, concepts in defect_concept_map.items():
        if not concepts:
            refined[defect_type] = []
            continue

        n_defect_imgs = len(image_groups.get(defect_type, []))
        if n_defect_imgs == 0:
            refined[defect_type] = list(concepts)
            continue

        # ── 4a. Cluster per dimension ──────────────────────────────────────────

        # Build image-level occurrence count for each concept name
        # FIXED: count how many defect images (of this type) have this concept True
        defect_anns = [
            a for a in all_annotations if a.get("anomaly_type") == defect_type
        ]
        image_freq: Counter = Counter()
        for ann in defect_anns:
            for name, val in ann.get("concept_vector", {}).items():
                if val:
                    image_freq[name] += 1

        dim_to_concepts: dict[str, list[dict]] = {}
        for c in concepts:
            dim = normalize_dimension(c.get("visual_dimension", ""))
            dim_to_concepts.setdefault(dim, []).append(c)

        # cluster_members_map tracks which original names merged into each canonical
        cluster_members_map: dict[str, list[str]] = {}
        all_kept: list[dict] = []

        for dim, dim_concepts in dim_to_concepts.items():
            names = [c["name"] for c in dim_concepts]
            threshold = get_dimension_threshold(dim, cluster_threshold)

            if (
                use_visual_grounding
                and CLIP_AVAILABLE
                and clip_model is not None
                and len(names) > 1
            ):
                # Combined text + visual similarity matrix
                prototypes = _compute_visual_prototypes(
                    names, defect_type, all_annotations, clip_model, clip_preprocess, clip_device
                )
                n_proto = sum(1 for n in names if n in prototypes)
                log.info(
                    "  Stage 4 [visual]: '%s' '%s' — %d/%d have prototypes",
                    defect_type, dim, n_proto, len(names),
                )
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
                    n_clusters=None,
                    distance_threshold=1.0 - threshold,
                    metric="precomputed",
                    linkage="average",
                )
                labels = clustering.fit_predict(dist)
            else:
                log.info(
                    "  Stage 4: '%s' '%s' — clustering %d concepts (thresh=%.2f)",
                    defect_type, dim, len(names), threshold,
                )
                if len(names) == 1:
                    labels = [0]
                else:
                    embs = embedder.encode(names, normalize_embeddings=True)
                    clustering = AgglomerativeClustering(
                        n_clusters=None,
                        distance_threshold=1.0 - threshold,
                        metric="cosine",
                        linkage="average",
                    )
                    labels = clustering.fit_predict(embs)

            # Pick canonical by image-level occurrence count (FIXED)
            clusters: dict[int, list[dict]] = {}
            for c, label in zip(dim_concepts, labels):
                clusters.setdefault(int(label), []).append(c)

            for members in clusters.values():
                # canonical = most image-frequent member; tie-break: shorter name
                canonical = max(
                    members,
                    key=lambda c: (image_freq.get(c["name"], 0), -len(c["name"])),
                )
                all_kept.append(canonical)
                cluster_members_map[canonical["name"]] = [m["name"] for m in members]

        log.info(
            "Stage 4: '%s' %d concepts → %d after clustering",
            defect_type, len(concepts), len(all_kept),
        )

        # ── 4b. Cluster-union frequency filter (FIXED) ─────────────────────────
        final_for_type: list[dict] = []
        for c in all_kept:
            members = cluster_members_map.get(c["name"], [c["name"]])
            # Count images where ANY cluster member is True (union)
            count = sum(
                1 for ann in defect_anns
                if any(ann.get("concept_vector", {}).get(m, False) for m in members)
            )
            freq = count / n_defect_imgs
            if freq >= min_defect_freq:
                final_for_type.append(c)
                log.info(
                    "    KEEP '%s'  freq=%.2f  (members: %s)",
                    c["name"], freq, members,
                )
            else:
                log.info(
                    "    DROP '%s'  freq=%.2f < %.2f",
                    c["name"], freq, min_defect_freq,
                )

        log.info(
            "  Stage 4: '%s' final: %s",
            defect_type, [c["name"] for c in final_for_type],
        )
        refined[defect_type] = final_for_type

    # ── 4c. Cross-defect generic concept discovery ────────────────────────────
    concept_to_defect_types: dict[str, set[str]] = {}
    for dt, concepts in refined.items():
        for c in concepts:
            concept_to_defect_types.setdefault(c["name"], set()).add(dt)

    discovered_generic: list[dict] = []
    for dt in list(refined.keys()):
        keep, promote = [], []
        for c in refined[dt]:
            if len(concept_to_defect_types.get(c["name"], set())) >= min_defect_types_for_generic:
                promote.append(c)
            else:
                keep.append(c)
        refined[dt] = keep
        for c in promote:
            if not any(g["name"] == c["name"] for g in discovered_generic):
                discovered_generic.append(c)
                log.info(
                    "  Stage 4: generic '%s' spans %s",
                    c["name"], sorted(concept_to_defect_types[c["name"]]),
                )

    log.info("Stage 4 complete: %d cross-defect generic concepts discovered", len(discovered_generic))
    return refined, discovered_generic
