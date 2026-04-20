"""
clip_filter.py — CLIP-based concept discriminability filter (Stage 2e).

For each candidate normal concept, measures how well it separates normal
images from anomalous images in CLIP similarity space. Concepts that score
high for both normal and defect images give the CBM no useful signal and
are dropped. Concepts that score distinctly higher for normal images are
genuinely capturing normal appearance attributes.

Discriminability metric: Cohen's d
    d = (mean_normal_sim - mean_defect_sim) / pooled_std

A concept with d > 0.2 is meaningfully more activated for normal images,
meaning it genuinely describes what normal looks like (not just what
everything looks like).

This filter runs AFTER Stage 2d (VLM audit) and BEFORE Stage 2b (generic
concept injection). It uses:
  - Already-computed CLIP image embeddings (normal + defect, from pipeline)
  - CLIP text encoder to embed concept name + description

No VLM calls, no new GPU work beyond what CLIP already does.
"""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)

try:
    import torch
    import clip
    from PIL import Image as _PILImage
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False


# ── CLIP text embedding ───────────────────────────────────────────────────────

def _embed_concepts_text(
    concepts: list[dict],
    clip_model,
    device: str,
) -> dict[str, np.ndarray]:
    """
    Embed each concept as a CLIP text vector using:
        "a photo showing {name}: {description}"
    Returns {concept_name: unit_embedding}.
    """
    if not CLIP_AVAILABLE or clip_model is None:
        return {}

    result: dict[str, np.ndarray] = {}
    for c in concepts:
        text = f"a photo showing {c['name'].replace('_', ' ')}: {c.get('description', '')}"
        tokens = clip.tokenize([text], truncate=True).to(device)
        with torch.no_grad():
            emb = clip_model.encode_text(tokens)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        result[c["name"]] = emb.cpu().numpy()[0]
    return result


# ── Discriminability score ────────────────────────────────────────────────────

def _cohen_d(normal_sims: np.ndarray, defect_sims: np.ndarray) -> float:
    """
    Cohen's d: how many standard deviations apart are the two distributions?
    Positive d means the concept scores higher for normal images (good).
    Negative d means the concept scores higher for defect images (bad — likely
    a defect-contaminated concept that slipped through the VLM audit).
    """
    if len(normal_sims) == 0 or len(defect_sims) == 0:
        return 0.0
    mu_n, mu_d = normal_sims.mean(), defect_sims.mean()
    var_n = normal_sims.var(ddof=1) if len(normal_sims) > 1 else 0.0
    var_d = defect_sims.var(ddof=1) if len(defect_sims) > 1 else 0.0
    pooled_std = np.sqrt((var_n + var_d) / 2.0)
    if pooled_std < 1e-8:
        return 0.0
    return float((mu_n - mu_d) / pooled_std)


# ── Main filter function ──────────────────────────────────────────────────────

def run(
    concepts: list[dict],
    normal_image_paths: list[str],
    defect_image_paths: list[str],
    clip_model,
    clip_preprocess,
    device: str,
    normal_embeddings: dict[str, np.ndarray],
    defect_embeddings: dict[str, np.ndarray],
    min_cohen_d: float = 0.10,
    top_k: int | None = None,
) -> list[dict]:
    """
    Filter concepts by CLIP discriminability.

    For each concept:
      1. Embed concept text with CLIP
      2. Compute cosine similarity to all normal images
      3. Compute cosine similarity to all defect images
      4. Keep concept if Cohen's d (normal vs defect similarity) >= min_cohen_d

    Parameters
    ----------
    concepts          : candidate concept dicts from Stage 2d
    normal_embeddings : {path: np.ndarray} already computed in pipeline
    defect_embeddings : {path: np.ndarray} already computed in pipeline
    min_cohen_d       : minimum Cohen's d to keep a concept (default 0.10)
                        0.10 = small but real effect
                        0.20 = moderate — use if you want stricter filtering
    top_k             : if set, keep top-K concepts by Cohen's d regardless of threshold
                        useful for ensuring a minimum concept count

    Returns
    -------
    Filtered list of concept dicts, each with an added "cohen_d" field for
    logging and thesis reporting.
    """
    if not CLIP_AVAILABLE or clip_model is None:
        log.warning("Stage 2e: CLIP not available — skipping discriminability filter")
        return concepts

    if not normal_embeddings or not defect_embeddings:
        log.warning("Stage 2e: image embeddings not available — skipping filter")
        return concepts

    log.info("Stage 2e: CLIP discriminability filter on %d concepts...", len(concepts))

    # Embed all concept texts
    text_embeddings = _embed_concepts_text(concepts, clip_model, device)

    # Stack image embeddings
    normal_paths = [p for p in normal_image_paths if p in normal_embeddings]
    defect_paths = [p for p in defect_image_paths if p in defect_embeddings]

    if not normal_paths or not defect_paths:
        log.warning("Stage 2e: no image embeddings found — skipping filter")
        return concepts

    normal_matrix = np.stack([normal_embeddings[p] for p in normal_paths])  # (N, D)
    defect_matrix = np.stack([defect_embeddings[p] for p in defect_paths])  # (M, D)

    # Score each concept
    scored: list[tuple[float, dict]] = []
    for c in concepts:
        name = c["name"]
        if name not in text_embeddings:
            scored.append((0.0, c))
            continue

        t = text_embeddings[name]                      # (D,)
        normal_sims = normal_matrix @ t                # (N,)
        defect_sims = defect_matrix @ t                # (M,)

        d = _cohen_d(normal_sims, defect_sims)
        c_with_score = dict(c, cohen_d=round(d, 3))
        scored.append((d, c_with_score))

        log.info(
            "  %-42s  d=%+.3f  μ_norm=%.3f  μ_def=%.3f",
            name, d, float(normal_sims.mean()), float(defect_sims.mean()),
        )

    # Sort by Cohen's d descending
    scored.sort(key=lambda x: -x[0])

    # Apply threshold filter
    kept = [(d, c) for d, c in scored if d >= min_cohen_d]

    # Apply top_k floor — ensures CBM always has enough concepts even if
    # threshold is aggressive (never return fewer than top_k concepts)
    if top_k and len(kept) < top_k:
        log.info(
            "  Stage 2e: threshold (d>=%.2f) left only %d concepts — "
            "expanding to top-%d by Cohen's d",
            min_cohen_d, len(kept), top_k,
        )
        kept = scored[:top_k]

    result = [c for _, c in kept]
    dropped = len(concepts) - len(result)

    log.info(
        "Stage 2e complete: %d → %d concepts "
        "(dropped %d with Cohen's d < %.2f)",
        len(concepts), len(result), dropped, min_cohen_d,
    )
    for d, c in kept:
        log.info("  KEEP %-42s  d=%+.3f", c["name"], d)
    for d, c in scored:
        if c not in result:
            log.info("  DROP %-42s  d=%+.3f", c["name"], d)

    return result