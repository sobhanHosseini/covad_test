"""
clip_utils.py — CLIP model loading, embedding computation (with disk cache),
                and CLIP-based normal reference selection for Stage 3.

All CLIP functionality is isolated here so the rest of the pipeline degrades
gracefully when CLIP is not installed.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

# ── Availability guard ────────────────────────────────────────────────────────

try:
    import torch
    import clip
    from PIL import Image as _PILImage
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    log.warning(
        "CLIP not found (pip install git+https://github.com/openai/CLIP.git). "
        "Normal reference selection will use a fixed middle image."
    )


# ── Model loading ─────────────────────────────────────────────────────────────

def load_clip(model_name: str = "ViT-B/32"):
    """
    Load CLIP model and preprocessor.
    Returns (None, None, "cpu") if CLIP is not available.
    """
    if not CLIP_AVAILABLE:
        return None, None, "cpu"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(model_name, device=device)
    model.eval()
    log.info("CLIP %s loaded on %s", model_name, device)
    return model, preprocess, device


# ── Embedding computation ─────────────────────────────────────────────────────

def compute_clip_embeddings(
    image_paths: list[str],
    model,
    preprocess,
    device: str,
    batch_size: int = 32,
) -> dict[str, np.ndarray]:
    """
    Compute normalised CLIP image embeddings.
    Returns {image_path: embedding_array}.
    Silently skips images that cannot be opened.
    """
    if not CLIP_AVAILABLE or model is None:
        return {}

    embeddings: dict[str, np.ndarray] = {}
    for i in range(0, len(image_paths), batch_size):
        batch = image_paths[i : i + batch_size]
        tensors, valid_paths = [], []
        for p in batch:
            try:
                t = preprocess(_PILImage.open(p).convert("RGB")).unsqueeze(0)
                tensors.append(t)
                valid_paths.append(p)
            except Exception as exc:
                log.warning("CLIP: skipping %s — %s", Path(p).name, exc)
        if not tensors:
            continue
        with torch.no_grad():
            emb = model.encode_image(torch.cat(tensors).to(device))
            emb = emb / emb.norm(dim=-1, keepdim=True)
        for path, e in zip(valid_paths, emb.cpu().numpy()):
            embeddings[path] = e
    return embeddings


def load_or_compute_clip_embeddings(
    image_paths: list[str],
    model,
    preprocess,
    device: str,
    cache_path: str | None = None,
    batch_size: int = 32,
) -> dict[str, np.ndarray]:
    """
    Load embeddings from disk cache if available and complete; otherwise compute
    and save to cache.
    """
    if not CLIP_AVAILABLE or model is None:
        return {}

    if cache_path and Path(cache_path).exists():
        data = np.load(cache_path, allow_pickle=True)
        cached: dict[str, np.ndarray] = dict(
            zip(data["paths"].tolist(), data["embeddings"])
        )
        missing = [p for p in image_paths if p not in cached]
        if not missing:
            log.info("CLIP: loaded %d embeddings from cache (%s)", len(cached), cache_path)
            return {p: cached[p] for p in image_paths if p in cached}
        log.info("CLIP cache incomplete (%d missing) — recomputing all", len(missing))

    log.info("CLIP: computing embeddings for %d images...", len(image_paths))
    embeddings = compute_clip_embeddings(image_paths, model, preprocess, device, batch_size)

    if cache_path and embeddings:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            cache_path,
            paths=np.array(list(embeddings.keys())),
            embeddings=np.stack(list(embeddings.values())),
        )
        log.info("CLIP embeddings cached → %s", cache_path)

    return embeddings


# ── Reference selection ───────────────────────────────────────────────────────

def select_normal_references(
    defect_embedding: np.ndarray,
    normal_embeddings: dict[str, np.ndarray],
    k: int = 3,
) -> list[str]:
    """
    Return the k normal images most visually similar to a defect image
    (CLIP cosine similarity).
    """
    if not normal_embeddings:
        return []
    paths = list(normal_embeddings.keys())
    embs = np.stack([normal_embeddings[p] for p in paths])
    sims = embs @ defect_embedding
    top_k = np.argsort(-sims)[: min(k, len(paths))]
    return [paths[i] for i in top_k]
