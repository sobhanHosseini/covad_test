"""
utils.py — Small utility helpers used across the package.

No imports from within the package — safe to import everywhere.
"""

from __future__ import annotations

import json
import re
import logging
from pathlib import Path

log = logging.getLogger(__name__)


# ── Text normalisation ────────────────────────────────────────────────────────

def to_snake_case(name: str) -> str:
    name = name.lower().strip()
    name = re.sub(r"[^a-z0-9\s_]", "", name)
    name = re.sub(r"\s+", "_", name)
    return name


def normalize_dimension(dim: str) -> str:
    """Map raw VLM dimension string to one of the canonical dimension keys."""
    from annotation_pipeline.config import DIMENSION_NORMALIZE  # local to avoid circular
    if not dim:
        return "unknown"
    key = dim.lower().strip()
    if key in DIMENSION_NORMALIZE:
        return DIMENSION_NORMALIZE[key]
    for k, v in DIMENSION_NORMALIZE.items():
        if k in key or key in k:
            return v
    return "unknown"


def get_dimension_threshold(dim: str, global_threshold: float) -> float:
    from annotation_pipeline.config import DIMENSION_THRESHOLDS
    return DIMENSION_THRESHOLDS.get(normalize_dimension(dim), global_threshold)


# ── JSON parsing ──────────────────────────────────────────────────────────────

def extract_json(text: str):
    """
    Strip markdown code fences and attempt to parse JSON.
    Falls back to regex extraction of the first [...] or {...} block.
    Returns None on failure.
    """
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        for pattern in (r"(\[.*\])", r"(\{.*\})"):
            match = re.search(pattern, text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1))
                except json.JSONDecodeError:
                    pass
    return None


# ── VLM call ──────────────────────────────────────────────────────────────────

def call_vlm(
    client,
    model_name: str,
    prompt: str,
    image_paths: list[str] | None = None,
    debug: bool = False,
) -> str:
    """
    Send a prompt (optionally with images) to the VLM via Ollama.
    Returns the model's raw text response.
    """
    message: dict = {"role": "user", "content": prompt}
    if image_paths:
        message["images"] = image_paths
    if debug:
        log.debug("VLM prompt:\n%s", prompt)
    response = client.chat(model=model_name, messages=[message])
    content = response["message"]["content"]
    if debug:
        log.debug("VLM response:\n%s", content)
    return content