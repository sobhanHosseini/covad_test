"""
vlm_interpreter.py — Query Qwen2.5-VL (via Ollama) with patch grid images to
extract human-readable anomaly concept names.

For each selected SAE atom:
1. Build a 3×3 grid of the top-9 anomaly patches (by SAE activation).
2. Send the grid to the VLM with a structured prompt.
3. Parse the structured response into (CONCEPT_NAME, COMMON_PATTERN, CONFIDENCE).
4. Save the grid image and accumulate results.
"""

from __future__ import annotations

import io
import json
import re
from pathlib import Path
from typing import Any

from PIL import Image

# Ollama is imported lazily inside each function so the module can be imported
# even if ollama is not installed (e.g. during unit tests).

# ── Prompt template ───────────────────────────────────────────────────────────

_PROMPT = """\
You are analyzing patches from industrial quality inspection images.

Each patch in this 3x3 grid comes from a confirmed defective region \
of an industrial object. All 9 patches strongly activate the same \
internal visual feature detector in a neural network.

Your task is to identify what visual anomaly pattern is COMMON \
across these patches.

Step 1 - Describe each patch briefly (1 sentence each):
Patch 1: ...
Patch 2: ...
...
Patch 9: ...

Step 2 - Identify the common visual pattern:
What visual characteristic or defect type appears consistently \
across most or all patches? Focus only on what is VISUALLY PRESENT \
in the patches — cracks, holes, stains, deformations, missing \
material, color changes, texture disruptions, etc.
Do NOT describe what is absent or what normal looks like.

Step 3 - Name the concept:
Give a short name (2-5 words) that describes this shared visual \
anomaly pattern. The name should be:
- Visually descriptive (what it looks like, not what caused it)
- General enough to apply across different object types
- Specific enough to distinguish from other defect types

Output your answer in exactly this format:
COMMON_PATTERN: [one sentence describing the shared visual pattern]
CONCEPT_NAME: [short name, lowercase, underscores instead of spaces]
CONFIDENCE: [high / medium / low]
REASON_FOR_LOW_CONFIDENCE: [only if confidence is low, else write none]\
"""

# ── Grid builder ──────────────────────────────────────────────────────────────

_GRID_COLS = 3
_PATCH_DISPLAY_PX = 112   # upscale each 14×14 patch 8× for visibility


def build_grid_image(
    atom_id: int,
    anomaly_patches: list[dict[str, Any]],
    top_k: int = 9,
) -> Image.Image:
    """Build a 3×3 grid of the top-k anomaly patches for a given SAE atom.

    Patches are ranked by their SAE activation magnitude for *atom_id* and
    upscaled 8× (14 px → 112 px per side) so the VLM can see them clearly.
    A grey background fills any empty cells when fewer than *top_k* patches
    are available.

    Args:
        atom_id: Index of the SAE atom (0 … 4095).
        anomaly_patches: List of dicts from ``extract_anomaly_patches``.
        top_k: Number of patches to include (must be a perfect square for an
            even grid; 9 gives the standard 3×3 layout).

    Returns:
        PIL RGB image of shape ``(3 * PATCH_DISPLAY_PX) × (3 * PATCH_DISPLAY_PX)``.
    """
    import torch

    # Rank patches by activation for this atom (descending)
    activations = [
        (i, float(p["sae_code"][atom_id]))
        for i, p in enumerate(anomaly_patches)
    ]
    activations.sort(key=lambda x: x[1], reverse=True)
    top_indices = [idx for idx, _ in activations[:top_k]]

    # Build grid canvas
    grid_w = _GRID_COLS * _PATCH_DISPLAY_PX
    grid_h = _GRID_COLS * _PATCH_DISPLAY_PX
    grid = Image.new("RGB", (grid_w, grid_h), color=(128, 128, 128))

    for cell_i, patch_idx in enumerate(top_indices):
        row = cell_i // _GRID_COLS
        col = cell_i % _GRID_COLS
        patch_img = anomaly_patches[patch_idx]["patch_image"]
        patch_resized = patch_img.resize(
            (_PATCH_DISPLAY_PX, _PATCH_DISPLAY_PX), Image.LANCZOS
        )
        grid.paste(patch_resized, (col * _PATCH_DISPLAY_PX, row * _PATCH_DISPLAY_PX))

    return grid


# ── VLM query ─────────────────────────────────────────────────────────────────

def _pil_to_bytes(img: Image.Image, fmt: str = "PNG") -> bytes:
    """Encode a PIL image to raw bytes for the Ollama API."""
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def _parse_response(text: str) -> dict[str, str]:
    """Extract structured fields from the VLM response text.

    Looks for the four tagged lines produced by the prompt. Falls back to
    ``"PARSE_ERROR"`` strings so downstream code always gets a complete dict.

    Args:
        text: Raw string returned by the VLM.

    Returns:
        Dict with keys ``concept_name``, ``common_pattern``, ``confidence``,
        ``reason_for_low_confidence``, and ``raw_response``.
    """
    def _extract(tag: str) -> str:
        pattern = rf"{tag}:\s*(.+)"
        m = re.search(pattern, text, re.IGNORECASE)
        return m.group(1).strip() if m else "PARSE_ERROR"

    return {
        "concept_name":              _extract("CONCEPT_NAME"),
        "common_pattern":            _extract("COMMON_PATTERN"),
        "confidence":                _extract("CONFIDENCE"),
        "reason_for_low_confidence": _extract("REASON_FOR_LOW_CONFIDENCE"),
        "raw_response":              text,
    }


def query_qwen(
    grid_image: Image.Image,
    atom_id: int,
    vlm_model: str = "gemma4:e4b",
    ollama_host: str = "http://localhost:6000",
) -> dict[str, str]:
    """Send a patch grid to Qwen2.5-VL and parse the concept name response.

    Calls the local Ollama server with the multimodal chat API.  The image is
    embedded directly as bytes (no temporary file needed).

    Args:
        grid_image: 3×3 PIL grid from ``build_grid_image``.
        atom_id: SAE atom index — included in the prompt for traceability.
        vlm_model: Ollama model tag (default ``"gemma4:e4b"``).
        ollama_host: Ollama server URL (default ``"http://localhost:6000"``).

    Returns:
        Dict with keys: ``atom_id``, ``concept_name``, ``common_pattern``,
        ``confidence``, ``reason_for_low_confidence``, ``raw_response``.
    """
    from ollama import Client  # lazy import — requires `pip install ollama`

    image_bytes = _pil_to_bytes(grid_image)
    client = Client(host=ollama_host)

    response = client.chat(
        model=vlm_model,
        messages=[{
            "role":    "user",
            "content": _PROMPT,
            "images":  [image_bytes],
        }],
    )

    raw_text = response["message"]["content"]
    parsed = _parse_response(raw_text)
    parsed["atom_id"] = str(atom_id)
    return parsed


# ── Batch runner ──────────────────────────────────────────────────────────────

def run_vlm_on_atoms(
    relevant_atoms: list[dict[str, Any]],
    anomaly_patches: list[dict[str, Any]],
    output_dir: str | Path,
    vlm_model: str = "gemma4:e4b",
    ollama_host: str = "http://localhost:6000",
    top_k: int = 9,
    run_name: str = "hazelnut",
) -> list[dict[str, Any]]:
    """Run VLM interpretation for every atom in *relevant_atoms*.

    For each atom:
    * Builds and saves the patch grid to ``output_dir/grids/atom_{id}.png``.
    * Queries Qwen2.5-VL and parses the response.
    * Accumulates all results and saves them to
      ``output_dir/hazelnut_vlm_results.json`` after each atom (so a crash
      mid-run does not lose progress).

    Args:
        relevant_atoms: Output of ``select_anomaly_relevant_atoms``.
        anomaly_patches: Output of ``extract_anomaly_patches``.
        output_dir: Root output directory (grids/ sub-dir is created here).
        vlm_model: Ollama model tag.
        ollama_host: Ollama server URL.
        top_k: Patches per grid (default 9 → 3×3).

    Returns:
        List of result dicts (one per atom), each containing all parsed VLM
        fields plus ``atom_id`` and ``discrimination_score``.
    """
    output_dir = Path(output_dir)
    grids_dir = output_dir / "grids"
    grids_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / f"{run_name}_vlm_results.json"

    all_results: list[dict[str, Any]] = []
    total = len(relevant_atoms)

    for i, atom_info in enumerate(relevant_atoms):
        atom_id = atom_info["atom_id"]
        print(
            f"[vlm_interpreter] [{i + 1}/{total}] atom {atom_id} "
            f"(disc={atom_info['discrimination_score']:.4f}) …",
            flush=True,
        )

        # Build and save grid
        grid_img = build_grid_image(atom_id, anomaly_patches, top_k=top_k)
        grid_path = grids_dir / f"atom_{atom_id}.png"
        grid_img.save(grid_path)

        # Query VLM
        try:
            parsed = query_qwen(grid_img, atom_id, vlm_model=vlm_model, ollama_host=ollama_host)
        except Exception as exc:
            print(f"  [WARN] VLM call failed: {exc}")
            parsed = {
                "atom_id":              str(atom_id),
                "concept_name":         "VLM_ERROR",
                "common_pattern":       str(exc),
                "confidence":           "none",
                "reason_for_low_confidence": "none",
                "raw_response":         str(exc),
            }

        result = {
            "atom_id":              atom_id,
            "discrimination_score": atom_info["discrimination_score"],
            "anomaly_count":        atom_info["anomaly_count"],
            **parsed,
        }
        all_results.append(result)

        concept = parsed.get("concept_name", "?")
        confidence = parsed.get("confidence", "?")
        print(f"  → concept: {concept!r}  confidence: {confidence}")

        # Save incrementally — safe against crashes mid-run
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2)

    print(f"[vlm_interpreter] Done. Results saved → {results_path}")
    return all_results
