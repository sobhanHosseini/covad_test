"""
vocabulary_builder.py — Consolidate per-atom VLM concept names into a
compact, deduplicated category vocabulary.

Two-stage pipeline:
  Stage 1 — LLM grouping:  send all raw concept names to a text LLM and ask
            it to group semantically similar names and elect one canonical name
            per group.  Falls back from qwen3.5:9b → gemma4:e4b.

  Stage 2 — CLIP dedup:  encode the canonical names with CLIP's text encoder
            and merge any pair whose cosine similarity exceeds *clip_threshold*
            (default 0.75).  This catches near-duplicates the LLM missed.

Target output: 15–25 canonical concepts.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

import torch


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — LLM grouping
# ─────────────────────────────────────────────────────────────────────────────

_GROUPING_PROMPT = """\
You are organizing a vocabulary of visual defect concepts discovered \
from industrial inspection images.

Below is a list of concept names, one per line. Each name describes \
a visual anomaly pattern found in industrial objects (cracks, holes, \
stains, deformations, etc.).

CONCEPT LIST:
{concept_list}

RULES:
1. Group semantically similar or duplicate concepts together.
   Every concept MUST appear in exactly one group or in DISCARDED.
2. For each group choose ONE canonical name (2-4 words, lowercase, underscores).
3. Discard ONLY concepts that are truly meaningless or object-specific.
   Do NOT discard more than 10% of the list.
4. Output ONLY the formatted blocks below — no preamble, no commentary.

OUTPUT FORMAT (repeat for every group, separated by ---):
CANONICAL_NAME: surface_crack
MEMBERS: linear_crack, thin_crack, surface_fracture, hairline_crack
---
CANONICAL_NAME: circular_void
MEMBERS: round_hole, circular_hole, punched_hole
---
DISCARDED: concept_x (reason), concept_y (reason)\
"""

# Patterns used by _parse_llm_grouping
_FENCE_RE     = re.compile(r"```[a-z]*\n?|```", re.IGNORECASE)
_THINK_RE     = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_SEP_RE       = re.compile(r"\n\s*(?:---|===|\*\*\*)\s*\n|\n{3,}")
_CANONICAL_RE = re.compile(r"CANONICAL_NAME\s*:\s*([^\n,]+)", re.IGNORECASE)
_MEMBERS_RE   = re.compile(
    r"MEMBERS\s*:\s*(.*?)(?=\nCANONICAL_NAME|\nDISCARDED|$)", re.IGNORECASE | re.DOTALL
)
_DISCARDED_RE = re.compile(r"DISCARDED\s*:\s*([^\n]+)", re.IGNORECASE)
_SNAKE_RE     = re.compile(r"\b[a-z][a-z0-9_]{3,}\b")


def _parse_json_grouping(text: str) -> list[dict[str, Any]] | None:
    """Try to parse JSON array of {canonical, members} dicts.  Returns None on failure."""
    try:
        start = text.index("[")
        data = json.loads(text[start:text.rindex("]") + 1])
        if isinstance(data, list) and data and "canonical" in data[0]:
            return [
                {"canonical": d["canonical"].strip().lower(),
                 "members":   [m.strip().lower() for m in d.get("members", [])]}
                for d in data if "canonical" in d
            ]
    except (ValueError, KeyError, TypeError):
        pass
    return None


def _parse_llm_grouping(text: str) -> tuple[list[dict[str, Any]], list[str]]:
    """Robustly parse the LLM grouping response into groups and discarded names.

    Handles:
    * Markdown code fences (```...```)
    * ``<think>...</think>`` blocks
    * JSON array format ``[{"canonical": ..., "members": [...]}]``
    * Varied block separators: ``---``, ``***``, ``===``, 3+ blank lines
    * Multi-line MEMBERS lists
    * DOTALL bug: DISCARDED no longer captures subsequent blocks

    Args:
        text: Raw LLM response string.

    Returns:
        ``(groups, discarded)`` — groups is list of ``{"canonical", "members"}``
        dicts; discarded is a list of snake_case concept name strings.
    """
    # Strip think tags and markdown fences
    text = _THINK_RE.sub("", text)
    text = _FENCE_RE.sub("", text)
    text = text.strip()

    groups: list[dict[str, Any]] = []
    discarded: list[str] = []

    # Try JSON first
    json_groups = _parse_json_grouping(text)
    if json_groups:
        return json_groups, []

    # Split into blocks on any separator style
    blocks = _SEP_RE.split(text)

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        # Extract DISCARDED line (single-line match only — no DOTALL)
        disc_m = _DISCARDED_RE.search(block)
        if disc_m:
            discarded += _SNAKE_RE.findall(disc_m.group(1))

        # Extract CANONICAL_NAME
        can_m = _CANONICAL_RE.search(block)
        if not can_m:
            continue

        canonical = can_m.group(1).strip().strip("*`").lower()
        # Remove trailing punctuation / whitespace artifacts
        canonical = re.sub(r"[^a-z0-9_]", "_", canonical).strip("_")

        # Extract MEMBERS (may be multi-line, stops before next directive)
        mem_m = _MEMBERS_RE.search(block)
        if not mem_m:
            continue

        raw_members = mem_m.group(1).replace("\n", " ")
        members = [
            re.sub(r"[^a-z0-9_]", "_", m.strip().lower()).strip("_")
            for m in raw_members.split(",")
            if m.strip() and not m.strip().startswith("#")
        ]
        members = [m for m in members if len(m) >= 3]

        if canonical and members:
            groups.append({"canonical": canonical, "members": members})

    return groups, discarded


def _llm_group_concepts(
    concept_names: list[str],
    ollama_host: str = "http://localhost:6000",
    primary_model: str = "qwen3.5:9b",
    fallback_model: str = "gemma4:e4b",
    min_groups: int = 5,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Send concept names to a text LLM for semantic grouping.

    Tries *primary_model* first.  If fewer than *min_groups* groups are
    returned, retries once with *fallback_model*.  Falls back to treating
    every concept as its own singleton group if both fail.

    Args:
        concept_names: List of raw concept name strings.
        ollama_host: Ollama server URL.
        primary_model: First-choice text model (no vision needed).
        fallback_model: Fallback if primary returns too few groups.
        min_groups: Minimum acceptable group count before trying fallback.

    Returns:
        (groups, discarded) as returned by ``_parse_llm_grouping``.
    """
    from ollama import Client

    prompt = _GROUPING_PROMPT.format(concept_list="\n".join(concept_names))
    client = Client(host=ollama_host)

    for model in (primary_model, fallback_model):
        print(f"[vocabulary_builder] LLM grouping with {model} …")
        try:
            t0 = time.time()
            # No num_predict cap: qwen3.5 thinking models spend tokens on an
            # internal <think> phase (in .thinking field) before writing the
            # answer into .content — capping tokens starves the content field.
            response = client.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
            )
            elapsed = time.time() - t0
            text = response.message.content
            groups, discarded = _parse_llm_grouping(text)

            if len(groups) < min_groups:
                print(
                    f"  [WARN] Only {len(groups)} groups parsed (min={min_groups}) "
                    f"— raw response snippet: {repr(text[:200])}"
                )
                raise ValueError(f"Too few groups: {len(groups)}")

            print(f"  → {len(groups)} groups, {len(discarded)} discarded  ({elapsed:.1f}s)")
            return groups, discarded

        except Exception as exc:
            print(f"  [WARN] {model} failed: {exc}")

    print("  [WARN] All LLM models failed — using singleton fallback")
    groups = [{"canonical": n, "members": [n]} for n in concept_names]
    return groups, []


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — Semantic text-embedding deduplication
# ─────────────────────────────────────────────────────────────────────────────

def _get_embeddings(
    names: list[str],
    ollama_host: str,
    embed_model: str = "nomic-embed-text",
) -> torch.Tensor | None:
    """Encode concept names via ollama's nomic-embed-text model.

    Returns a normalised float tensor of shape (N, D), or None on failure.
    CLIP ViT-B/32 text encoder is NOT used here: all "surface_xxx" defect
    names cluster above 0.75 in CLIP space because the model was trained on
    image–text pairs, not for fine-grained text similarity.  nomic-embed-text
    is a proper semantic text similarity model and separates defect categories
    much more reliably at the 0.75 threshold.
    """
    from ollama import Client
    try:
        client = Client(host=ollama_host)
        embs = []
        for name in names:
            r = client.embeddings(model=embed_model, prompt=name.replace("_", " "))
            embs.append(r["embedding"])
        t = torch.tensor(embs, dtype=torch.float32)
        t = t / t.norm(dim=-1, keepdim=True)
        return t
    except Exception as exc:
        print(f"[vocabulary_builder] nomic-embed-text failed: {exc}")
        return None


def _embed_dedup(
    canonical_concepts: list[dict[str, Any]],
    ollama_host: str = "http://localhost:6000",
    threshold: float = 0.75,
) -> list[dict[str, Any]]:
    """Merge canonical concepts whose text embeddings are too similar.

    Uses ``nomic-embed-text`` (via Ollama) for semantic text similarity.
    Any pair with cosine similarity > *threshold* is merged: the concept
    with more member atoms absorbs the smaller one.  Repeated until stable.

    Args:
        canonical_concepts: List of concept dicts with keys
            ``canonical``, ``atom_ids``, ``members``, ``patterns``.
        ollama_host: Ollama server URL for nomic-embed-text.
        threshold: Cosine similarity above which two concepts are merged
            (default 0.75; works well with nomic-embed-text).

    Returns:
        Deduplicated list of concept dicts.
    """
    if len(canonical_concepts) <= 1:
        return canonical_concepts

    print(f"[vocabulary_builder] Embedding dedup: {len(canonical_concepts)} concepts, threshold={threshold}")

    concepts = list(canonical_concepts)  # working copy

    changed = True
    while changed:
        changed = False
        names = [c["canonical"] for c in concepts]
        feats = _get_embeddings(names, ollama_host)
        if feats is None:
            print("  [WARN] Embedding failed — skipping dedup")
            break

        sim = feats @ feats.T  # (N, N)
        n = len(concepts)
        merged_into: dict[int, int] = {}

        for i in range(n):
            for j in range(i + 1, n):
                if i in merged_into or j in merged_into:
                    continue
                if float(sim[i, j]) > threshold:
                    keep, drop = (
                        (j, i) if len(concepts[j]["atom_ids"]) >= len(concepts[i]["atom_ids"])
                        else (i, j)
                    )
                    concepts[keep]["atom_ids"] += concepts[drop]["atom_ids"]
                    concepts[keep]["members"]  += concepts[drop]["members"] + [concepts[drop]["canonical"]]
                    concepts[keep]["patterns"] += concepts[drop]["patterns"]
                    merged_into[drop] = keep
                    print(
                        f"  merge: '{concepts[drop]['canonical']}' → "
                        f"'{concepts[keep]['canonical']}'  (sim={sim[i,j]:.3f})"
                    )
                    changed = True

        if changed:
            concepts = [c for idx, c in enumerate(concepts) if idx not in merged_into]

    print(f"[vocabulary_builder] After embedding dedup: {len(concepts)} concepts")
    return concepts


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def build_vocabulary(
    vlm_results: list[dict[str, Any]],
    exclude_confidence: set[str] | None = None,
    ollama_host: str = "http://localhost:6000",
    llm_primary: str = "qwen3.5:9b",
    llm_fallback: str = "gemma4:e4b",
    clip_threshold: float = 0.75,
    save_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build a compact anomaly vocabulary from per-atom VLM results.

    Pipeline:
        1. Collect valid raw concepts from *vlm_results*.
        2. LLM grouping: merge semantically similar names, elect canonical names.
        3. CLIP dedup: merge any remaining near-duplicate canonicals.

    Args:
        vlm_results: List of result dicts from ``run_vlm_on_atoms``.
        exclude_confidence: Confidence levels to drop (e.g. ``{"low"}``).
        ollama_host: Ollama server URL used for LLM grouping.
        llm_primary: First-choice text model for grouping.
        llm_fallback: Fallback text model if primary fails.
        clip_threshold: CLIP cosine similarity threshold for dedup (0.75
            recommended; lower = more aggressive merging).
        save_path: If given, saves the final vocabulary as JSON here.

    Returns:
        Dict::

            {
                "concepts": [
                    {
                        "name":      str,          # canonical concept name
                        "frequency": int,          # number of atoms
                        "atom_ids":  list[int],
                        "patterns":  list[str],
                        "members":   list[str],    # raw names merged here
                    }, ...
                ],
                "total_atoms_interpreted": int,
                "total_unique_concepts":   int,
                "discarded_by_llm":        list[str],
            }
    """
    exclude_confidence = exclude_confidence or set()

    # ── Step 0: collect valid raw concepts ───────────────────────────────────
    filtered = [
        r for r in vlm_results
        if r.get("confidence", "").lower() not in {c.lower() for c in exclude_confidence}
        and r.get("concept_name", "PARSE_ERROR") not in {"PARSE_ERROR", "VLM_ERROR"}
    ]
    print(f"[vocabulary_builder] {len(filtered)}/{len(vlm_results)} atoms pass confidence filter")

    # Build raw groups keyed by exact name (as before) to carry atom_ids + patterns
    raw: dict[str, dict[str, Any]] = {}
    for r in filtered:
        name = r["concept_name"].strip().lower()
        if name not in raw:
            raw[name] = {"name": name, "atom_ids": [], "patterns": []}
        raw[name]["atom_ids"].append(r["atom_id"])
        if r.get("common_pattern") and r["common_pattern"] != "PARSE_ERROR":
            raw[name]["patterns"].append(r["common_pattern"])

    raw_names = list(raw.keys())
    print(f"[vocabulary_builder] Raw unique concepts: {len(raw_names)}")

    # ── Step 1: LLM semantic grouping ────────────────────────────────────────
    llm_groups, discarded_by_llm = _llm_group_concepts(
        raw_names,
        ollama_host=ollama_host,
        primary_model=llm_primary,
        fallback_model=llm_fallback,
    )

    # Merge atom_ids and patterns from all member raw concepts into each group
    canonical_concepts: list[dict[str, Any]] = []
    covered_raw = set()
    for g in llm_groups:
        canonical = g["canonical"]
        members   = g["members"]
        atom_ids, patterns = [], []
        for m in members:
            m_clean = m.strip().lower()
            covered_raw.add(m_clean)
            if m_clean in raw:
                atom_ids += raw[m_clean]["atom_ids"]
                patterns += raw[m_clean]["patterns"]
        # Also check if the canonical itself is in raw (LLM sometimes reuses a member name)
        if canonical in raw and canonical not in covered_raw:
            atom_ids += raw[canonical]["atom_ids"]
            patterns += raw[canonical]["patterns"]
            covered_raw.add(canonical)

        canonical_concepts.append({
            "canonical": canonical,
            "atom_ids":  atom_ids,
            "patterns":  patterns,
            "members":   members,
        })

    # Any raw concept not covered by the LLM (e.g. if parsing missed it)
    for name, info in raw.items():
        if name not in covered_raw:
            print(f"  [INFO] LLM did not cover '{name}' — adding as singleton")
            canonical_concepts.append({
                "canonical": name,
                "atom_ids":  info["atom_ids"],
                "patterns":  info["patterns"],
                "members":   [name],
            })

    # ── Step 2: semantic embedding deduplication (nomic-embed-text) ─────────
    final_concepts = _embed_dedup(canonical_concepts, ollama_host=ollama_host, threshold=clip_threshold)

    # Sort by atom count descending
    final_concepts.sort(key=lambda x: len(x["atom_ids"]), reverse=True)

    # Reformat for output
    out_concepts = [
        {
            "name":      c["canonical"],
            "frequency": len(c["atom_ids"]),
            "atom_ids":  c["atom_ids"],
            "patterns":  c["patterns"],
            "members":   c["members"],
        }
        for c in final_concepts
    ]

    vocab = {
        "concepts":                out_concepts,
        "total_atoms_interpreted": len(filtered),
        "total_unique_concepts":   len(out_concepts),
        "discarded_by_llm":        discarded_by_llm,
    }

    print(f"\n[vocabulary_builder] Final vocabulary: {len(out_concepts)} concepts")
    for c in out_concepts:
        print(f"  {c['name']:<40s} ({c['frequency']} atoms)")

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(vocab, f, indent=2)
        print(f"[vocabulary_builder] Saved → {save_path}")

    return vocab
