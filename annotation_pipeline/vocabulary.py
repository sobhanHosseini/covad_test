"""
vocabulary.py — Build the three-tier concept vocabulary.

Tier 1  Normal attribute concepts     (from Stage 2)
Tier 2  Generic anomaly concepts      (5 fixed Stage 2b + discovered in Stage 4c)
Tier 3  Defect-specific concepts      (from Stage 4, per defect type)

Holdout logic: Tier 3 concepts for the held-out defect type are excluded.
               Tier 2 concepts are NEVER excluded (they are holdout-invariant by design).
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)


def build(
    normal_concepts: list[dict],
    generic_concepts: list[dict],
    defect_concept_map: dict[str, list[dict]],
    holdout_defect: str | None = None,
    discovered_generic: list[dict] | None = None,
) -> list[str]:
    """
    Assemble the final concept vocabulary as an ordered list of names.

    Parameters
    ----------
    normal_concepts   : Tier 1 — from Stage 2
    generic_concepts  : Tier 2 fixed — from Stage 2b (always 5 concepts)
    defect_concept_map: Tier 3 — {defect_type: [concept_dict]} from Stage 4
    holdout_defect    : if set, Tier 3 concepts for this defect are excluded
    discovered_generic: Tier 2 discovered — from Stage 4c (never held out)

    Returns
    -------
    Ordered list of concept name strings:
      [tier1..., tier2_fixed..., tier2_discovered..., tier3...]
    """
    vocab: list[str] = []

    # Tier 1 — normal
    for c in normal_concepts:
        if c["name"] not in vocab:
            vocab.append(c["name"])
    n_tier1 = len(vocab)

    # Tier 2 — fixed generic (never held out)
    for c in generic_concepts:
        if c["name"] not in vocab:
            vocab.append(c["name"])

    # Tier 2 — discovered cross-defect generic (never held out)
    for c in (discovered_generic or []):
        if c["name"] not in vocab:
            vocab.append(c["name"])
    n_tier2 = len(vocab) - n_tier1

    # Tier 3 — defect-specific (holdout-aware)
    excluded: set[str] = set()
    if holdout_defect:
        excluded = {c["name"] for c in defect_concept_map.get(holdout_defect, [])}
        log.info(
            "Holdout '%s': excluding %d Tier 3 concepts: %s",
            holdout_defect, len(excluded), sorted(excluded),
        )

    n_before = len(vocab)
    for dt, concepts in defect_concept_map.items():
        if dt == holdout_defect:
            continue
        for c in concepts:
            if c["name"] not in vocab and c["name"] not in excluded:
                vocab.append(c["name"])
    n_tier3 = len(vocab) - n_before

    log.info(
        "Vocabulary: %d Tier 1 + %d Tier 2 + %d Tier 3 = %d total%s",
        n_tier1, n_tier2, n_tier3, len(vocab),
        f"  (holdout={holdout_defect})" if holdout_defect else "",
    )
    return vocab
