"""Sentence-BERT concept deduplication.

Merges semantically near-duplicate concept names across tasks so we don't
grow the vocabulary with synonyms (e.g. 'shell crack' vs 'cracked shell').
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class DeduplicationResult:
    merged: dict[str, str]   # new_name → matched existing name
    genuinely_new: list[str]  # names that get a new concept head


class ConceptDeduplicator:
    """Embed concept names with sentence-transformers and merge near-duplicates.

    TODO — implement:
      - __init__(model_name, threshold):
          load SentenceTransformer(model_name)  # 'all-MiniLM-L6-v2' default
          threshold: cosine similarity cutoff (0.85 per spec)

      - deduplicate(new_names, existing_names) -> DeduplicationResult:
          1. embed all names
          2. for each new_name, find max cosine_sim with existing_names
          3. if max_sim > threshold → merge (map to existing name)
          4. else → genuinely_new

      - embed(names): returns normalised (N, D) numpy array
    """

    DEFAULT_MODEL = "all-MiniLM-L6-v2"
    DEFAULT_THRESHOLD = 0.85

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        threshold: float = DEFAULT_THRESHOLD,
    ):
        raise NotImplementedError("ConceptDeduplicator not yet implemented")

    def deduplicate(
        self,
        new_names: list[str],
        existing_names: list[str],
    ) -> DeduplicationResult:
        raise NotImplementedError
