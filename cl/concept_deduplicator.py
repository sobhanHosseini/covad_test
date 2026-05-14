"""Sentence-BERT concept deduplication.

When the annotation pipeline generates new concept names via --append_defect,
some names may be near-duplicates of concepts already in the vocabulary
(e.g. "shell_crack_line" vs "linear_shell_crack").  This module embeds all
names with a sentence-transformer and merges semantically close pairs.

For the pre-generated hazelnut CSVs this module is a no-op — all 42 concepts
are registered at Task 1 and no genuinely new names appear at Tasks 2–4.
It is needed for real CL deployment where the VLM pipeline runs on new images
and may generate overlapping vocabulary.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# Suppress benign sentence-transformers / HF Hub messages
warnings.filterwarnings("ignore", message=".*unauthenticated.*")
warnings.filterwarnings("ignore", message=".*UNEXPECTED.*")

from sentence_transformers import SentenceTransformer


@dataclass
class DeduplicationResult:
    """Result of processing one batch of new concept names."""

    merged: dict[str, str]          # new_name → existing_name it maps to
    genuinely_new: list[str]        # names that need a new FC head
    all_mappings: dict[str, str]    # complete map (new_name → final name used)
    similarities: dict[str, float]  # new_name → max cosine similarity (for logging)


class ConceptDeduplicator:
    """Embed concept names with sentence-transformers and merge near-duplicates.

    Lifecycle:
        dedup = ConceptDeduplicator()
        dedup.register_vocabulary(task1_concept_names)   # Task 1
        result = dedup.process_new_concepts(task2_names) # Task 2+
        # genuinely_new names are automatically registered after process_new_concepts
    """

    DEFAULT_MODEL = "all-MiniLM-L6-v2"
    DEFAULT_THRESHOLD = 0.85

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        threshold: float = DEFAULT_THRESHOLD,
    ):
        self.model = SentenceTransformer(
            model_name,
            processor_kwargs={"clean_up_tokenization_spaces": True},
        )
        self.model_name = model_name
        self.threshold = threshold
        self.existing_names: list[str] = []
        self.existing_embeddings: Optional[np.ndarray] = None  # (K, embed_dim)

    # ── vocabulary registration ───────────────────────────────────────────────

    def register_vocabulary(self, concept_names: list[str]) -> None:
        """Embed and store the initial concept vocabulary (called at Task 1).

        Args:
            concept_names: ordered list of K concept name strings.
        """
        self.existing_names = list(concept_names)
        self.existing_embeddings = self.model.encode(
            concept_names, normalize_embeddings=True, show_progress_bar=False
        )  # (K, 384)

    # ── new concept processing ────────────────────────────────────────────────

    def process_new_concepts(self, new_names: list[str]) -> DeduplicationResult:
        """Decide which new names are merges vs genuinely new vocabulary.

        For each name in new_names:
            embed → cosine similarity to all existing embeddings
            if max_sim ≥ threshold  → MERGE to the best-matching existing name
            else                    → GENUINELY NEW, add new FC head

        Genuinely new names are automatically appended to the registered
        vocabulary so subsequent calls see the expanded set.

        Args:
            new_names: concept names produced by the VLM pipeline for a new defect.

        Returns:
            DeduplicationResult with merged/genuinely_new/all_mappings/similarities.
        """
        if not new_names:
            return DeduplicationResult(
                merged={}, genuinely_new=[], all_mappings={}, similarities={}
            )

        new_embs = self.model.encode(
            new_names, normalize_embeddings=True, show_progress_bar=False
        )  # (n, 384)

        merged: dict[str, str] = {}
        genuinely_new: list[str] = []
        genuinely_new_embs: list[np.ndarray] = []
        all_mappings: dict[str, str] = {}
        similarities: dict[str, float] = {}

        for i, name in enumerate(new_names):
            emb = new_embs[i]  # (384,)

            if self.existing_embeddings is not None and len(self.existing_names) > 0:
                # cosine similarity = dot product of L2-normalised vectors
                sims = emb @ self.existing_embeddings.T   # (K,)
                best_idx = int(sims.argmax())
                max_sim = float(sims[best_idx])
                best_match = self.existing_names[best_idx]
            else:
                max_sim = 0.0
                best_match = None

            similarities[name] = max_sim

            if max_sim >= self.threshold:
                merged[name] = best_match
                all_mappings[name] = best_match
            else:
                genuinely_new.append(name)
                genuinely_new_embs.append(emb)
                all_mappings[name] = name   # keeps its own name as a new entry

        # ── register genuinely new names ──────────────────────────────────────
        if genuinely_new_embs:
            new_block = np.array(genuinely_new_embs)   # (n_new, 384)
            if self.existing_embeddings is not None:
                self.existing_embeddings = np.vstack(
                    [self.existing_embeddings, new_block]
                )
            else:
                self.existing_embeddings = new_block
            self.existing_names.extend(genuinely_new)

        return DeduplicationResult(
            merged=merged,
            genuinely_new=genuinely_new,
            all_mappings=all_mappings,
            similarities=similarities,
        )

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Save vocabulary state to a .pt file (model weights not saved — re-loaded on load)."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "existing_names": self.existing_names,
                "existing_embeddings": self.existing_embeddings,
                "threshold": self.threshold,
                "model_name": self.model_name,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "ConceptDeduplicator":
        """Restore vocabulary state; reloads the sentence-transformer model."""
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        obj = cls(model_name=ckpt["model_name"], threshold=ckpt["threshold"])
        obj.existing_names = ckpt["existing_names"]
        obj.existing_embeddings = ckpt["existing_embeddings"]
        return obj


# ── __main__ smoke test ───────────────────────────────────────────────────────

if __name__ == "__main__":
    import json
    import sys
    from pathlib import Path

    tier_path = Path("annotations/hazelnut/cl_tasks/concept_tier_map.json")
    if not tier_path.exists():
        sys.exit("Run task_csv_builder first, then run from project root.")

    print("=" * 68)
    print("ConceptDeduplicator — smoke test")
    print("=" * 68)

    # ── 1. load the 42 hazelnut concept names ─────────────────────────────────
    with open(tier_path) as f:
        tier_map = json.load(f)

    concept_names: list[str] = []
    for key in ["tier1_normal", "tier2_generic"] + [
        k for k in tier_map if k.startswith("tier3_")
    ]:
        for c in tier_map.get(key, []):
            if c not in concept_names:
                concept_names.append(c)

    print(f"\nLoading sentence-transformer (all-MiniLM-L6-v2) ...")
    dedup = ConceptDeduplicator(threshold=0.85)

    print(f"Registering vocabulary: {len(concept_names)} concepts")
    dedup.register_vocabulary(concept_names)
    vocab_before = len(dedup.existing_names)

    # ── 2+3. merge + new test cases ───────────────────────────────────────────
    # Probed actual similarity scores beforehand to confirm decisions:
    #   linear_crack_on_shell  0.9836 → linear_shell_crack     (MERGE ✓)
    #   reddish_brown_color    0.9486 → reddish_brown_hue      (MERGE ✓)
    #   inconsistent_texture   0.9110 → texture_inconsistency  (MERGE ✓)
    #   fissure_in_shell       0.9031 → deep_shell_fissure     (MERGE ✓)
    #   oil_stain_residue      0.4046 → [none close]           (NEW ✓)
    #   mold_growth_patch      0.4987 → [none close]           (NEW ✓)
    merge_inputs = [
        "linear_crack_on_shell",
        "reddish_brown_color",
        "inconsistent_texture",
        "fissure_in_shell",
    ]
    new_inputs = [
        "oil_stain_residue",
        "mold_growth_patch",
    ]
    all_inputs = merge_inputs + new_inputs

    result = dedup.process_new_concepts(all_inputs)

    # ── 4. print result table ─────────────────────────────────────────────────
    print()
    print(f"{'Input name':<30} {'Decision':<8} {'Maps to / new idx':<30} {'sim':>6}")
    print("-" * 78)
    for name in all_inputs:
        sim = result.similarities[name]
        if name in result.merged:
            decision = "MERGE"
            target = result.merged[name]
        else:
            decision = "NEW"
            new_idx = vocab_before + result.genuinely_new.index(name)
            target = f"new index {new_idx}"
        print(f"{name:<30} {decision:<8} {target:<30} {sim:>6.4f}")

    # ── 5. vocabulary growth check ────────────────────────────────────────────
    vocab_after = len(dedup.existing_names)
    print()
    print(f"Vocabulary size before : {vocab_before}")
    print(f"Vocabulary size after  : {vocab_after}")
    assert vocab_after == vocab_before + len(result.genuinely_new), \
        "Vocabulary growth mismatch"
    print(f"Growth: +{len(result.genuinely_new)} new concepts  ✓")

    # ── 6. verify merge and new decisions ─────────────────────────────────────
    assert set(result.merged.keys()) == set(merge_inputs), \
        f"Merge decisions wrong: {result.merged}"
    assert set(result.genuinely_new) == set(new_inputs), \
        f"New decisions wrong: {result.genuinely_new}"
    print(f"All merge decisions correct  ✓")
    print(f"All new decisions correct    ✓")

    # ── 7. save / load round-trip ─────────────────────────────────────────────
    print()
    tmp = Path("/tmp/concept_dedup_test.pt")
    dedup.save(tmp)
    restored = ConceptDeduplicator.load(tmp)

    assert restored.existing_names == dedup.existing_names
    assert np.allclose(restored.existing_embeddings, dedup.existing_embeddings)
    assert restored.threshold == dedup.threshold

    # Verify restored deduplicator produces identical decisions
    result2 = restored.process_new_concepts(["linear_crack_on_shell"])
    assert result2.merged == {"linear_crack_on_shell": "linear_shell_crack"}, \
        f"Restored deduplicator gave wrong result: {result2.merged}"

    print(f"Save/load round-trip         ✓")
    print(f"Restored deduplicator works  ✓")
    print(f"File size: {tmp.stat().st_size / 1024:.1f} KB")
    print()
    print("All assertions passed.")
