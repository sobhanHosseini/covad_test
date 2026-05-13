"""K independent sigmoid concept heads — weight tensors, updated by CONCIL only."""

from __future__ import annotations
import torch
import numpy as np


class ConceptHeads:
    """K independent binary concept classifiers.

    Each head k maps pooled z ∈ R^1536 → c_k ∈ [0, 1] via sigmoid(w_k^T z + b_k).
    Weights are plain numpy/torch tensors, never inside an nn.Module training loop.
    CONCIL solver writes directly to self.W and self.b.

    TODO — implement:
      - __init__(input_dim, concept_names): initialise W=(input_dim, K), b=(K,) to zeros
      - add_concept(name): append a new zero-initialised column to W and b
      - forward(z): return sigmoid(z @ W + b) → shape (B, K)
      - state: concept_names list keeps in sync with W columns
    """

    INPUT_DIM = 1536  # DINOv2 pooled dimension

    def __init__(self, concept_names: list[str]):
        raise NotImplementedError("ConceptHeads not yet implemented")

    @property
    def K(self) -> int:
        raise NotImplementedError

    def add_concept(self, name: str) -> int:
        """Append a new zero-weight head. Returns new index."""
        raise NotImplementedError

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """(B, 1536) → (B, K) concept activation probabilities."""
        raise NotImplementedError

    def __call__(self, z: torch.Tensor) -> torch.Tensor:
        return self.forward(z)
