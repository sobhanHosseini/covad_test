"""Linear anomaly head: y = sigmoid(w^T c + b), no hidden layers, no ReLU."""

from __future__ import annotations
import torch
import numpy as np


class LinearAnomalyHead:
    """Single linear layer: c ∈ [0,1]^K → y ∈ [0,1].

    Weight vector w ∈ R^K and scalar bias b are plain tensors.
    Updated by CONCIL closed-form solver, never by gradient descent.

    TODO — implement:
      - __init__(K): initialise w=(K,), b=scalar to zeros
      - resize(new_K): extend w with zeros when new concept heads are added
      - forward(c): return sigmoid(c @ w + b) → shape (B,)
    """

    def __init__(self, K: int):
        raise NotImplementedError("LinearAnomalyHead not yet implemented")

    def resize(self, new_K: int) -> None:
        """Extend weight vector when concept vocabulary grows."""
        raise NotImplementedError

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        """(B, K) → (B,) anomaly probabilities."""
        raise NotImplementedError

    def __call__(self, c: torch.Tensor) -> torch.Tensor:
        return self.forward(c)
