"""CONCIL closed-form ridge regression solver.

No torch.optim. No loss.backward(). Pure matrix operations.

All state (A, b) is accumulated across tasks and persisted in checkpoints.
"""

from __future__ import annotations
import numpy as np
import torch


class ConcilState:
    """Persistent accumulator for one regression target (concept head or anomaly head).

    TODO — implement:
      - __init__(input_dim, output_dim, lambda_ridge):
          A = zeros(input_dim, input_dim)   # Gram matrix
          b = zeros(input_dim, output_dim)  # cross-correlation

      - accumulate(Z, C):
          A += Z.T @ Z           # Z: (N, D), C: (N, K)
          b += Z.T @ C

      - solve() -> W:
          W = (A + lambda * I)^{-1} b     # shape (D, K)
          uses np.linalg.solve for numerical stability (NOT np.linalg.inv)

      - save(path) / load(path): persist A, b, lambda_ to .npz

    Constraint: old concept head weights are mathematically unchanged by
    adding new columns (ridge regression is independent per column).
    """

    def __init__(self, input_dim: int, output_dim: int, lambda_ridge: float = 1e-3):
        raise NotImplementedError("ConcilState not yet implemented")

    def accumulate(self, Z: np.ndarray, C: np.ndarray) -> None:
        """Accumulate Gram matrix and cross-correlation for one task's data."""
        raise NotImplementedError

    def solve(self) -> np.ndarray:
        """Solve ridge regression: W = (A + λI)^{-1} b."""
        raise NotImplementedError

    def expand_output_dim(self, new_output_dim: int) -> None:
        """Grow b when new concept columns are added (zero-pad new columns)."""
        raise NotImplementedError

    def save(self, path: str) -> None:
        raise NotImplementedError

    @classmethod
    def load(cls, path: str) -> "ConcilState":
        raise NotImplementedError
