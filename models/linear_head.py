"""Linear anomaly head: y = sigmoid(w^T c + b), no hidden layers, no ReLU.

Maps concept activations c ∈ [0,1]^K to an anomaly probability y ∈ [0,1].
Weights are SET by CONCIL — never trained by gradient descent.
Weight vector w_k is the learned contribution of concept k to anomaly score,
which makes the anomaly decision directly interpretable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import torch
import torch.nn as nn


class LinearAnomalyHead(nn.Module):
    """Single linear sigmoid layer: c ∈ [0,1]^K → y ∈ [0,1].

    Constraint: no hidden layers, no ReLU, no MLP — linear only.
    Weights are set by CONCIL ridge regression, never by optimiser.
    """

    def __init__(self, n_concepts: int):
        super().__init__()
        self.n_concepts = n_concepts

        self.linear = nn.Linear(n_concepts, 1, bias=True)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

        for p in self.parameters():
            p.requires_grad = False

    # ── forward ───────────────────────────────────────────────────────────────

    @torch.no_grad()
    def forward(self, c: torch.Tensor) -> torch.Tensor:
        """Map concept activations to anomaly probability.

        Args:
            c: (B, K) concept activation vector in [0, 1]
        Returns:
            y: (B,) anomaly scores in [0, 1]
        """
        return torch.sigmoid(self.linear(c.to(self.linear.weight.device))).squeeze(-1)

    # ── weight management (CONCIL interface) ──────────────────────────────────

    def set_weights(
        self,
        w: Union[np.ndarray, torch.Tensor],
        b: Union[np.ndarray, torch.Tensor],
    ) -> None:
        """Write CONCIL solution into the linear layer.

        Args:
            w: (K,) or (1, K) — anomaly head weight vector.
            b: scalar, (1,), or () — bias.
        """
        if isinstance(w, np.ndarray):
            w = torch.from_numpy(w).float()
        if isinstance(b, np.ndarray):
            b = torch.from_numpy(b).float()

        self.linear.weight.data.copy_(w.reshape(1, -1))
        self.linear.bias.data.copy_(b.reshape(1))

    def expand(self, new_n_concepts: int) -> None:
        """Expand weight vector when concept vocabulary grows.

        Old weights are preserved exactly; new weights are zero-initialised
        (CONCIL will update them on the next task's solve).

        Args:
            new_n_concepts: total number of concepts after expansion (must be ≥ current).
        """
        if new_n_concepts <= self.n_concepts:
            raise ValueError(
                f"new_n_concepts ({new_n_concepts}) must exceed "
                f"current ({self.n_concepts})"
            )
        new_linear = nn.Linear(new_n_concepts, 1, bias=True)
        nn.init.zeros_(new_linear.weight)
        nn.init.zeros_(new_linear.bias)

        with torch.no_grad():
            new_linear.weight[:, : self.n_concepts].copy_(self.linear.weight)
            new_linear.bias.copy_(self.linear.bias)

        for p in new_linear.parameters():
            p.requires_grad = False

        self.linear = new_linear
        self.n_concepts = new_n_concepts

    # ── interpretability ──────────────────────────────────────────────────────

    @property
    def concept_weights(self) -> np.ndarray:
        """Weight vector w as numpy array, shape (K,).

        Thesis use: w_k > 0 means concept k contributes to anomaly score;
        w_k < 0 means presence of concept k suppresses anomaly score.
        """
        return self.linear.weight.data.squeeze(0).cpu().numpy()

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "n_concepts": self.n_concepts,
                "state_dict": self.linear.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "LinearAnomalyHead":
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        obj = cls(ckpt["n_concepts"])
        obj.linear.load_state_dict(ckpt["state_dict"])
        for p in obj.parameters():
            p.requires_grad = False
        return obj

    def train(self, mode: bool = True) -> "LinearAnomalyHead":
        """Override: always stays in eval mode."""
        return super().train(False)
