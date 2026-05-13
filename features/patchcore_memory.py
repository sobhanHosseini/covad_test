"""PatchCore memory bank built from normal training images at Task 1, then frozen."""

from __future__ import annotations
import torch
import numpy as np
from typing import NamedTuple


class AnomalyScores(NamedTuple):
    s_novel: np.ndarray    # (N,) image-level anomaly scores
    anomaly_map: np.ndarray  # (N, H, W) patch-level distance maps


class PatchCoreMemory:
    """Nearest-neighbour memory bank over DINOv2 patch tokens.

    TODO — implement:
      - build(patch_tokens): coreset subsampling (greedy k-center or random),
        store as flat (M, 768) array; mark as frozen after this call
      - score(patch_tokens): for each image, find NN distance to memory bank;
        s_novel = max patch distance; anomaly_map = spatial patch distances
        resized to original resolution
      - save(path) / load(path): persist/restore the memory bank tensor
      - Use faiss (IndexFlatL2) for efficient NN search on GPU/CPU

    Constraint: memory bank content never changes after build().
    """

    def __init__(self, coreset_size: int = 1000, device: torch.device = None):
        raise NotImplementedError("PatchCoreMemory not yet implemented")

    def build(self, patch_tokens: torch.Tensor) -> None:
        """Build and freeze the memory bank from normal training patch tokens."""
        raise NotImplementedError

    def score(self, patch_tokens: torch.Tensor) -> AnomalyScores:
        """Compute s_novel and anomaly_map for a batch of images."""
        raise NotImplementedError

    def save(self, path: str) -> None:
        raise NotImplementedError

    @classmethod
    def load(cls, path: str) -> "PatchCoreMemory":
        raise NotImplementedError
