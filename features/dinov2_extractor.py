"""DINOv2 ViT-B/14 feature extractor — frozen, no gradients ever."""

from __future__ import annotations
import torch
import torch.nn as nn
from typing import NamedTuple


class DINOFeatures(NamedTuple):
    cls_token: torch.Tensor    # (B, 768)
    patch_tokens: torch.Tensor # (B, N_patches, 768)
    pooled: torch.Tensor       # (B, 1536)  = concat(cls, mean(patches))


class DINOv2Extractor(nn.Module):
    """Frozen DINOv2 ViT-B/14.

    TODO — implement:
      - torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
      - freeze all parameters (requires_grad = False everywhere)
      - forward(): run get_intermediate_layers, split CLS vs patch tokens
      - pooled = concat(cls_token, mean(patch_tokens)) → shape (B, 1536)
      - patch_tokens returned as (B, N, 768) for PatchCore
    """

    EMBED_DIM = 768
    POOLED_DIM = 1536  # CLS + mean(patches)

    def __init__(self, device: torch.device):
        super().__init__()
        raise NotImplementedError("DINOv2Extractor not yet implemented")

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> DINOFeatures:
        raise NotImplementedError
