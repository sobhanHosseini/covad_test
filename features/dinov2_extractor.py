"""DINOv2 feature extractor — frozen, no gradients ever.

Supports ViT-B/14 (default, backward compatible) and ViT-L/14.

    DINOv2Extractor()                    → ViT-B/14  (768-dim patches, 1536 pooled)
    DINOv2Extractor("dinov2_vitl14")     → ViT-L/14  (1024-dim patches, 2048 pooled)

Three extraction methods:
    extract_patch_tokens(images) → (B, N_patches, EMBED_DIM)
    extract_pooled(images)       → (B, POOLED_DIM)
    extract_both(images)         → (patch_tokens, pooled_z)

For 224×224 input: N_patches = (224 / 14)² = 256.
"""

from __future__ import annotations
import warnings
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

warnings.filterwarnings("ignore", message="xFormers is not available")

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

_MODEL_CONFIGS = {
    "dinov2_vitb14":     {"embed_dim": 768,  "pooled_dim": 1536, "patch_size": 14},
    "dinov2_vitl14":     {"embed_dim": 1024, "pooled_dim": 2048, "patch_size": 14},
    "dinov2_vitl14_reg": {"embed_dim": 1024, "pooled_dim": 2048, "patch_size": 14},
    "dinov2_vitg14":     {"embed_dim": 1536, "pooled_dim": 3072, "patch_size": 14},
}


class DINOv2Extractor(nn.Module):
    """Frozen DINOv2 feature extractor.

    All parameters have requires_grad=False at all times.
    Always in eval() mode. No gradient ever flows through this class.

    Args:
        model_name: one of "dinov2_vitb14" (default), "dinov2_vitl14",
                    "dinov2_vitg14"
    """

    def __init__(
        self,
        model_name: str = "dinov2_vitb14",
        device: torch.device | None = None,
    ):
        super().__init__()

        if model_name not in _MODEL_CONFIGS:
            raise ValueError(
                f"Unknown model '{model_name}'. "
                f"Choose from: {list(_MODEL_CONFIGS.keys())}"
            )

        cfg = _MODEL_CONFIGS[model_name]
        self.model_name = model_name
        self.EMBED_DIM  = cfg["embed_dim"]
        self.POOLED_DIM = cfg["pooled_dim"]
        self.PATCH_SIZE = cfg["patch_size"]

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        model = torch.hub.load(
            "facebookresearch/dinov2",
            model_name,
            pretrained=True,
            verbose=False,
        )
        for param in model.parameters():
            param.requires_grad = False
        model.eval()
        self.model = model.to(device)

        self._pil_to_tensor = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ])

    def _prepare(self, images) -> torch.Tensor:
        if isinstance(images, Image.Image):
            images = [images]
        if isinstance(images, list):
            x = torch.stack([self._pil_to_tensor(img) for img in images])
        else:
            x = images
        return x.to(self.device)

    @torch.no_grad()
    def _run(self, x: torch.Tensor):
        feats = self.model.forward_features(x)
        cls_token    = feats["x_norm_clstoken"]
        patch_tokens = feats["x_norm_patchtokens"]
        return cls_token, patch_tokens

    @torch.no_grad()
    def extract_patch_tokens(self, images) -> torch.Tensor:
        """(B, N_patches, EMBED_DIM)"""
        x = self._prepare(images)
        _, patch_tokens = self._run(x)
        return patch_tokens

    @torch.no_grad()
    def extract_pooled(self, images) -> torch.Tensor:
        """(B, POOLED_DIM) = concat(CLS, mean(patches))"""
        x = self._prepare(images)
        cls_token, patch_tokens = self._run(x)
        return torch.cat([cls_token, patch_tokens.mean(dim=1)], dim=1)

    @torch.no_grad()
    def extract_both(self, images):
        """Returns (patch_tokens, pooled_z)"""
        x = self._prepare(images)
        cls_token, patch_tokens = self._run(x)
        pooled_z = torch.cat([cls_token, patch_tokens.mean(dim=1)], dim=1)
        return patch_tokens, pooled_z

    def forward(self, images):
        return self.extract_both(images)

    def train(self, mode: bool = True) -> "DINOv2Extractor":
        return super().train(False)
