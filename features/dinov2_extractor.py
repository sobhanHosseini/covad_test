"""DINOv2 ViT-B/14 feature extractor — frozen, no gradients ever.

Three extraction methods:
    extract_patch_tokens(images) → (B, N_patches, 768)
    extract_pooled(images)       → (B, 1536)  = concat(CLS, mean(patches))
    extract_both(images)         → (patch_tokens, pooled_z)

For 224×224 input: N_patches = (224 / 14)² = 256.
EMBED_DIM = 768, POOLED_DIM = 1536.

xFormers is not required; DINOv2 falls back to standard attention with a
harmless UserWarning at first load. The warnings are suppressed here because
they are emitted at import time and would clutter every script that imports
this module.
"""

from __future__ import annotations

import warnings
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# Suppress xFormers availability warnings from the DINOv2 source tree.
# These are purely informational: the model falls back to standard attention.
warnings.filterwarnings("ignore", message="xFormers is not available")

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]
_PATCH_SIZE    = 14
_EMBED_DIM     = 768
_POOLED_DIM    = 1536   # CLS(768) + mean_patch(768)


class DINOv2Extractor(nn.Module):
    """Frozen DINOv2 ViT-B/14.

    All parameters have requires_grad=False at all times. The module is
    always in eval() mode. No gradient ever flows through this class.

    Accepts images as:
      - PIL.Image (single image)
      - list[PIL.Image] (batch)
      - torch.Tensor (B, 3, H, W) already normalised with ImageNet stats
    """

    PATCH_SIZE = _PATCH_SIZE
    EMBED_DIM  = _EMBED_DIM
    POOLED_DIM = _POOLED_DIM

    def __init__(self, device: torch.device | None = None):
        super().__init__()

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        model = torch.hub.load(
            "facebookresearch/dinov2",
            "dinov2_vitb14",
            pretrained=True,
            verbose=False,
        )

        # Freeze — hard guarantee: no parameter will ever have requires_grad=True
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

    # ── internal helpers ──────────────────────────────────────────────────────

    def _prepare(
        self,
        images: torch.Tensor | Image.Image | list[Image.Image],
    ) -> torch.Tensor:
        """Normalise and batch images to (B, 3, H, W) on self.device."""
        if isinstance(images, Image.Image):
            images = [images]
        if isinstance(images, list):
            x = torch.stack([self._pil_to_tensor(img) for img in images])
        else:
            x = images  # caller already provides a normalised tensor
        return x.to(self.device)

    @torch.no_grad()
    def _run(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Single forward pass → (cls_token (B,768), patch_tokens (B,N,768))."""
        feats = self.model.forward_features(x)
        cls_token    = feats["x_norm_clstoken"]    # (B, 768)
        patch_tokens = feats["x_norm_patchtokens"]  # (B, N, 768)
        return cls_token, patch_tokens

    # ── public API ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def extract_patch_tokens(
        self,
        images: torch.Tensor | Image.Image | list[Image.Image],
    ) -> torch.Tensor:
        """Return the spatial patch token grid: (B, N_patches, 768)."""
        x = self._prepare(images)
        _, patch_tokens = self._run(x)
        return patch_tokens

    @torch.no_grad()
    def extract_pooled(
        self,
        images: torch.Tensor | Image.Image | list[Image.Image],
    ) -> torch.Tensor:
        """Return pooled image representation: (B, 1536).

        pooled = concat(CLS_token, mean(patch_tokens), dim=-1)
        """
        x = self._prepare(images)
        cls_token, patch_tokens = self._run(x)
        mean_patch = patch_tokens.mean(dim=1)                    # (B, 768)
        return torch.cat([cls_token, mean_patch], dim=1)         # (B, 1536)

    @torch.no_grad()
    def extract_both(
        self,
        images: torch.Tensor | Image.Image | list[Image.Image],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Single forward pass returning both representations.

        Returns:
            patch_tokens: (B, N_patches, 768)  — for PatchCore memory bank
            pooled_z:     (B, 1536)             — for CONCIL concept heads
        """
        x = self._prepare(images)
        cls_token, patch_tokens = self._run(x)
        mean_patch = patch_tokens.mean(dim=1)
        pooled_z = torch.cat([cls_token, mean_patch], dim=1)
        return patch_tokens, pooled_z

    def forward(
        self,
        images: torch.Tensor | Image.Image | list[Image.Image],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Alias for extract_both."""
        return self.extract_both(images)

    # ── safety guard ──────────────────────────────────────────────────────────

    def train(self, mode: bool = True) -> "DINOv2Extractor":
        """Override: extractor stays in eval mode regardless of caller."""
        return super().train(False)


# ── __main__ smoke test ───────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from pathlib import Path
    import pandas as pd

    print("=" * 60)
    print("DINOv2Extractor — smoke test")
    print("=" * 60)

    # Resolve a real hazelnut image from the annotations CSV
    csv_path = Path("annotations/hazelnut/hazelnut.csv")
    if not csv_path.exists():
        sys.exit(f"CSV not found: {csv_path}. Run from project root.")

    image_path = Path(
        pd.read_csv(csv_path).query("anomaly_type == 'good'")
        .iloc[0]["image_path"]
    )
    if not image_path.exists():
        sys.exit(f"Image not found: {image_path}")

    print(f"Image : {image_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print()

    print("Loading DINOv2 ViT-B/14 ...")
    extractor = DINOv2Extractor(device=device)

    # ── frozen check ──────────────────────────────────────────────────────────
    all_frozen = all(not p.requires_grad for p in extractor.model.parameters())
    total_params = sum(p.numel() for p in extractor.model.parameters())
    print(f"All parameters frozen : {all_frozen}")
    print(f"Total parameters      : {total_params:,}")
    print()

    # ── single PIL image ──────────────────────────────────────────────────────
    img = Image.open(image_path).convert("RGB")
    print("── Single PIL image ──────────────────────────────")

    patch_tokens = extractor.extract_patch_tokens(img)
    print(f"extract_patch_tokens  : {tuple(patch_tokens.shape)}")

    pooled = extractor.extract_pooled(img)
    print(f"extract_pooled        : {tuple(pooled.shape)}")

    patch_t, pooled_z = extractor.extract_both(img)
    print(f"extract_both → patches: {tuple(patch_t.shape)}")
    print(f"             → pooled : {tuple(pooled_z.shape)}")

    print(f"Grad on output        : {pooled_z.requires_grad}")
    print()

    # ── batch of 4 PIL images ─────────────────────────────────────────────────
    imgs = [img] * 4
    print("── Batch (4 PIL images) ──────────────────────────")
    bp, bz = extractor.extract_both(imgs)
    print(f"extract_both → patches: {tuple(bp.shape)}")
    print(f"             → pooled : {tuple(bz.shape)}")
    print()

    # ── pre-normalised tensor ─────────────────────────────────────────────────
    print("── Pre-normalised tensor (B=2) ───────────────────")
    dummy = extractor._prepare([img, img])   # borrow prepare for a clean tensor
    tp, tz = extractor.extract_both(dummy)
    print(f"extract_both → patches: {tuple(tp.shape)}")
    print(f"             → pooled : {tuple(tz.shape)}")
    print()

    # ── dimension assertions ──────────────────────────────────────────────────
    assert pooled_z.shape == (1, 1536), f"Expected (1,1536), got {pooled_z.shape}"
    assert patch_t.shape  == (1, 256, 768), f"Expected (1,256,768), got {patch_t.shape}"
    assert not pooled_z.requires_grad, "Output must not require grad"
    assert all_frozen, "All parameters must be frozen"

    print("All assertions passed.")
    print(f"\nExpected 1536 = 768 (CLS) + 768 (mean patches) : {pooled_z.shape[1] == 1536}")
