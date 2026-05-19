"""
Sparse Autoencoder (SAE) for DINOv2 patch tokens.

TopK SAE: each patch token is reconstructed from exactly k active
dictionary atoms out of C total. Atoms are unit-norm vectors in
the patch token space (EMBED_DIM).

Default config targets DINOv2 ViT-L/14 (EMBED_DIM=1024).
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SAEConfig:
    d_input:  int = 1024   # DINOv2 ViT-L/14 patch dim
    d_hidden: int = 4096   # dictionary size (number of atoms)
    k:        int = 64     # TopK: active atoms per patch


class SparseAutoencoder(nn.Module):
    """TopK Sparse Autoencoder.

    Architecture (following EleutherAI/Anthropic conventions):
        encode: x → (x/‖x‖ - b_dec) @ W_enc + b_enc → TopK(ReLU(.)) → z
        decode: z @ W_dec + b_dec → x_hat

    Decoder columns W_dec[c] are kept on the unit sphere after every step.
    Input tokens are L2-normalised before encoding (matches DINOv2 output norm).
    """

    def __init__(self, config: SAEConfig):
        super().__init__()
        self.config = config
        d, C = config.d_input, config.d_hidden

        self.b_dec = nn.Parameter(torch.zeros(d))

        self.W_enc = nn.Parameter(torch.empty(d, C))
        nn.init.kaiming_uniform_(self.W_enc)
        self.b_enc = nn.Parameter(torch.zeros(C))

        self.W_dec = nn.Parameter(torch.empty(C, d))
        nn.init.kaiming_uniform_(self.W_dec)
        self._normalize_decoder()

    def _normalize_decoder(self):
        """Project decoder rows onto unit sphere (called after each opt step)."""
        with torch.no_grad():
            norms = self.W_dec.norm(dim=1, keepdim=True).clamp(min=1e-8)
            self.W_dec.data = self.W_dec.data / norms

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, d_input)  raw patch tokens
        z : (B, d_hidden) sparse codes — exactly k non-zero per row
        """
        x_norm = x / x.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        pre    = (x_norm - self.b_dec) @ self.W_enc + self.b_enc  # (B, C)
        topk_vals, topk_idx = torch.topk(pre.relu(), self.config.k, dim=-1)
        z = torch.zeros_like(pre)
        z.scatter_(-1, topk_idx, topk_vals)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """z : (B, d_hidden) → x_hat : (B, d_input)"""
        return z @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor):
        """Returns (z, x_hat, x_norm) — x_norm is the normalised input."""
        x_norm = x / x.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        z      = self.encode(x)
        x_hat  = self.decode(z)
        return z, x_hat, x_norm

    def get_atom(self, idx: int) -> torch.Tensor:
        """Return unit-norm dictionary atom idx. Shape: (d_input,)"""
        return self.W_dec[idx].detach()

    def save(self, path: str):
        torch.save({"config": self.config,
                    "state_dict": self.state_dict()}, path)
        print(f"SAE saved → {path}")

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "SparseAutoencoder":
        ckpt = torch.load(path, map_location=device, weights_only=False)
        m    = cls(ckpt["config"])
        m.load_state_dict(ckpt["state_dict"])
        return m
