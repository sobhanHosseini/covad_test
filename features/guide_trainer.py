"""Guide coefficient learner for SAE-CBM anomaly detection.

Learns two guide vectors in the SAE's 4096-dim latent space via recursive
ridge regression — same mathematical discipline as solvers/concil.py.

  g⁻  normal guide  — built ONCE from all normal patches, never updated
  g⁺  anomaly guide — updated sequentially as new defect types arrive

Anomaly score for a patch with sparse code z:
    score(z) = cos_sim(z, g⁺) − cos_sim(z, g⁻)
             = z_norm · g⁺ − z_norm · g⁻          (guides are pre-normalised)

Image-level score: max over all 256 patch scores.

Float64 is used for all Gram matrix accumulation (matching concil.py).
Gram matmuls are computed on GPU in float32 for speed, then cast to float64
on CPU for accumulation — the precision difference is negligible for guide vectors
of this size (4096-dim) relative to the ridge regularisation applied.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


class GuideCoeffTrainer:
    """Ridge-regression guide vectors in SAE latent space.

    Args:
        sae:        frozen SparseAutoencoder instance
        lambda_reg: ridge regularisation coefficient (default 1.0)
        device:     device for SAE encoding (default cuda:0)
    """

    _CHUNK = 10_000   # tokens per accumulation chunk

    def __init__(
        self,
        sae: nn.Module,
        lambda_reg: float = 1.0,
        device: str | torch.device = "cuda:0",
    ):
        self.sae        = sae
        self.sae.eval()
        self.lambda_reg = float(lambda_reg)
        self.device     = torch.device(device) if isinstance(device, str) else device

        C = sae.config.d_hidden   # 4096

        # float64 Gram matrices on CPU (matching concil.py discipline)
        self.A_neg: Optional[torch.Tensor] = None   # (C, C)
        self.b_neg: Optional[torch.Tensor] = None   # (C,)
        self.A_pos: Optional[torch.Tensor] = None   # (C, C)
        self.b_pos: Optional[torch.Tensor] = None   # (C,)

        # Solved, L2-normalised guide vectors — float32, on self.device
        self.g_neg: Optional[torch.Tensor] = None   # (C,)
        self.g_pos: Optional[torch.Tensor] = None   # (C,)

        self._neg_built   = False
        self.n_pos_tasks  = 0

    # ── internal helpers ──────────────────────────────────────────────────────

    def _solve(self, A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """g = (A + λI)^{-1} b.  Inputs and output in float64 on CPU."""
        C   = A.shape[0]
        reg = A + self.lambda_reg * torch.eye(C, dtype=torch.float64)
        return torch.linalg.solve(reg, b)   # (C,) float64

    @torch.no_grad()
    def _encode_and_accumulate(
        self,
        tokens: torch.Tensor,
        A: torch.Tensor,
        b: torch.Tensor,
    ) -> None:
        """Encode tokens through SAE and accumulate into A (Gram) and b (sum).

        Matmuls run on GPU in float32; results are cast to float64 before
        adding to the CPU accumulators — same result as pure float64 at a
        fraction of the CPU time for 4096-dim codes.

        tokens: (N, d_input) float32 CPU
        A:      (C, C) float64 CPU  — modified in-place
        b:      (C,)   float64 CPU  — modified in-place
        """
        for i in range(0, len(tokens), self._CHUNK):
            chunk = tokens[i : i + self._CHUNK].to(self.device)         # GPU f32
            z     = self.sae.encode(chunk)                               # (B, C) f32
            gram  = (z.T @ z).cpu().double()                             # (C, C) f64
            A.add_(gram)
            b.add_(z.cpu().sum(0).double())                              # (C,)   f64
            del chunk, z, gram

    # ── public API ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def build_normal_guide(self, normal_tokens: torch.Tensor) -> None:
        """Build g⁻ from all normal patch tokens.  Called ONCE per category.

        Args:
            normal_tokens: (N, d_input) float32 CPU — raw DINOv2 patch tokens
                           from the pre-extracted training-set normals.
        """
        C = self.sae.config.d_hidden
        print(f"  Building g⁻ from {len(normal_tokens):,} normal tokens …")

        self.A_neg = torch.zeros(C, C, dtype=torch.float64)
        self.b_neg = torch.zeros(C,    dtype=torch.float64)

        self._encode_and_accumulate(normal_tokens, self.A_neg, self.b_neg)

        g    = self._solve(self.A_neg, self.b_neg).float()   # (C,) f32
        norm = g.norm()
        g    = g / norm.clamp(min=1e-8)
        self.g_neg      = g.to(self.device)
        self._neg_built = True
        print(f"  g⁻ built  (raw norm before normalisation = {norm.item():.4f})")

    @torch.no_grad()
    def update_anomaly_guide(self, defect_patch_tokens: torch.Tensor) -> None:
        """Update g⁺ with one task's defect patch tokens (CONCIL-style recursion).

        Accumulates into A_pos / b_pos so earlier tasks are never forgotten.

        Args:
            defect_patch_tokens: (N, d_input) float32 CPU — raw DINOv2 tokens
                                 from the 80% training split of the new defect.
        """
        C = self.sae.config.d_hidden
        if self.A_pos is None:
            self.A_pos = torch.zeros(C, C, dtype=torch.float64)
            self.b_pos = torch.zeros(C,    dtype=torch.float64)

        self._encode_and_accumulate(defect_patch_tokens, self.A_pos, self.b_pos)

        g    = self._solve(self.A_pos, self.b_pos).float()
        norm = g.norm()
        g    = g / norm.clamp(min=1e-8)
        self.g_pos       = g.to(self.device)
        self.n_pos_tasks += 1

    @torch.no_grad()
    def score_image_patches(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        """Per-patch anomaly scores.

        Patches whose sparse code has zero L2-norm (all-zero vector) receive
        score=0 silently without dividing by zero.

        Args:
            patch_tokens: (N, d_input) float32 — raw DINOv2 patch tokens
        Returns:
            scores: (N,) float32 CPU — one score per patch
        """
        if self.g_pos is None or self.g_neg is None:
            raise RuntimeError(
                "g⁺ and g⁻ must both exist before scoring. "
                "Call build_normal_guide() and at least one update_anomaly_guide()."
            )

        patch_tokens = patch_tokens.to(self.device)
        z            = self.sae.encode(patch_tokens)     # (N, C) float32

        z_norms   = z.norm(dim=-1, keepdim=True)        # (N, 1)
        zero_mask = (z_norms.squeeze(-1) == 0)          # (N,) bool

        z_norm_safe = z / z_norms.clamp(min=1e-8)       # (N, C)
        scores      = z_norm_safe @ self.g_pos - z_norm_safe @ self.g_neg  # (N,)
        scores[zero_mask] = 0.0

        return scores.cpu()

    @torch.no_grad()
    def score_image(self, patch_tokens: torch.Tensor) -> float:
        """Image-level anomaly score: max over all patch scores.

        Args:
            patch_tokens: (N_patches, d_input) or (B, N_patches, d_input)
        Returns:
            scalar float — max patch score across the image(s)
        """
        if patch_tokens.dim() == 3:
            B, N, D       = patch_tokens.shape
            patch_tokens  = patch_tokens.reshape(B * N, D)
        return float(self.score_image_patches(patch_tokens).max().item())

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "lambda_reg":  self.lambda_reg,
                "A_neg":       self.A_neg,
                "b_neg":       self.b_neg,
                "A_pos":       self.A_pos,
                "b_pos":       self.b_pos,
                "g_neg":       self.g_neg.cpu() if self.g_neg is not None else None,
                "g_pos":       self.g_pos.cpu() if self.g_pos is not None else None,
                "n_pos_tasks": self.n_pos_tasks,
                "sae_config":  self.sae.config,
            },
            path,
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        sae: nn.Module,
        device: str | torch.device = "cuda:0",
    ) -> "GuideCoeffTrainer":
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        obj  = cls(sae, lambda_reg=ckpt["lambda_reg"], device=device)
        obj.A_neg       = ckpt["A_neg"]
        obj.b_neg       = ckpt["b_neg"]
        obj.A_pos       = ckpt["A_pos"]
        obj.b_pos       = ckpt["b_pos"]
        obj.n_pos_tasks = ckpt["n_pos_tasks"]
        obj._neg_built  = ckpt["g_neg"] is not None
        if ckpt["g_neg"] is not None:
            obj.g_neg = ckpt["g_neg"].to(obj.device)
        if ckpt["g_pos"] is not None:
            obj.g_pos = ckpt["g_pos"].to(obj.device)
        return obj
