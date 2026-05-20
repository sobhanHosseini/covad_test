"""
elastic_concil.py — ElasticConcilSolver: CONCIL with elastic net anomaly head.

The existing ConcilSolver uses ridge regression (L2 only):
    W = (A + λI)^{-1} b        closed-form, zero-forgetting

This class replaces the anomaly head solve with elastic net (L1 + L2):
    min_w  ½ wᵀÃw - b̃ᵀw + α·ρ·‖w_body‖₁ + ½·α(1-ρ)·‖w_body‖₂²

where Ã = A/N and b̃ = b/N are the Gram matrix and cross-correlation
normalised by total sample count N.  The bias (last dimension, from
_augment()) is NOT penalised by L1 or L2.

Zero-forgetting proof (elastic net via Gram coordinate descent):
    The coordinate descent update for variable j uses only A[j,:] and
    b[j], not the raw data.  Since A and b accumulate additively across
    tasks (same as ridge), the elastic net solution on accumulated A/N and
    b/N is IDENTICAL to training jointly on all data.  Zero-forgetting
    holds for any l1_ratio ∈ [0, 1] and any alpha > 0.

When l1_ratio = 0 (pure L2):
    Coordinate descent converges to the ridge solution:
        w[j] = (b̃[j] - Ã[j,¬j] @ w[¬j]) / (Ã[j,j] + alpha)
    which equals (A + alpha·N·I)^{-1} b — same as ConcilSolver with λ = alpha·N.

The Gram accumulation logic is UNCHANGED from ConcilSolver.
Only update_anomaly_head is overridden to replace _solve with _solve_elastic.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# Import parent class without modifying covad_test/
_COVAD = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_COVAD))
from solvers.concil import ConcilSolver   # noqa: E402


class ElasticConcilSolver(ConcilSolver):
    """CONCIL anomaly head with elastic net regularization.

    Concept heads (if used) still use L2 ridge from the parent class.
    Only the anomaly head is switched to elastic net.

    Args:
        input_dim: Feature dimension for concept heads (unused when only
            update_anomaly_head is called, but required by parent).
        alpha: Overall regularisation strength (scales both L1 and L2).
            Comparable to sklearn ElasticNet's `alpha`.
        l1_ratio: Mixing parameter ρ ∈ [0, 1].
            ρ = 0 → pure L2 ridge.
            ρ = 0.5 → balanced elastic net.
            ρ = 1.0 → pure L1 (lasso).
        lambda_concept: L2 strength for concept heads (unchanged from parent).
    """

    def __init__(
        self,
        input_dim: int = 13,
        alpha: float = 0.1,
        l1_ratio: float = 0.5,
        lambda_concept: float = 1e-4,
    ):
        # lambda_anomaly=1e-4 is kept in parent state but never used for the anomaly head
        super().__init__(input_dim=input_dim, lambda_concept=lambda_concept,
                         lambda_anomaly=1e-4)
        self.alpha    = float(alpha)
        self.l1_ratio = float(l1_ratio)
        self.N_anomaly: int = 0    # accumulated sample count for normalisation

    # ── elastic net coordinate descent ────────────────────────────────────────

    def _solve_elastic(
        self,
        A: torch.Tensor,     # (K+1, K+1) float64 accumulated Gram (unnormalised)
        b: torch.Tensor,     # (K+1,)     float64 accumulated cross-corr (unnormalised)
        n_samples: int,
        max_iter: int = 3000,
        tol: float = 1e-9,
    ) -> torch.Tensor:
        """Elastic net via coordinate descent on the normalised Gram matrix.

        Minimises (over w ∈ ℝ^{K+1}):
            ½ wᵀ (A/N) w  −  (b/N)ᵀ w
            + α·ρ · ‖w[0:K]‖₁              (L1, body only — bias not penalised)
            + ½·α·(1-ρ) · ‖w[0:K]‖₂²       (L2, body only)

        Coordinate descent update for variable j (< K, i.e. not bias):
            r_j = (b/N)[j] - (A/N)[j,:] @ w + (A/N)[j,j] * w[j]
            w[j] = S(r_j, α·ρ) / ((A/N)[j,j] + α·(1-ρ))

        where S(z, λ) = sign(z) · max(|z| - λ, 0) is the soft-threshold.

        For the bias (j = K):
            w[K] = r_K / (A/N)[K,K]          (no penalty)

        The Aw product is updated with rank-1 corrections after each
        coordinate update — O(K) per coordinate, O(K²) per sweep.

        Args:
            A: Accumulated Gram matrix (not divided by N yet).
            b: Accumulated cross-correlation vector.
            n_samples: Total N for normalisation.

        Returns:
            w: (K+1,) float64 solution vector.
        """
        K_aug = A.shape[0]    # K concepts + 1 bias
        l1    = self.alpha * self.l1_ratio
        l2    = self.alpha * (1.0 - self.l1_ratio)

        # Normalise by N so alpha has interpretable scale
        A_n = A / n_samples   # (K+1, K+1) float64
        b_n = b / n_samples   # (K+1,)     float64

        w   = torch.zeros(K_aug, dtype=torch.float64)
        Aw  = torch.zeros(K_aug, dtype=torch.float64)   # A_n @ w, updated cheaply

        for _ in range(max_iter):
            w_prev = w.clone()

            for j in range(K_aug):
                # Partial residual: what the coordinate j should predict
                r_j    = float(b_n[j] - Aw[j] + A_n[j, j] * w[j])
                old_wj = float(w[j])

                if j < K_aug - 1:
                    # Concept dimension: apply L1 + L2 penalty
                    denom = float(A_n[j, j]) + l2
                    if denom < 1e-14:
                        new_wj = 0.0
                    elif r_j > l1:
                        new_wj = (r_j - l1) / denom
                    elif r_j < -l1:
                        new_wj = (r_j + l1) / denom
                    else:
                        new_wj = 0.0          # soft-thresholded to zero
                else:
                    # Bias: no penalty
                    d = float(A_n[j, j])
                    new_wj = r_j / d if abs(d) > 1e-14 else 0.0

                w[j] = new_wj
                # Rank-1 update of Aw avoids recomputing A_n @ w each step
                Aw  += A_n[:, j] * (new_wj - old_wj)

            if (w - w_prev).abs().max().item() < tol:
                break

        return w

    # ── override update_anomaly_head ──────────────────────────────────────────

    @torch.no_grad()
    def update_anomaly_head(
        self,
        C_activated: torch.Tensor,
        y: torch.Tensor,
        vocabulary_expanded: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Accumulate Gram matrices and solve with elastic net.

        Interface is identical to ConcilSolver.update_anomaly_head.
        Gram accumulation is UNCHANGED.  Only the solve step differs.

        Args:
            C_activated: (N, K) float32 concept activation vectors.
            y:           (N,)   float32 binary anomaly labels.
            vocabulary_expanded: reset Gram if concept dimension changed.

        Returns:
            w_weight: (K,)  float32 — concept weights (may be sparse due to L1).
            w_bias:   (1,)  float32 — unpenalised bias.
        """
        C_activated = C_activated.double().detach().cpu()
        y           = y.double().detach().cpu()
        K_current   = C_activated.shape[1]
        N           = C_activated.shape[0]

        dim_mismatch = (
            self.A_anomaly is not None
            and self.A_anomaly.shape[0] != K_current + 1
        )

        if vocabulary_expanded or self.A_anomaly is None or dim_mismatch:
            self.A_anomaly = torch.zeros(K_current + 1, K_current + 1,
                                         dtype=torch.float64)
            self.b_anomaly = torch.zeros(K_current + 1, 1, dtype=torch.float64)
            self.N_anomaly = 0

        C_aug = self._augment(C_activated)        # (N, K+1)
        self.A_anomaly.add_(C_aug.T @ C_aug)      # accumulate Gram
        self.b_anomaly.add_(C_aug.T @ y.unsqueeze(1))  # accumulate cross-corr
        self.N_anomaly += N

        # Elastic net solve on accumulated (unnormalised) Gram
        b_flat = self.b_anomaly[:, 0]             # (K+1,) float64
        w = self._solve_elastic(self.A_anomaly, b_flat, self.N_anomaly)

        w_weight = w[:K_current].float().numpy()  # (K,)  may contain zeros
        w_bias   = np.array([float(w[K_current])], dtype=np.float32)
        return w_weight, w_bias

    @property
    def n_zero_weights(self) -> int:
        """Number of zero-valued concept weights (after most recent solve)."""
        if self.A_anomaly is None:
            return 0
        b_flat = self.b_anomaly[:, 0]
        w = self._solve_elastic(self.A_anomaly, b_flat, self.N_anomaly)
        K = self.A_anomaly.shape[0] - 1
        return int((w[:K].abs() < 1e-8).sum().item())
