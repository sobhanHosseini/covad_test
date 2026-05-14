"""CONCIL closed-form ridge regression solver.

No torch.optim. No loss.backward(). Pure matrix operations only.

Mathematical guarantee — zero-forgetting for concept heads:

    After accumulating T tasks the solution W_T is IDENTICAL to training
    jointly on all data D_1 ∪ ... ∪ D_T:

        A_T = Σ_t  Z_t_aug^T @ Z_t_aug      (feature-space Gram matrix)
        b_T = Σ_t  Z_t_aug^T @ C_t          (cross-correlation)
        W_T = (A_T + λI)^{-1} b_T           (ridge solution)

    Proof: A_T and b_T equal the joint-training Gram and cross-correlation
    because matrix addition is commutative and associative.  The solve depends
    only on the accumulated totals, not the order tasks arrive.

Gram matrices are stored in float64 for numerical stability.
Inputs (Z, C) are cast to float64 on entry; returned weights are float32.

Known limitation — anomaly head:
    The anomaly head's feature space is the concept activation vector c ∈ [0,1]^K.
    When K grows (new concepts added), the Gram matrix dimension changes and the
    old K×K accumulation cannot be re-used.  The anomaly head is therefore RESET
    on vocabulary expansion.  Concept heads retain zero-forgetting.  This is an
    accepted trade-off for the thesis experiment where vocabulary grows rarely.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch


class ConcilSolver:
    """Recursive ridge regression for concept heads and the linear anomaly head.

    All state is accumulated on CPU in float64.  The solver is stateless between
    task boundaries — it accumulates incrementally and each call to update_*
    returns the current best weights.
    """

    def __init__(
        self,
        input_dim: int = 1536,
        lambda_concept: float = 1e-4,
        lambda_anomaly: float = 1e-4,
    ):
        self.D = input_dim
        self.D_aug = input_dim + 1       # feature dimension + bias column
        self.lambda_c = lambda_concept
        self.lambda_a = lambda_anomaly
        self.K = 0                        # concept vocabulary size (grows with tasks)
        self.n_tasks_seen = 0

        self.A_concept: Optional[torch.Tensor] = None  # (D+1, D+1) float64
        self.b_concept: Optional[torch.Tensor] = None  # (D+1, K)   float64
        self.A_anomaly: Optional[torch.Tensor] = None  # (K+1, K+1) float64
        self.b_anomaly: Optional[torch.Tensor] = None  # (K+1, 1)   float64

        self._vocabulary_expanded: bool = False

    # ── internal helpers ──────────────────────────────────────────────────────

    @torch.no_grad()
    def _augment(self, X: torch.Tensor) -> torch.Tensor:
        """Append a bias column of ones: (N, D) → (N, D+1)."""
        ones = torch.ones(X.shape[0], 1, dtype=X.dtype, device=X.device)
        return torch.cat([X, ones], dim=1)

    @torch.no_grad()
    def _solve(
        self,
        A: torch.Tensor,
        b: torch.Tensor,
        lam: float,
    ) -> torch.Tensor:
        """W = (A + λI)^{-1} b via torch.linalg.solve (numerically stable)."""
        n = A.shape[0]
        reg = A + lam * torch.eye(n, dtype=A.dtype, device=A.device)
        return torch.linalg.solve(reg, b).cpu()

    # ── concept head update ────────────────────────────────────────────────────

    @torch.no_grad()
    def update_concept_heads(
        self,
        Z: torch.Tensor,
        C: torch.Tensor,
        new_concept_cols: Optional[int] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Accumulate one task's data and return updated concept head weights.

        Args:
            Z: (N, D) pooled DINOv2 features — no gradient required.
            C: (N, K_current) binary concept labels from task CSV.
            new_concept_cols: how many NEW concept columns are at the tail of C.
                If b_concept is already expanded (via expand_for_new_concepts),
                this argument is a cross-check and triggers no further expansion.

        Returns:
            W_weight: (K, D) float32 numpy → ConceptHeads.set_weights(W, b)
            W_bias:   (K,)   float32 numpy
        """
        Z = Z.double().detach().cpu()
        C = C.double().detach().cpu()

        N, K_c = C.shape
        Z_aug = self._augment(Z)          # (N, D+1)

        # ── initialise A on first call ────────────────────────────────────────
        if self.A_concept is None:
            self.A_concept = torch.zeros(self.D_aug, self.D_aug, dtype=torch.float64)

        # ── expand b when vocabulary grew (handles both pre-call and inline) ──
        if self.b_concept is not None:
            existing_K = self.b_concept.shape[1]
            if K_c > existing_K:
                n_expand = K_c - existing_K
                zeros = torch.zeros(self.D_aug, n_expand, dtype=torch.float64)
                self.b_concept = torch.cat([self.b_concept, zeros], dim=1)
        # (if b_concept is None it will be initialised below with correct K_c)

        if self.b_concept is None:
            self.b_concept = torch.zeros(self.D_aug, K_c, dtype=torch.float64)

        self.K = K_c

        # ── accumulate Gram matrix and cross-correlation ───────────────────────
        self.A_concept.add_(Z_aug.T @ Z_aug)   # (D+1, D+1)
        self.b_concept.add_(Z_aug.T @ C)       # (D+1, K)

        # ── solve ─────────────────────────────────────────────────────────────
        W = self._solve(self.A_concept, self.b_concept, self.lambda_c)  # (D+1, K)

        W_weight = W[: self.D, :].T.float().numpy()   # (K, D)
        W_bias   = W[self.D, :].float().numpy()       # (K,)

        self.n_tasks_seen += 1
        return W_weight, W_bias

    # ── anomaly head update ────────────────────────────────────────────────────

    @torch.no_grad()
    def update_anomaly_head(
        self,
        C_activated: torch.Tensor,
        y: torch.Tensor,
        vocabulary_expanded: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Accumulate concept activations + labels and return anomaly head weights.

        Args:
            C_activated: (N, K) concept activations from ConceptHeads.forward().
                         These are PREDICTED probabilities, not ground-truth labels.
            y:           (N,) binary anomaly labels (0=normal, 1=anomalous).
            vocabulary_expanded: if True, reset Gram matrices for the new K+1 dim.

        Returns:
            w_weight: (K,)  float32 numpy → LinearAnomalyHead.set_weights(w, b)
            w_bias:   (1,)  float32 numpy
        """
        C_activated = C_activated.double().detach().cpu()
        y = y.double().detach().cpu()

        K_current = C_activated.shape[1]
        dim_mismatch = (
            self.A_anomaly is not None
            and self.A_anomaly.shape[0] != K_current + 1
        )

        if vocabulary_expanded or self.A_anomaly is None or dim_mismatch:
            # Anomaly head reset on vocabulary expansion.
            # Concept heads retain zero-forgetting. Known limitation.
            self.A_anomaly = torch.zeros(
                K_current + 1, K_current + 1, dtype=torch.float64
            )
            self.b_anomaly = torch.zeros(K_current + 1, 1, dtype=torch.float64)
            self._vocabulary_expanded = False

        C_aug = self._augment(C_activated)              # (N, K+1)
        self.A_anomaly.add_(C_aug.T @ C_aug)
        self.b_anomaly.add_(C_aug.T @ y.unsqueeze(1))

        W = self._solve(self.A_anomaly, self.b_anomaly, self.lambda_a)  # (K+1, 1)

        w_weight = W[:K_current, 0].float().numpy()     # (K,)
        w_bias   = np.array([W[K_current, 0].item()], dtype=np.float32)
        return w_weight, w_bias

    # ── vocabulary expansion ───────────────────────────────────────────────────

    def expand_for_new_concepts(self, n_new: int) -> None:
        """Expand b_concept with n_new zero columns; A_concept is unchanged.

        Call this BEFORE update_concept_heads when new concept columns arrive.

        Zero-padding b for new columns is mathematically equivalent to those
        concepts having seen zero training data so far — correct, since they
        are genuinely new.  All previous columns of b are untouched, so old
        concept head weights are unaffected by expansion.
        """
        if n_new <= 0:
            return
        if self.b_concept is not None:
            zeros = torch.zeros(self.D_aug, n_new, dtype=torch.float64)
            self.b_concept = torch.cat([self.b_concept, zeros], dim=1)
        self._vocabulary_expanded = True
        self.K += n_new

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Save full solver state to a .pt file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "A_concept":    self.A_concept,
                "b_concept":    self.b_concept,
                "A_anomaly":    self.A_anomaly,
                "b_anomaly":    self.b_anomaly,
                "K":            self.K,
                "D":            self.D,
                "lambda_c":     self.lambda_c,
                "lambda_a":     self.lambda_a,
                "n_tasks_seen": self.n_tasks_seen,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "ConcilSolver":
        """Restore a saved solver. All Gram matrices are reloaded as-is."""
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        obj = cls(
            input_dim=ckpt["D"],
            lambda_concept=ckpt["lambda_c"],
            lambda_anomaly=ckpt["lambda_a"],
        )
        obj.A_concept    = ckpt["A_concept"]
        obj.b_concept    = ckpt["b_concept"]
        obj.A_anomaly    = ckpt["A_anomaly"]
        obj.b_anomaly    = ckpt["b_anomaly"]
        obj.K            = ckpt["K"]
        obj.n_tasks_seen = ckpt["n_tasks_seen"]
        return obj


# ── __main__ — verify correctness with synthetic data ────────────────────────

if __name__ == "__main__":
    import sys

    print("=" * 62)
    print("ConcilSolver — correctness tests (synthetic data)")
    print("=" * 62)

    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    D, K, N1, N2, N3 = 1536, 5, 50, 20, 15
    lam = 1e-4

    def _randbinary(n, k):
        return torch.from_numpy(rng.integers(0, 2, (n, k)).astype(np.float32))

    def _augment_np(X):
        return torch.cat([X, torch.ones(X.shape[0], 1, dtype=X.dtype)], dim=1)

    Z1 = torch.randn(N1, D)
    C1 = _randbinary(N1, K)
    y1 = _randbinary(N1, 1).squeeze(1)

    Z2 = torch.randn(N2, D)
    C2 = _randbinary(N2, K)
    y2 = _randbinary(N2, 1).squeeze(1)

    solver = ConcilSolver(input_dim=D, lambda_concept=lam, lambda_anomaly=lam)

    # ── TEST 1: Task 1 update ─────────────────────────────────────────────────
    print("\nTEST 1 — Task 1 update")
    print("-" * 40)

    W_c1, b_c1 = solver.update_concept_heads(Z1, C1)

    # Use zeroed concept heads for anomaly update (dummy activations for test)
    C_act1 = torch.sigmoid(torch.randn(N1, K))
    W_a1, b_a1 = solver.update_anomaly_head(C_act1, y1)

    ok_A = tuple(solver.A_concept.shape) == (D + 1, D + 1)
    ok_b = tuple(solver.b_concept.shape) == (D + 1, K)
    ok_W = W_c1.shape == (K, D)
    ok_Aa = tuple(solver.A_anomaly.shape) == (K + 1, K + 1)

    print(f"  A_concept shape : {tuple(solver.A_concept.shape)}  {'✓' if ok_A else '✗'}")
    print(f"  b_concept shape : {tuple(solver.b_concept.shape)}  {'✓' if ok_b else '✗'}")
    print(f"  W_concept returned shape : {W_c1.shape}  {'✓' if ok_W else '✗'}")
    print(f"  A_anomaly shape : {tuple(solver.A_anomaly.shape)}  {'✓' if ok_Aa else '✗'}")
    assert ok_A and ok_b and ok_W and ok_Aa, "TEST 1 FAILED"

    # ── TEST 2: Task 2 accumulation ───────────────────────────────────────────
    print("\nTEST 2 — Task 2 accumulation check")
    print("-" * 40)

    A_sum_before = solver.A_concept.sum().item()

    W_c2, b_c2 = solver.update_concept_heads(Z2, C2, new_concept_cols=0)
    C_act2 = torch.sigmoid(torch.randn(N2, K))
    W_a2, b_a2 = solver.update_anomaly_head(C_act2, y2)

    A_sum_after = solver.A_concept.sum().item()
    accumulated = A_sum_after > A_sum_before

    print(f"  A_concept.sum() before: {A_sum_before:.4f}")
    print(f"  A_concept.sum() after : {A_sum_after:.4f}")
    print(f"  A_concept accumulated (Task2 > Task1): {'✓' if accumulated else '✗'}")
    assert accumulated, "TEST 2 FAILED: A_concept did not grow"

    # ── TEST 3: zero-forgetting vs joint training (CRITICAL) ──────────────────
    print("\nTEST 3 — Zero-forgetting vs joint training  [CRITICAL]")
    print("-" * 40)

    # Recompute joint solution from scratch
    Z_joint  = torch.cat([Z1, Z2], dim=0).double()   # (N1+N2, D)
    C_joint  = torch.cat([C1, C2], dim=0).double()   # (N1+N2, K)
    Z_aug_j  = _augment_np(Z_joint)                  # (N1+N2, D+1)

    A_joint  = Z_aug_j.T @ Z_aug_j                   # (D+1, D+1) float64
    b_joint  = Z_aug_j.T @ C_joint                   # (D+1, K)   float64

    n = A_joint.shape[0]
    reg = A_joint + lam * torch.eye(n, dtype=torch.float64)
    W_joint = torch.linalg.solve(reg, b_joint)        # (D+1, K)
    W_joint_weight = W_joint[:D, :].T.float().numpy() # (K, D)

    max_diff = float(np.abs(W_joint_weight - W_c2).max())
    passed = max_diff < 1e-6

    print(f"  W_joint shape    : {W_joint_weight.shape}")
    print(f"  W_solver shape   : {W_c2.shape}")
    print(f"  Max diff vs joint training: {max_diff:.2e}")
    print(f"  Zero-forgetting verified  : {'✓' if passed else '✗ FAILED'}")
    assert passed, (
        f"Zero-forgetting test FAILED: max_diff = {max_diff:.2e} ≥ 1e-6\n"
        "This indicates a bug in the Gram accumulation."
    )

    # ── TEST 4: concept vocabulary expansion ──────────────────────────────────
    print("\nTEST 4 — Vocabulary expansion (K: 5 → 7)")
    print("-" * 40)

    K_before = solver.K
    b_cols_before = solver.b_concept.shape[1]

    solver.expand_for_new_concepts(n_new=2)

    K_mid = solver.K
    b_cols_mid = solver.b_concept.shape[1]
    print(f"  K before expand : {K_before}  →  after: {K_mid}")
    print(f"  b cols before   : {b_cols_before}  →  after: {b_cols_mid}")
    assert K_mid == K_before + 2, "K not updated correctly"
    assert b_cols_mid == b_cols_before + 2, "b not expanded correctly"
    # New columns of b must be zero
    assert solver.b_concept[:, K_before:].abs().max().item() == 0.0, \
        "New b columns not zero-initialised"
    print(f"  New b columns zero-init : ✓")

    Z3 = torch.randn(N3, D)
    C3 = _randbinary(N3, K_mid)                         # (15, 7)

    W_c3, b_c3 = solver.update_concept_heads(Z3, C3, new_concept_cols=2)
    C_act3 = torch.sigmoid(torch.randn(N3, K_mid))
    W_a3, b_a3 = solver.update_anomaly_head(
        C_act3, _randbinary(N3, 1).squeeze(1), vocabulary_expanded=True
    )

    shape_ok = W_c3.shape == (K_mid, D)
    print(f"  W_concept shape after expansion : {W_c3.shape}  {'✓' if shape_ok else '✗'}")
    assert shape_ok, "W shape wrong after expansion"

    # A_anomaly was reset; verify new size
    anomaly_ok = tuple(solver.A_anomaly.shape) == (K_mid + 1, K_mid + 1)
    print(f"  A_anomaly reset to ({K_mid+1},{K_mid+1}) : {'✓' if anomaly_ok else '✗'}")
    assert anomaly_ok

    # ── TEST 5: save / load round-trip ────────────────────────────────────────
    print("\nTEST 5 — Save / load round-trip")
    print("-" * 40)

    tmp = Path("/tmp/concil_test.pt")
    solver.save(tmp)
    solver2 = ConcilSolver.load(tmp)

    A_equal = torch.equal(solver.A_concept, solver2.A_concept)
    b_equal = torch.equal(solver.b_concept, solver2.b_concept)
    K_equal = solver.K == solver2.K
    t_equal = solver.n_tasks_seen == solver2.n_tasks_seen

    print(f"  A_concept identical : {'✓' if A_equal else '✗'}")
    print(f"  b_concept identical : {'✓' if b_equal else '✗'}")
    print(f"  K identical         : {'✓' if K_equal else '✗'}  ({solver2.K})")
    print(f"  n_tasks_seen        : {'✓' if t_equal else '✗'}  ({solver2.n_tasks_seen})")
    assert A_equal and b_equal and K_equal and t_equal, "Save/load round-trip FAILED"
    print(f"  Save/load round-trip : ✓  (file: {tmp.stat().st_size / 1024:.0f} KB)")

    print()
    print("All tests passed.")
