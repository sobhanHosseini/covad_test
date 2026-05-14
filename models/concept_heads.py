"""K independent sigmoid concept heads — weights set by CONCIL, never by gradients.

Each head k: z ∈ R^{1536} → sigmoid(w_k^T z + b_k) ∈ [0, 1].
All K heads share one nn.Linear(1536, K) for efficiency; they are
mathematically independent because CONCIL solves each column of W separately.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import torch
import torch.nn as nn


class ConceptHeads(nn.Module):
    """K independent sigmoid binary classifiers over DINOv2 pooled features.

    Weights are SET by CONCIL (closed-form ridge regression) — never trained
    by gradient descent.  requires_grad is False on all parameters at all times.
    """

    def __init__(
        self,
        concept_names: list[str],
        input_dim: int = 1536,
    ):
        super().__init__()
        self._concept_names: list[str] = list(concept_names)
        self.input_dim = input_dim

        K = len(concept_names)
        self.linear = nn.Linear(input_dim, K, bias=True)

        # Initialise to zero — CONCIL sets real values before first use
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

        # Freeze permanently
        for p in self.parameters():
            p.requires_grad = False

    # ── properties ────────────────────────────────────────────────────────────

    @property
    def n_concepts(self) -> int:
        return len(self._concept_names)

    @property
    def concept_names(self) -> list[str]:
        return list(self._concept_names)

    # ── forward ───────────────────────────────────────────────────────────────

    @torch.no_grad()
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Map pooled DINOv2 features to concept activations.

        Args:
            z: (B, 1536) — automatically moved to the module's device if needed.
        Returns:
            c: (B, K) values in [0, 1]
        """
        return torch.sigmoid(self.linear(z.to(self.linear.weight.device)))

    # ── weight management (CONCIL interface) ──────────────────────────────────

    def set_weights(
        self,
        W: Union[np.ndarray, torch.Tensor],
        b: Union[np.ndarray, torch.Tensor],
    ) -> None:
        """Write CONCIL solution into the linear layer.

        Args:
            W: (K, input_dim) or (input_dim, K) weight matrix.
               Accepted shapes: (K, D) — standard nn.Linear convention.
               If (D, K) is passed (CONCIL's column-major output), it is
               transposed automatically.
            b: (K,) bias vector.
        """
        if isinstance(W, np.ndarray):
            W = torch.from_numpy(W).float()
        if isinstance(b, np.ndarray):
            b = torch.from_numpy(b).float()

        # CONCIL solves W as (D, K); nn.Linear stores weight as (K, D)
        if W.shape == (self.input_dim, self.n_concepts):
            W = W.T                                   # (K, D)
        if W.shape != (self.n_concepts, self.input_dim):
            raise ValueError(
                f"Expected W shape ({self.n_concepts}, {self.input_dim}) "
                f"or ({self.input_dim}, {self.n_concepts}), got {tuple(W.shape)}"
            )

        self.linear.weight.data.copy_(W)
        self.linear.bias.data.copy_(b.reshape(-1))

    def add_concepts(self, new_names: list[str]) -> list[int]:
        """Expand the head set with new zero-initialised concepts.

        Old weights are preserved exactly.  New rows are zeroed (CONCIL fills
        them on the next update).  requires_grad stays False after expansion.

        Args:
            new_names: list of concept name strings to append.
        Returns:
            List of integer indices for the newly added concepts.
        """
        if not new_names:
            return []

        K_old = self.n_concepts
        K_new = K_old + len(new_names)

        # Build expanded linear layer (old weights preserved, new rows = 0)
        new_linear = nn.Linear(self.input_dim, K_new, bias=True)
        nn.init.zeros_(new_linear.weight)
        nn.init.zeros_(new_linear.bias)

        with torch.no_grad():
            new_linear.weight[:K_old].copy_(self.linear.weight)
            new_linear.bias[:K_old].copy_(self.linear.bias)

        # Freeze new layer
        for p in new_linear.parameters():
            p.requires_grad = False

        self.linear = new_linear
        self._concept_names.extend(new_names)

        return list(range(K_old, K_new))

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "concept_names": self._concept_names,
                "input_dim": self.input_dim,
                "state_dict": self.linear.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "ConceptHeads":
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        obj = cls(ckpt["concept_names"], input_dim=ckpt["input_dim"])
        obj.linear.load_state_dict(ckpt["state_dict"])
        for p in obj.parameters():
            p.requires_grad = False
        return obj

    def train(self, mode: bool = True) -> "ConceptHeads":
        """Override: always stays in eval mode — no training loop allowed."""
        return super().train(False)


# ── __main__ smoke test (also exercises LinearAnomalyHead) ───────────────────

if __name__ == "__main__":
    import json
    import sys
    from pathlib import Path

    import pandas as pd
    from PIL import Image

    from features.dinov2_extractor import DINOv2Extractor
    from models.linear_head import LinearAnomalyHead

    print("=" * 62)
    print("ConceptHeads + LinearAnomalyHead — smoke test")
    print("=" * 62)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # ── 1. load concept names from tier map ───────────────────────────────────
    tier_path = Path("annotations/hazelnut/cl_tasks/concept_tier_map.json")
    if not tier_path.exists():
        sys.exit("Run task_csv_builder first, then run from project root.")

    with open(tier_path) as f:
        tier_map = json.load(f)

    all_concepts: list[str] = []
    for key in ["tier1_normal", "tier2_generic"] + [
        k for k in tier_map if k.startswith("tier3_")
    ]:
        all_concepts.extend(tier_map.get(key, []))

    # deduplicate preserving order
    seen: set[str] = set()
    concept_names = [c for c in all_concepts if not (c in seen or seen.add(c))]
    print(f"Concept names loaded: {len(concept_names)} unique concepts")

    # ── 2. instantiate both models ────────────────────────────────────────────
    heads = ConceptHeads(concept_names, input_dim=1536)
    anomaly_head = LinearAnomalyHead(n_concepts=len(concept_names))

    print(f"ConceptHeads    : K={heads.n_concepts}, input_dim={heads.input_dim}")
    print(f"LinearAnomalyHead: K={anomaly_head.n_concepts}")
    print()

    # ── 3. extract pooled_z for 2 normal + 2 crack images ────────────────────
    csv_path = Path("annotations/hazelnut/hazelnut.csv")
    df = pd.read_csv(csv_path)

    normal_paths = df[df["anomaly_type"] == "good"]["image_path"].tolist()[:2]
    crack_paths  = df[df["anomaly_type"] == "crack"]["image_path"].tolist()[:2]

    print("Loading DINOv2 extractor...")
    extractor = DINOv2Extractor(device=device)

    imgs = [Image.open(p).convert("RGB") for p in normal_paths + crack_paths]
    pooled_z = extractor.extract_pooled(imgs)       # (4, 1536)
    print(f"Extracted pooled_z shape: {tuple(pooled_z.shape)}\n")

    # ── 4. set DUMMY weights and run forward ──────────────────────────────────
    rng = np.random.default_rng(42)
    K, D = heads.n_concepts, heads.input_dim

    W_dummy = rng.standard_normal((K, D)).astype(np.float32)
    b_dummy = np.zeros(K, dtype=np.float32)
    heads.set_weights(W_dummy, b_dummy)

    w_anom = np.ones(K, dtype=np.float32) / K
    b_anom = np.zeros(1, dtype=np.float32)
    anomaly_head.set_weights(w_anom, b_anom)

    c = heads(pooled_z.to(device))                  # (4, K)
    y = anomaly_head(c)                             # (4,)

    print("── Forward pass ─────────────────────────────────")
    print(f"concept_activations shape : {tuple(c.shape)}")
    print(f"anomaly_scores shape      : {tuple(y.shape)}")
    print(f"anomaly_scores values     : {y.cpu().numpy().round(4).tolist()}")
    print(f"c range                   : [{c.min().item():.4f}, {c.max().item():.4f}]")
    print()

    # ── 5. add_concepts expansion ─────────────────────────────────────────────
    print("── add_concepts test ────────────────────────────")
    K_before = heads.n_concepts
    W_old = heads.linear.weight.data[:K_before].clone()

    new_idx = heads.add_concepts(["test_concept_a", "test_concept_b"])
    K_after = heads.n_concepts

    print(f"n_concepts before → after : {K_before} → {K_after}")
    print(f"New concept indices       : {new_idx}")

    # old weights must be bit-exact
    assert torch.equal(heads.linear.weight.data[:K_before], W_old), \
        "Old weights changed after add_concepts!"
    print("Old weights preserved     : ✓")

    # new rows must be zero
    assert torch.equal(
        heads.linear.weight.data[K_before:],
        torch.zeros(2, heads.input_dim),
    ), "New concept weights not zero-initialised!"
    print("New rows zero-initialised : ✓")

    # ── 6. LinearAnomalyHead expand ───────────────────────────────────────────
    print()
    print("── LinearAnomalyHead.expand test ────────────────")
    w_old = anomaly_head.linear.weight.data.clone()
    anomaly_head.expand(K_after)
    print(f"n_concepts before → after : {K_before} → {anomaly_head.n_concepts}")
    assert torch.equal(anomaly_head.linear.weight.data[:, :K_before], w_old), \
        "Old anomaly weights changed after expand!"
    print("Old weights preserved     : ✓")
    assert torch.equal(
        anomaly_head.linear.weight.data[:, K_before:],
        torch.zeros(1, K_after - K_before),
    ), "New anomaly weights not zero!"
    print("New weights zero-init     : ✓")

    # ── 7. requires_grad check after expansion ────────────────────────────────
    print()
    print("── requires_grad check ──────────────────────────")
    heads_frozen = all(not p.requires_grad for p in heads.parameters())
    anom_frozen  = all(not p.requires_grad for p in anomaly_head.parameters())
    print(f"ConceptHeads frozen after add_concepts : {heads_frozen}")
    print(f"LinearAnomalyHead frozen after expand  : {anom_frozen}")
    assert heads_frozen, "ConceptHeads has requires_grad=True after expansion!"
    assert anom_frozen,  "LinearAnomalyHead has requires_grad=True after expand!"

    # ── 8. save / load round-trip ─────────────────────────────────────────────
    print()
    print("── save/load round-trip ─────────────────────────")
    tmp_heads = Path("/tmp/concept_heads_test.pt")
    tmp_anom  = Path("/tmp/linear_head_test.pt")

    heads.save(tmp_heads)
    restored_heads = ConceptHeads.load(tmp_heads)
    assert restored_heads.n_concepts == K_after
    assert torch.equal(
        restored_heads.linear.weight.data, heads.linear.weight.data
    ), "ConceptHeads weight mismatch after load!"
    assert all(not p.requires_grad for p in restored_heads.parameters()), \
        "Loaded ConceptHeads has requires_grad=True!"
    print("ConceptHeads save/load    : ✓")

    anomaly_head.save(tmp_anom)
    restored_anom = LinearAnomalyHead.load(tmp_anom)
    assert restored_anom.n_concepts == K_after
    assert torch.equal(
        restored_anom.linear.weight.data, anomaly_head.linear.weight.data
    ), "LinearAnomalyHead weight mismatch after load!"
    assert all(not p.requires_grad for p in restored_anom.parameters()), \
        "Loaded LinearAnomalyHead has requires_grad=True!"
    print("LinearAnomalyHead save/load: ✓")

    print()
    print("All assertions passed.")
