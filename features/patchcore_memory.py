"""PatchCore memory bank — fixed after Task 1, never updated.

At Task 1: build() stores a random coreset of L2-normalised patch vectors
           extracted from all normal training images by DINOv2Extractor.
At inference: score() computes per-patch nearest-neighbour distances to the
           memory bank and aggregates them to an image-level anomaly score
           (s_novel) and a 16×16 spatial anomaly map.

Distance formula (after L2 normalisation):
    ||a - b||² = 2 - 2·(a·bᵀ)
Reduces to a single matrix multiply — no faiss required.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import torch
import torch.nn.functional as F

_TOP_K_FRACTION = 0.01   # top-1% of 256 patch distances → image score (k=2)
_SPATIAL_SIDE   = 16     # 224 / patch_size_14 = 16 patches per side
_NN_CHUNK       = 2048   # rows per distance-matrix chunk (avoids OOM on large batches)


class PatchCoreMemory:
    """Fixed nearest-neighbour memory bank for anomaly scoring.

    Lifecycle:
        memory = PatchCoreMemory(coreset_size=10_000)
        memory.build(all_normal_patch_tokens)   # Task 1 only
        scores, maps = memory.score(test_patch_tokens)   # any time

    After build() the memory tensor is frozen; score() is read-only.
    """

    def __init__(
        self,
        coreset_size: int = 10_000,
        device: torch.device | None = None,
    ):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self._coreset_size = coreset_size

        self.memory: torch.Tensor | None = None   # (N_stored, 768) after build
        self.is_built: bool = False
        self._n_source_images: int = 0

    # ── properties ────────────────────────────────────────────────────────────

    @property
    def coreset_size(self) -> int:
        return self.memory.shape[0] if self.is_built else 0

    @property
    def memory_mb(self) -> float:
        if not self.is_built:
            return 0.0
        return self.memory.numel() * self.memory.element_size() / (1024 ** 2)

    # ── build ─────────────────────────────────────────────────────────────────

    def build(
        self,
        patch_tokens: Union[torch.Tensor, list[torch.Tensor]],
    ) -> None:
        """Build and freeze the memory bank from normal training patch tokens.

        Args:
            patch_tokens: (B, 256, 768) tensor or list of (b, 256, 768) tensors
                          collected from ALL normal training images at Task 1.
        Constraint: raises RuntimeError if called a second time.
        """
        if self.is_built:
            raise RuntimeError(
                "Memory bank already built. It is frozen for the rest of Scenario B."
            )

        # ── collect + flatten to (N_total_patches, 768) ───────────────────────
        if isinstance(patch_tokens, list):
            parts = [t.reshape(-1, t.shape[-1]) for t in patch_tokens]
            all_patches = torch.cat(parts, dim=0)
            self._n_source_images = sum(t.shape[0] for t in patch_tokens)
        else:
            self._n_source_images = patch_tokens.shape[0]
            all_patches = patch_tokens.reshape(-1, patch_tokens.shape[-1])

        all_patches = all_patches.to(self.device)        # move once

        # ── L2 normalise ──────────────────────────────────────────────────────
        all_patches = F.normalize(all_patches, p=2, dim=1)

        # ── random coreset subsampling ────────────────────────────────────────
        N_total = all_patches.shape[0]
        k = min(N_total, self._coreset_size)
        idx = torch.randperm(N_total, device=self.device)[:k]
        self.memory = all_patches[idx].contiguous()      # (k, 768)

        self.is_built = True

    # ── score ─────────────────────────────────────────────────────────────────

    def score(
        self,
        patch_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute anomaly scores for a batch of images.

        Args:
            patch_tokens: (B, N_patches, 768) from DINOv2Extractor.

        Returns:
            image_scores: (B,)      — s_novel (mean of top-1% patch distances)
            anomaly_maps: (B,16,16) — spatial patch-distance grid
        """
        if not self.is_built:
            raise RuntimeError("Call build() before score().")

        B, N_patches, D = patch_tokens.shape
        x = patch_tokens.to(self.device).reshape(B * N_patches, D)

        # ── L2 normalise test patches ─────────────────────────────────────────
        x = F.normalize(x, p=2, dim=1)                   # (B*N, 768)

        # ── nearest-neighbour distances (chunked to avoid OOM) ────────────────
        # After L2 norm: ||a-b||² = 2 - 2·(a·bᵀ)
        nn_parts: list[torch.Tensor] = []
        for start in range(0, x.shape[0], _NN_CHUNK):
            chunk = x[start : start + _NN_CHUNK]          # (c, 768)
            sim   = chunk @ self.memory.T                  # (c, N_stored)
            dist  = 2.0 - 2.0 * sim                       # (c, N_stored)
            nn_parts.append(dist.min(dim=1).values)        # (c,)
        nn_dist = torch.cat(nn_parts, dim=0)               # (B*N,)

        patch_distances = nn_dist.reshape(B, N_patches)    # (B, 256)

        # ── image score: mean of top-1% most anomalous patches ────────────────
        k_top = max(1, int(_TOP_K_FRACTION * N_patches))
        top_vals, _ = patch_distances.topk(k_top, dim=1)  # (B, k_top)
        image_scores = top_vals.mean(dim=1)                # (B,)

        # ── spatial anomaly map: (B, 16, 16) ─────────────────────────────────
        anomaly_maps = patch_distances.reshape(
            B, _SPATIAL_SIDE, _SPATIAL_SIDE
        )

        return image_scores, anomaly_maps

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Persist the memory bank to a .pt file."""
        if not self.is_built:
            raise RuntimeError("Cannot save: memory bank not built.")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "memory": self.memory.cpu(),
                "coreset_size": self.coreset_size,
                "n_source_images": self._n_source_images,
            },
            path,
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        device: torch.device | None = None,
    ) -> "PatchCoreMemory":
        """Restore a saved memory bank. Returns a built, frozen instance."""
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        obj = cls(coreset_size=ckpt["coreset_size"], device=device)
        obj.memory = ckpt["memory"].to(obj.device)
        obj._n_source_images = ckpt["n_source_images"]
        obj.is_built = True
        return obj


# ── __main__ smoke test ───────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from pathlib import Path

    import pandas as pd
    from PIL import Image

    from features.dinov2_extractor import DINOv2Extractor

    print("=" * 60)
    print("PatchCoreMemory — smoke test")
    print("=" * 60)

    csv_path = Path("annotations/hazelnut/hazelnut.csv")
    if not csv_path.exists():
        sys.exit("Run from project root: python -m features.patchcore_memory")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    df = pd.read_csv(csv_path)

    # ── extract patch tokens from all normal training images ──────────────────
    print("Loading DINOv2 extractor...")
    extractor = DINOv2Extractor(device=device)

    normal_paths = df[df["anomaly_type"] == "good"]["image_path"].tolist()
    print(f"Normal images  : {len(normal_paths)}")
    print(
        f"Expected patches: {len(normal_paths)} × 256 = "
        f"{len(normal_paths) * 256:,} total\n"
    )

    print("Extracting patch tokens from normal images...")
    all_patches: list[torch.Tensor] = []
    batch_size = 16
    n_batches = (len(normal_paths) + batch_size - 1) // batch_size

    for i in range(0, len(normal_paths), batch_size):
        batch = [Image.open(p).convert("RGB") for p in normal_paths[i : i + batch_size]]
        patches = extractor.extract_patch_tokens(batch).cpu()   # (b, 256, 768)
        all_patches.append(patches)
        b_idx = i // batch_size + 1
        if b_idx % 5 == 0 or b_idx == n_batches:
            print(f"  batch {b_idx:>3}/{n_batches}")

    all_patches_tensor = torch.cat(all_patches, dim=0)          # (431, 256, 768)
    print(f"\nCollected shape : {tuple(all_patches_tensor.shape)}")

    # ── build ─────────────────────────────────────────────────────────────────
    print("\nBuilding memory bank...")
    memory = PatchCoreMemory(coreset_size=10_000, device=device)
    memory.build(all_patches_tensor)
    print(
        f"Memory bank built: {memory.coreset_size:,} patches stored "
        f"({memory.memory_mb:.1f} MB on {device})"
    )

    # ── score test images ─────────────────────────────────────────────────────
    print("\nScoring test images:")

    def _score_one(path: str, label: str) -> float:
        img = Image.open(path).convert("RGB")
        patches = extractor.extract_patch_tokens(img)           # (1, 256, 768)
        s, maps = memory.score(patches)
        val = s[0].item()
        print(f"  {label:<22} s_novel = {val:.6f}   map shape: {tuple(maps.shape)}")
        return val

    # Use a good image from the test split (not seen during memory build)
    # and one from each defect type
    normal_test = df[df["anomaly_type"] == "good"].iloc[-1]["image_path"]
    crack_path  = df[df["anomaly_type"] == "crack"].iloc[0]["image_path"]
    hole_path   = df[df["anomaly_type"] == "hole"].iloc[0]["image_path"]

    s_normal = _score_one(normal_test, "normal (good)")
    s_crack  = _score_one(crack_path,  "defect (crack)")
    s_hole   = _score_one(hole_path,   "defect (hole)")

    # ── assertions ────────────────────────────────────────────────────────────
    print()
    assert s_crack > s_normal, (
        f"FAIL: crack score ({s_crack:.6f}) should exceed normal ({s_normal:.6f})"
    )
    assert s_hole > s_normal, (
        f"FAIL: hole score ({s_hole:.6f}) should exceed normal ({s_normal:.6f})"
    )
    print("Assertion: s_novel_crack  > s_novel_normal  ✓")
    print("Assertion: s_novel_hole   > s_novel_normal  ✓")

    # ── save / load round-trip ────────────────────────────────────────────────
    print("\nTesting save/load round-trip...")
    tmp = Path("/tmp/patchcore_hazelnut_test.pt")
    memory.save(tmp)
    restored = PatchCoreMemory.load(tmp, device=device)

    img_crack = Image.open(crack_path).convert("RGB")
    s2, _ = restored.score(extractor.extract_patch_tokens(img_crack))
    delta = abs(s2[0].item() - s_crack)
    assert delta < 1e-5, f"Save/load mismatch: delta = {delta}"
    print(f"Save/load round-trip  ✓  (delta = {delta:.2e})")
    print(f"\nFile size: {tmp.stat().st_size / 1024:.1f} KB")
