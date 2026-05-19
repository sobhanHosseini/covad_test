"""
System 3 — Stage 1: Concept Bottleneck Model (CBM) on SAE sparse codes.

Decision path (true CBM — no shortcut through z):
    patch_token (1024) → SAE.encode() → z (4096) → ConceptHeads → h (K) → score

Steps:
  1. Data-driven atom clustering: compute per-atom discrimination scores from
     pixel masks, K-means on cross-category discrimination profiles → K=20 concepts
  2. Pixel-mask patch labelling (atom-firing × mask overlap)
  3. Train K binary concept heads h_k : R^4096 → [0,1]
  4. Learn CONCIL guide vectors c+, c- in K-dim concept space
  5. Evaluate I-AUC per category; compare against System 2 baseline (0.971 avg)

Run from project root:
    python scripts/09_concept_heads.py
"""

from __future__ import annotations

import json
import sys
import warnings
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from features.dinov2_extractor import DINOv2Extractor
from features.sae import SparseAutoencoder

# ── Config ────────────────────────────────────────────────────────────────────

K_CONCEPTS          = 20
TOP_M_ATOMS         = 2000    # top discriminative atoms fed into K-means
SAE_PATH            = Path("sae_training/sae_vitl14reg_C4096_k64.pt")
ATOM_NAMES_PATH     = Path("sae_training/atom_names.json")
TOKENS_PATH         = Path("sae_training/mvtec_normal_patches_vitl14reg.pt")
INDEX_PATH          = Path("sae_training/mvtec_patch_index_reg.pt")
MVTEC_ROOT          = Path("/home/sobhan_hosseini/datasets/mvtec")
ANN_ROOT            = Path("annotations")
CONCEPT_HEADS_DIR   = Path("sae_training/concept_heads")
CONCEPT_MAP_PATH    = Path("sae_training/concept_mapping.json")
RESULTS_PATH        = Path("results/system3_stage1_results.json")

# CATEGORIES          = ["bottle", "capsule", "hazelnut", "metal_nut", "screw"]
CATEGORIES          = ["hazelnut"]
DEVICE              = torch.device("cuda:0")
MODEL_NAME          = "dinov2_vitl14_reg"
BATCH_SIZE          = 64      # images per DINOv2 batch
HEAD_BATCH_SIZE     = 512     # patches per concept-head training batch
TRAIN_EPOCHS        = 20
LR                  = 1e-3
LAMBDA_REG          = 1.0
DEFECT_TRAIN_RATIO  = 0.80
SEED                = 42

IMG_SIZE        = 224
PATCH_SIZE      = 14
N_SIDE          = IMG_SIZE // PATCH_SIZE   # 16
N_PATCHES       = N_SIDE ** 2             # 256
OVERLAP_THR     = 0.1                      # fraction of patch covered by mask

# ── ConceptHeads ──────────────────────────────────────────────────────────────

class ConceptHeads(nn.Module):
    """K binary linear concept heads: z (4096) → h (K) via sigmoid."""

    def __init__(self, n_atoms: int = 4096, n_concepts: int = 20):
        super().__init__()
        self.n_atoms    = n_atoms
        self.n_concepts = n_concepts
        self.W = nn.Parameter(torch.empty(n_concepts, n_atoms))
        self.b = nn.Parameter(torch.zeros(n_concepts))
        nn.init.xavier_uniform_(self.W)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, n_atoms) → h: (B, n_concepts) ∈ [0, 1]"""
        return torch.sigmoid(z @ self.W.T + self.b)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "W": self.W.detach().cpu(),
            "b": self.b.detach().cpu(),
            "n_atoms": self.n_atoms,
            "n_concepts": self.n_concepts,
        }, path)

    @classmethod
    def load(cls, path: str | Path, device="cpu") -> "ConceptHeads":
        ckpt = torch.load(path, map_location=device, weights_only=True)
        obj  = cls(ckpt["n_atoms"], ckpt["n_concepts"])
        obj.W.data = ckpt["W"].to(device)
        obj.b.data = ckpt["b"].to(device)
        return obj


# ── ConceptGuideTrainer ────────────────────────────────────────────────────────

class ConceptGuideTrainer:
    """Ridge-regression guide vectors in K-dim concept activation space."""

    def __init__(self, K: int, lambda_reg: float = 1.0, device="cuda:0"):
        self.K          = K
        self.lambda_reg = float(lambda_reg)
        self.device     = torch.device(device)
        self.A_neg: Optional[torch.Tensor] = None
        self.b_neg: Optional[torch.Tensor] = None
        self.A_pos: Optional[torch.Tensor] = None
        self.b_pos: Optional[torch.Tensor] = None
        self.c_neg: Optional[torch.Tensor] = None
        self.c_pos: Optional[torch.Tensor] = None
        self.n_pos_tasks = 0

    def _solve(self, A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        reg = A + self.lambda_reg * torch.eye(self.K, dtype=torch.float64)
        return torch.linalg.solve(reg, b)

    @torch.no_grad()
    def build_normal_guide(self, h_normal: torch.Tensor) -> None:
        h = h_normal.double().cpu()
        self.A_neg = h.T @ h
        self.b_neg = h.sum(0)
        c    = self._solve(self.A_neg, self.b_neg).float()
        norm = c.norm().clamp(min=1e-8)
        self.c_neg = (c / norm).to(self.device)
        print(f"    c⁻ built  (raw norm = {norm.item():.4f})")

    @torch.no_grad()
    def update_anomaly_guide(self, h_anomaly: torch.Tensor) -> None:
        h = h_anomaly.double().cpu()
        if self.A_pos is None:
            self.A_pos = torch.zeros(self.K, self.K, dtype=torch.float64)
            self.b_pos = torch.zeros(self.K, dtype=torch.float64)
        self.A_pos.add_(h.T @ h)
        self.b_pos.add_(h.sum(0))
        c    = self._solve(self.A_pos, self.b_pos).float()
        norm = c.norm().clamp(min=1e-8)
        self.c_pos = (c / norm).to(self.device)
        self.n_pos_tasks += 1

    @torch.no_grad()
    def score_patches(self, h: torch.Tensor) -> torch.Tensor:
        h = h.to(self.device)
        return (h @ self.c_pos - h @ self.c_neg).cpu()

    def score_image(self, h: torch.Tensor) -> float:
        return float(self.score_patches(h).max().item())

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "K": self.K, "lambda_reg": self.lambda_reg,
            "A_neg": self.A_neg, "b_neg": self.b_neg,
            "A_pos": self.A_pos, "b_pos": self.b_pos,
            "c_neg": self.c_neg.cpu() if self.c_neg is not None else None,
            "c_pos": self.c_pos.cpu() if self.c_pos is not None else None,
            "n_pos_tasks": self.n_pos_tasks,
        }, path)

    @classmethod
    def load(cls, path: str | Path, device="cuda:0") -> "ConceptGuideTrainer":
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        obj  = cls(ckpt["K"], ckpt["lambda_reg"], device=device)
        obj.A_neg      = ckpt["A_neg"]
        obj.b_neg      = ckpt["b_neg"]
        obj.A_pos      = ckpt["A_pos"]
        obj.b_pos      = ckpt["b_pos"]
        obj.n_pos_tasks = ckpt["n_pos_tasks"]
        if ckpt["c_neg"] is not None:
            obj.c_neg = ckpt["c_neg"].to(obj.device)
        if ckpt["c_pos"] is not None:
            obj.c_pos = ckpt["c_pos"].to(obj.device)
        return obj


# ── Step 1: Data-driven atom clustering ───────────────────────────────────────

@torch.no_grad()
def step1_compute_concept_mapping(
    sae:         SparseAutoencoder,
    extractor:   DINOv2Extractor,
    all_tokens:  torch.Tensor,       # (N_total, 1024) pre-extracted normals
    patch_index: list[dict],
    tasks_per_cat: dict[str, list[dict]],   # category → task list
) -> tuple[dict[str, list[int]], list[str], torch.Tensor]:
    """Fully data-driven concept clustering using pixel-mask discrimination.

    Algorithm:
      For each category:
        1. Compute mean SAE code of normal patches:     z_norm_mean  (4096,)
        2. Compute mean SAE code of defect patches      z_def_mean   (4096,)
           (only patches where mask overlap > OVERLAP_THR)
        3. disc_score[cat, atom] = z_def_mean[atom] - z_norm_mean[atom]

      Aggregate: combined_score[atom] = 0.5·mean_disc + 0.5·max_disc over cats

      Select top-M=2000 atoms by combined_score.

      K-means (K=20) on L2-normalised (M, n_cats) discrimination profiles
      → cluster assignment for top-M atoms.

      Name each cluster from the most common top CLIP name among named atoms
      in that cluster.

    Returns:
        concept_mapping : concept_name → sorted list of atom_ids
        concept_names   : ordered list of K concept name strings
        disc_profiles   : (4096, n_cats) float32 — stored for warm-start init
    """
    import pandas as pd

    print("\n" + "=" * 66)
    print("  Step 1 — Data-driven atom discrimination + K-means clustering")
    print("=" * 66)

    n_atoms = sae.config.d_hidden   # 4096
    n_cats  = len(CATEGORIES)

    disc_profiles = torch.zeros(n_atoms, n_cats)   # (4096, 5)

    for cat_idx, category in enumerate(CATEGORIES):
        print(f"\n  [{cat_idx+1}/{n_cats}] {category.upper()}")

        # ── normal mean ───────────────────────────────────────────────────────
        cat_info      = next(e for e in patch_index if e["category"] == category)
        norm_tokens   = all_tokens[cat_info["row_start"]:cat_info["row_end"]]
        z_norm_sum    = torch.zeros(n_atoms)
        n_norm        = 0

        for i in tqdm(range(0, len(norm_tokens), 4096),
                      desc=f"    {category} normal z", leave=False):
            chunk      = norm_tokens[i:i+4096].to(DEVICE)
            z_chunk    = sae.encode(chunk)               # (B, 4096)
            z_norm_sum += z_chunk.sum(0).cpu()
            n_norm     += len(chunk)
        z_norm_mean = z_norm_sum / max(n_norm, 1)        # (4096,)
        print(f"    Normal:  {n_norm:>8,} patches")

        # ── defect mean (masked patches only) ─────────────────────────────────
        z_def_sum = torch.zeros(n_atoms)
        n_def     = 0

        for task in tasks_per_cat[category]:
            df         = pd.read_csv(task["csv_path"])
            defect_df  = df[df["label_index"] == 1].reset_index(drop=True)

            for _, row in defect_df.iterrows():
                img_path  = row["image_path"]
                mask_path = row.get("mask_path", "")

                img = Image.open(img_path).convert("RGB")
                tok = extractor.extract_patch_tokens([img])  # (1, 256, 1024)
                z   = sae.encode(tok.reshape(N_PATCHES, -1).to(DEVICE))  # (256, 4096)

                if (mask_path and isinstance(mask_path, str)
                        and mask_path.strip() and Path(mask_path).exists()):
                    flags    = mask_to_patch_flags(mask_path)        # (256,)
                    def_mask = torch.tensor(flags, dtype=torch.bool)
                else:
                    def_mask = torch.ones(N_PATCHES, dtype=torch.bool)

                if def_mask.any():
                    z_def_sum += z[def_mask].sum(0).cpu()
                    n_def     += int(def_mask.sum())

        z_def_mean = z_def_sum / max(n_def, 1)           # (4096,)
        print(f"    Defect:  {n_def:>8,} masked patches")

        disc_profiles[:, cat_idx] = z_def_mean - z_norm_mean

    # ── Rank atoms ────────────────────────────────────────────────────────────
    mean_disc     = disc_profiles.mean(dim=1)           # (4096,)
    max_disc      = disc_profiles.max(dim=1).values     # (4096,)
    combined      = 0.5 * mean_disc + 0.5 * max_disc    # (4096,)

    top_atom_ids  = combined.argsort(descending=True)[:TOP_M_ATOMS].tolist()
    print(f"\n  Top-{TOP_M_ATOMS} atoms selected  "
          f"(min disc score = {combined[top_atom_ids[-1]]:.4f}, "
          f"max = {combined[top_atom_ids[0]]:.4f})")

    # ── K-means on L2-normalised discrimination profiles ─────────────────────
    top_profiles = disc_profiles[top_atom_ids].numpy()   # (M, 5)
    norms        = np.linalg.norm(top_profiles, axis=1, keepdims=True) + 1e-8
    top_normed   = top_profiles / norms

    print(f"  K-means (K={K_CONCEPTS}) on {TOP_M_ATOMS}×{n_cats} profiles …")
    kmeans = KMeans(n_clusters=K_CONCEPTS, random_state=SEED, n_init=20, max_iter=500)
    cluster_labels = kmeans.fit_predict(top_normed)     # (M,)

    cluster_to_atoms: dict[int, list[int]] = {k: [] for k in range(K_CONCEPTS)}
    for i, atom_id in enumerate(top_atom_ids):
        cluster_to_atoms[cluster_labels[i]].append(atom_id)

    # ── Auto-name clusters from atom_names.json ───────────────────────────────
    with open(ATOM_NAMES_PATH) as f:
        atom_data = json.load(f)

    concept_names: list[str]         = []
    concept_mapping: dict[str, list] = {}
    seen_names: set[str]             = set()

    for k in range(K_CONCEPTS):
        atoms_in_cluster = cluster_to_atoms[k]
        top_names_in_cluster = [
            atom_data[str(a)]["names"][0]
            for a in atoms_in_cluster
            if str(a) in atom_data
        ]

        if top_names_in_cluster:
            cnt       = Counter(top_names_in_cluster)
            best_name = cnt.most_common(1)[0][0]
            # Convert to a clean identifier
            clean = best_name.lower()
            for ch in " -/()":
                clean = clean.replace(ch, "_")
            clean = "".join(c for c in clean if c.isalnum() or c == "_")[:28]
        else:
            clean = f"concept_{k:02d}"

        # Ensure uniqueness
        base  = clean
        suffix = 0
        while clean in seen_names:
            suffix += 1
            clean = f"{base}_{suffix}"
        seen_names.add(clean)

        concept_names.append(clean)
        concept_mapping[clean] = sorted(atoms_in_cluster)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n  {'Concept':<32} {'Atoms':>6}  {'Top name'}")
    print("  " + "─" * 66)
    for k, cn in enumerate(concept_names):
        aids  = concept_mapping[cn]
        sample_names = [atom_data[str(a)]["names"][0]
                        for a in aids[:5] if str(a) in atom_data]
        top  = sample_names[0] if sample_names else "(unnamed)"
        print(f"  {cn:<32} {len(aids):>6}  {top}")
    print("  " + "─" * 66)
    print(f"  {'TOTAL':<32} {sum(len(v) for v in concept_mapping.values()):>6}")
    print(f"  (remaining {4096-TOP_M_ATOMS} atoms excluded from clusters; "
          f"W still learns them via BCE)")

    # Save
    CONCEPT_MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CONCEPT_MAP_PATH, "w") as f:
        json.dump(concept_mapping, f, indent=2)
    print(f"\n  concept_mapping.json → {CONCEPT_MAP_PATH}")

    return concept_mapping, concept_names, disc_profiles


# ── Mask helpers ──────────────────────────────────────────────────────────────

def mask_to_patch_flags(mask_path: str | Path) -> np.ndarray:
    """Load PNG mask → (N_PATCHES,) float32 binary.

    Patch flagged as 1 when >OVERLAP_THR fraction of its 14×14 pixels are masked.
    """
    mask_img = Image.open(mask_path).convert("L").resize(
        (IMG_SIZE, IMG_SIZE), Image.NEAREST
    )
    mask_np = (np.array(mask_img) > 0)   # (224, 224)

    flags = np.zeros(N_PATCHES, dtype=np.float32)
    for i in range(N_SIDE):
        for j in range(N_SIDE):
            patch = mask_np[i*PATCH_SIZE:(i+1)*PATCH_SIZE,
                            j*PATCH_SIZE:(j+1)*PATCH_SIZE]
            flags[i * N_SIDE + j] = float(
                patch.sum() / (PATCH_SIZE * PATCH_SIZE) > OVERLAP_THR
            )
    return flags


# ── DINOv2 + SAE helpers ──────────────────────────────────────────────────────

@torch.no_grad()
def z_to_h_batched(
    z: torch.Tensor,
    heads: ConceptHeads,
    batch_size: int = 4096,
) -> torch.Tensor:
    """Run frozen ConceptHeads on z in batches. Returns (N, K) float32 CPU."""
    heads.eval()
    out = []
    for i in range(0, len(z), batch_size):
        chunk = z[i:i+batch_size].to(DEVICE)
        out.append(heads(chunk).cpu())
    return torch.cat(out, dim=0)


# ── Step 3: Build defect dataset with atom-firing concept labels ──────────────

def patch_concept_labels(
    z_img:           torch.Tensor,       # (256, 4096) CPU
    patch_flags:     np.ndarray,         # (256,) float32
    cluster_atoms:   list[list[int]],    # K lists of atom_ids
) -> torch.Tensor:
    """Compute per-patch, per-concept binary labels.

    label[p, k] = 1  iff  patch p is in the defect mask
                       AND any atom in cluster k is active (z > 0) on patch p.

    Using z > 0 as "active" is exact for TopK-ReLU SAE: non-top-k entries are
    exactly zero, so z > 0 ↔ atom is among the top-k for this patch.
    """
    K    = len(cluster_atoms)
    labs = np.zeros((N_PATCHES, K), dtype=np.float32)
    z_np = z_img.numpy()                 # (256, 4096)

    for k, aids in enumerate(cluster_atoms):
        if not aids:
            continue
        # any atom in cluster fires on each patch
        fires = (z_np[:, aids] > 0).any(axis=1)   # (256,) bool
        labs[:, k] = (fires & (patch_flags > 0)).astype(np.float32)

    return torch.tensor(labs, dtype=torch.float32)


def build_defect_dataset(
    category:      str,
    all_tasks:     list[dict],
    extractor:     DINOv2Extractor,
    sae:           SparseAutoencoder,
    cluster_atoms: list[list[int]],      # K lists of atom_ids
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collect (z, concept_labels) for defect training images of one category.

    Uses the 80 % training split (same as System 2).

    Returns:
        z_defect      : (N_patches, 4096) float32 CPU
        labels_defect : (N_patches, K)    float32 CPU
    """
    import pandas as pd

    z_chunks: list[torch.Tensor] = []
    l_chunks: list[torch.Tensor] = []

    for task in all_tasks:
        defect_type = task["defect"]
        df          = pd.read_csv(task["csv_path"])
        defect_df   = df[df["label_index"] == 1].reset_index(drop=True)
        n_defect    = len(defect_df)
        n_train     = max(1, int(n_defect * DEFECT_TRAIN_RATIO))
        rng         = np.random.RandomState(SEED)
        train_rows  = defect_df.iloc[rng.permutation(n_defect)[:n_train]]

        for _, row in tqdm(train_rows.iterrows(), total=len(train_rows),
                           desc=f"  {defect_type}", leave=False):
            img_path  = row["image_path"]
            mask_path = row.get("mask_path", "")

            img      = Image.open(img_path).convert("RGB")
            tok      = extractor.extract_patch_tokens([img])      # (1, 256, 1024)
            tok_flat = tok.reshape(N_PATCHES, -1).to(DEVICE)
            z_img    = sae.encode(tok_flat).cpu()                 # (256, 4096)

            if (mask_path and isinstance(mask_path, str)
                    and mask_path.strip() and Path(mask_path).exists()):
                patch_flags = mask_to_patch_flags(mask_path)
            else:
                patch_flags = np.ones(N_PATCHES, dtype=np.float32)

            labs = patch_concept_labels(z_img, patch_flags, cluster_atoms)
            z_chunks.append(z_img)
            l_chunks.append(labs)
            del img, tok, tok_flat, z_img

    if not z_chunks:
        K = len(cluster_atoms)
        return torch.empty(0, 4096), torch.empty(0, K)

    return torch.cat(z_chunks, dim=0), torch.cat(l_chunks, dim=0)


def step3_train_concept_heads(
    category:      str,
    sae:           SparseAutoencoder,
    extractor:     DINOv2Extractor,
    normal_tokens: torch.Tensor,         # (N_norm, 1024) CPU
    all_tasks:     list[dict],
    concept_mapping: dict[str, list[int]],
    concept_names: list[str],
    disc_profiles: torch.Tensor,         # (4096, n_cats) for warm-start
) -> tuple[ConceptHeads, dict]:
    """Train K binary concept heads for one category.

    Training data:
      Normal patches : z from normal_tokens, all concept labels = 0
      Defect patches : z from 80 % train split; labels from atom-firing × mask

    Returns: (frozen ConceptHeads on DEVICE, concept_accs dict)
    """
    K = len(concept_names)
    cluster_atoms = [concept_mapping[cn] for cn in concept_names]

    print(f"\n  Step 3 — Training concept heads for {category.upper()}")

    # ── 1. Encode normal tokens → z ──────────────────────────────────────────
    print(f"    Encoding {len(normal_tokens):,} normal tokens through SAE …")
    sae.eval()
    z_norm_chunks: list[torch.Tensor] = []
    for i in tqdm(range(0, len(normal_tokens), 4096),
                  desc="    normal z", leave=False):
        chunk = normal_tokens[i:i+4096].to(DEVICE)
        z_norm_chunks.append(sae.encode(chunk).cpu())
    z_normal = torch.cat(z_norm_chunks, dim=0)       # (N_norm, 4096)
    del z_norm_chunks

    # Subsample normals to cap imbalance (≤50 k patches)
    n_cap  = min(len(z_normal), 50_000)
    gen    = torch.Generator().manual_seed(SEED)
    perm   = torch.randperm(len(z_normal), generator=gen)
    z_normal = z_normal[perm[:n_cap]]
    l_normal = torch.zeros(len(z_normal), K, dtype=torch.float32)
    print(f"    Normal patches: {len(z_normal):,} (subsampled)")

    # ── 2. Collect defect patches with atom-firing concept labels ─────────────
    print(f"    Collecting defect patches from {len(all_tasks)} task(s) …")
    z_defect, l_defect = build_defect_dataset(
        category, all_tasks, extractor, sae, cluster_atoms
    )
    n_anom = int((l_defect.max(1).values > 0).sum().item())
    print(f"    Defect patches: {len(z_defect):,}  "
          f"(patches with ≥1 concept label = {n_anom:,})")

    # ── 3. Combine ────────────────────────────────────────────────────────────
    z_all = torch.cat([z_normal, z_defect], dim=0)   # (N, 4096)
    l_all = torch.cat([l_normal, l_defect], dim=0)   # (N, K)
    N     = len(z_all)

    # ── 4. Warm-start: initialise W_k from mean disc score of cluster atoms ──
    heads = ConceptHeads(n_atoms=sae.config.d_hidden, n_concepts=K)
    # Category-averaged discrimination score gives the sign & magnitude of
    # each atom's contribution to anomaly vs normal.
    mean_disc = disc_profiles.mean(dim=1)             # (4096,)
    with torch.no_grad():
        nn.init.zeros_(heads.W)
        for k, aids in enumerate(cluster_atoms):
            if aids:
                weights = mean_disc[aids].clamp(min=0)  # only positive contribution
                total   = weights.sum().item()
                if total > 1e-8:
                    heads.W.data[k, aids] = (weights / total).float()
    heads = heads.to(DEVICE)

    # ── 5. Compute class weight per concept ───────────────────────────────────
    # Use per-concept positive rate to set pos_weight
    pos_counts = l_all.sum(0).clamp(min=1)           # (K,)
    neg_counts = N - pos_counts
    pos_weight = (neg_counts / pos_counts).clamp(max=50.0).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(heads.parameters(), lr=LR, weight_decay=1e-5)

    perm  = torch.randperm(N, generator=torch.Generator().manual_seed(SEED))
    z_all = z_all[perm]
    l_all = l_all[perm]

    # ── 6. Training loop (all epochs reported) ────────────────────────────────
    print(f"    Training {TRAIN_EPOCHS} epochs, N={N:,} patches …")
    print(f"    {'Epoch':>6}  {'loss':>10}  {'avg_h_defect':>14}  {'avg_h_normal':>13}")
    print("    " + "─" * 48)

    for epoch in range(1, TRAIN_EPOCHS + 1):
        heads.train()
        epoch_loss = 0.0
        n_batches  = 0
        for i in range(0, N, HEAD_BATCH_SIZE):
            zb     = z_all[i:i+HEAD_BATCH_SIZE].to(DEVICE)
            lb     = l_all[i:i+HEAD_BATCH_SIZE].to(DEVICE)
            logits = zb @ heads.W.T + heads.b
            loss   = criterion(logits, lb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches  += 1

        # Quick activations on a sample to monitor learning
        if epoch % 2 == 0 or epoch == 1 or epoch == TRAIN_EPOCHS:
            heads.eval()
            with torch.no_grad():
                n_sample = min(N, 4096)
                z_s  = z_all[:n_sample].to(DEVICE)
                l_s  = l_all[:n_sample]
                h_s  = heads(z_s).cpu()
                def_mask  = l_s.max(1).values > 0
                norm_mask = ~def_mask
                avg_h_def  = h_s[def_mask].mean().item()  if def_mask.any()  else float("nan")
                avg_h_norm = h_s[norm_mask].mean().item() if norm_mask.any() else float("nan")
            print(f"    {epoch:>6}  {epoch_loss/n_batches:>10.4f}  "
                  f"{avg_h_def:>14.4f}  {avg_h_norm:>13.4f}")

    # ── 7. Concept-level precision / recall / F1 on defect patches ────────────
    heads.eval()
    print("\n    Concept-level metrics on defect training patches:")
    print(f"    {'Concept':<32} {'P':>6} {'R':>6} {'F1':>6} {'n_pos':>7}")
    print("    " + "─" * 58)

    concept_accs: dict = {}
    if len(z_defect) > 0:
        with torch.no_grad():
            h_def = z_to_h_batched(z_defect, heads)   # (N_def, K)
        h_bin = (h_def.numpy() > 0.5)
        l_np  = l_defect.numpy()

        for k, cn in enumerate(concept_names):
            n_pos = int(l_np[:, k].sum())
            if n_pos < 2:
                concept_accs[cn] = {"precision": None, "recall": None,
                                    "f1": None, "n_pos": n_pos}
                continue
            tp = int(((h_bin[:, k]) & (l_np[:, k] > 0)).sum())
            fp = int(((h_bin[:, k]) & (l_np[:, k] == 0)).sum())
            fn = int(((~h_bin[:, k]) & (l_np[:, k] > 0)).sum())
            prec = tp / (tp + fp + 1e-8)
            rec  = tp / (tp + fn + 1e-8)
            f1   = 2 * prec * rec / (prec + rec + 1e-8)
            concept_accs[cn] = {"precision": round(prec, 4), "recall": round(rec, 4),
                                 "f1": round(f1, 4), "n_pos": n_pos}
            print(f"    {cn:<32} {prec:>6.3f} {rec:>6.3f} {f1:>6.3f} {n_pos:>7}")

    # Save
    out_path = CONCEPT_HEADS_DIR / f"{category}_heads.pt"
    heads.save(out_path)
    print(f"\n    Concept heads → {out_path}")

    heads.eval()
    for p in heads.parameters():
        p.requires_grad_(False)

    return heads, concept_accs


# ── Steps 4+5: Guide learning + sequential CL evaluation ──────────────────────

def load_defect_split(task_csv_path: str):
    """80/20 defect split — identical to System 2 (script 05)."""
    import pandas as pd
    df        = pd.read_csv(task_csv_path)
    defect_df = df[df["label_index"] == 1].reset_index(drop=True)
    n_defect  = len(defect_df)
    n_train   = max(1, int(n_defect * DEFECT_TRAIN_RATIO))
    rng       = np.random.RandomState(SEED)
    idx       = rng.permutation(n_defect)
    return (defect_df.iloc[idx[:n_train]]["image_path"].tolist(),
            defect_df.iloc[idx[n_train:]]["image_path"].tolist())


def compute_iauc(normal_scores: list[float], defect_scores: list[float]) -> float:
    if not defect_scores:
        return float("nan")
    y_true  = [0] * len(normal_scores) + [1] * len(defect_scores)
    y_score = normal_scores + defect_scores
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(roc_auc_score(y_true, y_score))


def run_category_eval(
    category:      str,
    sae:           SparseAutoencoder,
    heads:         ConceptHeads,
    normal_tokens: torch.Tensor,
    extractor:     DINOv2Extractor,
    concept_names: list[str],
) -> list[dict]:
    """Steps 4+5: guide learning + sequential CL eval (identical protocol to System 2)."""
    K = len(concept_names)
    print(f"\n  Category: {category.upper()}")
    print(f"  {'─' * 62}")

    # ── Build c⁻ from normal patch concept activations ────────────────────────
    print(f"    Encoding {len(normal_tokens):,} normal tokens → z …")
    sae.eval()
    z_norm_list: list[torch.Tensor] = []
    for i in tqdm(range(0, len(normal_tokens), 4096),
                  desc="    normal z", leave=False):
        chunk = normal_tokens[i:i+4096].to(DEVICE)
        z_norm_list.append(sae.encode(chunk).cpu())
    z_normal = torch.cat(z_norm_list, dim=0)
    del z_norm_list

    print(f"    Computing h for {len(z_normal):,} normal patches …")
    h_normal = z_to_h_batched(z_normal, heads)    # (N, K) CPU
    del z_normal

    guide = ConceptGuideTrainer(K=K, lambda_reg=LAMBDA_REG, device=str(DEVICE))
    guide.build_normal_guide(h_normal)
    del h_normal

    # ── Load task sequence ────────────────────────────────────────────────────
    task_seq_path = ANN_ROOT / category / "cl_tasks" / "task_sequence.json"
    with open(task_seq_path) as f:
        tasks = json.load(f)
    print(f"    Tasks: {[t['defect'] for t in tasks]}")

    # ── Pre-extract normal test h vectors ─────────────────────────────────────
    test_good_dir     = MVTEC_ROOT / category / "test" / "good"
    test_normal_paths = sorted(test_good_dir.glob("*.png"))
    print(f"    Pre-extracting {len(test_normal_paths)} normal test images …")

    h_norm_test_list: list[torch.Tensor] = []
    for p in tqdm(test_normal_paths, desc="    normal test", leave=False):
        img = Image.open(p).convert("RGB")
        tok = extractor.extract_patch_tokens([img])
        z   = sae.encode(tok.reshape(N_PATCHES, -1).to(DEVICE))
        h   = heads(z).cpu()
        h_norm_test_list.append(h)
    h_normal_test = torch.stack(h_norm_test_list, dim=0)   # (n_norm, 256, K)
    n_normal_test = len(test_normal_paths)
    del h_norm_test_list

    # ── Sequential task loop ──────────────────────────────────────────────────
    held_h_per_defect: dict[str, torch.Tensor] = {}
    n_held_per_defect: dict[str, int]           = {}
    results: list[dict] = []

    for task in tasks:
        task_id = task["task_id"]
        defect  = task["defect"]
        print(f"\n    Task {task_id}: {defect.upper()}")

        train_paths, held_paths = load_defect_split(task["csv_path"])
        print(f"      Split: {len(train_paths)} train / {len(held_paths)} held")

        # Cache held-out h
        if held_paths:
            h_held_list: list[torch.Tensor] = []
            for p in tqdm(held_paths, desc=f"      held {defect}", leave=False):
                img = Image.open(p).convert("RGB")
                tok = extractor.extract_patch_tokens([img])
                z   = sae.encode(tok.reshape(N_PATCHES, -1).to(DEVICE))
                h_held_list.append(heads(z).cpu())
            held_h_per_defect[defect] = torch.stack(h_held_list, dim=0)  # (n,256,K)
            n_held_per_defect[defect] = len(held_paths)
        else:
            held_h_per_defect[defect] = torch.empty(0, N_PATCHES, K)
            n_held_per_defect[defect] = 0

        # Update c⁺ from train images
        print(f"      Updating c⁺ from {len(train_paths)} defect images …")
        h_train_list: list[torch.Tensor] = []
        for p in tqdm(train_paths, desc=f"      train {defect}", leave=False):
            img = Image.open(p).convert("RGB")
            tok = extractor.extract_patch_tokens([img])
            z   = sae.encode(tok.reshape(N_PATCHES, -1).to(DEVICE))
            h_train_list.append(heads(z).cpu())
        if h_train_list:
            h_train = torch.cat(h_train_list, dim=0)   # (n*256, K)
            guide.update_anomaly_guide(h_train)
        del h_train_list

        # Evaluate all seen defects
        normal_scores = [guide.score_image(h_normal_test[i])
                         for i in range(n_normal_test)]

        for seen_task in tasks[:task_id]:
            seen_defect  = seen_task["defect"]
            seen_task_id = seen_task["task_id"]
            n_held       = n_held_per_defect.get(seen_defect, 0)

            if n_held == 0:
                i_auc = float("nan")
                print(f"        {seen_defect:<16} I-AUC=nan (no held images)")
            else:
                h_held        = held_h_per_defect[seen_defect]
                defect_scores = [guide.score_image(h_held[i]) for i in range(n_held)]
                i_auc         = compute_iauc(normal_scores, defect_scores)
                print(f"        {seen_defect:<16} I-AUC={i_auc:.4f}  "
                      f"({n_normal_test} norm + {n_held} def)")

            results.append({
                "task_id":              seen_task_id,
                "defect":               seen_defect,
                "evaluated_after_task": task_id,
                "i_auc":                i_auc,
                "n_normal_test":        n_normal_test,
                "n_defect_test":        n_held,
            })

        guide_ckpt_dir = Path("sae_training/concept_guides") / category
        guide_ckpt_dir.mkdir(parents=True, exist_ok=True)
        guide.save(guide_ckpt_dir / f"task_{task_id:02d}.pt")

    return results


# ── BWT (identical to System 2) ───────────────────────────────────────────────

def compute_bwt(results: list[dict]) -> tuple[dict, float]:
    _NAN = float("nan")
    if len(results) < 2:
        return {}, _NAN
    T      = max(r["evaluated_after_task"] for r in results)
    lookup = {(r["defect"], r["evaluated_after_task"]): r["i_auc"] for r in results}
    t_first_for: dict[str, int] = {}
    for r in results:
        d = r["defect"]
        if d not in t_first_for or r["task_id"] < t_first_for[d]:
            t_first_for[d] = r["task_id"]
    bwt: dict[str, float] = {}
    for d, tf in t_first_for.items():
        if tf >= T:
            continue
        r0 = lookup.get((d, tf), _NAN)
        rT = lookup.get((d, T),  _NAN)
        bwt[d] = float(rT - r0) if not (np.isnan(r0) or np.isnan(rT)) else _NAN
    valid = [v for v in bwt.values() if not np.isnan(v)]
    return bwt, float(np.mean(valid)) if valid else _NAN


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    for p in [SAE_PATH, TOKENS_PATH, INDEX_PATH, ATOM_NAMES_PATH]:
        if not p.exists():
            print(f"ERROR: required file missing: {p}")
            sys.exit(1)

    print("=" * 66)
    print("  System 3 Stage 1 — SAE-CBM Concept Head Pipeline (data-driven)")
    print("=" * 66)

    # ── Shared resources ──────────────────────────────────────────────────────
    print("\nLoading SAE …")
    sae = SparseAutoencoder.load(str(SAE_PATH), device="cpu")
    sae.to(DEVICE).eval()
    for p in sae.parameters():
        p.requires_grad_(False)
    print(f"  SAE: d_input={sae.config.d_input}, "
          f"d_hidden={sae.config.d_hidden}, k={sae.config.k}")

    print("Loading DINOv2 extractor …")
    extractor = DINOv2Extractor(MODEL_NAME, device=DEVICE)
    extractor.eval()

    print("Loading pre-extracted normal tokens …")
    all_tokens  = torch.load(TOKENS_PATH, map_location="cpu", weights_only=True)
    patch_index = torch.load(INDEX_PATH,  map_location="cpu", weights_only=False)
    print(f"  Token tensor: {all_tokens.shape}  "
          f"({all_tokens.nbytes / 1e9:.2f} GB)")

    # Gather task sequences for Step 1
    tasks_per_cat: dict[str, list[dict]] = {}
    for cat in CATEGORIES:
        task_seq_path = ANN_ROOT / cat / "cl_tasks" / "task_sequence.json"
        with open(task_seq_path) as f:
            tasks_per_cat[cat] = json.load(f)

    # ── Step 1: Data-driven concept mapping (runs once, all categories) ───────
    concept_mapping, concept_names, disc_profiles = step1_compute_concept_mapping(
        sae, extractor, all_tokens, patch_index, tasks_per_cat
    )
    print(f"\n  K={K_CONCEPTS} concept names: {concept_names}")

    # ── Per-category Steps 3–5 ────────────────────────────────────────────────
    CONCEPT_HEADS_DIR.mkdir(parents=True, exist_ok=True)
    Path("results").mkdir(parents=True, exist_ok=True)

    all_results:  dict = {}
    summary_rows: list = []

    for category in CATEGORIES:
        print(f"\n{'='*66}")
        print(f"  CATEGORY: {category.upper()}")
        print(f"{'='*66}")

        cat_info      = next(e for e in patch_index if e["category"] == category)
        normal_tokens = all_tokens[cat_info["row_start"]:cat_info["row_end"]]
        print(f"  Normal tokens: {len(normal_tokens):,} patches "
              f"({cat_info['n_images']} images × {N_PATCHES})")

        tasks = tasks_per_cat[category]

        # Step 3: train concept heads
        heads, concept_accs = step3_train_concept_heads(
            category, sae, extractor, normal_tokens, tasks,
            concept_mapping, concept_names, disc_profiles,
        )

        # Steps 4+5: guide learning + sequential CL eval
        results = run_category_eval(
            category, sae, heads, normal_tokens, extractor, concept_names
        )

        bwt_d, mean_bwt = compute_bwt(results)
        T_cat      = max(r["evaluated_after_task"] for r in results)
        final_aucs = [r["i_auc"] for r in results
                      if r["evaluated_after_task"] == T_cat
                      and not np.isnan(r["i_auc"])]
        mean_auc = float(np.mean(final_aucs)) if final_aucs else float("nan")

        print(f"\n  {category.upper()} — Final I-AUC: {mean_auc:.4f}  "
              f"BWT: {mean_bwt:+.4f}")

        all_results[category] = {
            "tasks":          results,
            "mean_iauc":      mean_auc,
            "bwt_per_defect": {k: (None if np.isnan(v) else v)
                               for k, v in bwt_d.items()},
            "mean_bwt":       None if np.isnan(mean_bwt) else mean_bwt,
            "concept_accs":   concept_accs,
        }
        summary_rows.append({"category": category, "mean_iauc": mean_auc,
                              "mean_bwt": mean_bwt})
        torch.cuda.empty_cache()

    # ── Cross-category summary ────────────────────────────────────────────────
    SYSTEM2_BASELINE = 0.971
    print("\n" + "=" * 66)
    print("  CROSS-CATEGORY SUMMARY")
    print("=" * 66)
    print(f"  {'category':<14} {'I-AUC':>8}  {'BWT':>8}")
    print("  " + "─" * 36)
    aucs = []
    for row in summary_rows:
        auc_s = f"{row['mean_iauc']:.4f}" if not np.isnan(row['mean_iauc']) else "   nan"
        bwt_v = row.get("mean_bwt", float("nan"))
        bwt_s = f"{bwt_v:+.4f}" if not np.isnan(bwt_v) else "    nan"
        print(f"  {row['category']:<14} {auc_s:>8}  {bwt_s:>8}")
        if not np.isnan(row["mean_iauc"]):
            aucs.append(row["mean_iauc"])
    avg_auc = float(np.mean(aucs)) if aucs else float("nan")
    print("  " + "─" * 36)
    print(f"  {'Average':<14} {avg_auc:.4f}")
    print(f"\n  System 2 baseline  : {SYSTEM2_BASELINE:.4f}")
    delta     = avg_auc - SYSTEM2_BASELINE
    direction = "improvement" if delta >= 0 else "degradation"
    print(f"  System 3 Stage 1   : {avg_auc:.4f}  ({delta:+.4f} {direction})")
    print("=" * 66)

    # ── Save results ──────────────────────────────────────────────────────────
    all_results["summary"] = {
        "avg_iauc":      avg_auc,
        "system2_iauc":  SYSTEM2_BASELINE,
        "delta":         avg_auc - SYSTEM2_BASELINE,
        "K_concepts":    K_CONCEPTS,
        "concept_names": concept_names,
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults → {RESULTS_PATH}")


if __name__ == "__main__":
    main()
