"""
scripts/10_prototype_sanity.py

One-day prototype anomaly detection experiment on HAZELNUT only.

Tests whether DINOv2 ViT-L/14-reg4 patch features support prototype-based
anomaly scoring before committing to a full System 3 redesign.

Three methods:
  A — Random prototype bank   (N=256 sampled normals, no learning)
  B — K-means prototype bank  (K=64 centroids, no learning)
  C — CONCIL-weighted sims    (K=64 prototypes + ridge regression on sim space)

Sequential CL protocol: tasks arrive one at a time, 80/20 split.
Evaluation after ALL tasks seen: I-AUC per defect, mean I-AUC, BWT.

Prints diagnostic shapes/values at each step.
Prints final comparison table + go/no-go decision for full System 3.

Run from project root:
    python scripts/10_prototype_sanity.py
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from features.dinov2_extractor import DINOv2Extractor
from solvers.concil import ConcilSolver

# ── Config ────────────────────────────────────────────────────────────────────

CATEGORY            = "hazelnut"
K_PROTO             = 64      # cluster centroids for Methods B + C
N_RANDOM            = 256     # random prototypes for Method A
LAMBDA_ANOMALY      = 1.0     # ridge regularisation for Method C
DEFECT_TRAIN_RATIO  = 0.80
SEED                = 42

MODEL_NAME   = "dinov2_vitl14_reg"
EMBED_DIM    = 1024
N_PATCHES    = 256              # 16×16 patches for 224×224 image with patch_size=14
DEVICE       = torch.device("cuda:0")

MVTEC_ROOT   = Path("/home/sobhan_hosseini/datasets/mvtec")
ANN_ROOT     = Path("annotations")
TOKENS_PATH  = Path("sae_training/mvtec_normal_patches_vitl14reg.pt")
INDEX_PATH   = Path("sae_training/mvtec_patch_index_reg.pt")

SYSTEM2_HAZELNUT_IAUC = 0.995
SYSTEM2_HAZELNUT_BWT  = 0.0    # not published; use 0 as placeholder

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_defect_split(csv_path: str):
    """80/20 defect split — identical to System 2 (seed=42)."""
    import pandas as pd
    df        = pd.read_csv(csv_path)
    defect_df = df[df["label_index"] == 1].reset_index(drop=True)
    n         = len(defect_df)
    n_train   = max(1, int(n * DEFECT_TRAIN_RATIO))
    rng       = np.random.RandomState(SEED)
    idx       = rng.permutation(n)
    return (defect_df.iloc[idx[:n_train]]["image_path"].tolist(),
            defect_df.iloc[idx[n_train:]]["image_path"].tolist())


@torch.no_grad()
def l2_norm(t: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return F.normalize(t, p=2, dim=dim)


@torch.no_grad()
def patches_to_sim_matrix(
    patches:    torch.Tensor,    # (N_patches, D) or (B, N_patches, D)
    prototypes: torch.Tensor,    # (K, D) — already L2-normalised
) -> torch.Tensor:
    """Cosine similarity between every patch and every prototype.

    Returns (N_patches, K) if patches is 2-D,
            (B, N_patches, K) if patches is 3-D.
    """
    p_norm = l2_norm(patches)
    return torch.matmul(p_norm, prototypes.T)    # (…, K)


@torch.no_grad()
def image_feature_c(
    patches:    torch.Tensor,    # (N_patches, D) or (256, D)
    prototypes: torch.Tensor,    # (K, D) L2-normed
) -> torch.Tensor:
    """Method C image feature: max-sim per prototype over all patches → (K,).

    f[k] = max_{p=1..256} cos_sim(patch_p, prototype_k)

    Captures "best coverage" of each prototype by the image.
    """
    sim = patches_to_sim_matrix(patches, prototypes)   # (256, K)
    return sim.max(dim=0).values                        # (K,)


@torch.no_grad()
def image_score_ab(
    patches:    torch.Tensor,    # (N_patches, D)
    prototypes: torch.Tensor,    # (K, D) L2-normed
) -> float:
    """Methods A/B anomaly score: max over patches of (1 – nearest-prototype-sim).

    Higher score → more anomalous.  Same logic as PatchCore with cosine distance.
    """
    sim = patches_to_sim_matrix(patches, prototypes)   # (256, K)
    max_sim_per_patch = sim.max(dim=1).values           # (256,)  best match per patch
    anomaly_per_patch = 1.0 - max_sim_per_patch         # (256,)
    return float(anomaly_per_patch.max().item())


@torch.no_grad()
def extract_patch_tokens(
    img_path: str | Path,
    extractor: DINOv2Extractor,
) -> torch.Tensor:
    """Load one image → (N_patches, D) float32 CPU."""
    img = Image.open(img_path).convert("RGB")
    tok = extractor.extract_patch_tokens([img])   # (1, 256, D)
    return tok.squeeze(0).cpu()                   # (256, D)


@torch.no_grad()
def tokens_to_feature_c(
    tokens_3d:  torch.Tensor,    # (B, 256, D) CPU
    prototypes: torch.Tensor,    # (K, D) L2-normed, CPU
    batch_size: int = 32,
) -> torch.Tensor:
    """Compute Method C feature vectors for a batch of images.

    Processes in mini-batches to stay within GPU memory.
    Returns (B, K) float32 CPU.
    """
    proto_gpu = prototypes.to(DEVICE)
    feats     = []
    for i in range(0, len(tokens_3d), batch_size):
        chunk  = tokens_3d[i:i+batch_size].to(DEVICE)   # (b, 256, D)
        b      = chunk.shape[0]
        flat   = chunk.reshape(b * N_PATCHES, EMBED_DIM) # (b*256, D)
        sim    = patches_to_sim_matrix(flat, proto_gpu)   # (b*256, K)
        sim_3d = sim.reshape(b, N_PATCHES, -1)            # (b, 256, K)
        f      = sim_3d.max(dim=1).values                 # (b, K)
        feats.append(f.cpu())
    return torch.cat(feats, dim=0)                        # (B, K)


def compute_iauc(norm_scores: list[float], def_scores: list[float]) -> float:
    if not def_scores:
        return float("nan")
    y_true  = [0] * len(norm_scores) + [1] * len(def_scores)
    y_score = norm_scores + def_scores
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(roc_auc_score(y_true, y_score))


# ── Step 1: Load resources ────────────────────────────────────────────────────

def load_resources():
    print("=" * 66)
    print("  Prototype Sanity Check — HAZELNUT")
    print("=" * 66)

    print("\n[1] Loading pre-extracted normal tokens …")
    all_tokens  = torch.load(TOKENS_PATH, map_location="cpu", weights_only=True)
    patch_index = torch.load(INDEX_PATH,  map_location="cpu", weights_only=False)
    print(f"    all_tokens : {tuple(all_tokens.shape)}  "
          f"({all_tokens.nbytes / 1e6:.0f} MB)")

    cat_info      = next(e for e in patch_index if e["category"] == CATEGORY)
    normal_tokens = all_tokens[cat_info["row_start"]:cat_info["row_end"]]
    n_norm_images = cat_info["n_images"]
    print(f"    {CATEGORY} normal : {tuple(normal_tokens.shape)}  "
          f"({n_norm_images} images × {N_PATCHES} patches)")

    # Reshape to (n_images, 256, D) for batch feature computation
    normal_tokens_3d = normal_tokens.reshape(n_norm_images, N_PATCHES, EMBED_DIM)
    print(f"    normal_tokens_3d : {tuple(normal_tokens_3d.shape)}")

    print("\n[1] Loading DINOv2 extractor …")
    extractor = DINOv2Extractor(MODEL_NAME, device=DEVICE)
    extractor.eval()
    print(f"    model: {MODEL_NAME}  embed_dim={extractor.EMBED_DIM}")

    print("\n[1] Loading task sequence …")
    task_seq_path = ANN_ROOT / CATEGORY / "cl_tasks" / "task_sequence.json"
    with open(task_seq_path) as f:
        tasks = json.load(f)
    print(f"    Tasks: {[t['defect'] for t in tasks]}")

    return normal_tokens, normal_tokens_3d, n_norm_images, extractor, tasks


# ── Step 2: Build prototypes ──────────────────────────────────────────────────

def build_prototypes(normal_tokens: torch.Tensor):
    """Returns L2-normalised prototypes for Methods A and B."""
    print("\n[2] Building prototype banks …")

    # ── Method A: random sample ───────────────────────────────────────────────
    rng   = np.random.RandomState(SEED)
    idx_a = rng.choice(len(normal_tokens), size=N_RANDOM, replace=False)
    proto_a = l2_norm(normal_tokens[idx_a].float())
    print(f"    Method A: {N_RANDOM} random patches  → {tuple(proto_a.shape)}")
    mean_norm = normal_tokens[idx_a].float().norm(dim=1).mean().item()
    print(f"    Mean raw norm of sampled tokens: {mean_norm:.4f}")

    # ── Method B/C: K-means ───────────────────────────────────────────────────
    # Subsample for K-means speed
    max_km_samples = 80_000
    if len(normal_tokens) > max_km_samples:
        idx_km = rng.choice(len(normal_tokens), size=max_km_samples, replace=False)
        tokens_km = normal_tokens[idx_km].float().numpy()
        print(f"    K-means input: {max_km_samples:,} sampled patches (full={len(normal_tokens):,})")
    else:
        tokens_km = normal_tokens.float().numpy()
        print(f"    K-means input: {len(tokens_km):,} patches")

    print(f"    Running MiniBatchKMeans(K={K_PROTO}) …")
    km = MiniBatchKMeans(n_clusters=K_PROTO, random_state=SEED,
                         batch_size=4096, n_init=5, max_iter=300)
    km.fit(tokens_km)
    centroids   = torch.tensor(km.cluster_centers_, dtype=torch.float32)
    proto_b     = l2_norm(centroids)           # (K, D)
    inertia_str = f"{km.inertia_:.2e}" if km.inertia_ is not None else "n/a"
    print(f"    Method B/C: {K_PROTO} K-means centroids → {tuple(proto_b.shape)}")
    print(f"    K-means inertia: {inertia_str}")

    # Quick sanity: mean max-sim of normal patches to their nearest centroid
    sample_idx = rng.choice(len(normal_tokens), size=2000, replace=False)
    sample     = l2_norm(normal_tokens[sample_idx].float())
    sim_sample = (sample @ proto_b.T).max(dim=1).values   # (2000,)
    print(f"    Sanity — mean max-sim(normal → B centroid): {sim_sample.mean():.4f}  "
          f"min={sim_sample.min():.4f}  max={sim_sample.max():.4f}")

    return proto_a, proto_b


# ── Step 3: Pre-extract features ──────────────────────────────────────────────

@torch.no_grad()
def preextract_all_features(
    normal_tokens_3d: torch.Tensor,
    tasks:            list[dict],
    extractor:        DINOv2Extractor,
    proto_a:          torch.Tensor,    # (N_RANDOM, D)
    proto_b:          torch.Tensor,    # (K_PROTO, D)
):
    """Pre-extract all features needed for evaluation.

    Returns a dict with keys:
      "f_norm_train_a/b"     : (n_images, N_RANDOM / K_PROTO) from pre-extracted tokens
      "f_norm_test_a/b"      : (n_test, N_RANDOM / K_PROTO)   from test images
      "train_paths_per_task" : {defect: [paths]}
      "held_paths_per_task"  : {defect: [paths]}
      "f_held_a/b_per_task"  : {defect: (n_held, N_RANDOM / K_PROTO)}
      "ab_score_norm_test"   : {method: (n_test,)}
      "ab_score_held"        : {method: {defect: (n_held,)}}
    """
    print("\n[3] Pre-extracting features …")

    # ── Normal training features (from pre-extracted tokens) ──────────────────
    print(f"    Normal train tokens → features  ({len(normal_tokens_3d)} images) …")
    f_norm_train_a = tokens_to_feature_c(normal_tokens_3d, proto_a)   # (n_img, N_RANDOM)
    f_norm_train_b = tokens_to_feature_c(normal_tokens_3d, proto_b)   # (n_img, K_PROTO)
    print(f"    f_norm_train_a : {tuple(f_norm_train_a.shape)}  "
          f"mean={f_norm_train_a.mean():.4f}  std={f_norm_train_a.std():.4f}")
    print(f"    f_norm_train_b : {tuple(f_norm_train_b.shape)}  "
          f"mean={f_norm_train_b.mean():.4f}  std={f_norm_train_b.std():.4f}")

    # ── Normal test images ────────────────────────────────────────────────────
    test_good_dir      = MVTEC_ROOT / CATEGORY / "test" / "good"
    normal_test_paths  = sorted(test_good_dir.glob("*.png"))
    print(f"    Extracting {len(normal_test_paths)} normal test images …")

    ab_score_norm_a: list[float] = []
    ab_score_norm_b: list[float] = []
    f_norm_test_a_list: list[torch.Tensor] = []
    f_norm_test_b_list: list[torch.Tensor] = []

    for p in tqdm(normal_test_paths, desc="    normal test", leave=False):
        patches = extract_patch_tokens(p, extractor).to(DEVICE)
        ab_score_norm_a.append(image_score_ab(patches, proto_a.to(DEVICE)))
        ab_score_norm_b.append(image_score_ab(patches, proto_b.to(DEVICE)))
        f_norm_test_a_list.append(image_feature_c(patches, proto_a.to(DEVICE)).cpu())
        f_norm_test_b_list.append(image_feature_c(patches, proto_b.to(DEVICE)).cpu())

    f_norm_test_a = torch.stack(f_norm_test_a_list)   # (n_test, N_RANDOM)
    f_norm_test_b = torch.stack(f_norm_test_b_list)   # (n_test, K_PROTO)
    print(f"    f_norm_test_a  : {tuple(f_norm_test_a.shape)}  "
          f"mean={f_norm_test_a.mean():.4f}")
    print(f"    f_norm_test_b  : {tuple(f_norm_test_b.shape)}  "
          f"mean={f_norm_test_b.mean():.4f}")

    # ── Defect held-out features ──────────────────────────────────────────────
    train_paths_per_task: dict[str, list[str]]       = {}
    held_paths_per_task:  dict[str, list[str]]       = {}
    f_held_a_per_task:    dict[str, torch.Tensor]    = {}
    f_held_b_per_task:    dict[str, torch.Tensor]    = {}
    ab_score_held_a:      dict[str, list[float]]     = {}
    ab_score_held_b:      dict[str, list[float]]     = {}

    for task in tasks:
        defect       = task["defect"]
        train_ps, held_ps = load_defect_split(task["csv_path"])
        train_paths_per_task[defect] = train_ps
        held_paths_per_task[defect]  = held_ps
        print(f"    {defect}: {len(train_ps)} train / {len(held_ps)} held")

        fa_list: list[torch.Tensor] = []
        fb_list: list[torch.Tensor] = []
        sa_list: list[float]        = []
        sb_list: list[float]        = []

        for p in tqdm(held_ps, desc=f"    held {defect}", leave=False):
            patches = extract_patch_tokens(p, extractor).to(DEVICE)
            sa_list.append(image_score_ab(patches, proto_a.to(DEVICE)))
            sb_list.append(image_score_ab(patches, proto_b.to(DEVICE)))
            fa_list.append(image_feature_c(patches, proto_a.to(DEVICE)).cpu())
            fb_list.append(image_feature_c(patches, proto_b.to(DEVICE)).cpu())

        if fa_list:
            f_held_a_per_task[defect] = torch.stack(fa_list)   # (n_held, N_RANDOM)
            f_held_b_per_task[defect] = torch.stack(fb_list)   # (n_held, K_PROTO)
            ab_score_held_a[defect]   = sa_list
            ab_score_held_b[defect]   = sb_list
            print(f"      f_held_b shape: {tuple(f_held_b_per_task[defect].shape)}  "
                  f"mean={f_held_b_per_task[defect].mean():.4f}  "
                  f"(vs normal train mean={f_norm_train_b.mean():.4f})")
        else:
            f_held_a_per_task[defect] = torch.empty(0, N_RANDOM)
            f_held_b_per_task[defect] = torch.empty(0, K_PROTO)
            ab_score_held_a[defect]   = []
            ab_score_held_b[defect]   = []

    return {
        "f_norm_train_a":      f_norm_train_a,
        "f_norm_train_b":      f_norm_train_b,
        "f_norm_test_a":       f_norm_test_a,
        "f_norm_test_b":       f_norm_test_b,
        "train_paths_per_task": train_paths_per_task,
        "held_paths_per_task":  held_paths_per_task,
        "f_held_a_per_task":   f_held_a_per_task,
        "f_held_b_per_task":   f_held_b_per_task,
        "ab_score_norm_a":     ab_score_norm_a,
        "ab_score_norm_b":     ab_score_norm_b,
        "ab_score_held_a":     ab_score_held_a,
        "ab_score_held_b":     ab_score_held_b,
        "normal_test_paths":   normal_test_paths,
    }


# ── Step 4: Method A & B evaluation ──────────────────────────────────────────

def evaluate_ab(data: dict, tasks: list[dict]) -> dict[str, dict]:
    """Evaluate Methods A and B (fixed prototypes, no learning, no CL).

    Returns per-defect I-AUC for each method.
    """
    print("\n[4] Methods A & B — fixed prototype evaluation …")

    results: dict[str, dict] = {"A": {}, "B": {}}

    for method, norm_scores, held_scores_per_defect in [
        ("A", data["ab_score_norm_a"], data["ab_score_held_a"]),
        ("B", data["ab_score_norm_b"], data["ab_score_held_b"]),
    ]:
        for task in tasks:
            defect      = task["defect"]
            def_scores  = held_scores_per_defect.get(defect, [])
            i_auc       = compute_iauc(norm_scores, def_scores)
            results[method][defect] = i_auc
            print(f"    Method {method}  {defect:<12}  I-AUC={i_auc:.4f}  "
                  f"(n_norm={len(norm_scores)}, n_def={len(def_scores)})")

    return results


# ── Step 5: Method C — CONCIL sequential CL ──────────────────────────────────

def evaluate_c(
    data:      dict,
    tasks:     list[dict],
    extractor: DINOv2Extractor,
    proto_b:   torch.Tensor,
) -> dict:
    """Method C: CONCIL ridge regression in prototype-similarity space.

    Sequential CL: accumulate normal first, then defect tasks one by one.
    Evaluates after EACH task to enable BWT computation.

    Returns {defect: {task_id_at_eval: I-AUC}} and final I-AUC per defect.
    """
    print("\n[5] Method C — CONCIL sequential CL …")

    solver = ConcilSolver(input_dim=1, lambda_anomaly=LAMBDA_ANOMALY)
    f_norm_train = data["f_norm_train_b"]   # (n_norm_images, K_PROTO)
    f_norm_test  = data["f_norm_test_b"]    # (n_test, K_PROTO)

    # ── Accumulate normal data (y=0) ─────────────────────────────────────────
    y_norm = torch.zeros(len(f_norm_train))
    print(f"    Accumulating {len(f_norm_train)} normal images (y=0)  "
          f"feature shape: {tuple(f_norm_train.shape)}")
    w, b_vec = solver.update_anomaly_head(f_norm_train, y_norm)
    print(f"    After normal data  →  w: shape={w.shape}  "
          f"mean={w.mean():.4f}  std={w.std():.4f}  "
          f"b={b_vec[0]:.4f}")

    # ── Sequential defect tasks ───────────────────────────────────────────────
    # task_auc[defect][task_id_at_eval] = I-AUC
    task_auc:   dict[str, dict[int, float]] = {t["defect"]: {} for t in tasks}
    final_w:    np.ndarray | None           = None
    final_b:    float                       = 0.0

    for task in tasks:
        task_id = task["task_id"]
        defect  = task["defect"]
        print(f"\n    Task {task_id}: {defect.upper()}")

        # Extract defect train features on-the-fly
        train_paths = data["train_paths_per_task"][defect]
        print(f"      Extracting {len(train_paths)} defect train images …")
        f_def_train_list: list[torch.Tensor] = []
        for p in tqdm(train_paths, desc=f"      {defect} train", leave=False):
            patches = extract_patch_tokens(p, extractor).to(DEVICE)
            f_def_train_list.append(image_feature_c(patches, proto_b.to(DEVICE)).cpu())
        f_def_train = torch.stack(f_def_train_list)   # (n_train, K_PROTO)
        y_def       = torch.ones(len(f_def_train))

        print(f"      f_def_train : {tuple(f_def_train.shape)}  "
              f"mean={f_def_train.mean():.4f}  "
              f"(normal mean={f_norm_train.mean():.4f})")

        # Update CONCIL
        w, b_vec = solver.update_anomaly_head(f_def_train, y_def)
        print(f"      Updated w  : mean={w.mean():.4f}  std={w.std():.4f}  "
              f"min={w.min():.4f}  max={w.max():.4f}  b={b_vec[0]:.4f}")
        print(f"      n_pos={w[w > 0].sum():.0f} pos-weight atoms, "
              f"n_neg={w[w < 0].sum():.0f} neg-weight atoms  "
              f"(of {len(w)} total)")

        final_w = w
        final_b = float(b_vec[0])

        # Compute scores under current solution
        w_t  = torch.tensor(w,       dtype=torch.float32)
        b_t  = float(b_vec[0])
        norm_scores_c = (f_norm_test  @ w_t + b_t).tolist()

        # Evaluate all defects seen so far
        for seen_task in tasks[:task_id]:
            seen_defect = seen_task["defect"]
            f_held      = data["f_held_b_per_task"].get(seen_defect)
            if f_held is None or len(f_held) == 0:
                task_auc[seen_defect][task_id] = float("nan")
                continue
            def_scores_c = (f_held @ w_t + b_t).tolist()
            auc          = compute_iauc(norm_scores_c, def_scores_c)
            task_auc[seen_defect][task_id] = auc
            n_def = len(def_scores_c)
            print(f"        {seen_defect:<14}  I-AUC={auc:.4f}  "
                  f"(n_norm={len(norm_scores_c)}, n_def={n_def})")

    # Final scores (after all tasks) — already in task_auc at T=max task_id
    T = max(t["task_id"] for t in tasks)
    final_per_defect: dict[str, float] = {
        t["defect"]: task_auc[t["defect"]].get(T, float("nan"))
        for t in tasks
    }

    # ── BWT ──────────────────────────────────────────────────────────────────
    print("\n    Method C BWT:")
    bwt_per_defect: dict[str, float] = {}
    for t in tasks:
        defect  = t["defect"]
        t_first = t["task_id"]
        if t_first >= T:
            continue
        r_first = task_auc[defect].get(t_first, float("nan"))
        r_final = task_auc[defect].get(T,       float("nan"))
        if np.isnan(r_first) or np.isnan(r_final):
            bwt_per_defect[defect] = float("nan")
        else:
            bwt_per_defect[defect] = r_final - r_first
        print(f"      {defect:<14}  BWT={bwt_per_defect.get(defect, float('nan')):+.4f}  "
              f"(first={r_first:.4f}  final={r_final:.4f})")
    valid_bwt = [v for v in bwt_per_defect.values() if not np.isnan(v)]
    mean_bwt  = float(np.mean(valid_bwt)) if valid_bwt else float("nan")

    return {
        "final_per_defect": final_per_defect,
        "task_auc":         task_auc,
        "bwt_per_defect":   bwt_per_defect,
        "mean_bwt":         mean_bwt,
        "final_w":          final_w,
        "final_b":          final_b,
    }


# ── Final comparison table ────────────────────────────────────────────────────

def print_comparison(
    tasks:      list[dict],
    results_ab: dict[str, dict],
    results_c:  dict,
):
    defects = [t["defect"] for t in tasks]

    # Compute means (ignoring NaN)
    def mean_auc(d: dict) -> float:
        vals = [d[k] for k in defects if k in d and not np.isnan(d[k])]
        return float(np.mean(vals)) if vals else float("nan")

    def bwt_str(v) -> str:
        return f"{v:+.3f}" if (v is not None and not np.isnan(v)) else "  n/a"

    mean_a = mean_auc(results_ab["A"])
    mean_b = mean_auc(results_ab["B"])
    mean_c = mean_auc(results_c["final_per_defect"])
    bwt_c  = results_c["mean_bwt"]

    # ── Per-defect table ──────────────────────────────────────────────────────
    W = 72
    print("\n" + "=" * W)
    print("  HAZELNUT — PER-DEFECT I-AUC")
    print("=" * W)
    print(f"  {'Defect':<14}  {'Method A':>9}  {'Method B':>9}  "
          f"{'Method C':>9}  {'Sys2 ref':>9}")
    print("  " + "─" * (W - 2))
    for defect in defects:
        a = results_ab["A"].get(defect, float("nan"))
        b = results_ab["B"].get(defect, float("nan"))
        c = results_c["final_per_defect"].get(defect, float("nan"))
        a_s = f"{a:.4f}" if not np.isnan(a) else "   nan"
        b_s = f"{b:.4f}" if not np.isnan(b) else "   nan"
        c_s = f"{c:.4f}" if not np.isnan(c) else "   nan"
        print(f"  {defect:<14}  {a_s:>9}  {b_s:>9}  {c_s:>9}  {'(see avg)':>9}")
    print("  " + "─" * (W - 2))

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "=" * W)
    print("  SUMMARY")
    print("=" * W)
    header = f"  {'Method':<28}  {'mean I-AUC':>11}  {'BWT':>8}  {'Notes'}"
    print(header)
    print("  " + "─" * (W - 2))

    def fmt(mean, bwt_v, notes):
        m_s = f"{mean:.3f}" if not np.isnan(mean) else "  nan"
        b_s = bwt_str(bwt_v)
        print(f"  {notes:<28}  {m_s:>11}  {b_s:>8}")

    fmt(mean_a, float("nan"), "A: Random (N=256)")
    fmt(mean_b, float("nan"), f"B: K-means (K={K_PROTO})")
    fmt(mean_c, bwt_c,        f"C: CONCIL (K={K_PROTO} proto + ridge)")
    print("  " + "─" * (W - 2))
    fmt(SYSTEM2_HAZELNUT_IAUC, SYSTEM2_HAZELNUT_BWT, "System 2 (SAE+guide, ref)")
    print("=" * W)

    # ── Decision rule ──────────────────────────────────────────────────────────
    print("\n" + "─" * W)
    print("  DECISION")
    print("─" * W)
    if np.isnan(mean_c):
        verdict = "INSUFFICIENT DATA — cannot evaluate"
    elif mean_c >= 0.92:
        verdict = ("PROTOTYPE DIRECTION VIABLE — "
                   "proceed to full System 3")
    elif mean_c >= 0.85:
        verdict = ("MARGINAL — "
                   "prototype space is weak, consider hybrid")
    else:
        verdict = ("PROTOTYPE DIRECTION FAILS — "
                   "stay with SAE space")

    print(f"\n  Method C mean I-AUC = {mean_c:.4f}")
    print(f"  >> {verdict}")
    print("─" * W)

    return mean_a, mean_b, mean_c


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    for p in [TOKENS_PATH, INDEX_PATH]:
        if not p.exists():
            print(f"ERROR: missing {p}")
            sys.exit(1)

    normal_tokens, normal_tokens_3d, n_norm_images, extractor, tasks = load_resources()
    proto_a, proto_b = build_prototypes(normal_tokens)
    data             = preextract_all_features(
                           normal_tokens_3d, tasks, extractor, proto_a, proto_b)
    results_ab       = evaluate_ab(data, tasks)
    results_c        = evaluate_c(data, tasks, extractor, proto_b)
    mean_a, mean_b, mean_c = print_comparison(tasks, results_ab, results_c)

    # Save lightweight results
    out = {
        "category":       CATEGORY,
        "K_proto":        K_PROTO,
        "N_random":       N_RANDOM,
        "lambda_anomaly": LAMBDA_ANOMALY,
        "method_A_iauc":  {d: (None if np.isnan(v) else v)
                           for d, v in results_ab["A"].items()},
        "method_B_iauc":  {d: (None if np.isnan(v) else v)
                           for d, v in results_ab["B"].items()},
        "method_C_iauc":  {d: (None if np.isnan(v) else v)
                           for d, v in results_c["final_per_defect"].items()},
        "method_C_bwt":   {d: (None if np.isnan(v) else v)
                           for d, v in results_c["bwt_per_defect"].items()},
        "mean_iauc": {
            "A": None if np.isnan(mean_a) else round(mean_a, 4),
            "B": None if np.isnan(mean_b) else round(mean_b, 4),
            "C": None if np.isnan(mean_c) else round(mean_c, 4),
            "system2_ref": SYSTEM2_HAZELNUT_IAUC,
        },
        "mean_bwt_C": None if np.isnan(results_c["mean_bwt"]) else round(results_c["mean_bwt"], 4),
    }
    out_path = Path("results/prototype_sanity_hazelnut.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults → {out_path}")


if __name__ == "__main__":
    main()
