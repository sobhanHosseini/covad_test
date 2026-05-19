"""
Phase 3: Guide Coefficient Learning for SAE-CBM.

Sequential CL experiment using GuideCoeffTrainer across 5 MVTec categories.
Replicates the same CSV reading, 80/20 defect split (seed=42), evaluation
protocol, and BWT formula used in cl/cl_trainer.py and evaluators/evaluator_cl.py.

Required inputs (must exist before running):
  sae_training/mvtec_normal_patches_vitl14reg.pt   (run scripts/01_extract_tokens.py)
  sae_training/mvtec_patch_index_reg.pt
  sae_training/sae_vitl14reg_C4096_k64.pt          (run scripts/02_train_sae.py)

Outputs:
  sae_training/guides/{category}/task_{t:02d}.pt   per-task checkpoints
  sae_training/guides/results_summary.json          all metrics

Run from project root:
    python scripts/05_train_guide_coefficients.py
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from features.dinov2_extractor import DINOv2Extractor
from features.sae              import SparseAutoencoder
from features.guide_trainer    import GuideCoeffTrainer

# ── config ───────────────────────────────────────────────────────────────────

MVTEC_ROOT        = Path("/home/sobhan_hosseini/datasets/mvtec")
ANN_ROOT          = Path("annotations")
GUIDES_DIR        = Path("sae_training/guides")
TOKENS_PATH       = Path("sae_training/mvtec_normal_patches_vitl14reg.pt")
INDEX_PATH        = Path("sae_training/mvtec_patch_index_reg.pt")
SAE_PATH          = Path("sae_training/sae_vitl14reg_C4096_k64.pt")
OUT_SUMMARY       = GUIDES_DIR / "results_summary.json"

CATEGORIES        = ["bottle", "capsule", "hazelnut", "metal_nut", "screw"]
DEFECT_TRAIN_RATIO = 0.80
SEED              = 42
LAMBDA_REG        = 1.0
MODEL_NAME        = "dinov2_vitl14_reg"
BATCH_SIZE        = 16
DEVICE            = torch.device("cuda:0")

_META_COLS = frozenset(
    ["image_path", "label_index", "mask_path", "anomaly_type", "split"]
)

# ── preflight check ───────────────────────────────────────────────────────────

def preflight():
    missing = [p for p in [TOKENS_PATH, INDEX_PATH, SAE_PATH] if not p.exists()]
    if missing:
        print("ERROR — required files not found:")
        for p in missing:
            print(f"  {p}")
        print()
        print("Run in order:")
        print("  python scripts/01_extract_tokens.py   (backbone: dinov2_vitl14_reg)")
        print("  python scripts/02_train_sae.py")
        sys.exit(1)

# ── data helpers ─────────────────────────────────────────────────────────────

def load_defect_split(task_csv_path: str):
    """Load defect images from a task CSV using the same 80/20 split as cl_trainer.py.

    Returns:
        train_paths:  list[str] — 80% for guide learning
        held_paths:   list[str] — 20% for evaluation
    """
    df        = pd.read_csv(task_csv_path)
    defect_df = df[df["label_index"] == 1].reset_index(drop=True)
    n_defect  = len(defect_df)
    n_train   = max(1, int(n_defect * DEFECT_TRAIN_RATIO))

    rng          = np.random.RandomState(SEED)
    shuffled_idx = rng.permutation(n_defect)
    train_idx    = shuffled_idx[:n_train]
    held_idx     = shuffled_idx[n_train:]

    train_paths = defect_df.iloc[train_idx]["image_path"].tolist()
    held_paths  = defect_df.iloc[held_idx]["image_path"].tolist()
    return train_paths, held_paths


@torch.no_grad()
def extract_patch_tokens_batched(
    image_paths: list[str | Path],
    extractor: DINOv2Extractor,
    batch_size: int = BATCH_SIZE,
    desc: str = "",
) -> torch.Tensor:
    """Extract DINOv2 patch tokens for a list of images.

    Returns: (len(image_paths) * 256, EMBED_DIM) float32 CPU tensor.
    """
    all_patches = []
    it = range(0, len(image_paths), batch_size)
    if desc:
        it = tqdm(it, desc=desc, leave=False)
    for i in it:
        batch = [Image.open(p).convert("RGB") for p in image_paths[i:i+batch_size]]
        p     = extractor.extract_patch_tokens(batch)              # (B, 256, D)
        all_patches.append(p.cpu().reshape(-1, extractor.EMBED_DIM))
    return torch.cat(all_patches, dim=0)


def score_images_from_tokens(
    tokens_per_image: torch.Tensor,    # (N_imgs, 256, d_input)
    trainer: GuideCoeffTrainer,
) -> list[float]:
    """Score each image using max-pooling over patch scores."""
    return [trainer.score_image(tokens_per_image[i]) for i in range(len(tokens_per_image))]


def compute_iauc(
    normal_scores: list[float],
    defect_scores: list[float],
) -> float:
    """I-AUC: normal=0, defect=1."""
    if not defect_scores:
        return float("nan")
    y_true  = [0] * len(normal_scores) + [1] * len(defect_scores)
    y_score = normal_scores + defect_scores
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(roc_auc_score(y_true, y_score))

# ── BWT (Lopez-Paz & Ranzato 2017) ───────────────────────────────────────────

def compute_bwt(
    results: list[dict],
) -> tuple[dict[str, float], float]:
    """Replicate evaluators/evaluator_cl.py per_defect_bwt() exactly.

    BWT_d = R(T, d) - R(t_first(d), d)
    where T = final task index, t_first(d) = task_id when d was first evaluated.
    """
    _NAN = float("nan")
    if len(results) < 2:
        return {}, _NAN

    T = max(r["evaluated_after_task"] for r in results)
    lookup: dict[tuple[str, int], float] = {
        (r["defect"], r["evaluated_after_task"]): r["i_auc"]
        for r in results
    }
    t_first_for: dict[str, int] = {}
    for r in results:
        d = r["defect"]
        if d not in t_first_for or r["task_id"] < t_first_for[d]:
            t_first_for[d] = r["task_id"]

    bwt: dict[str, float] = {}
    for d, t_first in t_first_for.items():
        if t_first >= T:
            continue
        r_first = lookup.get((d, t_first), _NAN)
        r_final = lookup.get((d, T),       _NAN)
        if not (np.isnan(r_first) or np.isnan(r_final)):
            bwt[d] = float(r_final - r_first)
        else:
            bwt[d] = _NAN

    valid  = [v for v in bwt.values() if not np.isnan(v)]
    mean_b = float(np.mean(valid)) if valid else _NAN
    return bwt, mean_b

# ── summary table (matches evaluators/evaluator_cl.py format) ─────────────────

def print_summary(category: str, results: list[dict], bwt_d: dict, mean_bwt: float):
    W   = 66
    sep = "─" * W
    hdr = (
        f"{'T':>2}  {'defect':<14} {'@':>2}  "
        f"{'I-AUC(SAE)':>12}  {'n_norm':>6}  {'n_def':>5}"
    )
    lines = [
        "=" * W,
        f"  CONVAD-SAE Guide Coefficients — {category}",
        "=" * W,
        hdr,
        sep,
    ]
    for r in sorted(results, key=lambda x: (x["evaluated_after_task"], x["task_id"])):
        i_auc = r["i_auc"]
        lines.append(
            f"{r['task_id']:>2}  {r['defect']:<14} {r['evaluated_after_task']:>2}  "
            f"{i_auc:>12.4f}  {r['n_normal_test']:>6}  {r['n_defect_test']:>5}"
        )
    lines.append(sep)
    lines.append(f"  Standard BWT (Lopez-Paz 2017): {mean_bwt:+.4f}")
    _NAN = float("nan")
    for d, v in sorted(bwt_d.items()):
        tag = "" if np.isnan(v) else ("  ← forgetting" if v < -0.01 else "  ← stable")
        lines.append(f"    BWT[{d}] = {v:+.4f}{tag}")
    lines.append("=" * W)
    print("\n".join(lines))

# ── per-category runner ───────────────────────────────────────────────────────

def run_category(
    category: str,
    sae: SparseAutoencoder,
    all_tokens: torch.Tensor,
    patch_index: list[dict],
    extractor: DINOv2Extractor,
) -> list[dict]:
    """Full sequential CL experiment for one category.

    Returns list of result dicts (one per (defect, evaluated_after_task) pair).
    """
    print(f"\n{'='*66}")
    print(f"  Category: {category.upper()}")
    print(f"{'='*66}")

    # ── 1. Slice pre-extracted normal tokens for this category ────────────────
    cat_info     = next(e for e in patch_index if e["category"] == category)
    normal_tokens = all_tokens[cat_info["row_start"] : cat_info["row_end"]]
    print(f"  Normal tokens: {len(normal_tokens):,} patches "
          f"({cat_info['n_images']} images × 256)")

    # ── 2. Initialise trainer and build g⁻ ────────────────────────────────────
    trainer = GuideCoeffTrainer(sae, lambda_reg=LAMBDA_REG, device=DEVICE)
    trainer.build_normal_guide(normal_tokens)

    # ── 3. Load task sequence ─────────────────────────────────────────────────
    task_seq_path = ANN_ROOT / category / "cl_tasks" / "task_sequence.json"
    with open(task_seq_path) as f:
        tasks = json.load(f)
    print(f"  Tasks: {[t['defect'] for t in tasks]}")

    # ── 4. Pre-extract normal test tokens (reused every evaluation round) ─────
    test_good_dir    = MVTEC_ROOT / category / "test" / "good"
    test_normal_paths = sorted(test_good_dir.glob("*.png"))
    print(f"  Pre-extracting {len(test_normal_paths)} normal test images …")
    normal_test_flat = extract_patch_tokens_batched(
        test_normal_paths, extractor, desc="normal test"
    )
    n_normal_test           = len(test_normal_paths)
    normal_test_per_image   = normal_test_flat.reshape(n_normal_test, 256, extractor.EMBED_DIM)

    # ── 5. Per-task state ─────────────────────────────────────────────────────
    held_tokens_per_defect: dict[str, torch.Tensor] = {}   # defect → (n*256, D)
    n_held_per_defect:      dict[str, int]           = {}
    results: list[dict] = []
    ckpt_dir = GUIDES_DIR / category
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── 6. Sequential task loop ───────────────────────────────────────────────
    for task in tasks:
        task_id = task["task_id"]
        defect  = task["defect"]
        print(f"\n  Task {task_id}: {defect.upper()}")

        # a. 80/20 defect split (matching cl_trainer.py exactly)
        train_paths, held_paths = load_defect_split(task["csv_path"])
        print(f"    Split: {len(train_paths)} train / {len(held_paths)} held-out")

        # b. Pre-extract and store held-out tokens for evaluation
        if held_paths:
            held_flat = extract_patch_tokens_batched(
                held_paths, extractor, desc=f"held {defect}"
            )
            held_tokens_per_defect[defect] = held_flat
            n_held_per_defect[defect]      = len(held_paths)
        else:
            held_tokens_per_defect[defect] = torch.empty(0, extractor.EMBED_DIM)
            n_held_per_defect[defect]      = 0

        # c. Extract defect training tokens and update g⁺
        print(f"    Extracting {len(train_paths)} defect training images …")
        train_tokens = extract_patch_tokens_batched(
            train_paths, extractor, desc=f"train {defect}"
        )
        trainer.update_anomaly_guide(train_tokens)
        del train_tokens

        # d. Evaluate ALL defects seen so far (re-score with updated g⁺)
        print(f"    Evaluating {task_id} defect(s) …")
        normal_scores = score_images_from_tokens(normal_test_per_image, trainer)

        for seen_task in tasks[:task_id]:
            seen_defect  = seen_task["defect"]
            seen_task_id = seen_task["task_id"]

            n_held = n_held_per_defect.get(seen_defect, 0)
            if n_held == 0:
                i_auc = float("nan")
                print(f"      {seen_defect:<14} I-AUC=nan  (no held-out images)")
            else:
                held_flat = held_tokens_per_defect[seen_defect]
                held_per_img = held_flat.reshape(n_held, 256, extractor.EMBED_DIM)
                defect_scores = score_images_from_tokens(held_per_img, trainer)
                i_auc = compute_iauc(normal_scores, defect_scores)
                print(f"      {seen_defect:<14} I-AUC={i_auc:.4f}  "
                      f"({n_normal_test} normals + {n_held} defects)")

            results.append({
                "task_id":             seen_task_id,
                "defect":              seen_defect,
                "evaluated_after_task": task_id,
                "i_auc":               i_auc,
                "n_normal_test":       n_normal_test,
                "n_defect_test":       n_held,
            })

        # e. Checkpoint
        trainer.save(ckpt_dir / f"task_{task_id:02d}.pt")
        print(f"    Checkpoint → {ckpt_dir}/task_{task_id:02d}.pt")

    return results

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    preflight()

    print("=" * 66)
    print("  Phase 3: SAE-CBM Guide Coefficient Learning")
    print("=" * 66)

    # Load shared resources once
    print("\nLoading SAE …")
    sae = SparseAutoencoder.load(str(SAE_PATH), device="cpu")
    sae.to(DEVICE).eval()

    print("Loading pre-extracted normal tokens …")
    all_tokens  = torch.load(TOKENS_PATH, map_location="cpu", weights_only=True)
    patch_index = torch.load(INDEX_PATH,  map_location="cpu", weights_only=False)
    print(f"  Token tensor: {all_tokens.shape}  ({all_tokens.nbytes/1e9:.2f} GB)")

    print("Loading DINOv2 extractor (dinov2_vitl14_reg) …")
    extractor = DINOv2Extractor(MODEL_NAME, device=DEVICE)
    extractor.eval()

    GUIDES_DIR.mkdir(parents=True, exist_ok=True)

    # Run all categories
    all_results: dict[str, list[dict]] = {}
    summary_output: dict = {}

    for category in CATEGORIES:
        results = run_category(category, sae, all_tokens, patch_index, extractor)
        all_results[category] = results

        bwt_d, mean_bwt = compute_bwt(results)
        print()
        print_summary(category, results, bwt_d, mean_bwt)

        summary_output[category] = {
            "tasks":          results,
            "bwt_per_defect": {k: (None if np.isnan(v) else v) for k, v in bwt_d.items()},
            "mean_bwt":       None if np.isnan(mean_bwt) else mean_bwt,
        }

    # Save full summary
    with open(OUT_SUMMARY, "w") as f:
        json.dump(summary_output, f, indent=2)
    print(f"\nResults saved → {OUT_SUMMARY}")

    # Cross-category summary
    print("\n" + "=" * 66)
    print("  CROSS-CATEGORY SUMMARY")
    print("=" * 66)
    print(f"{'category':<14} {'tasks':>5}  {'mean I-AUC':>11}  {'mean BWT':>9}")
    print("─" * 50)
    for cat, res in summary_output.items():
        tasks_list  = res["tasks"]
        # Mean I-AUC: diagonal entries (evaluated_after_task == task_id of last task)
        T_cat       = max(r["evaluated_after_task"] for r in tasks_list)
        final_aucs  = [r["i_auc"] for r in tasks_list
                       if r["evaluated_after_task"] == T_cat
                       and not np.isnan(r["i_auc"])]
        mean_auc    = float(np.mean(final_aucs)) if final_aucs else float("nan")
        mean_bwt_v  = res["mean_bwt"]
        bwt_str     = f"{mean_bwt_v:+.4f}" if mean_bwt_v is not None else "   nan"
        n_tasks     = len(set(r["task_id"] for r in tasks_list))
        print(f"{cat:<14} {n_tasks:>5}  {mean_auc:>11.4f}  {bwt_str:>9}")
    print("=" * 66)


if __name__ == "__main__":
    main()
