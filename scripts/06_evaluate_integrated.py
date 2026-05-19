"""
Integrated evaluation: Branch 1 (PatchCore) + Branch 2 (guide coefficients).

For each of the 5 categories evaluates all defect types at final model state,
combining both branches and reporting per-defect I-AUC for B1 alone, B2 alone,
and a 0.5/0.5 min-max-normalised blend.

Required inputs:
  sae_training/mvtec_normal_patches_vitl14reg.pt
  sae_training/mvtec_patch_index_reg.pt
  sae_training/sae_vitl14reg_C4096_k64.pt
  sae_training/guides/{category}/task_XX.pt      (last checkpoint per category)
  sae_training/guides/results_summary.json       (Phase 3 BWT reference)

Outputs:
  sae_training/integrated_results.json

Run from project root:
    python scripts/06_evaluate_integrated.py
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
from features.dinov2_extractor  import DINOv2Extractor
from features.sae               import SparseAutoencoder
from features.guide_trainer     import GuideCoeffTrainer
from features.patchcore_memory  import PatchCoreMemory

# ── config ────────────────────────────────────────────────────────────────────

MVTEC_ROOT   = Path("/home/sobhan_hosseini/datasets/mvtec")
ANN_ROOT     = Path("annotations")
GUIDES_DIR   = Path("sae_training/guides")
TOKENS_PATH  = Path("sae_training/mvtec_normal_patches_vitl14reg.pt")
INDEX_PATH   = Path("sae_training/mvtec_patch_index_reg.pt")
SAE_PATH     = Path("sae_training/sae_vitl14reg_C4096_k64.pt")
PHASE3_JSON  = GUIDES_DIR / "results_summary.json"
OUT_JSON     = Path("sae_training/integrated_results.json")

CATEGORIES       = ["bottle", "capsule", "hazelnut", "metal_nut", "screw"]
DEFECT_TRAIN_RATIO = 0.80
SEED             = 42
CORESET_SIZE     = 10_000
EMBED_DIM        = 1024
MODEL_NAME       = "dinov2_vitl14_reg"
DEVICE           = torch.device("cuda:0")

# ── preflight ─────────────────────────────────────────────────────────────────

def preflight():
    required = [TOKENS_PATH, INDEX_PATH, SAE_PATH, PHASE3_JSON]
    missing  = [p for p in required if not p.exists()]
    if missing:
        print("ERROR — required files not found:")
        for p in missing:
            print(f"  {p}")
        sys.exit(1)
    for cat in CATEGORIES:
        if not list((GUIDES_DIR / cat).glob("task_*.pt")):
            print(f"ERROR — no guide checkpoint found for {cat}")
            sys.exit(1)

# ── data helpers ──────────────────────────────────────────────────────────────

def load_defect_split(task_csv_path: str):
    """80/20 split matching cl_trainer.py exactly (seed=42 per task)."""
    df        = pd.read_csv(task_csv_path)
    defect_df = df[df["label_index"] == 1].reset_index(drop=True)
    n_defect  = len(defect_df)
    n_train   = max(1, int(n_defect * DEFECT_TRAIN_RATIO))
    rng          = np.random.RandomState(SEED)
    shuffled_idx = rng.permutation(n_defect)
    held_idx     = shuffled_idx[n_train:]
    return defect_df.iloc[held_idx]["image_path"].tolist()


@torch.no_grad()
def score_path(img_path, memory, trainer, extractor):
    """Score one image through both branches. Returns (s_novel, s_guide)."""
    img    = Image.open(img_path).convert("RGB")
    tokens = extractor.extract_patch_tokens([img])     # (1, 256, D)
    s_novel, _ = memory.score(tokens)
    s_guide    = trainer.score_image(tokens.squeeze(0))
    return float(s_novel[0].item()), float(s_guide)


def minmax(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo + 1e-8)


def iauc(y_true, y_score) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(roc_auc_score(y_true, y_score))

# ── per-category evaluation ───────────────────────────────────────────────────

def evaluate_category(
    category: str,
    sae: SparseAutoencoder,
    all_tokens: torch.Tensor,
    patch_index: list[dict],
    extractor: DINOv2Extractor,
    phase3: dict,
) -> list[dict]:
    print(f"\n{'='*66}")
    print(f"  {category.upper()}")
    print(f"{'='*66}")

    # ── Branch 1: build PatchCore from pre-extracted reg4 tokens ─────────────
    cat_info           = next(e for e in patch_index if e["category"] == category)
    n_images           = cat_info["n_images"]
    normal_tokens_flat = all_tokens[cat_info["row_start"]:cat_info["row_end"]]
    normal_tokens_3d   = normal_tokens_flat.reshape(n_images, 256, EMBED_DIM)

    memory = PatchCoreMemory(coreset_size=CORESET_SIZE, device=DEVICE, seed=42)
    memory.build(normal_tokens_3d)
    print(f"  PatchCore: {memory.coreset_size:,} patches ({memory.memory_mb:.1f} MB)")

    # ── Branch 2: load last-task guide checkpoint ─────────────────────────────
    last_ckpt = sorted((GUIDES_DIR / category).glob("task_*.pt"))[-1]
    trainer   = GuideCoeffTrainer.load(last_ckpt, sae, device=DEVICE)
    print(f"  Guide:     loaded {last_ckpt.name}")

    # ── task sequence ─────────────────────────────────────────────────────────
    task_seq_path = ANN_ROOT / category / "cl_tasks" / "task_sequence.json"
    with open(task_seq_path) as f:
        tasks = json.load(f)

    # ── collect all test image paths ──────────────────────────────────────────
    normal_paths = sorted((MVTEC_ROOT / category / "test" / "good").glob("*.png"))
    n_normal     = len(normal_paths)

    all_paths:  list[str] = [str(p) for p in normal_paths]
    all_labels: list[int] = [0] * n_normal
    defect_ranges: dict[str, tuple[int, int, int]] = {}   # defect → (start, end, task_id)

    for task in tasks:
        defect     = task["defect"]
        held_paths = load_defect_split(task["csv_path"])
        start      = len(all_paths)
        all_paths.extend(held_paths)
        all_labels.extend([1] * len(held_paths))
        defect_ranges[defect] = (start, len(all_paths), task["task_id"])

    # ── score every test image through both branches ──────────────────────────
    print(f"  Scoring {len(all_paths)} test images …")
    s_novel_all: list[float] = []
    s_guide_all: list[float] = []

    for path in tqdm(all_paths, desc=f"  {category}", leave=False):
        sn, sg = score_path(path, memory, trainer, extractor)
        s_novel_all.append(sn)
        s_guide_all.append(sg)

    s1 = np.array(s_novel_all)
    s2 = np.array(s_guide_all)
    s_combined = 0.5 * minmax(s1) + 0.5 * minmax(s2)

    # ── compute per-defect I-AUC ──────────────────────────────────────────────
    all_labels_arr = np.array(all_labels)
    normal_idx     = list(range(n_normal))

    results: list[dict] = []
    for task in tasks:
        defect           = task["defect"]
        start, end, tid  = defect_ranges[defect]
        n_held           = end - start
        if n_held == 0:
            continue

        sel    = normal_idx + list(range(start, end))
        labels = np.array([0] * n_normal + [1] * n_held)

        b1_auc   = iauc(labels, s1[sel])
        b2_auc   = iauc(labels, s2[sel])
        comb_auc = iauc(labels, s_combined[sel])

        results.append({
            "task_id":       tid,
            "defect":        defect,
            "b1_iauc":       round(b1_auc,   4),
            "b2_iauc":       round(b2_auc,   4),
            "combined_iauc": round(comb_auc, 4),
            "n_normal_test": n_normal,
            "n_defect_test": n_held,
        })
        print(f"    {defect:<18}  B1={b1_auc:.4f}  B2={b2_auc:.4f}  "
              f"Comb={comb_auc:.4f}  (n_def={n_held})")

    return results


# ── summary table ─────────────────────────────────────────────────────────────

def print_summary(all_results: dict[str, list[dict]], phase3: dict):
    W   = 74
    sep = "─" * W
    print("\n" + "=" * W)
    print("  INTEGRATED EVALUATION SUMMARY")
    print("=" * W)
    print(f"{'Category':<14} {'Defects':>7}  "
          f"{'B1 mean':>8}  {'B2 mean':>8}  {'Comb mean':>10}  {'BWT(B2)':>8}")
    print(sep)
    for cat, results in all_results.items():
        n_def  = len(results)
        b1_m   = np.mean([r["b1_iauc"]       for r in results])
        b2_m   = np.mean([r["b2_iauc"]       for r in results])
        c_m    = np.mean([r["combined_iauc"]  for r in results])
        bwt_b2 = phase3.get(cat, {}).get("mean_bwt", float("nan"))
        bwt_s  = f"{bwt_b2:+.4f}" if bwt_b2 is not None and not np.isnan(bwt_b2) \
                 else "   nan"
        print(f"{cat:<14} {n_def:>7}  {b1_m:>8.4f}  {b2_m:>8.4f}  "
              f"{c_m:>10.4f}  {bwt_s:>8}")
    print("=" * W)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    preflight()

    print("=" * 66)
    print("  Phase 4: Integrated Evaluation (B1 + B2)")
    print("=" * 66)

    with open(PHASE3_JSON) as f:
        phase3 = json.load(f)

    print("\nLoading SAE …")
    sae = SparseAutoencoder.load(str(SAE_PATH), device="cpu")
    sae.to(DEVICE).eval()

    print("Loading pre-extracted tokens …")
    all_tokens  = torch.load(TOKENS_PATH, map_location="cpu", weights_only=True)
    patch_index = torch.load(INDEX_PATH,  map_location="cpu", weights_only=False)

    print("Loading DINOv2 extractor …")
    extractor = DINOv2Extractor(MODEL_NAME, device=DEVICE)
    extractor.eval()

    all_results: dict[str, list[dict]] = {}
    for cat in CATEGORIES:
        all_results[cat] = evaluate_category(
            cat, sae, all_tokens, patch_index, extractor, phase3
        )

    print_summary(all_results, phase3)

    output = {
        cat: {
            "defects":      results,
            "b1_mean_iauc": round(float(np.mean([r["b1_iauc"]      for r in results])), 4),
            "b2_mean_iauc": round(float(np.mean([r["b2_iauc"]      for r in results])), 4),
            "comb_mean_iauc": round(float(np.mean([r["combined_iauc"] for r in results])), 4),
            "b2_mean_bwt":  phase3.get(cat, {}).get("mean_bwt"),
        }
        for cat, results in all_results.items()
    }
    with open(OUT_JSON, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved → {OUT_JSON}")


if __name__ == "__main__":
    main()
