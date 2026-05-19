"""
08_cl_dual_branch.py — Sequential CL experiment with dual-branch scoring.

Combined score = 0.9 * SAE_guide_score  +  0.1 * MA-CBM_score

SAE guide accumulates zero-forgetting (recursive ridge regression on g⁺).
MA-CBM CONCIL anomaly head accumulates zero-forgetting.
Both use full 15-task accumulated models for R(T,t) and per-task snapshots for R(t,t).

Architecture
────────────
  Global g⁻ : built ONCE from ALL 15 categories' train/good/ normal patches.
  g⁺        : accumulated sequentially (tasks 1 → 15) from train defect patches.
  CONCIL head: accumulated sequentially from concept vector cache.
  Combined  : 0.9 * norm(SAE) + 0.1 * norm(MA-CBM) per category.

Metrics
───────
  R(t,t)   = I-AUC of category t right after training on task t (initial)
  R(T,t)   = I-AUC of category t after all 15 tasks (final)
  BWT(t)   = R(T,t) - R(t,t)
  Mean BWT (overall, surface, structural)

Efficiency
──────────
  Patch tokens are extracted ONCE and cached in memory.
  Sequential guide updates + scoring = SAE encode + dot product (fast).
  Total DINOv2-L time ≈ 15–25 min (one pass over all images).

Usage:
    cd /mnt/nvme1/sobhan_hosseini/covad_test
    uv run mac/scripts/08_cl_dual_branch.py
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from sklearn.metrics import roc_auc_score

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT.parent))
sys.path.insert(0, str(_ROOT))

from features.dinov2_extractor import DINOv2Extractor
from features.sae import SparseAutoencoder
from features.guide_trainer import GuideCoeffTrainer
from solvers.concil import ConcilSolver

MVTEC_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper",
]
SURFACE = {"carpet","grid","hazelnut","leather","metal_nut","pill","tile","wood","zipper"}
ALPHA   = 0.9
K       = 6       # concept dimensions


# ── helpers ───────────────────────────────────────────────────────────────────

def _auc(y: np.ndarray, s: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y, s))
    except ValueError:
        return float("nan")


def _norm01(x: np.ndarray) -> np.ndarray:
    lo, hi = x.min(), x.max()
    return np.zeros_like(x) if hi - lo < 1e-9 else (x - lo) / (hi - lo)


@torch.no_grad()
def _extract_patches_flat(
    paths: list[Path],
    dino: DINOv2Extractor,
    batch_size: int = 4,
) -> torch.Tensor:
    """Extract ViT-L/14 patch tokens and return (N_images * 256, 1024) on CPU."""
    parts = []
    for i in range(0, len(paths), batch_size):
        imgs = [Image.open(p).convert("RGB") for p in paths[i:i+batch_size]]
        tok  = dino.extract_patch_tokens(imgs).cpu()   # (b, 256, 1024)
        b, n, d = tok.shape
        parts.append(tok.reshape(b * n, d))
    return torch.cat(parts, dim=0)   # (N*256, 1024)


@torch.no_grad()
def _score_with_guide(
    patch_tokens_flat: torch.Tensor,   # (N_images * 256, 1024)
    guide: GuideCoeffTrainer,
    n_images: int,
) -> np.ndarray:
    """Score n_images using max-patch guide score. Returns (n_images,)."""
    patch_scores = guide.score_image_patches(patch_tokens_flat)  # (N*256,)
    return patch_scores.reshape(n_images, -1).max(dim=1).values.numpy()


# ── build eval index from concept vector cache ────────────────────────────────

def _eval_index(cc: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return (eval_vecs, eval_y) matching script 06/07 eval set."""
    vecs = cc["vecs"]
    y    = np.array(cc["y_all"])
    nt   = cc["n_train_normal"]
    nd   = cc["n_defect_train"]
    nn   = cc["n_test_normal"]
    eval_vecs = np.concatenate([vecs[nt:nt+nd], vecs[nt+nd:nt+nd+nn], vecs[nt+nd+nn:]])
    eval_y    = np.concatenate([y[nt:nt+nd],    y[nt+nd:nt+nd+nn],    y[nt+nd+nn:]])
    return eval_vecs.astype(np.float32), eval_y.astype(np.int32)


def _eval_paths(cc: dict) -> list[Path]:
    """Return eval image paths in same order as _eval_index."""
    paths = cc["all_paths"]
    nt, nd, nn = cc["n_train_normal"], cc["n_defect_train"], cc["n_test_normal"]
    idx = list(range(nt, nt+nd)) + list(range(nt+nd, nt+nd+nn)) + list(range(nt+nd+nn, len(paths)))
    return [Path(paths[i]) for i in idx]


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    cfg_path = _ROOT / "configs" / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    mvtec_root = Path(cfg["mvtec_root"])
    device     = cfg.get("device", "cuda")
    out_dir    = Path(cfg["output_dir"]) / "cl_dual_branch"
    out_dir.mkdir(parents=True, exist_ok=True)
    tok_cache  = out_dir / "patch_token_cache.pkl"

    print("=" * 68)
    print("  CL Dual-Branch: SAE guide + MA-CBM  (α=0.9)")
    print("=" * 68)

    # ── Load models ───────────────────────────────────────────────────────────
    print("\n[init] Loading ViT-L/14-reg + SAE …")
    dino = DINOv2Extractor("dinov2_vitl14_reg", device=torch.device(device))
    sae  = SparseAutoencoder.load(cfg["sae_weights"], device=device)
    sae  = sae.to(device).eval()

    # ── Load concept vector cache ─────────────────────────────────────────────
    print("[init] Loading concept vector cache …")
    cache = pickle.load(open(
        Path(cfg["output_dir"]) / "concil" / "concept_vector_cache.pkl", "rb"
    ))

    # ── Phase 0: Extract & cache all patch tokens (one DINOv2-L pass) ─────────
    if tok_cache.exists():
        print(f"\n[Phase 0] Loading patch token cache from {tok_cache.name} …")
        tok_data = pickle.load(open(tok_cache, "rb"))
    else:
        print("\n[Phase 0] Extracting ViT-L patch tokens for all images …")
        tok_data: dict[str, dict] = {}
        for cat in MVTEC_CATEGORIES:
            cc     = cache[cat]
            paths  = cc["all_paths"]
            nt, nd = cc["n_train_normal"], cc["n_defect_train"]
            nn     = cc["n_test_normal"]

            # Paths
            normal_train = [Path(paths[i]) for i in range(nt)]
            defect_train = [Path(paths[i]) for i in range(nt, nt+nd)]
            eval_paths   = _eval_paths(cc)

            print(f"  [{cat}] normal_train={len(normal_train)}, "
                  f"defect_train={len(defect_train)}, eval={len(eval_paths)}")

            tok_data[cat] = {
                "normal_train": _extract_patches_flat(normal_train, dino, batch_size=4),
                "defect_train": _extract_patches_flat(defect_train, dino, batch_size=4),
                "eval":         _extract_patches_flat(eval_paths,   dino, batch_size=4),
                "n_eval":       len(eval_paths),
            }
        pickle.dump(tok_data, open(tok_cache, "wb"))
        print(f"  Cached → {tok_cache}")

    # ── Phase 1: Build global g⁻ from ALL normals ─────────────────────────────
    print("\n[Phase 1] Building global g⁻ from all 15 categories' train normals …")
    guide = GuideCoeffTrainer(sae, lambda_reg=1.0, device=device)
    all_normal_tokens = torch.cat([tok_data[c]["normal_train"] for c in MVTEC_CATEGORIES])
    print(f"  Total normal tokens: {all_normal_tokens.shape[0]:,}")
    guide.build_normal_guide(all_normal_tokens)
    del all_normal_tokens

    # ── Phase 2: Sequential loop ──────────────────────────────────────────────
    print("\n[Phase 2] Sequential training loop …\n")
    solver = ConcilSolver(input_dim=K, lambda_anomaly=1e-4)

    init_sae:   dict[str, np.ndarray] = {}
    init_macbm: dict[str, np.ndarray] = {}
    init_y:     dict[str, np.ndarray] = {}

    hdr = f"  {'T':<3} {'Category':<13} {'SAE_init':>9} {'MAC_init':>9} {'Comb_init':>10}"
    print(hdr);  print("  " + "─" * 48)

    for t_idx, cat in enumerate(MVTEC_CATEGORIES):
        cc = cache[cat]

        # -- Update SAE g⁺ with this task's defect train patches --
        guide.update_anomaly_guide(tok_data[cat]["defect_train"])

        # -- SAE score for this category's eval set --
        eval_toks = tok_data[cat]["eval"]
        n_eval    = tok_data[cat]["n_eval"]
        sae_s     = _score_with_guide(eval_toks, guide, n_eval)

        # -- MA-CBM score: update CONCIL with this task's concept vecs --
        eval_vecs, eval_y = _eval_index(cc)
        nt = cc["n_train_normal"];  nd = cc["n_defect_train"]
        train_vecs = cc["vecs"][np.array(cc["is_train_mask"])].astype(np.float32)
        train_y    = np.array(cc["y_all"])[np.array(cc["is_train_mask"])]
        C_t = torch.tensor(train_vecs, dtype=torch.float32)
        y_t = torch.tensor(train_y,   dtype=torch.float32)
        w_t, b_t_arr = solver.update_anomaly_head(C_t, y_t)
        macbm_s = (eval_vecs @ w_t + float(b_t_arr[0])).astype(np.float32)

        # -- Combined AUC (normalized per category eval set) --
        comb_s  = ALPHA * _norm01(sae_s) + (1 - ALPHA) * _norm01(macbm_s)

        init_sae[cat]   = sae_s
        init_macbm[cat] = macbm_s
        init_y[cat]     = eval_y

        auc_sae  = _auc(eval_y, sae_s)
        auc_mac  = _auc(eval_y, macbm_s)
        auc_comb = _auc(eval_y, comb_s)
        print(f"  T{t_idx+1:<2} {cat:<13} {auc_sae:>9.4f} {auc_mac:>9.4f} {auc_comb:>10.4f}")

    # ── Phase 3: Final scoring with full guide (all 15 tasks) ─────────────────
    print("\n[Phase 3] Final scoring with full guide (after T15) …")

    # Final CONCIL weights
    W_final = solver._solve(solver.A_anomaly, solver.b_anomaly, solver.lambda_a)
    w_final = W_final[:K, 0].float().numpy()
    b_final = float(W_final[K, 0])

    final_sae:   dict[str, np.ndarray] = {}
    final_macbm: dict[str, np.ndarray] = {}

    for cat in MVTEC_CATEGORIES:
        cc     = cache[cat]
        eval_toks  = tok_data[cat]["eval"]
        n_eval     = tok_data[cat]["n_eval"]
        eval_vecs, eval_y = _eval_index(cc)

        sae_s   = _score_with_guide(eval_toks, guide, n_eval)
        macbm_s = (eval_vecs @ w_final + b_final).astype(np.float32)

        final_sae[cat]   = sae_s
        final_macbm[cat] = macbm_s

        comb_s = ALPHA * _norm01(sae_s) + (1 - ALPHA) * _norm01(macbm_s)
        print(f"  {cat:<13}  SAE={_auc(eval_y,sae_s):.4f}  "
              f"MAC={_auc(eval_y,macbm_s):.4f}  "
              f"Comb={_auc(eval_y,comb_s):.4f}")

    # ── Phase 4: BWT ──────────────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print("  FINAL RESULTS — CL Dual-Branch (α=0.9)")
    print("=" * 68)

    print(f"\n  {'Category':<14} {'I-AUC(init)':>12} {'I-AUC(T15)':>11} "
          f"{'BWT':>8}  {'Group'}")
    print("  " + "─" * 60)

    bwt_all: list[float] = []
    bwt_surface: list[float] = []
    bwt_struct:  list[float] = []

    results: dict[str, dict] = {}

    for t_idx, cat in enumerate(MVTEC_CATEGORIES[:-1]):   # exclude last (no BWT)
        y       = init_y[cat]
        comb_i  = ALPHA * _norm01(init_sae[cat])   + (1-ALPHA)*_norm01(init_macbm[cat])
        comb_f  = ALPHA * _norm01(final_sae[cat])  + (1-ALPHA)*_norm01(final_macbm[cat])
        auc_i   = _auc(y, comb_i)
        auc_f   = _auc(y, comb_f)
        bwt     = auc_f - auc_i
        grp     = "S" if cat in SURFACE else "T"
        print(f"  {cat:<14} {auc_i:>12.4f} {auc_f:>11.4f} {bwt:>+8.4f}  [{grp}]")
        bwt_all.append(bwt)
        (bwt_surface if cat in SURFACE else bwt_struct).append(bwt)
        results[cat] = {"init": auc_i, "final": auc_f, "bwt": bwt, "group": grp}

    # Last task
    cat_last = MVTEC_CATEGORIES[-1]
    y_last   = init_y[cat_last]
    comb_i_l = ALPHA * _norm01(init_sae[cat_last])  + (1-ALPHA)*_norm01(init_macbm[cat_last])
    auc_i_l  = _auc(y_last, comb_i_l)
    grp_l    = "S" if cat_last in SURFACE else "T"
    print(f"  {cat_last:<14} {auc_i_l:>12.4f} {'(last)':>11} {'—':>8}  [{grp_l}]")
    results[cat_last] = {"init": auc_i_l, "final": None, "bwt": None, "group": grp_l}

    print("  " + "─" * 60)
    mean_bwt_all  = float(np.mean(bwt_all))
    mean_bwt_sur  = float(np.mean(bwt_surface)) if bwt_surface else float("nan")
    mean_bwt_stc  = float(np.mean(bwt_struct))  if bwt_struct  else float("nan")

    print(f"\n  Mean BWT  (overall)    : {mean_bwt_all:+.4f}")
    print(f"  Mean BWT  (surface/S)  : {mean_bwt_sur:+.4f}")
    print(f"  Mean BWT  (structural/T): {mean_bwt_stc:+.4f}")

    print(f"\n  {'─'*50}")
    print(f"  Comparison:")
    print(f"    SAE guide  standalone BWT : +0.0140  (reference)")
    print(f"    MA-CBM     standalone BWT : -0.0360  (reference)")
    print(f"    Dual-branch (α=0.9)  BWT : {mean_bwt_all:+.4f}  (measured)")
    expected = 0.9 * 0.014 + 0.1 * (-0.036)
    print(f"    Expected (linear approx) : {expected:+.4f}  "
          f"= 0.9×(+0.014) + 0.1×(-0.036)")
    delta = abs(mean_bwt_all - expected)
    print(f"    Deviation from expected  : {delta:.4f}")
    print(f"  {'─'*50}")

    # Save
    summary = {
        "alpha": ALPHA,
        "mean_bwt_overall":    mean_bwt_all,
        "mean_bwt_surface":    mean_bwt_sur,
        "mean_bwt_structural": mean_bwt_stc,
        "per_category": results,
        "reference_sae_bwt":   0.014,
        "reference_macbm_bwt": -0.036,
        "expected_combined":   expected,
    }
    with open(out_dir / "cl_dual_branch_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Results → {out_dir}/cl_dual_branch_results.json")
    print("=" * 68)


if __name__ == "__main__":
    main()
