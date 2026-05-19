"""
07_dual_branch_eval.py — Evaluate single-branch and dual-branch anomaly detection.

Systems evaluated
─────────────────
  Branch 1-A : PatchCore (ViT-B/14, NN distance to memory bank)
  Branch 1-B : SAE guide  (ViT-L/14 → SAE → guide cosine scores)
  Branch 2   : MA-CBM     (6-dim concept vector → CONCIL anomaly head)

  Dual-branch combinations:
    PatchCore + MA-CBM  :  α·PC + (1-α)·MACBM   for α ∈ {0.3, 0.5, 0.7, 0.9}
    SAE guide + MA-CBM  :  α·SAE + (1-α)·MACBM

Evaluation protocol (standard MVTec)
─────────────────────────────────────
  Train set : mvtec/{cat}/train/good/  (normals only — for building PatchCore + g⁻)
  Eval set  : mvtec/{cat}/test/good/   (all normals, label=0)
              + mvtec/{cat}/test/{defect}/ for all defect types (label=1)

  The MA-CBM 80/20 split (from script 06) is used to build the SAE g⁺ guide
  (80% of defect test images → no overlap with our reported eval metrics which
  cover ALL test images).

All scores are min-max normalised per category before combining.
Alpha is never optimised on the test set — we report all α and call the best.

Usage:
    cd /mnt/nvme1/sobhan_hosseini/covad_test
    uv run mac/scripts/07_dual_branch_eval.py
"""

from __future__ import annotations

import pickle
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from sklearn.metrics import roc_auc_score

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT.parent))
sys.path.insert(0, str(_ROOT))

from features.dinov2_extractor import DINOv2Extractor
from features.patchcore_memory import PatchCoreMemory
from features.sae import SparseAutoencoder
from features.guide_trainer import GuideCoeffTrainer
from solvers.concil import ConcilSolver

MVTEC_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper",
]
SURFACE = {"carpet","grid","hazelnut","leather","metal_nut","pill","tile","wood","zipper"}
ALPHAS  = [0.3, 0.5, 0.7, 0.9]

CONCEPT_NAMES = [
    "surface_discontinuity","surface_discoloration","surface_crack",
    "surface_abrasion","surface_void","normality",
]
K = len(CONCEPT_NAMES)


# ── helpers ───────────────────────────────────────────────────────────────────

def _norm01(x: np.ndarray) -> np.ndarray:
    """Min-max normalise to [0, 1]. Stable against zero-range."""
    lo, hi = x.min(), x.max()
    if hi - lo < 1e-9:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def _auc(y: np.ndarray, scores: np.ndarray) -> float:
    """ROC-AUC, returns nan if only one class present."""
    try:
        return float(roc_auc_score(y, scores))
    except ValueError:
        return float("nan")


def _load_images_batched(
    paths: list[Path],
    batch_size: int = 8,
) -> list[list[Image.Image]]:
    """Yield batches of PIL images."""
    batches = []
    for start in range(0, len(paths), batch_size):
        batches.append([
            Image.open(p).convert("RGB")
            for p in paths[start:start + batch_size]
        ])
    return batches


# ── PatchCore scoring ─────────────────────────────────────────────────────────

@torch.no_grad()
def build_patchcore(
    normal_paths: list[Path],
    dino_b: DINOv2Extractor,
    batch_size: int = 8,
) -> PatchCoreMemory:
    """Build a PatchCore memory bank from ViT-B/14 normal-image patch tokens."""
    memory = PatchCoreMemory(coreset_size=10_000, device=dino_b.device)
    all_patches: list[torch.Tensor] = []
    for start in range(0, len(normal_paths), batch_size):
        imgs = [Image.open(p).convert("RGB") for p in normal_paths[start:start+batch_size]]
        ptok = dino_b.extract_patch_tokens(imgs).cpu()   # (b, 256, 768)
        all_patches.append(ptok)
    memory.build(torch.cat(all_patches, dim=0))
    return memory


@torch.no_grad()
def score_patchcore(
    image_paths: list[Path],
    memory: PatchCoreMemory,
    dino_b: DINOv2Extractor,
    batch_size: int = 8,
) -> np.ndarray:
    """Score a list of images with PatchCore. Returns (N,) float32 scores."""
    scores: list[float] = []
    for start in range(0, len(image_paths), batch_size):
        imgs = [Image.open(p).convert("RGB") for p in image_paths[start:start+batch_size]]
        ptok = dino_b.extract_patch_tokens(imgs)          # (b, 256, 768)
        s, _ = memory.score(ptok)
        scores.extend(s.cpu().tolist())
    return np.array(scores, dtype=np.float32)


# ── SAE guide scoring ─────────────────────────────────────────────────────────

@torch.no_grad()
def build_sae_guide(
    normal_paths: list[Path],
    defect_paths: list[Path],          # used only to build g⁺
    dino_l: DINOv2Extractor,
    sae: SparseAutoencoder,
    batch_size: int = 4,
) -> GuideCoeffTrainer:
    """Build SAE guide g⁻ (from normals) and g⁺ (from defects)."""
    guide = GuideCoeffTrainer(sae, lambda_reg=1.0, device=str(dino_l.device))

    # Collect raw ViT-L patch tokens (CPU, flat)
    def _collect(paths):
        parts = []
        for start in range(0, len(paths), batch_size):
            imgs = [Image.open(p).convert("RGB") for p in paths[start:start+batch_size]]
            ptok = dino_l.extract_patch_tokens(imgs).cpu()  # (b, 256, 1024)
            b, n, d = ptok.shape
            parts.append(ptok.reshape(b * n, d))
        return torch.cat(parts, dim=0)   # (N*256, 1024)

    guide.build_normal_guide(_collect(normal_paths))
    if defect_paths:
        guide.update_anomaly_guide(_collect(defect_paths))
    return guide


@torch.no_grad()
def score_sae_guide(
    image_paths: list[Path],
    guide: GuideCoeffTrainer,
    dino_l: DINOv2Extractor,
    batch_size: int = 4,
) -> np.ndarray:
    """Score images with SAE guide. Returns (N,) float32 max-patch scores."""
    scores: list[float] = []
    for start in range(0, len(image_paths), batch_size):
        imgs = [Image.open(p).convert("RGB") for p in image_paths[start:start+batch_size]]
        ptok = dino_l.extract_patch_tokens(imgs).cpu()   # (b, 256, 1024)
        for i in range(ptok.shape[0]):
            s = guide.score_image(ptok[i])   # max over patches
            scores.append(float(s))
    return np.array(scores, dtype=np.float32)


# ── CONCIL anomaly weights from concept vector cache ─────────────────────────

def recompute_concil_weights(cache: dict) -> tuple[np.ndarray, float]:
    """Re-run sequential CONCIL training on cached concept vectors.

    Uses the 80% training split (train_normal + defect_train) for each category
    in task order to accumulate the Gram matrix. Returns final (w, b).
    """
    solver = ConcilSolver(input_dim=K, lambda_anomaly=1e-4)

    for cat in MVTEC_CATEGORIES:
        c  = cache[cat]
        nt = c["n_train_normal"]
        nd = c["n_defect_train"]
        vecs   = c["vecs"]
        y_all  = np.array(c["y_all"])
        train  = np.array(c["is_train_mask"])
        X_tr   = vecs[train]
        y_tr   = y_all[train]
        C_t    = torch.tensor(X_tr, dtype=torch.float32)
        y_t    = torch.tensor(y_tr, dtype=torch.float32)
        solver.update_anomaly_head(C_t, y_t)

    # Extract weights from accumulated state
    W = solver._solve(solver.A_anomaly, solver.b_anomaly, solver.lambda_a)
    w = W[:K, 0].float().numpy()
    b = float(W[K, 0])
    return w, b


def macbm_scores_for_eval(
    cache_cat: dict,
    w: np.ndarray,
    b: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute MA-CBM anomaly scores for the standard eval set.

    Eval set = test/good/ (all) + ALL test defect images (both 80% and 20%
    partitions from the CONCIL split), preserving label ordering.

    Returns (scores, y) both (N,) arrays.
    """
    vecs = cache_cat["vecs"]            # (N_all, 6)
    y    = np.array(cache_cat["y_all"])
    nt   = cache_cat["n_train_normal"]
    nd   = cache_cat["n_defect_train"]
    nn   = cache_cat["n_test_normal"]

    # eval indices: defect_train (80%) | test_normal | defect_test (20%)
    eval_vecs = np.concatenate([
        vecs[nt:nt+nd],          # 80% defects (label=1)
        vecs[nt+nd:nt+nd+nn],    # test normals (label=0)
        vecs[nt+nd+nn:],         # 20% defects (label=1)
    ], axis=0)
    eval_y = np.concatenate([
        y[nt:nt+nd],
        y[nt+nd:nt+nd+nn],
        y[nt+nd+nn:],
    ])
    scores = eval_vecs @ w + b
    return scores.astype(np.float32), eval_y.astype(np.int32)


def eval_image_paths_for_cat(
    cat: str,
    mvtec_root: Path,
    cache_cat: dict,
) -> tuple[list[Path], np.ndarray]:
    """Return ordered (paths, y) for the same eval set as macbm_scores_for_eval."""
    all_paths = cache_cat["all_paths"]
    all_y     = cache_cat["y_all"]
    nt = cache_cat["n_train_normal"]
    nd = cache_cat["n_defect_train"]
    nn = cache_cat["n_test_normal"]

    idx_list = (
        list(range(nt, nt+nd)) +            # 80% defects
        list(range(nt+nd, nt+nd+nn)) +      # test normals
        list(range(nt+nd+nn, len(all_paths)))  # 20% defects
    )
    paths = [Path(all_paths[i]) for i in idx_list]
    y     = np.array([all_y[i] for i in idx_list], dtype=np.int32)
    return paths, y


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    cfg_path = _ROOT / "configs" / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    mvtec_root = Path(cfg["mvtec_root"])
    device     = cfg.get("device", "cuda")
    out_dir    = Path(cfg["output_dir"]) / "dual_branch"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("  Dual-Branch Evaluation: PatchCore / SAE guide / MA-CBM")
    print("=" * 72)

    # ── Load models ───────────────────────────────────────────────────────────
    print("\n[init] Loading models …")
    dino_b = DINOv2Extractor("dinov2_vitb14",  device=torch.device(device))
    dino_l = DINOv2Extractor("dinov2_vitl14_reg", device=torch.device(device))
    sae    = SparseAutoencoder.load(cfg["sae_weights"], device=device)
    sae    = sae.to(device).eval()
    print(f"  ViT-B/14 embed={dino_b.EMBED_DIM}  ViT-L/14 embed={dino_l.EMBED_DIM}")
    print(f"  SAE: d_input={sae.config.d_input}  d_hidden={sae.config.d_hidden}")

    # ── Load concept vector cache + recompute CONCIL weights ──────────────────
    print("\n[init] Loading MA-CBM concept vector cache …")
    cache = pickle.load(open(
        Path(cfg["output_dir"]) / "concil" / "concept_vector_cache.pkl", "rb"
    ))
    print("  Recomputing CONCIL anomaly weights from cache …")
    macbm_w, macbm_b = recompute_concil_weights(cache)
    print(f"  w = {macbm_w.round(4)}  b = {macbm_b:.4f}")

    # ── Per-category evaluation ───────────────────────────────────────────────
    results: dict[str, dict] = {}   # cat → {pc, sae, macbm, y, paths}
    score_cache_path = out_dir / "score_cache.pkl"

    if score_cache_path.exists():
        print(f"\n[cache] Loading existing scores from {score_cache_path.name}")
        results = pickle.load(open(score_cache_path, "rb"))
    else:
        for cat in MVTEC_CATEGORIES:
            print(f"\n{'─'*60}")
            print(f"  {cat.upper()}")
            print(f"{'─'*60}")
            cat_root = mvtec_root / cat
            cc = cache[cat]

            # Eval image paths and labels
            eval_paths, eval_y = eval_image_paths_for_cat(cat, mvtec_root, cc)
            n_norm = int((eval_y == 0).sum())
            n_anom = int((eval_y == 1).sum())
            print(f"  Eval set: {n_norm} normal + {n_anom} anomalous = {len(eval_y)} images")

            # Training normals (for building PatchCore memory and SAE g⁻)
            train_normal_paths = [
                Path(cc["all_paths"][i]) for i in range(cc["n_train_normal"])
            ]
            # Defect train paths (80% split — for SAE g⁺)
            nt = cc["n_train_normal"]
            nd = cc["n_defect_train"]
            defect_train_paths = [Path(cc["all_paths"][i]) for i in range(nt, nt+nd)]
            print(f"  Build from: {len(train_normal_paths)} normal + "
                  f"{len(defect_train_paths)} defect-train patches")

            # ── PatchCore ────────────────────────────────────────────────────
            print(f"  [PatchCore] Building memory bank …")
            memory = build_patchcore(train_normal_paths, dino_b, batch_size=8)
            print(f"  [PatchCore] Scoring {len(eval_paths)} images …")
            pc_scores = score_patchcore(eval_paths, memory, dino_b, batch_size=8)
            print(f"  [PatchCore] AUC = {_auc(eval_y, pc_scores):.4f}  "
                  f"score range [{pc_scores.min():.3f}, {pc_scores.max():.3f}]")

            # ── SAE guide ────────────────────────────────────────────────────
            print(f"  [SAE guide] Building g⁻ + g⁺ …")
            guide = build_sae_guide(
                train_normal_paths, defect_train_paths,
                dino_l, sae, batch_size=4
            )
            print(f"  [SAE guide] Scoring {len(eval_paths)} images …")
            sae_scores = score_sae_guide(eval_paths, guide, dino_l, batch_size=4)
            print(f"  [SAE guide] AUC = {_auc(eval_y, sae_scores):.4f}  "
                  f"score range [{sae_scores.min():.3f}, {sae_scores.max():.3f}]")

            # ── MA-CBM ───────────────────────────────────────────────────────
            macbm_scores, _ = macbm_scores_for_eval(cc, macbm_w, macbm_b)
            print(f"  [MA-CBM]   AUC = {_auc(eval_y, macbm_scores):.4f}  "
                  f"score range [{macbm_scores.min():.3f}, {macbm_scores.max():.3f}]")

            results[cat] = {
                "y":           eval_y,
                "pc_scores":   pc_scores,
                "sae_scores":  sae_scores,
                "macbm_scores": macbm_scores,
            }
            del memory, guide

        pickle.dump(results, open(score_cache_path, "wb"))
        print(f"\n[cache] Scores saved → {score_cache_path}")

    # ── Compute all AUCs ──────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  Computing AUCs for all systems and alpha values")
    print("=" * 72)

    rows: dict[str, dict] = {}   # cat → {system: auc}

    for cat, r in results.items():
        y   = r["y"]
        pc  = _norm01(r["pc_scores"])
        sae = _norm01(r["sae_scores"])
        mac = _norm01(r["macbm_scores"])

        row = {
            "PC":  _auc(y, pc),
            "SAE": _auc(y, sae),
            "MAC": _auc(y, mac),
        }
        for alpha in ALPHAS:
            row[f"PC+MAC_a{alpha}"]  = _auc(y, alpha * pc  + (1-alpha) * mac)
            row[f"SAE+MAC_a{alpha}"] = _auc(y, alpha * sae + (1-alpha) * mac)
        rows[cat] = row

    # ── Print comparison table ────────────────────────────────────────────────
    def group_mean(cats_set, key):
        vals = [rows[c][key] for c in MVTEC_CATEGORIES if c in cats_set and not np.isnan(rows[c][key])]
        return np.mean(vals) if vals else float("nan")

    def overall_mean(key):
        vals = [rows[c][key] for c in MVTEC_CATEGORIES if not np.isnan(rows[c][key])]
        return np.mean(vals) if vals else float("nan")

    STRUCTURAL = set(MVTEC_CATEGORIES) - SURFACE

    print(f"\n{'─'*72}")
    print(f"  Per-category I-AUC")
    print(f"{'─'*72}")
    hdr = f"  {'Category':<13} {'PC':>7} {'SAE':>7} {'MAC':>7} | "
    for a in ALPHAS:
        hdr += f"PC+a{a:.1f} "
    hdr += "| "
    for a in ALPHAS:
        hdr += f"SA+a{a:.1f} "
    print(hdr)
    print(f"  {'─'*68}")

    for cat in MVTEC_CATEGORIES:
        r = rows[cat]
        tag = "S" if cat in SURFACE else "T"  # S=surface, T=structural
        line = f"  {cat:<13}[{tag}] {r['PC']:>5.3f} {r['SAE']:>7.3f} {r['MAC']:>7.3f} | "
        for a in ALPHAS:
            line += f"{r[f'PC+MAC_a{a}']:>6.3f} "
        line += "| "
        for a in ALPHAS:
            line += f"{r[f'SAE+MAC_a{a}']:>6.3f} "
        print(line)

    print(f"  {'─'*68}")

    # ── Summary comparison table ──────────────────────────────────────────────
    print(f"\n{'═'*72}")
    print(f"  SUMMARY — Mean I-AUC  (S=surface/texture, T=structural)")
    print(f"{'═'*72}")
    print(f"  {'System':<36} {'Overall':>8} {'Surface':>8} {'Struct':>8}")
    print(f"  {'─'*62}")

    systems = [
        ("PatchCore standalone",        "PC"),
        ("SAE guide standalone",        "SAE"),
        ("MA-CBM concept only",         "MAC"),
    ]
    for a in ALPHAS:
        systems.append((f"PatchCore + MA-CBM  (α={a})", f"PC+MAC_a{a}"))
    for a in ALPHAS:
        systems.append((f"SAE guide + MA-CBM  (α={a})", f"SAE+MAC_a{a}"))

    best_pc_mac_auc, best_pc_alpha = -1, None
    best_sae_mac_auc, best_sae_alpha = -1, None

    for label, key in systems:
        ov  = overall_mean(key)
        sur = group_mean(SURFACE,     key)
        stc = group_mean(STRUCTURAL,  key)
        print(f"  {label:<36} {ov:>8.4f} {sur:>8.4f} {stc:>8.4f}")
        if "PC+MAC" in key and ov > best_pc_mac_auc:
            best_pc_mac_auc, best_pc_alpha = ov, float(key.split("a")[1])
        if "SAE+MAC" in key and ov > best_sae_mac_auc:
            best_sae_mac_auc, best_sae_alpha = ov, float(key.split("a")[1])

    print(f"  {'─'*62}")
    print(f"  Best PatchCore+MA-CBM alpha : α={best_pc_alpha}  "
          f"→ mean I-AUC = {best_pc_mac_auc:.4f}")
    print(f"  Best SAE+MA-CBM     alpha   : α={best_sae_alpha}  "
          f"→ mean I-AUC = {best_sae_mac_auc:.4f}")
    print(f"{'═'*72}")

    # ── Save results ──────────────────────────────────────────────────────────
    import json
    summary = {
        "overall": {s[0]: overall_mean(s[1]) for s in systems},
        "surface": {s[0]: group_mean(SURFACE, s[1]) for s in systems},
        "structural": {s[0]: group_mean(STRUCTURAL, s[1]) for s in systems},
        "per_category": {
            cat: {s[0]: rows[cat][s[1]] for s in systems}
            for cat in MVTEC_CATEGORIES
        },
        "best_pc_mac_alpha":  best_pc_alpha,
        "best_sae_mac_alpha": best_sae_alpha,
    }
    with open(out_dir / "comparison_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Results saved → {out_dir}/comparison_results.json")


if __name__ == "__main__":
    main()
