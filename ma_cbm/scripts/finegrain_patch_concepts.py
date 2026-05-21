"""
finegrain_patch_concepts.py — Fine-grained sub-concept discovery within each
of the 5 coarse patch-level defect concepts.

Pipeline:
  1. Assign anomaly patches to each of the 5 coarse concepts
     (patches where any member atom fires above 70th-pct threshold).
  2. Sub-cluster each coarse concept with K-means (adaptive K).
  3. Build 3×3 patch grids (14×14 → 112×112 per cell) and query VLM
     with a fine-graining prompt that asks for sub-patterns WITHIN the
     parent concept.
  4. Consolidate: nomic-embed-text dedup across all new concept names,
     including cross-parent dedup.
  5. Train fine-grained LogReg heads on full 4096-dim SAE codes.
  6. Build v5 concept cache (new patch dims replace dims 0-4).
  7. Run 15-task CONCIL and report I-AUC + BWT.
  8. Re-run C-AUC against CONVAD annotations.

All intermediate results saved after each VLM call.
"""

from __future__ import annotations

import io, json, math, pickle, re, sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

_COVAD = Path(__file__).resolve().parents[2]
_MABCM = _COVAD / "ma_cbm"
_OUT   = _MABCM / "outputs" / "fine_grained"
_OUT.mkdir(parents=True, exist_ok=True)
(_OUT / "grids").mkdir(exist_ok=True)

sys.path.insert(0, str(_COVAD))
sys.path.insert(0, str(_COVAD / "mac"))

# ── Paths ─────────────────────────────────────────────────────────────────────
_VOCAB5       = _COVAD / "mac/outputs/cross_category/cross_category_vocabulary_5concept.json"
_ATOMS_JSON   = _COVAD / "mac/outputs/cross_category/cross_category_relevant_atoms.json"
_PATCHES_PKL  = _COVAD / "mac/outputs/cross_category/cross_category_anomaly_patches.pkl"
_NORM_PATCHES = _COVAD / "sae_training/mvtec_normal_patches_vitl14reg.pt"
_NORM_INDEX   = _COVAD / "sae_training/mvtec_patch_index_reg.pt"
_SAE_W        = _COVAD / "sae_training/sae_vitl14reg_C4096_k64.pt"
_V2_CACHE     = _MABCM / "outputs/v2_concept_cache.pkl"
_MP_CACHE     = _MABCM / "outputs/allcat_maxpool_cache.pkl"
_IMG_HEADS    = _MABCM / "outputs/allcat_image_level_heads_v2.pkl"
_NOPAT_HEADS  = _MABCM / "outputs/norm_patch_concept_heads.pkl"
_NOIMG_HEADS  = _MABCM / "outputs/norm_image_concept_heads.pkl"
_VLM_V2       = _MABCM / "outputs/image_level_vlm_results_allcat_v2.json"
_SAE_SC       = _COVAD / "mac/outputs/dual_branch/score_cache.pkl"
_ANN_DIR      = Path("/home/sobhan_hosseini/cbm_data/mvtec")
_MVTEC        = Path("/home/sobhan_hosseini/datasets/mvtec")

_OLLAMA  = "http://localhost:6000"
_VLM_MDL = "gemma4:e4b"
_CELL_PX = 112   # 14px patch × 8 upscale
_GRID_PX = 3 * _CELL_PX   # 336×336

MVTEC_15 = ["bottle","cable","capsule","carpet","grid","hazelnut","leather",
             "metal_nut","pill","screw","tile","toothbrush","transistor","wood","zipper"]

# Adaptive K per coarse concept (based on patch count ≥100 → K=3, else K=2)
_K_MAP = {"surface_discontinuity":3, "surface_discoloration":3,
          "surface_crack":3, "surface_abrasion":3, "surface_void":2}

_VLM_PROMPT_TEMPLATE = """\
You are analyzing patches from industrial quality inspection images.

Each patch in this 3x3 grid comes from a confirmed defective region \
of an industrial object. All 9 patches strongly activate the same \
internal visual feature detector in a neural network.

These patches are a SUBSET of the broader defect category \
called "{parent}". Your task is to identify what MORE SPECIFIC \
visual anomaly pattern distinguishes these patches WITHIN this \
broader category.

Step 1 - Describe each patch briefly (1 sentence each).

Step 2 - What specific visual sub-pattern is common to these patches \
that distinguishes them from other patches in the same broad \
category "{parent}"? Be more specific than "{parent}".

Step 3 - Give a short specific sub-concept name (2-5 words).

Output format:
COMMON_PATTERN: [specific description of the sub-pattern]
CONCEPT_NAME: [lowercase_underscores]
CONFIDENCE: [high / medium / low]
REASON_FOR_LOW_CONFIDENCE: [only if confidence is low, else none]\
"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _pil_bytes(img):
    buf = io.BytesIO(); img.save(buf, format="PNG"); return buf.getvalue()

def _parse_vlm(text):
    def _g(tag):
        m = re.search(rf"{tag}:\s*(.+)", text, re.IGNORECASE)
        return m.group(1).strip() if m else "PARSE_ERROR"
    return {"concept_name": _g("CONCEPT_NAME"), "common_pattern": _g("COMMON_PATTERN"),
            "confidence": _g("CONFIDENCE").lower(), "raw": text}

def _query_vlm(grid, parent_name):
    from ollama import Client
    prompt = _VLM_PROMPT_TEMPLATE.format(parent=parent_name)
    try:
        r = Client(host=_OLLAMA).chat(model=_VLM_MDL, messages=[{
            "role":"user","content":prompt,"images":[_pil_bytes(grid)]}])
        return _parse_vlm(r["message"]["content"])
    except Exception as e:
        return {"concept_name":"VLM_ERROR","common_pattern":str(e),
                "confidence":"low","raw":str(e)}

def _build_patch_grid(patch_images: list) -> Image.Image:
    """Build 3×3 grid from up to 9 PIL patch images (14×14 → 112×112 each)."""
    grid = Image.new("RGB", (_GRID_PX, _GRID_PX), (128,128,128))
    for i, img in enumerate(patch_images[:9]):
        up = img.resize((_CELL_PX, _CELL_PX), Image.LANCZOS)
        r, c = divmod(i, 3)
        grid.paste(up, (c*_CELL_PX, r*_CELL_PX))
    return grid

def _assign_patches(patches, member_atom_ids, all_codes, pos_percentile=70.0):
    """Return boolean mask of patches assigned to this concept.

    Positive: any member atom fires above its pos_percentile threshold.
    """
    acts = all_codes[:, member_atom_ids].float()         # (N, n_members)
    thresholds = torch.quantile(acts, pos_percentile/100.0, dim=0)
    return (acts > thresholds.unsqueeze(0)).any(dim=1).numpy()


# ── Step 1: Load data ─────────────────────────────────────────────────────────

def load_data():
    print("[load] Reading 5-concept vocabulary …")
    vocab5 = json.load(open(_VOCAB5))

    print("[load] Reading anomaly patches …")
    patches = pickle.load(open(_PATCHES_PKL, "rb"))
    all_codes = torch.stack([p["sae_code"] for p in patches]).float()
    print(f"  {len(patches)} patches, SAE codes: {all_codes.shape}")
    return vocab5, patches, all_codes


# ── Step 2: Sub-cluster each coarse concept ───────────────────────────────────

def subcluster_concept(name, member_ids, patches, all_codes, K, svd_n=128):
    """Assign patches → PCA → K-means → return list of sub-cluster dicts."""
    mask = _assign_patches(patches, member_ids, all_codes)
    idx  = np.where(mask)[0]
    print(f"  {name}: {len(idx)} assigned patches → K-means K={K}")

    codes_sub = all_codes[idx].numpy()      # (M, 4096)
    codes_norm = normalize(codes_sub, norm="l2")    # L2-normalise sparse codes
    svd = TruncatedSVD(n_components=min(svd_n, codes_norm.shape[1]-1), random_state=42)
    codes_red = svd.fit_transform(codes_norm)       # (M, 128)

    km = MiniBatchKMeans(n_clusters=K, random_state=42, n_init=5, max_iter=300)
    labels = km.fit_predict(codes_red)
    centroids = km.cluster_centers_                 # (K, 128)

    sub_clusters = []
    for k in range(K):
        sub_mask = labels == k
        sub_idx  = idx[sub_mask]
        if sub_mask.sum() < 5:
            print(f"    sub-cluster {k}: too small ({sub_mask.sum()}) — skip")
            continue
        # 9 closest to centroid in PCA space
        dists = np.linalg.norm(codes_red[sub_mask] - centroids[k], axis=1)
        top9  = sub_idx[np.argsort(dists)[:9]]
        rep_patches = [patches[i] for i in top9]
        sub_clusters.append({
            "parent": name, "k": k, "n": int(sub_mask.sum()),
            "rep_global_idx": top9.tolist(),
            "all_global_idx": sub_idx.tolist(),
        })
    return sub_clusters


# ── Step 3: VLM naming ────────────────────────────────────────────────────────

def name_subclusters(sub_clusters, patches, saved_path: Path):
    """Query VLM for each sub-cluster. Saves incrementally."""
    if saved_path.exists():
        results = json.load(open(saved_path))
        done_keys = {(r["parent"], r["k"]) for r in results}
        print(f"  Loaded {len(results)} cached VLM results")
    else:
        results, done_keys = [], set()

    for sc in sub_clusters:
        key = (sc["parent"], sc["k"])
        if key in done_keys:
            r = next(r for r in results if r["parent"]==sc["parent"] and r["k"]==sc["k"])
            print(f"  [{sc['parent']} k={sc['k']}] cached → '{r['concept_name']}' [{r['confidence']}]")
            continue

        # Build grid
        rep_patches = [patches[i] for i in sc["rep_global_idx"]]
        imgs = [p["patch_image"] for p in rep_patches]
        grid = _build_patch_grid(imgs)
        grid_path = _OUT / "grids" / f"{sc['parent']}_{sc['k']}.png"
        grid.save(grid_path)

        print(f"  [{sc['parent']} k={sc['k']} n={sc['n']}] VLM …", flush=True)
        vlm = _query_vlm(grid, sc["parent"])
        vlm.update({"parent": sc["parent"], "k": sc["k"], "n": sc["n"],
                    "all_global_idx": sc["all_global_idx"],
                    "rep_global_idx": sc["rep_global_idx"]})
        results.append(vlm)
        json.dump(results, open(saved_path, "w"), indent=2)
        conf = vlm["confidence"]
        print(f"    → [{conf}] '{vlm['concept_name']}'")

    return results


# ── Step 4: Consolidate vocabulary ────────────────────────────────────────────

def consolidate(vlm_results, threshold=0.75, min_confidence={"high","medium"}):
    """Nomic dedup across all fine-grained names. Returns kept results."""
    # Keep only high/medium confidence
    kept = [r for r in vlm_results if r["confidence"] in min_confidence
            and r["concept_name"] not in {"VLM_ERROR","PARSE_ERROR"}]
    dropped = [r for r in vlm_results if r not in kept]
    if dropped:
        print(f"  Dropped {len(dropped)} low-confidence sub-concepts:")
        for r in dropped:
            print(f"    {r['parent']} k={r['k']} → '{r['concept_name']}' [{r['confidence']}]")

    if len(kept) <= 1:
        return kept, {}

    names = [r["concept_name"] for r in kept]
    print(f"  Running nomic dedup on {len(names)} concept names …")
    try:
        from ollama import Client
        client = Client(host=_OLLAMA)
        embs = []
        for n in names:
            resp = client.embeddings(model="nomic-embed-text", prompt=n.replace("_"," "))
            embs.append(resp["embedding"])
        t = torch.tensor(embs, dtype=torch.float32)
        t = t / t.norm(dim=-1, keepdim=True)
        sim = (t @ t.T).numpy()
    except Exception as e:
        print(f"  [WARN] nomic failed: {e} — no dedup")
        return kept, {}

    merged_into = {}
    for i in range(len(kept)):
        for j in range(i+1, len(kept)):
            if i in merged_into or j in merged_into:
                continue
            if sim[i,j] > threshold:
                # Merge j into i (keep the one with more patches)
                if kept[j]["n"] > kept[i]["n"]:
                    merged_into[i] = j
                    print(f"  MERGE '{kept[i]['concept_name']}' → '{kept[j]['concept_name']}' (sim={sim[i,j]:.3f})")
                else:
                    merged_into[j] = i
                    print(f"  MERGE '{kept[j]['concept_name']}' → '{kept[i]['concept_name']}' (sim={sim[i,j]:.3f})")

    final = [r for idx, r in enumerate(kept) if idx not in merged_into]
    return final, merged_into


# ── Step 5: Train fine-grained heads ─────────────────────────────────────────

def train_finegrained_heads(final_results, patches, all_codes, norm_sae_codes):
    """Train one LogReg head per fine-grained concept.

    Positive: patches in this sub-cluster.
    Negative: all normal patches + patches from all other fine-grained concepts.
    """
    # Build anom index sets per concept
    name_to_idx = {}
    for r in final_results:
        name = r["concept_name"]
        idxs = r["all_global_idx"]
        if name not in name_to_idx:
            name_to_idx[name] = set()
        name_to_idx[name].update(idxs)

    all_anom_idx = set().union(*name_to_idx.values())
    all_codes_np = all_codes.numpy()

    heads_dict   = {}
    head_metrics = []
    print(f"\n  {'Concept':<45} {'N+':>6} {'N-':>7} {'F1':>7} {'AUC':>7}")
    print("  " + "─"*70)

    for r in final_results:
        name = r["concept_name"]
        pos_idx = list(name_to_idx[name])
        neg_idx = list(all_anom_idx - name_to_idx[name])

        X_pos = all_codes_np[pos_idx]
        X_neg_anom = all_codes_np[neg_idx]
        X = np.concatenate([X_pos, X_neg_anom, norm_sae_codes])
        y = np.array([1]*len(X_pos) + [0]*(len(X_neg_anom)+len(norm_sae_codes)), dtype=np.int32)

        if len(X_pos) < 10:
            print(f"  {name:<45} SKIP — {len(X_pos)} positives")
            continue

        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000,
                                 solver="saga", n_jobs=-1, random_state=42)
        clf.fit(X_tr, y_tr)
        y_p = clf.predict(X_te); y_pb = clf.predict_proba(X_te)[:, 1]
        f1  = float(f1_score(y_te, y_p, zero_division=0))
        auc = float(roc_auc_score(y_te, y_pb))

        heads_dict[name] = {"clf": clf, "parent": r["parent"],
                             "n_pos": len(X_pos), "f1": f1, "auc": auc}
        head_metrics.append({"concept": name, "parent": r["parent"],
                              "n_pos": len(X_pos), "f1": f1, "auc": auc})
        flag = " ⚠ F1<0.70" if f1 < 0.70 else ""
        print(f"  {name:<45} {len(X_pos):>6} {len(X_neg_anom)+len(norm_sae_codes):>7} "
              f"{f1:>7.3f} {auc:>7.3f}{flag}")

    return heads_dict, head_metrics


# ── Step 6: Build v5 concept cache ────────────────────────────────────────────

def build_v5_cache(heads_dict, v2_cache, mp_cache, sae, dino,
                   img_head_names, img_heads, nopat_heads, nopat_names,
                   noimg_heads, noimg_names, batch_size=4):
    """Build per-image concept vectors with fine-grained patch dims.

    Dims 0..N-1     : N fine-grained patch defect dims (replaces original 5)
    Dims N..N+12    : 13 image defect dims (unchanged)
    Dims N+13..N+24 : 12 patch normality dims (unchanged)
    Dims N+25..N+34 : 10 image normality dims (unchanged)
    """
    from features.sae import SparseAutoencoder
    from features.dinov2_extractor import DINOv2Extractor

    fg_names = list(heads_dict.keys())
    N_fg     = len(fg_names)
    N_img    = len(img_head_names)       # 13
    N_npat   = len(nopat_names)          # 12
    N_nimg   = len(noimg_names)          # 10
    N_CONC   = N_fg + N_img + N_npat + N_nimg
    print(f"\n[v5 cache] Building {N_CONC}-dim concept vectors "
          f"({N_fg} fg-patch + {N_img} img-def + {N_npat} norm-patch + {N_nimg} norm-img)")

    sae_dev = next(sae.parameters()).device
    cache_v5 = {}

    for cat in MVTEC_15:
        v2 = v2_cache[cat]
        mp = mp_cache[cat]
        N  = len(v2["all_paths"])

        # Dims 5-17 (image defect) and 30-39 (norm-image): from mp cache
        mp_feats = mp["maxpool_feats"].astype(np.float32)
        img_vecs  = np.zeros((N, N_img),  dtype=np.float32)
        nimg_vecs = np.zeros((N, N_nimg), dtype=np.float32)
        for k, name in enumerate(img_head_names):
            clf = img_heads.get(name)
            if clf: img_vecs[:, k] = clf.predict_proba(mp_feats)[:, 1]
        for k, name in enumerate(noimg_names):
            clf = noimg_heads.get(name)
            if clf: nimg_vecs[:, k] = clf.predict_proba(mp_feats)[:, 1]

        # Dims 0..N_fg-1 (fine-grained patch defect) and N_fg+13..N_fg+24 (norm-patch):
        # Need fresh DINOv2+SAE inference
        fg_vecs   = np.zeros((N, N_fg),   dtype=np.float32)
        npat_vecs = np.zeros((N, N_npat), dtype=np.float32)

        print(f"  [{cat}] SAE inference for {N} images …", end="", flush=True)
        for start in range(0, N, batch_size):
            batch_paths = v2["all_paths"][start:start+batch_size]
            imgs = [Image.open(p).convert("RGB") for p in batch_paths]
            with torch.no_grad():
                x = dino._prepare(imgs)
                _, patches = dino._run(x)                    # (b, 256, 1024)
                b = patches.shape[0]
                flat  = patches.reshape(b*256, 1024)
                codes = sae.encode(flat.to(sae_dev)).detach().cpu().float().numpy()
                codes = codes.reshape(b, 256, 4096)

            for bi in range(b):
                c256 = codes[bi]   # (256, 4096)
                for k, name in enumerate(fg_names):
                    clf = heads_dict[name]["clf"]
                    fg_vecs[start+bi, k] = clf.predict_proba(c256)[:, 1].max()
                for k, name in enumerate(nopat_names):
                    clf = nopat_heads.get(name)
                    if clf: npat_vecs[start+bi, k] = clf.predict_proba(c256)[:, 1].max()
        print(" done")

        vecs = np.concatenate([fg_vecs, img_vecs, npat_vecs, nimg_vecs], axis=1)
        cache_v5[cat] = {
            "all_paths":    v2["all_paths"],
            "y_all":        v2["y_all"],
            "is_train_mask": v2["is_train_mask"],
            "vecs":         vecs,
            "n_train_normal":  v2["n_train_normal"],
            "n_defect_train":  v2["n_defect_train"],
            "n_test_normal":   v2["n_test_normal"],
            "n_defect_test":   v2["n_defect_test"],
        }

    return cache_v5, N_CONC, N_fg


# ── Step 7: CONCIL ────────────────────────────────────────────────────────────

def run_concil(cache_v5, N_CONC):
    from solvers.concil import ConcilSolver
    from sklearn.metrics import roc_auc_score

    def _eval_set(cc):
        vecs = cc["vecs"]; y = np.array(cc["y_all"])
        nt, nd, nn = cc["n_train_normal"], cc["n_defect_train"], cc["n_test_normal"]
        ev = np.concatenate([vecs[nt:nt+nd], vecs[nt+nd:nt+nd+nn], vecs[nt+nd+nn:]])
        ey = np.concatenate([y[nt:nt+nd],   y[nt+nd:nt+nd+nn],    y[nt+nd+nn:]])
        return ev.astype(np.float32), ey.astype(np.int32)

    solver = ConcilSolver(input_dim=N_CONC, lambda_anomaly=1e-4)
    init_scores, init_y = {}, {}
    SURFACE = {"carpet","grid","hazelnut","leather","metal_nut","pill","tile","wood","zipper"}

    print(f"\n[CONCIL] Sequential 15-task experiment ({N_CONC}-dim) …\n")
    hdr = f"  {'T':<3} {'Category':<14} {'I-AUC':>8}"
    print(hdr); print("  "+"─"*28)

    for t, cat in enumerate(MVTEC_15):
        cc = cache_v5[cat]
        mask = np.array(cc["is_train_mask"])
        X_tr = cc["vecs"][mask].astype(np.float32)
        y_tr = np.array(cc["y_all"])[mask].astype(np.float32)
        w, b_arr = solver.update_anomaly_head(torch.tensor(X_tr), torch.tensor(y_tr))
        b = float(b_arr[0])
        ev, ey = _eval_set(cc)
        sc = ev @ w + b
        init_scores[cat] = sc; init_y[cat] = ey
        auc = float(roc_auc_score(ey, sc))
        print(f"  T{t+1:<2} {cat:<14} {auc:>8.4f}")

    W_f = solver._solve(solver.A_anomaly, solver.b_anomaly[:, 0], solver.lambda_a)
    w_f = W_f[:N_CONC].float().numpy(); b_f = float(W_f[N_CONC])
    final_scores = {}
    for cat in MVTEC_15:
        ev, _ = _eval_set(cache_v5[cat])
        final_scores[cat] = ev @ w_f + b_f

    bwts, v5c, v5cs, v5ct = [], [], [], []
    print(f"\n  {'Cat':<14} {'init':>7} {'final':>7} {'BWT':>8}")
    print("  " + "─"*42)
    for cat in MVTEC_15:
        y = init_y[cat]
        ai = float(roc_auc_score(y, init_scores[cat]))
        af = float(roc_auc_score(y, final_scores[cat]))
        bwt = (af - ai) if cat != MVTEC_15[-1] else float("nan")
        bwt_s = f"{bwt:>+8.4f}" if not math.isnan(bwt) else "  (last)"
        print(f"  {cat:<14} {ai:>7.4f} {af:>7.4f} {bwt_s}")
        v5c.append(af)
        if cat in SURFACE: v5cs.append(af)
        else:               v5ct.append(af)
        if not math.isnan(bwt): bwts.append(bwt)

    print(f"\n  Mean: overall={np.mean(v5c):.4f}  "
          f"surface={np.mean(v5cs):.4f}  struct={np.mean(v5ct):.4f}  "
          f"BWT={np.mean(bwts):+.4f}")
    print(f"  v4 baseline: overall=1.000  surface=1.000  struct=1.000  BWT=0.000")

    return init_scores, final_scores, init_y, np.mean(bwts)


# ── Step 8: C-AUC ─────────────────────────────────────────────────────────────

def run_cauc(cache_v5, fg_names, img_head_names, nopat_names, noimg_names, init_y,
             final_scores):
    """Rerun C-AUC with the fine-grained vocabulary."""
    from sklearn.metrics import roc_auc_score

    # Build v5 concept lookup
    lookup_v5 = {}
    for cat in MVTEC_15:
        cc = cache_v5[cat]
        for path, vec in zip(cc["all_paths"], cc["vecs"]):
            lookup_v5[path] = vec

    all_names_v5 = fg_names + img_head_names + nopat_names + noimg_names

    META = {"split","image_path","label_index","mask_path","anomaly_type"}
    def _auc(y,s):
        try:    return float(roc_auc_score(y,s))
        except: return float("nan")

    act_aucs, sup_aucs, uni_aucs = [], [], []
    print(f"\n  {'Category':<13} {'n_act':>6} {'n_sup':>6} {'act':>8} {'sup':>8} {'unified':>9}")
    print("  "+"─"*57)

    for cat in MVTEC_15:
        csv_path = _ANN_DIR / f"{cat}_dataset_automated.csv"
        if not csv_path.exists(): continue
        import pandas as pd
        df = pd.read_csv(csv_path)
        eval_df = df[df["split"]=="test"].copy()
        if (eval_df["label_index"]>0).sum() < 5: eval_df = df.copy()
        concept_cols = [c for c in df.columns if c not in META]
        local_paths  = [str(_MVTEC / p.split("mvtec/")[1]) for p in eval_df["image_path"]]
        matched = [i for i,p in enumerate(local_paths) if p in lookup_v5]
        if not matched: continue
        vecs = np.stack([lookup_v5[local_paths[i]] for i in matched])
        conv = eval_df.iloc[matched][concept_cols].values
        y    = (eval_df.iloc[matched]["label_index"].values > 0).astype(int)

        act_r, sup_r = [], []
        for j, cname in enumerate(concept_cols):
            m0 = conv[y==0,j].mean(); m1 = conv[y==1,j].mean()
            rtype = "activation" if m1 >= m0 else "suppression"
            if conv[:,j].sum() == 0: continue
            best = max((_auc(conv[:,j], vecs[:,k]) for k in range(vecs.shape[1])),
                       default=float("nan"))
            best_inv = max((_auc(conv[:,j], 1-vecs[:,k]) for k in range(vecs.shape[1])),
                           default=float("nan"))
            best = max(best, best_inv)
            weight = int(conv[:,j].sum())
            (act_r if rtype=="activation" else sup_r).append((best, weight))

        def _wm(results):
            v = [(a,w) for a,w in results if not math.isnan(a) and w>0]
            return float(np.average([a for a,_ in v], weights=[w for _,w in v])) if v else float("nan")
        a_auc = _wm(act_r); s_auc = _wm(sup_r)
        all_w = [(a,w) for a,w in act_r+sup_r if not math.isnan(a) and w>0]
        u_auc = float(np.average([a for a,_ in all_w], weights=[w for _,w in all_w])) if all_w else float("nan")
        print(f"  {cat:<13} {len(act_r):>6} {len(sup_r):>6} {a_auc:>8.4f} {s_auc:>8.4f} {u_auc:>9.4f}")
        if not math.isnan(a_auc): act_aucs.append(a_auc)
        if not math.isnan(s_auc): sup_aucs.append(s_auc)
        if not math.isnan(u_auc): uni_aucs.append(u_auc)

    m_act = float(np.mean(act_aucs)); m_sup = float(np.mean(sup_aucs))
    m_uni = float(np.mean(uni_aucs))
    print("  "+"─"*57)
    print(f"  {'MEAN':<13}             {m_act:>8.4f} {m_sup:>8.4f} {m_uni:>9.4f}")
    print(f"\n  Previous (v4): act=0.8706  sup=0.8962  unified=0.8822")
    print(f"  CONVAD reference:                              0.8600")
    return m_act, m_sup, m_uni


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print("  Fine-grained Patch Concept Discovery")
    print("=" * 72)

    vocab5, patches, all_codes = load_data()

    # Normal patches for negative training
    print("[load] Sampling normal SAE codes …")
    from features.sae import SparseAutoencoder
    from features.dinov2_extractor import DINOv2Extractor
    sae  = SparseAutoencoder.load(str(_SAE_W), device="cuda"); sae = sae.to("cuda").eval()
    dino = DINOv2Extractor("dinov2_vitl14_reg", device=torch.device("cuda"))

    normal_tensor = torch.load(_NORM_PATCHES, map_location="cpu", weights_only=False)
    rng = np.random.RandomState(42)
    n_sample = min(3000, normal_tensor.shape[0])
    idx_sample = rng.choice(normal_tensor.shape[0], n_sample, replace=False)
    with torch.no_grad():
        norm_codes = sae.encode(normal_tensor[idx_sample].to("cuda")).cpu().float().numpy()
    del normal_tensor
    print(f"  Normal SAE codes: {norm_codes.shape}")

    # ── Steps 2-3: Sub-cluster + VLM naming ──────────────────────────────────
    vlm_save = _OUT / "vlm_results.json"
    all_subclusters = []

    for concept in vocab5["concepts"]:
        name       = concept["name"]
        member_ids = [int(a) for a in concept["atom_ids"]]
        K          = _K_MAP.get(name, 3)
        print(f"\n[Step 2] {name} ({len(member_ids)} atoms) …")
        sub = subcluster_concept(name, member_ids, patches, all_codes, K)
        all_subclusters.extend(sub)

    print(f"\n[Step 3] VLM naming …")
    vlm_results = name_subclusters(all_subclusters, patches, vlm_save)

    # Print vocabulary
    print(f"\n  Parent concept → Sub-concepts:")
    for cname in [c["name"] for c in vocab5["concepts"]]:
        sub = [r for r in vlm_results if r["parent"]==cname]
        print(f"  {cname}:")
        for r in sub:
            print(f"    [{r['confidence']}] {r['concept_name']}  (n={r['n']})")

    # ── Step 4: Consolidate ───────────────────────────────────────────────────
    print(f"\n[Step 4] Consolidating vocabulary …")
    final_results, merged = consolidate(vlm_results)
    print(f"\n  Final vocabulary ({len(final_results)} fine-grained concepts):")
    for r in final_results:
        print(f"  {r['parent']:<25} → {r['concept_name']}  n={r['n']}  [{r['confidence']}]")

    json.dump(final_results, open(_OUT/"final_vocabulary.json","w"), indent=2)

    # ── Step 5: Train heads ───────────────────────────────────────────────────
    print(f"\n[Step 5] Training fine-grained concept heads …\n")
    heads_dict, head_metrics = train_finegrained_heads(
        final_results, patches, all_codes, norm_codes)
    fg_names = [r["concept_name"] for r in final_results if r["concept_name"] in heads_dict]

    clfs_only = {n: v["clf"] for n,v in heads_dict.items()}
    pickle.dump(clfs_only, open(_OUT/"finegrained_patch_heads.pkl","wb"))
    json.dump(head_metrics, open(_OUT/"finegrained_head_metrics.json","w"), indent=2)
    print(f"\n  Saved {len(clfs_only)} fine-grained heads")

    # ── Step 6: Build v5 cache ────────────────────────────────────────────────
    img_heads   = pickle.load(open(_IMG_HEADS,"rb"))
    nopat_heads = pickle.load(open(_NOPAT_HEADS,"rb"))
    noimg_heads = pickle.load(open(_NOIMG_HEADS,"rb"))
    v2_cache    = pickle.load(open(_V2_CACHE,"rb"))
    mp_cache    = pickle.load(open(_MP_CACHE,"rb"))

    vlm2 = json.load(open(_VLM_V2))
    img_head_names  = [r["concept_name"] for r in sorted(vlm2, key=lambda x: x["cluster_id"])]
    nopat_names = list(nopat_heads.keys())
    noimg_names  = list(noimg_heads.keys())

    print(f"\n[Step 6] Building v5 concept cache …")
    v5_cache_path = _OUT / "v5_concept_cache.pkl"
    if v5_cache_path.exists():
        print(f"  Loading from {v5_cache_path.name} …")
        cache_v5 = pickle.load(open(v5_cache_path,"rb"))
        N_CONC = next(iter(cache_v5.values()))["vecs"].shape[1]
        N_fg   = len(fg_names)
        print(f"  {N_CONC}-dim concept vectors")
    else:
        cache_v5, N_CONC, N_fg = build_v5_cache(
            heads_dict, v2_cache, mp_cache, sae, dino,
            img_head_names, img_heads, nopat_heads, nopat_names,
            noimg_heads, noimg_names)
        pickle.dump(cache_v5, open(v5_cache_path,"wb"))
        print(f"  Saved → {v5_cache_path.name}")

    # ── Step 7: CONCIL ────────────────────────────────────────────────────────
    print(f"\n[Step 7] CONCIL sequential experiment …")
    init_scores, final_scores, init_y, mean_bwt = run_concil(cache_v5, N_CONC)

    # ── Step 8: C-AUC ─────────────────────────────────────────────────────────
    print(f"\n[Step 8] C-AUC measurement …")
    m_act, m_sup, m_uni = run_cauc(
        cache_v5, fg_names, img_head_names, nopat_names, noimg_names,
        init_y, final_scores)

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  FINAL SUMMARY")
    print(f"{'='*72}")
    print(f"  Fine-grained patch concepts:  {N_fg} (replaced 5 coarse)")
    print(f"  Total concept dimensions:     {N_CONC}")
    print(f"  I-AUC (concept-only mean):    (see per-category above)")
    print(f"  BWT:                          {mean_bwt:+.4f}")
    print(f"  Unified C-AUC:                {m_uni:.4f}")
    print(f"  Previous v4 unified C-AUC:    0.8822")
    print(f"  CONVAD reference:             0.8600")
    delta = m_uni - 0.8822
    print(f"  Improvement vs v4:            {delta:+.4f}")
    print("=" * 72)

    json.dump({"n_fg_concepts":N_fg,"n_total_dims":N_CONC,
               "bwt":mean_bwt,"c_auc_act":m_act,"c_auc_sup":m_sup,
               "c_auc_unified":m_uni,"delta_vs_v4":delta},
              open(_OUT/"summary.json","w"), indent=2)


if __name__ == "__main__":
    main()
