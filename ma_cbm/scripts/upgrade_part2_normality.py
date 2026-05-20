"""
upgrade_part2_normality.py — Discover and train normality concept heads.

Part 2A — Patch-level normality atoms (SAE space):
  - Select top 50 SAE atoms by mean_activation × std_across_categories on normal patches.
  - VLM autointerpretability with a normality-specific prompt.
  - Consolidate to 5-8 normality patch concepts.
  - Train LogReg heads: positive=normal patches, negative=anomaly patches.

Part 2B — Image-level normality clusters (DINOv2 max-pool space):
  - Select high-variance DINOv2 dimensions (distinguishing per-category normals).
  - K-means K=10 on all normal images.
  - VLM on full normal image grids.
  - Train LogReg heads: positive=normal images in cluster, negative=all anomaly images.
"""

from __future__ import annotations

import io, json, pickle, re, sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

_COVAD  = Path(__file__).resolve().parents[2]
_MABCM  = _COVAD / "ma_cbm"
_OUT    = _MABCM / "outputs"
sys.path.insert(0, str(_COVAD))
sys.path.insert(0, str(_COVAD / "mac"))

from features.dinov2_extractor import DINOv2Extractor
from features.sae import SparseAutoencoder

# ── Paths ─────────────────────────────────────────────────────────────────────
_NORMAL_TENSOR = _COVAD / "sae_training/mvtec_normal_patches_vitl14reg.pt"
_NORMAL_INDEX  = _COVAD / "sae_training/mvtec_patch_index_reg.pt"
_SAE_WEIGHTS   = _COVAD / "sae_training/sae_vitl14reg_C4096_k64.pt"
_MP_CACHE      = _OUT / "allcat_maxpool_cache.pkl"
_MVTEC_ROOT    = Path("/home/sobhan_hosseini/datasets/mvtec")

_NORM_PATCH_VLM   = _OUT / "norm_patch_vlm_results.json"
_NORM_IMG_VLM     = _OUT / "norm_image_vlm_results.json"
_NORM_PATCH_HEADS = _OUT / "norm_patch_concept_heads.pkl"
_NORM_IMG_HEADS   = _OUT / "norm_image_concept_heads.pkl"
_NORM_PATCH_GRIDS = _OUT / "norm_patch_grids"
_NORM_IMG_GRIDS   = _OUT / "norm_image_grids"
for d in (_NORM_PATCH_GRIDS, _NORM_IMG_GRIDS):
    d.mkdir(parents=True, exist_ok=True)

_DEVICE   = "cuda"
_BATCH    = 8
_CELL_PX  = 224
_OLLAMA   = "http://localhost:6000"
_VLM_MDL  = "gemma4:e4b"
_TOP_ATOMS = 50
_K_NORM_IMG = 10

MVTEC_15 = [
    "bottle","cable","capsule","carpet","grid","hazelnut","leather",
    "metal_nut","pill","screw","tile","toothbrush","transistor","wood","zipper",
]
_PATCH_PX = 14   # DINOv2 patch size in pixels at 224-px input
_PATCHES_PER_SIDE = 16

_PROMPT_PATCH = """\
You are analyzing patches from industrial objects that are in PERFECT \
CONDITION — no defects, no anomalies. All 9 patches strongly activate \
the same visual feature detector in a neural network.

Your task: identify what visual property of NORMAL, INTACT industrial \
surfaces is common across these patches.

Step 1: Describe each patch briefly.
Step 2: What visual property of NORMAL appearance is consistently \
present? Focus on what makes these patches look correct and intact: \
texture regularity, color consistency, surface smoothness, grain \
uniformity, etc. Do NOT describe defects.
Step 3: Name this normal visual property.

Output format:
COMMON_PATTERN: [one sentence about normal appearance]
CONCEPT_NAME: [lowercase_underscores]
CONFIDENCE: [high / medium / low]\
"""

_PROMPT_IMG = """\
You are analyzing industrial objects that are COMPLETELY NORMAL — \
perfectly manufactured, no defects whatsoever.

These 9 images all show the same type of normal industrial object \
appearance. They all activate the same global visual pattern detector.

Step 1: Describe what you see briefly.
Step 2: What makes these objects look NORMAL and INTACT at a global \
level? Consider: overall shape completeness, component arrangement, \
structural integrity, expected geometry.
Step 3: Name this normal global property.

Output format:
COMMON_PATTERN: [one sentence]
CONCEPT_NAME: [lowercase_underscores]
CONFIDENCE: [high / medium / low]\
"""


# ── VLM helpers ───────────────────────────────────────────────────────────────

def _pil_bytes(img):
    buf = io.BytesIO();  img.save(buf, format="PNG");  return buf.getvalue()

def _parse(text):
    def _g(tag):
        m = re.search(rf"{tag}:\s*(.+)", text, re.IGNORECASE)
        return m.group(1).strip() if m else "PARSE_ERROR"
    return {"concept_name": _g("CONCEPT_NAME"), "common_pattern": _g("COMMON_PATTERN"),
            "confidence": _g("CONFIDENCE").lower(), "raw": text}

def _query_vlm(grid, prompt):
    from ollama import Client
    try:
        r = Client(host=_OLLAMA).chat(model=_VLM_MDL, messages=[{
            "role":"user","content":prompt,"images":[_pil_bytes(grid)]}])
        return _parse(r["message"]["content"])
    except Exception as e:
        return {"concept_name":"VLM_ERROR","common_pattern":str(e),"confidence":"low","raw":str(e)}

def _build_grid_from_paths(paths, cell=_CELL_PX):
    grid = Image.new("RGB",(3*cell,3*cell),(128,128,128))
    for i,p in enumerate(paths[:9]):
        img = Image.open(p).convert("RGB").resize((cell,cell),Image.LANCZOS)
        r,c = divmod(i,3)
        grid.paste(img,(c*cell,r*cell))
    return grid

def _build_grid_from_crops(crops: list, cell=_CELL_PX):
    """crops: list of PIL images (14×14 patch crops)."""
    grid = Image.new("RGB",(3*cell,3*cell),(128,128,128))
    for i,crop in enumerate(crops[:9]):
        img = crop.resize((cell,cell),Image.LANCZOS)
        r,c = divmod(i,3)
        grid.paste(img,(c*cell,r*cell))
    return grid


# ── Part 2A: Patch-level normality ────────────────────────────────────────────

def _compute_atom_stats(sae, normal_tensor, index):
    """Per-category mean SAE activation for each atom.

    Returns: (15, 4096) float32 array — mean activation per category per atom.
    """
    C = sae.config.d_hidden   # 4096
    n_cats = len(index)
    cat_means = np.zeros((n_cats, C), dtype=np.float32)

    sae_dev = next(sae.parameters()).device
    CHUNK = 8192

    for ci, entry in enumerate(index):
        cat = entry["category"]
        print(f"  [{cat}] encoding {entry['n_patches']:,} patches …", flush=True)
        raw = normal_tensor[entry["row_start"]:entry["row_end"]]  # (N, 1024) CPU
        acts_list = []
        with torch.no_grad():
            for start in range(0, raw.shape[0], CHUNK):
                batch = raw[start:start+CHUNK].to(sae_dev)
                z = sae.encode(batch)                 # (b, 4096)
                acts_list.append(z.cpu().float().numpy())
        cat_acts = np.concatenate(acts_list, axis=0)  # (N, 4096)
        cat_means[ci] = cat_acts.mean(axis=0)
        del cat_acts, acts_list

    return cat_means   # (15, 4096)


@torch.no_grad()
def _extract_normal_patch_samples(dino, sae, mp_cache, n_per_cat=30, seed=42):
    """Extract per-image patch tokens + SAE codes for sampled normal images.

    Returns: list of dicts {path, img_224, sae_codes: (256, 4096), patches: (256, 1024)}
    """
    rng = np.random.RandomState(seed)
    results = []
    for cat in MVTEC_15:
        mp = mp_cache[cat]
        y  = np.array(mp["y_all"])
        normal_paths = [mp["all_paths"][i] for i in np.where(y==0)[0]]
        sample = rng.permutation(len(normal_paths))[:n_per_cat]
        chosen = [normal_paths[i] for i in sample]

        for start in range(0, len(chosen), _BATCH):
            batch_paths = chosen[start:start+_BATCH]
            imgs = [Image.open(p).convert("RGB") for p in batch_paths]
            x    = dino._prepare(imgs)
            _, patches = dino._run(x)              # (b, 256, 1024)
            b = patches.shape[0]
            flat = patches.reshape(b*256, 1024)
            codes = sae.encode(flat.to(sae.b_dec.device)).detach().cpu().float()
            codes = codes.reshape(b, 256, 4096).numpy()
            patches_np = patches.detach().cpu().float().numpy()

            for bi, (path, img) in enumerate(zip(batch_paths, imgs)):
                # Build 224px unnorm image for crop extraction
                from torchvision import transforms
                spatial = transforms.Compose([
                    transforms.Resize(224),
                    transforms.CenterCrop(224),
                ])
                img_224 = spatial(img)
                results.append({"path": path, "img_224": img_224,
                                 "sae_codes": codes[bi],    # (256, 4096)
                                 "patches": patches_np[bi]}) # (256, 1024)
        print(f"  [{cat}] sampled {min(len(chosen), n_per_cat)} images", end="\r")
    print()
    return results


def run_part2a(sae, dino, mp_cache):
    print("\n" + "="*70)
    print("  Part 2A — Patch-level normality concept discovery")
    print("="*70)

    # Load precomputed normal patch tensor
    print("\n[2A-1] Computing per-category per-atom mean activations …")
    normal_tensor = torch.load(_NORMAL_TENSOR, map_location="cpu", weights_only=False)
    index = torch.load(_NORMAL_INDEX, map_location="cpu", weights_only=False)
    cat_means = _compute_atom_stats(sae, normal_tensor, index)   # (15, 4096)
    del normal_tensor

    # Score: mean × cross-category std
    overall_mean = cat_means.mean(axis=0)         # (4096,)
    cat_std      = cat_means.std(axis=0)          # (4096,) std across categories
    score        = overall_mean * cat_std          # (4096,) — high = activated everywhere but differently
    top_atoms    = np.argsort(score)[::-1][:_TOP_ATOMS]

    print(f"\n  Top {_TOP_ATOMS} normality atoms selected.")
    print(f"  Score range: {score[top_atoms[-1]]:.4f} – {score[top_atoms[0]]:.4f}")

    # Sample normal images for patch visualization
    print(f"\n[2A-2] Sampling normal images for patch visualization …")
    norm_samples = _extract_normal_patch_samples(dino, sae, mp_cache, n_per_cat=25)
    print(f"  Sampled {len(norm_samples)} normal images total")

    # For each atom: find top-9 patches across all sampled images
    print(f"\n[2A-3] Building patch grids and querying VLM …")

    # Load partial results if any
    if _NORM_PATCH_VLM.exists():
        norm_patch_vlm = json.load(open(_NORM_PATCH_VLM))
        done_atoms = {r["atom_id"] for r in norm_patch_vlm}
        print(f"  Loaded {len(norm_patch_vlm)} cached results")
    else:
        norm_patch_vlm, done_atoms = [], set()

    for rank, atom_id in enumerate(top_atoms):
        if int(atom_id) in done_atoms:
            continue

        # Collect (activation, image_idx, patch_idx) for this atom
        entries = []
        for img_i, sample in enumerate(norm_samples):
            patch_acts = sample["sae_codes"][:, atom_id]   # (256,)
            best_pi    = int(np.argmax(patch_acts))
            entries.append((float(patch_acts[best_pi]), img_i, best_pi))
        entries.sort(key=lambda x: -x[0])

        # Build crops from top-9
        crops = []
        for act, img_i, pi in entries[:9]:
            sample  = norm_samples[img_i]
            img_224 = sample["img_224"]
            r_p, c_p = divmod(pi, _PATCHES_PER_SIDE)
            left  = c_p * _PATCH_PX;  upper = r_p * _PATCH_PX
            crop  = img_224.crop((left, upper, left+_PATCH_PX, upper+_PATCH_PX))
            crops.append(crop)

        if not crops:
            continue

        grid = _build_grid_from_crops(crops)
        grid.save(_NORM_PATCH_GRIDS / f"atom_{atom_id}.png")

        result = _query_vlm(grid, _PROMPT_PATCH)
        result.update({"atom_id": int(atom_id), "rank": rank,
                       "score": float(score[atom_id]),
                       "mean_activation": float(overall_mean[atom_id]),
                       "std_across_cats": float(cat_std[atom_id])})
        norm_patch_vlm.append(result)
        json.dump(norm_patch_vlm, open(_NORM_PATCH_VLM,"w"), indent=2)

        conf = result["confidence"]
        print(f"  Atom {atom_id:4d} [rank {rank+1:2d}]  "
              f"conf={conf:<8}  '{result['concept_name']}'")

    # ── Consolidate patch normality vocabulary ────────────────────────────────
    print(f"\n[2A-4] Consolidating patch normality vocabulary …")
    from concepts.vocabulary_builder import build_vocabulary

    # Adapt format for vocabulary_builder (expects 'concept_name', 'confidence', 'atom_id')
    vlm_adapted = [{"concept_name": r["concept_name"],
                    "common_pattern": r.get("common_pattern",""),
                    "confidence": r["confidence"],
                    "atom_id": r["atom_id"],
                    "raw_response": r.get("raw","")} for r in norm_patch_vlm]

    # Filter to high/medium only for vocabulary
    vlm_adapted_hm = [v for v in vlm_adapted if v["confidence"] in ("high","medium")]
    print(f"  High/medium: {len(vlm_adapted_hm)}/{len(vlm_adapted)} atoms")

    norm_patch_vocab = build_vocabulary(
        vlm_results=vlm_adapted_hm,
        ollama_host=_OLLAMA,
        clip_threshold=0.75,
        save_path=_OUT/"norm_patch_vocabulary.json",
    )

    # ── Train patch normality heads ───────────────────────────────────────────
    print(f"\n[2A-5] Training patch normality concept heads …")

    # Collect all normal SAE codes (from precomputed tensor)
    normal_tensor2 = torch.load(_NORMAL_TENSOR, map_location="cpu", weights_only=False)
    # Sample 2000 normal patches for training (balanced)
    rng = np.random.RandomState(42)
    sample_idx = rng.choice(normal_tensor2.shape[0], size=min(2000, normal_tensor2.shape[0]), replace=False)
    norm_sample_raw = normal_tensor2[sample_idx]  # (2000, 1024)
    sae_dev = next(sae.parameters()).device
    with torch.no_grad():
        norm_sae_codes = sae.encode(norm_sample_raw.to(sae_dev)).cpu().float().numpy()
    del normal_tensor2, norm_sample_raw

    # Collect anomaly SAE codes (sample)
    # We'll re-use the patch-level anomaly SAE codes from the mac pipeline
    # Using the existing cross-category normal SAE codes from the training pipeline
    # And anomaly patches from the existing anomaly patches cache
    anom_pkl = _COVAD / "mac/outputs/cross_category/cross_category_anomaly_patches.pkl"
    if anom_pkl.exists():
        anom_patches = pickle.load(open(anom_pkl,"rb"))
        anom_sae_codes = np.stack([p["sae_code"].numpy() for p in anom_patches])
        anom_sae_codes = anom_sae_codes[rng.choice(len(anom_sae_codes),
                                                    size=min(2000,len(anom_sae_codes)),replace=False)]
    else:
        anom_sae_codes = np.zeros((0, 4096), dtype=np.float32)

    norm_patch_heads_dict = {}
    norm_patch_head_metrics = []
    hdr = f"  {'Concept':<40} {'Atoms':>5} {'F1':>7} {'AUC':>7}"
    print(hdr);  print("  "+"─"*60)

    for concept in norm_patch_vocab.get("concepts", []):
        name = concept["name"]
        member_ids = [int(a) for a in concept["atom_ids"]]

        # Positive: normal patch samples where any member atom fires > threshold
        acts_on_norm = norm_sae_codes[:, member_ids]   # (N_norm, n_members)
        thresholds   = np.percentile(acts_on_norm, 70, axis=0)
        pos_mask     = (acts_on_norm > thresholds).any(axis=1)

        # Negative: anomaly patch SAE codes
        X_pos = norm_sae_codes[pos_mask]
        X_neg = anom_sae_codes
        if len(X_pos) < 5 or len(X_neg) < 5:
            print(f"  {name:<40} SKIP")
            continue

        X = np.concatenate([X_pos, X_neg])
        y = np.array([1]*len(X_pos)+[0]*len(X_neg), dtype=np.int32)
        X_tr, X_te, y_tr, y_te = train_test_split(X,y,test_size=0.2,stratify=y,random_state=42)
        clf = LogisticRegression(C=1.0,class_weight="balanced",max_iter=1000,solver="lbfgs",random_state=42)
        clf.fit(X_tr, y_tr)
        y_p = clf.predict(X_te);  y_pb = clf.predict_proba(X_te)[:,1]
        f1  = float(f1_score(y_te,y_p,zero_division=0))
        auc = float(roc_auc_score(y_te,y_pb))
        norm_patch_heads_dict[name] = clf
        norm_patch_head_metrics.append({"concept":name,"f1":f1,"auc":auc,
                                         "n_atoms":len(member_ids),"n_pos":len(X_pos)})
        print(f"  {name:<40} {len(member_ids):>5} {f1:>7.3f} {auc:>7.3f}")

    pickle.dump(norm_patch_heads_dict, open(_NORM_PATCH_HEADS,"wb"))
    json.dump(norm_patch_head_metrics, open(_OUT/"norm_patch_head_metrics.json","w"), indent=2)
    print(f"\n  Normality patch heads saved → {_NORM_PATCH_HEADS.name}")
    return norm_patch_heads_dict, norm_patch_head_metrics, norm_patch_vocab


# ── Part 2B: Image-level normality ────────────────────────────────────────────

def run_part2b(mp_cache):
    print("\n" + "="*70)
    print("  Part 2B — Image-level normality concept discovery")
    print("="*70)

    # Collect all normal images' max-pool features
    norm_feats_all, norm_items_all = [], []
    cat_means_norm = {}

    for cat in MVTEC_15:
        mp = mp_cache[cat]
        y  = np.array(mp["y_all"])
        mask_norm = y == 0
        feats_cat = mp["maxpool_feats"][mask_norm]
        paths_cat = [mp["all_paths"][i] for i in np.where(mask_norm)[0]]
        cat_means_norm[cat] = feats_cat.mean(axis=0)
        for f, p in zip(feats_cat, paths_cat):
            norm_feats_all.append(f)
            norm_items_all.append({"path": p, "category": cat})

    norm_feats_all = np.stack(norm_feats_all)   # (N_norm, 1024)
    print(f"\n  Total normal images: {len(norm_items_all)}")

    # High-variance dimensions: std of per-category means
    cat_mean_matrix = np.stack(list(cat_means_norm.values()))   # (15, 1024)
    dim_var  = cat_mean_matrix.std(axis=0)                       # (1024,)
    top_dims = np.argsort(dim_var)[::-1][:100]
    print(f"  High-variance dims range: {top_dims.min()}–{top_dims.max()}")

    # K-means on normal images
    print(f"\n[2B-2] K-means (K={_K_NORM_IMG}) on normal images …")
    norm_proj = norm_feats_all[:, top_dims]         # (N_norm, 100)
    km = KMeans(n_clusters=_K_NORM_IMG, random_state=42, n_init=10)
    norm_labels = km.fit_predict(norm_proj)
    centroids   = km.cluster_centers_

    for k in range(_K_NORM_IMG):
        cats_k = sorted(set(norm_items_all[i]["category"] for i in np.where(norm_labels==k)[0]))
        print(f"  Cluster {k:2d}: {(norm_labels==k).sum():>5} images  cats: {cats_k}")

    # VLM naming
    print(f"\n[2B-3] Building image grids and querying VLM …")

    if _NORM_IMG_VLM.exists():
        norm_img_vlm = json.load(open(_NORM_IMG_VLM))
        done_ks = {r["cluster_id"] for r in norm_img_vlm}
        print(f"  Loaded {len(norm_img_vlm)} cached results")
    else:
        norm_img_vlm, done_ks = [], set()

    # Collect anomaly max-pool feats for head training
    anom_feats_all = np.concatenate([
        mp_cache[c]["maxpool_feats"][np.array(mp_cache[c]["y_all"])==1]
        for c in MVTEC_15])

    for k in range(_K_NORM_IMG):
        if k in done_ks:
            name = next(r["concept_name"] for r in norm_img_vlm if r["cluster_id"]==k)
            print(f"  Cluster {k:2d}: [cached] → '{name}'")
            continue

        mask_k = norm_labels == k
        idx_k  = np.where(mask_k)[0]
        dists  = np.linalg.norm(norm_proj[idx_k] - centroids[k], axis=1)
        top9   = idx_k[np.argsort(dists)[:9]]
        rep_paths = [norm_items_all[i]["path"] for i in top9]
        cats_k    = sorted(set(norm_items_all[i]["category"] for i in idx_k))

        grid = _build_grid_from_paths(rep_paths)
        grid.save(_NORM_IMG_GRIDS / f"cluster_{k}.png")

        print(f"  Cluster {k:2d} ({mask_k.sum()} images) — VLM …", flush=True)
        result = _query_vlm(grid, _PROMPT_IMG)
        result.update({"cluster_id": k, "cluster_size": int(mask_k.sum()),
                       "categories": cats_k, "representative_images": rep_paths})
        norm_img_vlm.append(result)
        json.dump(norm_img_vlm, open(_NORM_IMG_VLM,"w"), indent=2)
        print(f"  → [{result['confidence']}] '{result['concept_name']}'  cats: {cats_k}")

    # ── Train image-level normality heads ─────────────────────────────────────
    print(f"\n[2B-4] Training image-level normality concept heads …")
    norm_img_heads_dict   = {}
    norm_img_head_metrics = []
    hdr = f"  {'Concept':<45} {'N+':>5} {'N-':>6} {'F1':>7} {'AUC':>7}"
    print(hdr);  print("  "+"─"*70)

    for result in norm_img_vlm:
        k    = result["cluster_id"]
        name = result["concept_name"]
        if result["confidence"] not in ("high","medium"):
            print(f"  {name:<45} SKIP (low confidence)")
            continue

        mask_pos = norm_labels == k
        X_pos = norm_feats_all[mask_pos]
        X_neg = anom_feats_all
        X = np.concatenate([X_pos, X_neg])
        y = np.array([1]*len(X_pos)+[0]*len(X_neg), dtype=np.int32)

        if len(X_pos) < 5:
            print(f"  {name:<45} SKIP (too few positives)")
            continue

        X_tr, X_te, y_tr, y_te = train_test_split(X,y,test_size=0.2,stratify=y,random_state=42)
        clf = LogisticRegression(C=1.0,class_weight="balanced",max_iter=1000,
                                 solver="lbfgs",random_state=42)
        clf.fit(X_tr, y_tr)
        y_p = clf.predict(X_te);  y_pb = clf.predict_proba(X_te)[:,1]
        f1  = float(f1_score(y_te, y_p, zero_division=0))
        auc = float(roc_auc_score(y_te, y_pb))
        norm_img_heads_dict[name] = clf
        norm_img_head_metrics.append({"concept":name,"cluster_id":k,"f1":f1,"auc":auc,
                                       "n_pos":len(X_pos),"categories":result["categories"]})
        print(f"  {name:<45} {len(X_pos):>5} {len(X_neg):>6} {f1:>7.3f} {auc:>7.3f}")

    pickle.dump(norm_img_heads_dict, open(_NORM_IMG_HEADS,"wb"))
    json.dump(norm_img_head_metrics, open(_OUT/"norm_image_head_metrics.json","w"), indent=2)
    print(f"\n  Normality image heads saved → {_NORM_IMG_HEADS.name}")
    return norm_img_heads_dict, norm_img_head_metrics, norm_img_vlm


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  Part 2 — Normality Concept Discovery (patch + image level)")
    print("=" * 70)

    print("\n[init] Loading models …")
    dino = DINOv2Extractor("dinov2_vitl14_reg", device=torch.device(_DEVICE))
    sae  = SparseAutoencoder.load(str(_SAE_WEIGHTS), device=_DEVICE)
    sae  = sae.to(_DEVICE).eval()
    print(f"  DINOv2 embed={dino.EMBED_DIM}  SAE d_hidden={sae.config.d_hidden}")

    print("[init] Loading max-pool cache …")
    mp_cache = pickle.load(open(_MP_CACHE,"rb"))

    # Part 2A
    norm_patch_heads, norm_patch_metrics, norm_patch_vocab = run_part2a(sae, dino, mp_cache)

    # Part 2B
    norm_img_heads, norm_img_metrics, norm_img_vlm = run_part2b(mp_cache)

    # Summary
    print("\n" + "="*70)
    print("  NORMALITY CONCEPTS SUMMARY")
    print("="*70)
    print(f"\n  Patch-level normality concepts: {len(norm_patch_heads)}")
    for m in sorted(norm_patch_metrics, key=lambda x: -x.get("auc",0)):
        print(f"    {m['concept']:<40}  F1={m['f1']:.3f}  AUC={m['auc']:.3f}")
    print(f"\n  Image-level normality concepts: {len(norm_img_heads)}")
    for m in sorted(norm_img_metrics, key=lambda x: -x.get("auc",0)):
        print(f"    {m['concept']:<40}  F1={m['f1']:.3f}  AUC={m['auc']:.3f}  "
              f"cats={m['categories']}")
    print("="*70)


if __name__ == "__main__":
    main()
