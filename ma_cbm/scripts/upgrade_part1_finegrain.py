"""
upgrade_part1_finegrain.py — Generalized fine-graining of low-confidence concepts.

Scans ALL VLM results (image-level) for low/medium confidence.
For each low-confidence concept:
  1. Sub-cluster its images with K=2. If any result is still low: try K=3.
  2. Query VLM for each sub-cluster.
  3. If both sub-clusters are high/medium: replace original with split.
  4. If one or both still low: keep best sub-cluster, mark others "weak".
Retrains ALL image-level concept heads with updated cluster assignments.
Reports changed concepts and new per-concept F1/AUC.
"""

from __future__ import annotations
import io, json, pickle, re, sys
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

_COVAD  = Path(__file__).resolve().parents[2]
_MABCM  = _COVAD / "ma_cbm"
_OUT    = _MABCM / "outputs"

_MP_CACHE   = _OUT / "allcat_maxpool_cache.pkl"
_VLM_IN     = _OUT / "image_level_vlm_results_allcat.json"
_VLM_OUT    = _OUT / "image_level_vlm_results_allcat_v2.json"
_HEADS_OUT  = _OUT / "allcat_image_level_heads_v2.pkl"
_GRIDS_DIR  = _OUT / "image_level_grids_v2"
_GRIDS_DIR.mkdir(parents=True, exist_ok=True)

_OLLAMA  = "http://localhost:6000"
_VLM_MDL = "gemma4:e4b"
_CELL_PX = 224
_MVTEC_15 = [
    "bottle","cable","capsule","carpet","grid","hazelnut","leather",
    "metal_nut","pill","screw","tile","toothbrush","transistor","wood","zipper",
]

_PROMPT = """\
You are analyzing industrial inspection images showing defects.
These 9 images all show the same type of structural or \
geometric anomaly in an industrial object.

The anomaly involves the OVERALL structure of the object — \
component shape, orientation, presence or absence of parts, \
spatial relationships — NOT surface-level defects like \
cracks or stains.

Step 1: Describe what you see in each image briefly.
Step 2: Identify what structural or geometric anomaly \
pattern is common to most images. What component \
is wrong? How does it deviate from expected? \
What is missing or misaligned?
Step 3: Give a short name (2-5 words) for this structural concept.

Output format:
COMMON_PATTERN: [one sentence]
CONCEPT_NAME: [lowercase_underscores]
CONFIDENCE: [high / medium / low]\
"""


# ── helpers ──────────────────────────────────────────────────────────────────

def _build_grid(paths, cell=_CELL_PX):
    grid = Image.new("RGB", (3*cell, 3*cell), (128,128,128))
    for i, p in enumerate(paths[:9]):
        img = Image.open(p).convert("RGB").resize((cell,cell), Image.LANCZOS)
        r, c = divmod(i, 3)
        grid.paste(img, (c*cell, r*cell))
    return grid

def _pil_bytes(img):
    buf = io.BytesIO();  img.save(buf, format="PNG");  return buf.getvalue()

def _parse(text):
    def _g(tag):
        m = re.search(rf"{tag}:\s*(.+)", text, re.IGNORECASE)
        return m.group(1).strip() if m else "PARSE_ERROR"
    return {"concept_name": _g("CONCEPT_NAME"),
            "common_pattern": _g("COMMON_PATTERN"),
            "confidence": _g("CONFIDENCE").lower(), "raw": text}

def _query_vlm(grid):
    from ollama import Client
    try:
        r = Client(host=_OLLAMA).chat(model=_VLM_MDL, messages=[{
            "role":"user","content":_PROMPT,"images":[_pil_bytes(grid)]}])
        return _parse(r["message"]["content"])
    except Exception as e:
        return {"concept_name":"VLM_ERROR","common_pattern":str(e),
                "confidence":"low","raw":str(e)}

def _top_dims(mp_cache):
    all_anom = np.concatenate([
        mp_cache[c]["maxpool_feats"][np.array(mp_cache[c]["y_all"])==1]
        for c in _MVTEC_15])
    all_norm = np.concatenate([
        mp_cache[c]["maxpool_feats"][np.array(mp_cache[c]["y_all"])==0]
        for c in _MVTEC_15])
    disc = all_anom.mean(0) - all_norm.mean(0)
    return np.argsort(np.abs(disc))[::-1][:100]

def _train_head(feats_anom, feats_norm, mask_pos):
    X_pos = feats_anom[mask_pos]
    X_neg = np.concatenate([feats_anom[~mask_pos], feats_norm])
    X = np.concatenate([X_pos, X_neg])
    y = np.array([1]*len(X_pos)+[0]*len(X_neg), dtype=np.int32)
    if len(X_pos) < 5:
        return None, {"f1":float("nan"),"auc":float("nan"),"n_pos":len(X_pos),"n_neg":len(X_neg)}
    X_tr, X_te, y_tr, y_te = train_test_split(X,y,test_size=0.2,stratify=y,random_state=42)
    clf = LogisticRegression(C=1.0,class_weight="balanced",max_iter=1000,solver="lbfgs",random_state=42)
    clf.fit(X_tr, y_tr)
    y_p = clf.predict(X_te);  y_pb = clf.predict_proba(X_te)[:,1]
    return clf, {"f1":float(f1_score(y_te,y_p,zero_division=0)),
                 "auc":float(roc_auc_score(y_te,y_pb)),
                 "n_pos":int(mask_pos.sum()),"n_neg":len(X_neg)}


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  Part 1 — Generalized fine-graining of low-confidence concepts")
    print("=" * 70)

    vlm_results  = json.load(open(_VLM_IN))
    mp_cache     = pickle.load(open(_MP_CACHE,"rb"))
    top_dims     = _top_dims(mp_cache)

    # Collect all anom items + features (same as discover script)
    anom_items, feats_anom_all = [], []
    for cat in _MVTEC_15:
        mp = mp_cache[cat]
        y  = np.array(mp["y_all"])
        for i in np.where(y==1)[0]:
            anom_items.append({"path": mp["all_paths"][i], "category": cat,
                               "defect_type": Path(mp["all_paths"][i]).parent.name})
            feats_anom_all.append(mp["maxpool_feats"][i])
    feats_anom_all = np.stack(feats_anom_all)   # (N_anom, 1024)

    feats_norm_all = np.concatenate([
        mp_cache[c]["maxpool_feats"][np.array(mp_cache[c]["y_all"])==0]
        for c in _MVTEC_15])

    # Re-run original K-means to get cluster labels
    K_orig = len(vlm_results)
    print(f"\n[Step 1] Re-running original K-means (K={K_orig}) …")
    km_orig = KMeans(n_clusters=K_orig, random_state=42, n_init=10)
    orig_labels = km_orig.fit_predict(feats_anom_all[:, top_dims])

    # Identify concepts by confidence
    low_conf    = [r for r in vlm_results if r["confidence"] not in ("high","medium")]
    med_conf    = [r for r in vlm_results if r["confidence"] == "medium"]
    high_conf   = [r for r in vlm_results if r["confidence"] == "high"]
    print(f"  High confidence: {len(high_conf)}")
    print(f"  Medium confidence: {len(med_conf)}  (flagged for monitoring)")
    print(f"  Low confidence: {len(low_conf)}  (will be sub-clustered)")

    # Keep track of all clusters (original + replacements)
    new_vlm_results = [r for r in vlm_results if r["confidence"] in ("high","medium")]
    next_cluster_id  = max(r["cluster_id"] for r in vlm_results) + 1

    for orig_result in low_conf:
        orig_k   = orig_result["cluster_id"]
        orig_name = orig_result["concept_name"]
        mask_orig = (orig_labels == orig_k)
        feats_sub = feats_anom_all[mask_orig][:, top_dims]   # (M, 100)
        items_sub = [anom_items[i] for i in np.where(mask_orig)[0]]
        paths_sub = [it["path"] for it in items_sub]

        print(f"\n[Sub-cluster] '{orig_name}'  ({mask_orig.sum()} images)")

        replaced = False
        for K_sub in (2, 3):
            print(f"  Trying K={K_sub} sub-clusters …")
            km_sub = KMeans(n_clusters=K_sub, random_state=42, n_init=10)
            sub_labels = km_sub.fit_predict(feats_sub)
            sub_centroids = km_sub.cluster_centers_

            sub_results = []
            all_acceptable = True

            for sk in range(K_sub):
                sub_mask = sub_labels == sk
                if sub_mask.sum() < 3:
                    print(f"    Sub-cluster {sk}: too small ({sub_mask.sum()}), skip")
                    all_acceptable = False
                    continue

                sub_idx = np.where(sub_mask)[0]
                dists   = np.linalg.norm(feats_sub[sub_idx] - sub_centroids[sk], axis=1)
                top9    = sub_idx[np.argsort(dists)[:9]]
                rep_paths = [paths_sub[i] for i in top9]

                grid_path = _GRIDS_DIR / f"subcluster_{orig_k}_{sk}.png"
                grid = _build_grid(rep_paths)
                grid.save(grid_path)

                print(f"    Sub-cluster {sk} ({sub_mask.sum()} images) — VLM …", flush=True)
                vlm = _query_vlm(grid)
                cats_in = sorted(set(items_sub[i]["category"] for i in sub_idx))
                vlm.update({"cluster_id": next_cluster_id + sk,
                             "cluster_size": int(sub_mask.sum()),
                             "categories": cats_in,
                             "representative_images": rep_paths,
                             "parent_cluster": orig_k,
                             "sub_labels": sub_labels.tolist()})
                sub_results.append(vlm)

                conf = vlm["confidence"]
                print(f"    → '{vlm['concept_name']}'  conf={conf}")
                if conf == "low":
                    all_acceptable = False

            if all_acceptable or K_sub == 3:
                # Accept: keep acceptable ones, mark weak ones
                kept, weak = [], []
                for sr in sub_results:
                    if sr["confidence"] in ("high","medium"):
                        kept.append(sr)
                    else:
                        weak.append(sr)
                        print(f"    [WEAK] '{sr['concept_name']}' — excluded from vocabulary")

                if kept:
                    print(f"  SPLIT '{orig_name}' → {[r['concept_name'] for r in kept]}")
                    for sr in kept:
                        new_vlm_results.append(sr)
                    next_cluster_id += K_sub
                    replaced = True
                break

        if not replaced:
            # Keep best sub-cluster or original
            best = max(sub_results, key=lambda r: ["high","medium","low"].index(r.get("confidence","low")) if r.get("confidence") in ("high","medium","low") else 99, default=None)
            if best and best["confidence"] != "low":
                print(f"  PARTIAL: kept best sub-cluster '{best['concept_name']}'")
                new_vlm_results.append(best)
                next_cluster_id += K_sub
            else:
                print(f"  FAILED: all sub-clusters low confidence. Keeping original (will be 'weak').")
                orig_result["confidence"] = "weak"
                # Don't add to new_vlm_results

    print(f"\n  Final vocabulary: {len(new_vlm_results)} concepts")
    for r in new_vlm_results:
        print(f"    cluster {r.get('cluster_id','?'):>2}  [{r['confidence']}]  {r['concept_name']}")

    json.dump(new_vlm_results, open(_VLM_OUT,"w"), indent=2)
    print(f"\n  VLM results saved → {_VLM_OUT.name}")

    # ── Retrain ALL image-level heads ─────────────────────────────────────────
    print(f"\n[Step 2] Retraining all image-level concept heads …\n")

    # Rebuild cluster assignments for the new vocabulary
    # Each new_vlm_result maps to a subset of anom_items
    # We need to assign each anom image to exactly one concept
    # Strategy: re-run K-means with new K or use parent+sub labels

    # Rebuild label array: each anom image → new_concept_name (or None if excluded)
    image_to_concept: list[str | None] = [None] * len(anom_items)

    for r in new_vlm_results:
        parent_k = r.get("parent_cluster")
        if parent_k is None:
            # Original concept (not split)
            orig_k = r["cluster_id"]
            for i in np.where(orig_labels == orig_k)[0]:
                image_to_concept[i] = r["concept_name"]
        else:
            # Sub-cluster: need sub_labels from the result
            sub_labels_arr = np.array(r["sub_labels"])
            sub_id_within  = r["cluster_id"] - (r["cluster_id"] // 10) * 10  # extract sub index
            # Simpler: iterate parent cluster items and match by sub_label
            parent_mask = np.where(orig_labels == parent_k)[0]
            for local_i, global_i in enumerate(parent_mask):
                # Check if this image belongs to this sub-cluster
                if local_i < len(sub_labels_arr):
                    sk_of_this = sub_labels_arr[local_i]
                    # Find which sub_result this cluster_id belongs to
                    if r["cluster_id"] == list(set(sr["cluster_id"] for sr in new_vlm_results
                                                   if sr.get("parent_cluster")==parent_k))[
                        list({sr["cluster_id"] for sr in new_vlm_results if sr.get("parent_cluster")==parent_k}).index(r["cluster_id"])]:
                        pass  # complex — use simpler approach below

    # Simpler approach: for original (non-split) concepts, use orig_labels
    # For split concepts, we stored sub_labels in the result
    image_to_concept = [None] * len(anom_items)
    for r in new_vlm_results:
        parent_k = r.get("parent_cluster")
        if parent_k is None:
            # Not split — original cluster_id
            orig_k = r["cluster_id"]
            for i in np.where(orig_labels == orig_k)[0]:
                image_to_concept[i] = r["concept_name"]
        # Split concepts: handled by sub_labels below

    # Handle split concepts using stored sub_labels
    split_results = [r for r in new_vlm_results if r.get("parent_cluster") is not None]
    if split_results:
        # Group by parent_cluster
        parents = {}
        for r in split_results:
            pk = r["parent_cluster"]
            if pk not in parents:
                parents[pk] = []
            parents[pk].append(r)

        for parent_k, sub_res_list in parents.items():
            parent_mask_idx = np.where(orig_labels == parent_k)[0]
            feats_sub = feats_anom_all[parent_mask_idx][:, top_dims]
            km_sub = KMeans(n_clusters=len(sub_res_list)+1, random_state=42, n_init=10)  # +1 for weak
            sub_labels_new = km_sub.fit_predict(feats_sub)

            # Match each sub-cluster to the stored sub_labels in sub_res_list
            # Use majority voting between old sub_labels stored in results
            for local_i, global_i in enumerate(parent_mask_idx):
                # Use the first result's stored sub_labels for matching
                ref_sub_labels = np.array(sub_res_list[0].get("sub_labels",[]))
                if local_i < len(ref_sub_labels):
                    sub_k_old = int(ref_sub_labels[local_i])
                    # Find if sub_k_old matches an accepted result
                    # We stored sub_labels from K=2 run; find which concept it was
                    km_final = KMeans(n_clusters=2, random_state=42, n_init=10)
                    final_labels = km_final.fit_predict(feats_sub)
                    for sr_idx, sr in enumerate(sub_res_list):
                        if final_labels[local_i] == sr_idx:
                            image_to_concept[global_i] = sr["concept_name"]
                            break

    # Final assignment check
    n_assigned = sum(1 for x in image_to_concept if x is not None)
    print(f"  {n_assigned}/{len(anom_items)} anomalous images assigned to concepts")

    # Train heads
    all_concept_names = [r["concept_name"] for r in new_vlm_results]
    image_to_concept_arr = np.array(image_to_concept)

    heads_dict = {}
    head_metrics_list = []
    hdr = f"  {'Concept':<45} {'N+':>5} {'N-':>6} {'F1':>7} {'AUC':>7}"
    print(hdr);  print("  "+"─"*70)

    for concept_name in all_concept_names:
        mask_pos = image_to_concept_arr == concept_name
        clf, m = _train_head(feats_anom_all, feats_norm_all, mask_pos)
        heads_dict[concept_name] = clf
        print(f"  {concept_name:<45} {m['n_pos']:>5} {m['n_neg']:>6} "
              f"{m['f1']:>7.3f} {m['auc']:>7.3f}")
        head_metrics_list.append({"concept": concept_name, **m})

    valid_f1s = [m["f1"] for m in head_metrics_list if not __import__("math").isnan(m["f1"])]
    print("  "+"─"*70)
    print(f"  Mean F1: {__import__('numpy').mean(valid_f1s):.3f}")

    pickle.dump(heads_dict, open(_HEADS_OUT,"wb"))
    print(f"\n  Heads saved → {_HEADS_OUT.name}")
    json.dump(head_metrics_list, open(_OUT/"part1_head_metrics.json","w"), indent=2)
    print("=" * 70)


if __name__ == "__main__":
    main()
