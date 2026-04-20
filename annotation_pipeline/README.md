# annotation_pipeline

VLM-based concept annotation pipeline for CONVAD.  
MSc Thesis — Explainable Visual Anomaly Detection via CBMs  
University of Padova, 2025-2026

---

## Package Structure

```
annotation_pipeline/
├── config.py          ← ALL constants + ALL prompt templates (edit here)
├── utils.py           ← Shared helpers (JSON parsing, VLM call, dimension utils)
├── dataset.py         ← MVTec image discovery (generic, any category)
├── clip_utils.py      ← CLIP loading, embedding computation, reference selection
├── checkpoints.py     ← Stage output save/load (enables fast holdout + CL)
├── vocabulary.py      ← Three-tier vocabulary assembly
├── csv_builder.py     ← CONVAD-compatible CSV output
├── pipeline.py        ← Orchestrator + CLI argument definitions
├── __main__.py        ← Entry point: python -m annotation_pipeline
└── stages/
    ├── stage1.py      ← Per-image normal concept extraction (VLM)
    ├── stage2.py      ← Normal concept post-processing (clustering + audit)
    ├── stage3.py      ← Per-image defect annotation with normal references
    └── stage4.py      ← Defect concept refinement (clustering + freq filter)
```

---

## Three-Tier Vocabulary

| Tier | What | Source | Held out? |
|------|------|--------|-----------|
| 1 — Normal concepts | Describe healthy specimen appearance | Stage 2 | Never |
| 2 — Generic anomaly concepts | 5 fixed + cross-defect discovered | Stage 2b + Stage 4c | Never |
| 3 — Defect-specific concepts | Per defect type, visually grounded | Stage 4 | Yes (holdout defect only) |

---

## Common Usage

### 1. Full run (first time, any category)

```bash
python -m annotation_pipeline \
    --dataset_path /path/to/mvtec \
    --category hazelnut \
    --save_path ./annotations/hazelnut_v6.csv \
    --model_name gemma4:e4b \
    --n_normal_sample 50
```

This runs all 4 stages and saves three checkpoint files:
```
annotations/hazelnut_stage1_raw.json          (~1 MB, fast to recompute)
annotations/hazelnut_stage2_normal_dict.json  (~5 KB)
annotations/hazelnut_stage3_annotations.json  (~2 MB, expensive — 8-10h)
annotations/hazelnut_clip_normal.npz          (CLIP cache for normal images)
```

---

### 2. Fast holdout variants (load from Stage 3 checkpoint)

After the full run above, generate ALL holdout CSVs in minutes each:

```bash
# Holdout: print
python -m annotation_pipeline \
    --dataset_path /path/to/mvtec \
    --category hazelnut \
    --save_path ./annotations/hazelnut_holdout_print.csv \
    --load_from_stage 3 \
    --holdout_defect print

# Holdout: crack
python -m annotation_pipeline \
    --dataset_path /path/to/mvtec \
    --category hazelnut \
    --save_path ./annotations/hazelnut_holdout_crack.csv \
    --load_from_stage 3 \
    --holdout_defect crack

# Holdout: cut
python -m annotation_pipeline \
    --dataset_path /path/to/mvtec \
    --category hazelnut \
    --save_path ./annotations/hazelnut_holdout_cut.csv \
    --load_from_stage 3 \
    --holdout_defect cut

# Holdout: hole
python -m annotation_pipeline \
    --dataset_path /path/to/mvtec \
    --category hazelnut \
    --save_path ./annotations/hazelnut_holdout_hole.csv \
    --load_from_stage 3 \
    --holdout_defect hole
```

**What `--load_from_stage 3` does:**
- Loads Stage 3 annotation checkpoint (skips Stages 1-3 entirely)
- Reruns only Stage 4 (fast: clustering only, ~seconds)
- Rebuilds vocabulary excluding the holdout defect's Tier 3 concepts
- Writes a new CSV — no VLM calls needed

---

### 3. New category (e.g. capsule)

```bash
python -m annotation_pipeline \
    --dataset_path /path/to/mvtec \
    --category capsule \
    --save_path ./annotations/capsule_v6.csv \
    --model_name gemma4:e4b \
    --n_normal_sample 50
```

Everything is generic. No code changes needed between categories.

---

### 4. v2-mode (no Stage 4 clustering — maximum concepts, best C-AUC baseline)

```bash
python -m annotation_pipeline \
    --dataset_path /path/to/mvtec \
    --category hazelnut \
    --save_path ./annotations/hazelnut_v2mode.csv \
    --load_from_stage 3 \
    --skip_stage4
```

Reproduces v2 behaviour: raw defect concepts, no clustering, more redundancy but
historically best C-AUC (0.958 on hazelnut).

---

### 5. Continual learning — append new defect type

```bash
python -m annotation_pipeline \
    --dataset_path /path/to/mvtec \
    --category hazelnut \
    --save_path ./annotations/hazelnut_cl_updated.csv \
    --load_from_stage 3 \
    --append_defect new_defect_name
```

**What this does:**
1. Loads existing Stage 3 checkpoint
2. Annotates only the new defect type's images (Stage 3 for new images only)
3. Merges new annotations into the checkpoint (saved back to disk)
4. Reruns Stage 4 on the full merged set → updated vocabulary
5. Writes updated CSV

The normal vocabulary (Stages 1-2) never changes — this is realistic for CL
where the healthy appearance of the object does not change.

---

## Key Parameters

| Parameter | Default | When to change |
|-----------|---------|----------------|
| `--n_normal_sample` | all | Set to 50 to match validated hazelnut setup |
| `--min_concept_freq` | 0.20 | Lower to get more normal concepts (try 0.15) |
| `--cluster_threshold` | 0.65 | Higher = harder to merge = more normal concepts |
| `--max_new_defect_concepts` | 6 | Max new concepts VLM extracts per defect image |
| `--min_defect_freq` | 0.20 | Lower to keep rarer defect concepts |
| `--stage4_cluster_threshold` | 0.65 | Controls Stage 4 merging aggressiveness |
| `--skip_stage4` | off | Enable for v2-mode (max concepts, no clustering) |
| `--n_normal_refs` | 3 | CLIP-selected references per defect image |
| `--n_workers` | 1 | Increase for faster Stage 3 if GPU available |
| `--use_visual_grounding` | off | Enable CLIP visual similarity in Stage 4 |

---

## Checkpoints Explained

```
{category}_stage1_raw.json
  Raw per-image VLM output. One entry per (image × concept).
  Safe to delete — Stage 1 is fast (~1 min for 50 images).

{category}_stage2_normal_dict.json
  Refined normal concept dictionary. Typically 8-15 concepts.
  Safe to delete — Stage 2 takes <1 min.

{category}_stage3_annotations.json   ← THE VALUABLE ONE
  Per-image concept vectors for ALL images (normal + all defects).
  This represents 8-10h of VLM API calls. Never delete without reason.
  Used by --load_from_stage 3 for fast holdout and CL scenarios.

{category}_clip_normal.npz
  CLIP embedding cache for normal training images.
  Safe to delete — recomputed in ~1 min on GPU, ~5 min on CPU.
```

---

## Adding a New Prompt

All prompts live in `config.py`. To modify the defect annotation prompt:
1. Open `config.py`
2. Edit `PROMPT_DEFECT_ANNOTATION`
3. No other files need changing

To add a new prompt (e.g. for a new stage):
1. Add the string constant to `config.py`
2. Import it in the relevant stage file: `from annotation_pipeline.config import MY_PROMPT`

---

## Known Limitations

- Only one VLM backend supported (Ollama). Other backends require changes to `utils.py`.
- Stage 3 is the bottleneck (~1 min/image on CPU). Use `--n_workers` to parallelise.
- `all-MiniLM-L6-v2` is not trained on industrial vocabulary; clustering quality
  is limited for highly domain-specific concept names.
- Print detection ceiling at 58.8% is a MobileNetV2 feature extractor limitation,
  not a pipeline issue. No amount of concept engineering can overcome it.
