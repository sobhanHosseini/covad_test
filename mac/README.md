# MA-CBM — Mask-Anchored Autointerpretable Concept Bottleneck Model

Proof-of-concept validation pipeline for the hazelnut category.
Lives inside `covad_test/mac/` and shares `covad_test/features/` with the rest of the project.

## Structure

```
covad_test/
├── features/
│   ├── dinov2_extractor.py      ← shared — frozen DINOv2 extractor
│   └── sae.py                   ← shared — TopK Sparse Autoencoder
└── mac/
    ├── configs/
    │   └── config.yaml          ← edit mvtec_root here before running
    ├── data/
    │   └── patch_extractor.py   ← anomaly + normal patch extraction
    ├── concepts/
    │   ├── atom_selector.py     ← discrimination scoring
    │   ├── vlm_interpreter.py   ← Qwen2.5-VL querying + parsing
    │   └── vocabulary_builder.py
    ├── scripts/
    │   ├── 01_extract_patches.py
    │   ├── 02_select_atoms.py
    │   └── 03_run_vlm.py
    └── outputs/                 ← auto-created at runtime
        ├── hazelnut_anomaly_patches.pkl
        ├── hazelnut_relevant_atoms.json
        ├── hazelnut_vlm_results.json
        ├── hazelnut_vocabulary.json
        └── grids/
            └── atom_<id>.png
```

## Import resolution

Each script sets up `sys.path` so that:

| `sys.path` entry   | resolves                          |
|--------------------|-----------------------------------|
| `covad_test/`      | `features.dinov2_extractor`, `features.sae` |
| `covad_test/mac/`  | `data.patch_extractor`, `concepts.*`        |

## Setup

1. **Edit `mac/configs/config.yaml`** — set `mvtec_root` to your MVTec path.
2. **Dependencies** — `torch`, `torchvision`, `Pillow`, `numpy`, `pyyaml`, `ollama`
3. **Ollama** — verify with: `ollama run qwen2.5vl:7b "hello"`

## Running

```bash
cd /mnt/nvme1/sobhan_hosseini/covad_test

# Step 1 — extract anomaly patches and SAE-encode them (~5 min on A6000)
uv run mac/scripts/01_extract_patches.py

# Step 2 — score all 4096 atoms and select top-50 (seconds)
uv run mac/scripts/02_select_atoms.py

# Step 3 — query Qwen2.5-VL for each atom, build vocabulary (~50 VLM calls)
uv run mac/scripts/03_run_vlm.py
```

## Key design notes

- **Patch grid is 16×16 = 256 patches** per image (224 px ÷ 14 px patch = 16, not 14).
- **Normal patches** come from the precomputed tensor `sae_training/mvtec_normal_patches_vitl14reg.pt` — DINOv2 does not re-run on the normal split.
- **Grid upscale**: 14×14 px crops are upscaled 8× → 112 px per side → 336×336 px 3×3 grid.
- **Incremental VLM save**: results written after every atom — a crash never loses progress.
