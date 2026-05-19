"""
Phase 2: SAE Atom Naming via dino.txt

For each usable SAE atom, finds its top-activating patch crops,
encodes them with dino.txt, and scores against a concept vocabulary.

Input:
  sae_training/sae_vitl14reg_C4096_k64.pt
  sae_training/sae_vitl14reg_atom_stats_k64.pt
  sae_training/mvtec_normal_patches_vitl14reg.pt
  sae_training/mvtec_patch_index_reg.pt

Output:
  sae_training/atom_names.json   {atom_idx: {names, scores, freq}}
  sae_training/atom_names_full.pt  full similarity matrix

Run from project root:
    uv run python scripts/04_name_atoms.py
"""

import torch
import torch.nn.functional as F
import json
import sys
from pathlib import Path
from PIL import Image
from functools import lru_cache
import torchvision.transforms as T

sys.path.insert(0, str(Path(__file__).parent.parent))
from features.sae import SparseAutoencoder

# ── config ───────────────────────────────────────────────────────────
MVTEC_ROOT  = Path("/home/sobhan_hosseini/datasets/mvtec")
SAE_PATH    = Path("sae_training/sae_vitl14reg_C4096_k64.pt")
STATS_PATH  = Path("sae_training/sae_vitl14reg_atom_stats_k64.pt")
TOKENS_PATH = Path("sae_training/mvtec_normal_patches_vitl14reg.pt")
INDEX_PATH  = Path("sae_training/mvtec_patch_index_reg.pt")
OUT_JSON    = Path("sae_training/atom_names.json")
OUT_FULL    = Path("sae_training/atom_names_full.pt")

FREQ_LOW    = 0.001   # atoms below this → too rare / dead
FREQ_HIGH   = 0.50    # atoms above this → too generic (super-atoms)
TOP_PATCHES = 25      # patches per atom for visual representation
ATOM_CHUNK  = 150     # atoms processed per chunk (RAM control)
TOP_N_NAMES = 5       # concept names to store per atom
DEVICE      = torch.device("cuda:0")

CATEGORIES = [
    "bottle","cable","capsule","carpet","grid","hazelnut",
    "leather","metal_nut","pill","screw","tile","toothbrush",
    "transistor","wood","zipper",
]

PATCHES_PER_IMG  = 256
PATCHES_PER_SIDE = 16
PATCH_SIZE       = 14

# ── image transform (same as extraction) ────────────────────────────
_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]
_transform = T.Compose([
    T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=_MEAN, std=_STD),
])
_crop_transform = T.Compose([
    T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=_MEAN, std=_STD),
])

# ── concept vocabulary ───────────────────────────────────────────────
CONCEPTS = [
    # MVTec object materials
    "glass bottle surface", "transparent glass", "rubber cable",
    "pharmaceutical capsule", "carpet fiber", "metal grid mesh",
    "hazelnut shell surface", "leather embossed texture",
    "metal nut hexagonal", "pill tablet surface", "screw thread metal",
    "ceramic tile surface", "toothbrush bristles", "electronic transistor",
    "wood grain surface", "zipper teeth fabric",
    # surface textures
    "smooth uniform surface", "rough granular texture",
    "grainy speckled texture", "woven fabric texture",
    "embossed raised pattern", "polished reflective surface",
    "matte flat surface", "porous open texture",
    "fine detailed texture", "coarse irregular texture",
    # visual patterns
    "regular grid pattern", "repeating periodic pattern",
    "diagonal line pattern", "circular round shape",
    "hexagonal geometric pattern", "stripe linear pattern",
    "uniform solid color", "gradient brightness",
    # colors and brightness
    "dark black surface", "bright white surface",
    "brown warm color", "silver metallic color",
    "dark background region", "light gray surface",
    "orange warm material", "blue color surface",
    "dark brown material", "beige light color",
    "green color surface", "red color surface",
    # geometry and shape
    "curved rounded edge", "sharp straight edge",
    "cylindrical tube surface", "flat planar surface",
    "corner boundary region", "circular hole opening",
    "thin wire cable", "thick solid object",
    # surface properties
    "glossy shiny surface", "matte dull finish",
    "wet liquid surface", "dry solid material",
    "transparent clear material", "opaque solid material",
    # industrial inspection - normal
    "intact normal surface", "regular uniform appearance",
    "clean uncontaminated surface", "undamaged material",
    "properly manufactured part", "normal product surface",
    # defect types - anomalies
    "surface crack fracture", "hole pit defect",
    "scratch mark surface", "stain contamination",
    "color discoloration", "missing broken material",
    "dent deformation surface", "chipped damaged edge",
    "print mark surface", "inclusion foreign material",
    "bent deformed shape", "corroded degraded surface",
    "poke puncture mark", "glue adhesive residue",
    "thread defect fabric", "cut sliced material",
    # visual anomaly indicators
    "irregular unusual texture", "anomalous surface region",
    "unexpected color change", "structural discontinuity",
    "manufacturing defect mark", "quality control defect",
    # spatial/contextual
    "object boundary edge", "background region",
    "center object region", "corner image region",
    "foreground object", "surface transition",
]


# ── helpers ──────────────────────────────────────────────────────────

def build_image_list():
    paths = []
    for cat in CATEGORIES:
        good_dir = MVTEC_ROOT / cat / "train" / "good"
        cat_paths = (sorted(good_dir.glob("*.png")) +
                     sorted(good_dir.glob("*.jpg")))
        paths.extend(cat_paths)
    return paths


def token_to_location(token_idx, image_paths):
    img_idx   = token_idx // PATCHES_PER_IMG
    patch_idx = token_idx  % PATCHES_PER_IMG
    row = patch_idx // PATCHES_PER_SIDE
    col = patch_idx  % PATCHES_PER_SIDE
    return image_paths[img_idx], row, col


@lru_cache(maxsize=256)
def load_image_tensor(path_str):
    """Load and transform image. Cached to avoid reloading."""
    return _transform(Image.open(path_str).convert("RGB"))


def extract_patch_crop(img_tensor, patch_row, patch_col, context=3):
    """
    Extract a context window around (patch_row, patch_col) from
    the 224x224 image tensor, resize to 224x224 for encoding.
    img_tensor: (3, 224, 224) normalised tensor
    Returns: (3, 224, 224) ready for encode_image
    """
    r0 = max(0, patch_row - context) * PATCH_SIZE
    r1 = min(PATCHES_PER_SIDE, patch_row + context + 1) * PATCH_SIZE
    c0 = max(0, patch_col - context) * PATCH_SIZE
    c1 = min(PATCHES_PER_SIDE, patch_col + context + 1) * PATCH_SIZE
    crop = img_tensor[:, r0:r1, c0:c1]          # (3, H, W)
    crop = F.interpolate(crop.unsqueeze(0),
                         size=(224, 224),
                         mode="bicubic",
                         align_corners=False).squeeze(0)
    return crop


def compute_chunk_activations(tokens_centered, W_enc_chunk, b_enc_chunk):
    """
    tokens_centered: (N, d) CPU tensor
    W_enc_chunk: (d, chunk) CPU
    b_enc_chunk: (chunk,)  CPU
    Returns: (N, chunk) ReLU activations
    """
    BATCH = 50_000
    results = []
    for i in range(0, len(tokens_centered), BATCH):
        t = tokens_centered[i:i+BATCH]
        pre = t @ W_enc_chunk + b_enc_chunk
        results.append(pre.relu())
    return torch.cat(results, dim=0)


def get_text_embeddings(model, concepts, device):
    """Encode concept vocabulary with dino.txt text encoder."""
    from transformers import CLIPTokenizer
    print("Loading CLIP tokenizer...")
    try:
        tok = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    except Exception:
        print("  CLIPTokenizer failed, trying get_tokenizer()...")
        from dinov2.hub.dinotxt import get_tokenizer
        tok = get_tokenizer()

    print(f"Encoding {len(concepts)} concepts...")
    all_embeds = []
    BATCH = 64
    model.eval()
    with torch.no_grad():
        for i in range(0, len(concepts), BATCH):
            batch_concepts = concepts[i:i+BATCH]
            if hasattr(tok, '__call__'):
                # transformers CLIPTokenizer
                encoded = tok(
                    batch_concepts,
                    padding="max_length",
                    max_length=77,
                    truncation=True,
                    return_tensors="pt",
                )
                token_ids = encoded["input_ids"].to(device)
            else:
                # CLIP-style tokenizer returns tensor directly
                token_ids = tok(batch_concepts).to(device)
            emb = model.encode_text(token_ids, normalize=True)
            all_embeds.append(emb.cpu())
    return torch.cat(all_embeds, dim=0)   # (N_concepts, 2048)


def get_atom_visual_embedding(model, atom_token_indices,
                               image_paths, device, batch_size=32):
    """
    Encode top-K patches for one atom through dino.txt.
    Returns: (2048,) normalised embedding.
    """
    crops = []
    for tok_idx in atom_token_indices:
        path, row, col = token_to_location(int(tok_idx), image_paths)
        img_t = load_image_tensor(str(path))
        crop  = extract_patch_crop(img_t, row, col, context=3)
        crops.append(crop)

    all_embeds = []
    with torch.no_grad():
        for i in range(0, len(crops), batch_size):
            batch = torch.stack(crops[i:i+batch_size]).to(device)
            emb   = model.encode_image(batch, normalize=True)  # (B, 2048)
            all_embeds.append(emb.cpu())

    atom_emb = torch.cat(all_embeds, dim=0).mean(dim=0)
    return F.normalize(atom_emb, dim=0)


# ── main ─────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Phase 2: SAE Atom Naming via dino.txt")
    print("=" * 60)

    # Load everything
    print("\nLoading SAE...")
    sae   = SparseAutoencoder.load(str(SAE_PATH), device="cpu")
    stats = torch.load(STATS_PATH, map_location="cpu", weights_only=False)
    freq  = stats["activation_freq"]

    print("Loading patch tokens...")
    tokens = torch.load(TOKENS_PATH, map_location="cpu", weights_only=True)

    image_paths = build_image_list()
    print(f"Image list: {len(image_paths)} images")

    print("Loading dino.txt model...")
    dinotxt = torch.hub.load(
        "facebookresearch/dinov2",
        "dinov2_vitl14_reg4_dinotxt_tet1280d20h24l",
        verbose=False,
    ).to(DEVICE)
    dinotxt.eval()
    print("dino.txt loaded.")

    # Filter usable atoms
    usable_mask = (freq > FREQ_LOW) & (freq < FREQ_HIGH)
    usable_atoms = usable_mask.nonzero(as_tuple=True)[0].tolist()
    print(f"\nUsable atoms: {len(usable_atoms)} "
          f"({FREQ_LOW} < freq < {FREQ_HIGH})")

    # Get text embeddings for concept vocabulary
    print()
    text_embeddings = get_text_embeddings(
        dinotxt, CONCEPTS, DEVICE)          # (N_concepts, 2048)
    print(f"Text embeddings: {text_embeddings.shape}")

    # Pre-compute centered tokens (for activation scoring)
    print("\nPre-computing normalised + centred tokens...")
    tokens_norm     = tokens / tokens.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    tokens_centered = tokens_norm - sae.b_dec.cpu()
    W_enc_cpu = sae.W_enc.cpu()
    b_enc_cpu = sae.b_enc.cpu()

    # Process atoms in chunks
    n_atoms      = len(usable_atoms)
    atom_embeds  = torch.zeros(len(freq), 2048)   # (C, 2048) — only usable filled
    names_dict   = {}

    print(f"\nNaming {n_atoms} atoms in chunks of {ATOM_CHUNK}...")
    for chunk_start in range(0, n_atoms, ATOM_CHUNK):
        chunk_atoms = usable_atoms[chunk_start:chunk_start + ATOM_CHUNK]
        chunk_end   = chunk_start + len(chunk_atoms)
        pct         = chunk_end / n_atoms * 100
        print(f"  Atoms {chunk_start:4d}–{chunk_end:4d} / {n_atoms}  "
              f"({pct:.0f}%)...", end=" ", flush=True)

        # Activation scores for this chunk
        W_chunk = W_enc_cpu[:, chunk_atoms]        # (d, chunk)
        b_chunk = b_enc_cpu[chunk_atoms]           # (chunk,)
        acts    = compute_chunk_activations(
            tokens_centered, W_chunk, b_chunk)     # (N, chunk)

        # For each atom in chunk: get top-K patches + encode
        chunk_embeds = []
        for local_idx, atom_c in enumerate(chunk_atoms):
            top_acts, top_idx = acts[:, local_idx].topk(TOP_PATCHES)
            # Filter: only truly activated (non-zero)
            active_mask = top_acts > 0
            if active_mask.sum() == 0:
                chunk_embeds.append(torch.zeros(2048))
                continue
            top_idx = top_idx[active_mask][:TOP_PATCHES]
            emb = get_atom_visual_embedding(
                dinotxt, top_idx.tolist(), image_paths, DEVICE)
            chunk_embeds.append(emb)
            atom_embeds[atom_c] = emb

        # Free activation memory
        del acts
        print("done")

    # Score all named atoms against concept vocabulary
    print("\nScoring atoms against concept vocabulary...")
    named_atom_idx  = torch.tensor(usable_atoms)
    named_atom_embs = atom_embeds[named_atom_idx]          # (n_usable, 2048)
    similarity      = named_atom_embs @ text_embeddings.T  # (n_usable, N_concepts)

    # Build output dict
    print("Building output...")
    for i, atom_c in enumerate(usable_atoms):
        sim_row    = similarity[i]                         # (N_concepts,)
        top_scores, top_idx = sim_row.topk(TOP_N_NAMES)
        names_dict[int(atom_c)] = {
            "names":  [CONCEPTS[j] for j in top_idx.tolist()],
            "scores": [round(float(s), 4) for s in top_scores.tolist()],
            "freq":   round(float(freq[atom_c]), 6),
        }

    # Save
    with open(OUT_JSON, "w") as f:
        json.dump(names_dict, f, indent=2)
    print(f"\nSaved {len(names_dict)} named atoms → {OUT_JSON}")

    torch.save({
        "similarity":    similarity,
        "atom_indices":  named_atom_idx,
        "concepts":      CONCEPTS,
        "atom_embeds":   named_atom_embs,
        "text_embeds":   text_embeddings,
    }, OUT_FULL)
    print(f"Saved full similarity matrix → {OUT_FULL}")

    # Quick summary
    print("\n── Sample named atoms ──────────────────────────────────")
    sample_keys = list(names_dict.keys())[:10]
    for k in sample_keys:
        entry = names_dict[k]
        top   = entry["names"][0]
        score = entry["scores"][0]
        f_val = entry["freq"]
        print(f"  Atom {k:4d}  freq={f_val:.4f}  → {top!r}  "
              f"(score={score:.3f})")
    print("────────────────────────────────────────────────────────")


if __name__ == "__main__":
    main()
