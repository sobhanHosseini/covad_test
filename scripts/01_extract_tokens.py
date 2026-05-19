"""
Extract DINOv2 ViT-L/14 patch tokens from all MVTec AD normal images.

Input:  /home/sobhan_hosseini/datasets/mvtec/{cat}/train/good/*.png
Output: sae_training/mvtec_normal_patches_vitl14.pt   shape (N, 1024)
        sae_training/mvtec_patch_index.pt             per-image metadata

Run from project root:
    python scripts/01_extract_tokens.py
"""

import torch
import sys
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from features.dinov2_extractor import DINOv2Extractor

MVTEC_ROOT  = Path("/home/sobhan_hosseini/datasets/mvtec")
OUTPUT_DIR  = Path("sae_training")
MODEL_NAME  = "dinov2_vitl14_reg"
EMBED_DIM   = 1024
BATCH_SIZE  = 32
NUM_WORKERS = 4

CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper",
]

transform = T.Compose([
    T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class ImageDataset(Dataset):
    def __init__(self, paths):
        self.paths = paths
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, i):
        return transform(Image.open(self.paths[i]).convert("RGB"))


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    device    = torch.device("cuda:0")
    extractor = DINOv2Extractor(MODEL_NAME).to(device)
    print(f"Backbone : {MODEL_NAME}")
    print(f"EMBED_DIM: {extractor.EMBED_DIM}")
    print(f"Device   : {device}\n")

    all_patches = []
    index       = []   # (category, image_path, start_row, end_row)
    row         = 0

    for cat in CATEGORIES:
        good_dir = MVTEC_ROOT / cat / "train" / "good"
        paths    = sorted(good_dir.glob("*.png")) + \
                   sorted(good_dir.glob("*.jpg"))

        loader = DataLoader(
            ImageDataset(paths), batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS, pin_memory=True,
        )

        cat_patches = []
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"{cat:12s}", leave=False):
                p = extractor.extract_patch_tokens(batch.to(device))
                cat_patches.append(p.reshape(-1, EMBED_DIM).cpu().float())

        cat_tensor = torch.cat(cat_patches)
        end_row    = row + cat_tensor.shape[0]
        index.append({
            "category": cat,
            "n_images": len(paths),
            "n_patches": cat_tensor.shape[0],
            "row_start": row,
            "row_end":   end_row,
        })
        all_patches.append(cat_tensor)
        row = end_row
        print(f"{cat:12s}: {len(paths):4d} imgs "
              f"→ {cat_tensor.shape[0]:8,} patches")

    all_tokens = torch.cat(all_patches)
    print(f"\n{'TOTAL':12s}: {sum(e['n_images'] for e in index):4d} imgs "
          f"→ {all_tokens.shape[0]:8,} patches")
    print(f"Shape  : {all_tokens.shape}")
    print(f"Memory : {all_tokens.nbytes / 1e9:.2f} GB (float32)")

    out_tokens = OUTPUT_DIR / "mvtec_normal_patches_vitl14reg.pt"
    out_index  = OUTPUT_DIR / "mvtec_patch_index_reg.pt"
    torch.save(all_tokens, out_tokens)
    torch.save(index,      out_index)
    print(f"\nSaved tokens → {out_tokens}")
    print(f"Saved index  → {out_index}")

    norms = all_tokens.norm(dim=-1)
    print(f"\nPatch token stats:")
    print(f"  mean      : {all_tokens.mean():.4f}")
    print(f"  std       : {all_tokens.std():.4f}")
    print(f"  norm mean : {norms.mean():.4f}  (expect ≈ 1.0 — DINOv2 normalises)")
    print(f"  norm std  : {norms.std():.4f}")


if __name__ == "__main__":
    main()
