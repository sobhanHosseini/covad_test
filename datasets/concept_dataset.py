import os
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import pandas as pd

from torchvision.transforms import transforms
from torch.utils.data.dataset import Dataset
from torchvision.transforms.functional import InterpolationMode

EXCLUDE_COLS = ["image_path", "label_index", "mask_path", "split", "anomaly_type", "view"]


class ConceptDataset(Dataset):
    """Load a pre-annotated concept CSV into a PyTorch Dataset.

    The CSV must have columns: image_path, label_index, mask_path, split,
    anomaly_type, plus one binary column per concept.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        split: str,
        load_image: bool = True,
        apply_transformation: bool = True,
        img_size=(224, 224),
        use_attr: bool = True,
        load_mask: bool = False,
    ) -> None:
        super().__init__()

        self.split = split
        self.load_image = load_image
        self.apply_transformation = apply_transformation
        self.use_attr = use_attr
        self.load_mask = load_mask

        self.df = dataframe[dataframe["split"] == split].reset_index(drop=True)
        self.attr_cols = [col for col in self.df.columns if col not in EXCLUDE_COLS]

        self.pre_transform = transforms.Compose(
            [transforms.Resize(img_size), transforms.ToTensor()]
        )
        self.transform = transforms.Compose(
            [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        )
        self.transform_mask = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(
                    img_size, antialias=True, interpolation=InterpolationMode.NEAREST
                ),
            ]
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image = Image.open(row["image_path"]).convert("RGB")
        image = self.pre_transform(image)
        if self.apply_transformation:
            image = self.transform(image)

        label = int(row["label_index"])

        if self.load_mask:
            mask_path = row["mask_path"]
            if isinstance(mask_path, str) and mask_path:
                mask = transforms.ToTensor()(Image.open(mask_path).convert("L"))
            else:
                mask = torch.zeros(1, *image.shape[-2:])

        if self.use_attr:
            attr_label = torch.tensor(
                row[self.attr_cols].values.astype(np.float32), dtype=torch.float32
            )
            if self.load_image:
                return image, attr_label, label
            return attr_label, label

        if self.load_mask:
            return image, label, mask

        return image, label

    def find_class_imbalance(self, kind="main"):
        num_total = len(self.df)
        if kind == "main":
            counts = self.df["label_index"].value_counts().to_dict()
            num_pos = counts.get(1, 0)
            return num_total / max(num_pos, 1) - 1, counts.get(1, 0) / max(counts.get(0, 1), 1)
        if kind == "attributes":
            ratios = []
            for attr in self.attr_cols:
                num_pos = self.df[attr].sum()
                ratios.append(num_total / max(num_pos, 1) - 1)
            return ratios
        raise ValueError(f"Unknown kind: {kind}")
