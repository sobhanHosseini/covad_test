import os
import pandas as pd
from PIL import Image
import torch
from torchvision.transforms import transforms
from typing import Optional
from torchvision.transforms.functional import InterpolationMode
from pathlib import Path

from datasets.iad_dataset import IadDataset
from utils.configurations import Split, LabelName

IMG_EXTENSIONS = (".png", ".PNG")


class MVTecDataset(IadDataset):
    def __init__(
        self,
        root: str,
        category: str,
        split: Split,
        norm: bool = True,
        img_size=(224, 224),
        gt_mask_size: Optional[tuple] = None,
        preload_imgs: bool = True,
    ) -> None:
        super(MVTecDataset)

        gt_mask_size = img_size if gt_mask_size is None else gt_mask_size

        self.img_size = img_size
        self.gt_mask_size = gt_mask_size
        self.root_category = Path(root) / Path(category)
        self.category = category
        self.split = split
        self.samples: pd.DataFrame = None
        self.preload_imgs = preload_imgs

        t_list = [
            transforms.ToTensor(),
            transforms.Resize(img_size, antialias=True),
        ]
        if norm:
            t_list.append(
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            )
        self.transform_image = transforms.Compose(t_list)

        self.transform_mask = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(
                    gt_mask_size,
                    antialias=True,
                    interpolation=InterpolationMode.NEAREST,
                ),
            ]
        )

    def contains(self, item) -> bool:
        return self.samples["image_path"].eq(item["image_path"]).any()

    def load_dataset(self, use_gen_anomalies: bool = False):
        root = Path(self.root_category)

        samples_list = [
            (str(root),) + f.parts[-3:]
            for f in root.glob(r"**/*")
            if f.suffix in IMG_EXTENSIONS and "generated_anomalies" not in f.parts
        ]

        if not samples_list:
            raise RuntimeError(f"Found 0 images in {root}")

        samples = pd.DataFrame(
            samples_list, columns=["path", "split", "label", "image_path"]
        )

        samples["image_path"] = (
            samples.path + "/" + samples.split + "/" + samples.label + "/" + samples.image_path
        )

        samples.loc[(samples.label == "good"), "label_index"] = LabelName.NORMAL
        samples.loc[(samples.label != "good"), "label_index"] = LabelName.ABNORMAL
        samples.label_index = samples.label_index.astype(int)

        if self.split == Split.TEST:
            mask_samples = samples.loc[samples.split == "ground_truth"].sort_values(
                by="image_path", ignore_index=True
            )
            samples = samples[samples.split != "ground_truth"].sort_values(
                by="image_path", ignore_index=True
            )
            samples["mask_path"] = ""
            samples.loc[
                (samples.split == "test") & (samples.label_index == LabelName.ABNORMAL),
                "mask_path",
            ] = mask_samples.image_path.to_numpy()

        if self.split == Split.TEST:
            self.samples = samples[samples.split.str.startswith("test")].reset_index(drop=True)
        else:
            self.samples = samples[samples.split == self.split].reset_index(drop=True)

        if self.preload_imgs:
            self.data = [
                self.transform_image(
                    Image.open(self.samples.iloc[index].image_path).convert("RGB")
                )
                for index in range(len(self.samples))
            ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        if self.preload_imgs:
            image = self.data[index]
        else:
            image = self.transform_image(
                Image.open(self.samples.iloc[index].image_path).convert("RGB")
            )

        if self.split == Split.TRAIN:
            return image

        label = self.samples.iloc[index].label_index
        path = self.samples.iloc[index].image_path
        if label == LabelName.ABNORMAL:
            mask = Image.open(self.samples.iloc[index].mask_path).convert("L")
            mask = self.transform_mask(mask)
        else:
            mask = torch.zeros(1, *self.gt_mask_size)

        return image, label, mask.int(), path

    # --- abstract stubs required by IadDataset ---
    def set_category(self, category: str):
        self.category = category

    def compute_contamination_ratio(self) -> float:
        raise NotImplementedError

    def contaminate(self, source, ratio, seed=42):
        raise NotImplementedError
