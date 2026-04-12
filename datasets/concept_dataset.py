import os
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import pandas as pd

from torchvision.transforms import transforms
from torch.utils.data.dataset import Dataset
from torchvision.transforms.functional import InterpolationMode


class ConceptDataset(Dataset):
    def __init__(
        self,
        dataframe,
        split: str,
        load_image: bool = True,
        multiclass: bool = False,
        apply_transformation: bool = True,
        img_size=(224, 224),
        use_attr: bool = True,
        load_mask: bool = False,
        n_class_attr: int = 2,
        random_state: int = 42,
        contaminate: bool = False,
        subsample_anomalies: bool = False,
        n_per_type: int = 0,
        original_df: pd.DataFrame = None,
    ) -> None:
        super(ConceptDataset)

        self.split = split
        self.load_image = load_image
        self.multiclass = multiclass
        self.apply_transformation = apply_transformation
        self.use_attr = use_attr
        self.load_mask = load_mask
        self.n_class_attr = n_class_attr
        self.subsample_anomalies = subsample_anomalies
        self.random_state = random_state
        self.n_per_type = n_per_type
        self.df = dataframe[dataframe["split"] == split].reset_index(drop=True)

        if contaminate and split == "train" and original_df is not None:
            self.df = self.contaminate_with_originals(
                self.df,
                original_df,
                n_per_type=self.n_per_type,
                random_state=random_state,
            )

        if split == "train" and self.subsample_anomalies:
            self.df = self._subsample_anomalies(self.df, random_state)

        if self.multiclass:
            self.num_classes = self.df["category_index"].nunique()

        exclude_cols = [
            "image_path",
            "label_index",
            "mask_path",
            "split",
            "anomaly_type",
            "view"
        ]

        self.attr_cols = [col for col in self.df.columns if col not in exclude_cols]

        pre_transform = transforms.Compose(
            [
                transforms.Resize(img_size),
                transforms.ToTensor(),
            ]
        )

        transform = (
            transforms.Compose(
                [
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
            if apply_transformation
            else None
        )

        transform_augment = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=25),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        self.pre_transform = pre_transform
        self.transform = transform
        self.transform_augment = transform_augment

        # logo mask removal
        self.transform_mask = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(
                    img_size,
                    antialias=True,
                    interpolation=InterpolationMode.NEAREST,
                ),
            ]
        )
        mask_gemini_logo_path = os.environ.get("GEMINI_LOGO_MASK_PATH")
        if isinstance(mask_gemini_logo_path, str):
            mask_gemini_logo_path = mask_gemini_logo_path.strip()
            self.mask_gemini_logo_path = Path(mask_gemini_logo_path) if mask_gemini_logo_path else None
        else:
            self.mask_gemini_logo_path = mask_gemini_logo_path
        # load and save the gemini logo mask if provided
        if self.mask_gemini_logo_path is not None:
            if not self.mask_gemini_logo_path.exists() or not self.mask_gemini_logo_path.is_file():
                raise FileNotFoundError(
                    f"Gemini logo mask file not found: {self.mask_gemini_logo_path}"
                )
            # load the image with PIL
            logo_mask_img = Image.open(self.mask_gemini_logo_path).convert("L") 
            # transform the mask
            self.gemini_logo_mask = self.transform_mask(logo_mask_img)
        else:
            self.gemini_logo_mask = None

    def _subsample_anomalies(self, df, random_state):
        normal_df = df[df["label_index"] == 0]
        anomalous_df = df[df["label_index"] == 1]

        if anomalous_df.empty:
            return df

        subsampled_anomalies = anomalous_df.groupby(
            "anomaly_type", group_keys=False
        ).apply(
            lambda x: x.sample(n=min(self.n_per_type, len(x)), random_state=random_state)
        )

        new_df = (
            pd.concat([normal_df, subsampled_anomalies], axis=0)
            .sample(frac=1.0, random_state=random_state)
            .reset_index(drop=True)
        )

        print(
            f"Reduced training set: kept only {self.n_per_type} anomalous images per defect type."
        )

        return new_df

    def contaminate_with_originals(
        self, df_generated, df_original, n_per_type=3, random_state=42
    ):
        np.random.seed(random_state)

        # Only use anomaly rows (exclude "good")
        original_anomalies = df_original[df_original["anomaly_type"] != "good"]

        contaminated_samples = []
        for anomaly_type, group in original_anomalies.groupby("anomaly_type"):
            if len(group) >= n_per_type:
                sampled = group.sample(n=n_per_type, random_state=random_state)
            else:
                sampled = group  # if fewer than 3 available, take all
            contaminated_samples.append(sampled)

        if not contaminated_samples:
            return df_generated  # nothing to add

        contamination_df = pd.concat(contaminated_samples, axis=0)
        combined_df = (
            pd.concat([df_generated, contamination_df], axis=0)
            .sample(frac=1.0, random_state=random_state)
            .reset_index(drop=True)
        )

        print(
            f"Contaminated training set: added {len(contamination_df)} original anomaly images."
        )
        return combined_df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = row["image_path"]
        mask_path = row["mask_path"]

        image = Image.open(image_path).convert("RGB")

        if self.load_mask:
            if isinstance(mask_path, str):
                mask = Image.open(mask_path).convert("L")
            else:
                mask = Image.new("L", image.size, color=0)
            mask = transforms.ToTensor()(mask)

        label = row["category_index"] if self.multiclass else row["label_index"]

        image = self.pre_transform(image)

        if self.gemini_logo_mask is not None:
            image = image * self.gemini_logo_mask

        if self.apply_transformation:
            image = (
                self.transform_augment(image)
                if self.split == "train"
                else self.transform(image)
            )

        if self.use_attr:
            attr_label = torch.Tensor(row[self.attr_cols].values.astype(np.float32))
            if self.load_image:
                return image, attr_label, label
            return attr_label, label

        if self.load_mask:
            return image, label, mask

        return image, label

    def find_class_imbalance(self, type="main"):
        num_total = len(self.df)
        if type == "main":
            label_counts = self.df["label_index"].value_counts().to_dict()
            num_positives = label_counts.get(1, 0)

            imbalance_ratio = num_total / num_positives - 1
            contamination_ratio = label_counts[1] / label_counts[0]

        elif type == "attributes":
            imbalance_ratio = []
            for attr in self.attr_cols:
                num_positives = self.df[attr].sum()

                if num_positives > 0:
                    imbalance_ratio.append(num_total / num_positives - 1)
                else:
                    imbalance_ratio.append(0)

        if type == "main":
            return imbalance_ratio, contamination_ratio
        elif type == "attributes":
            return imbalance_ratio
