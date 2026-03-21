import os
import torch
import pandas as pd
import numpy as np
import torchxrayvision as xrv

from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, random_split


# Chest X-ray pathology labels
LABELS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Effusion", "Emphysema", "Fibrosis", "Hernia",
    "Infiltration", "Mass", "No Finding", "Nodule",
    "Pleural_Thickening", "Pneumonia", "Pneumothorax"
]

NUM_CLASSES = len(LABELS)


class NIHChestDataset(Dataset):
    """
    NIH ChestX-ray14 dataset.

    Expects:
        csv_path  : path to Data_Entry_2017.csv
        img_dir   : root directory containing all PNG images
        transform : torchvision transform pipeline
    
    Label encoding:
        Pipe-delimited strings ("Pneumonia|Effusion") are parsed into
        a float32 binary vector of length NUM_CLASSES.
    """
    def __init__(self, df: pd.DataFrame, img_dir: str, transform=None, model_name: str = "densenet121"):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.model_name = model_name

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["Image Index"])
        if self.model_name == "densenet121-xrv":
            image = Image.open(img_path).convert("L")
            image = np.array(image, dtype=np.float32)
            image = image[np.newaxis, ...]
            image = xrv.datasets.normalize(image, maxval=255, reshape=True)
        else:
            image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # Parse pipe-delimited labels into binary vector
        findings = row["Finding Labels"].split("|")
        label= torch.zeros(NUM_CLASSES, dtype=torch.float32)
        # print(f"NUM_CLASSES: {NUM_CLASSES}, LABEEL SHAPE: {label.shape}")
        for finding in findings:
            finding = finding.strip()
            if finding in LABELS:
                label[LABELS.index(finding)] = 1.0

        return image, label


def make_transforms(img_size: int, model_name: str = "densenet121"):
    if model_name == "densenet121-xrv":
        # XRV pipeline: grayscale, XRV normalization to [-1024, 1024]
        # XRayCenterCrop and XRayResizer handle sizing
        train_tf = transforms.Compose([
            xrv.datasets.XRayResizer(img_size),
            xrv.datasets.XRayCenterCrop(),
            # transforms.RandomHorizontalFlip(p=0.5),
            # transforms.RandomRotation(10),
        ])
        eval_tf = transforms.Compose([
            xrv.datasets.XRayResizer(img_size),
            xrv.datasets.XRayCenterCrop(),
        ])
    else:
        # Standard ImageNet pipeline for ResNet
        train_tf = transforms.Compose([
            transforms.Resize(int(img_size * 1.2)),
            transforms.CenterCrop(img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),  # minor patient positioning shifts
            transforms.ColorJitter(brightness=0.2, contrast=0.2),        # exposure/contrast variation between machines
            transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3), # varies image clarity
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225])
        ])
        eval_tf = transforms.Compose([
            transforms.Resize(int(img_size * 1.2)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225])
        ])
    return train_tf, eval_tf


def make_loaders(data_dir: str, img_size: int, batch_size: int, num_workers: int, model_name: str = "densenet121"):
    train_tf, eval_tf = make_transforms(img_size, model_name)

    # Load and split csv metadata
    csv_path = os.path.join(data_dir, "Data_Entry_2017.csv")
    df = pd.read_csv(csv_path)
    df = df[df["View Position"] == "PA"].reset_index(drop=True)
    # Patient-level train/val/test split: 70/15/15
    # Splitting by Patient ID prevents the same patient appearing in multiple
    # splits, which would allow the model to learn patient identity shortcuts
    # rather than genuine pathological features (patient leakage).
    patient_ids = df["Patient ID"].unique()
    rng = np.random.default_rng(seed=42)
    rng.shuffle(patient_ids)

    n = len(patient_ids)
    train_ids = set(patient_ids[:int(0.70 * n)])
    val_ids   = set(patient_ids[int(0.70 * n):int(0.85 * n)])
    test_ids  = set(patient_ids[int(0.85 * n):])

    train_df = df[df["Patient ID"].isin(train_ids)].reset_index(drop=True)
    val_df   = df[df["Patient ID"].isin(val_ids)].reset_index(drop=True)
    test_df  = df[df["Patient ID"].isin(test_ids)].reset_index(drop=True)

    img_dir = os.path.join(data_dir, "images")
    train_ds = NIHChestDataset(train_df, img_dir, transform=train_tf, model_name=model_name)
    val_ds   = NIHChestDataset(val_df,   img_dir, transform=eval_tf, model_name=model_name)
    test_ds  = NIHChestDataset(test_df,  img_dir, transform=eval_tf, model_name=model_name)

    # Multi-label Weighted sampling to handle class imbalance
    # Strategy: weight each sample by the rarity of its rarest positive label
    # This ensures uncommon conditions get adequate representation during training
    label_matrix = _build_label_matrix(train_df)                 # (N, NUM_CLASSES)
    label_counts = label_matrix.sum(axis=0).clip(min=1)          # count per label
    label_weights = 1.0 / np.sqrt(label_counts)                  # inverse frequency smoothed by sqrt
    sample_weights = label_matrix.dot(label_weights)             # sum weights of positive labels
    sample_weights = sample_weights / sample_weights.sum()       # normalize

    sampler = WeightedRandomSampler(
        weights=sample_weights.tolist(),
        num_samples=len(train_ds),
        replacement=True
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True,
                            persistent_workers=num_workers > 0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True,
                             persistent_workers=num_workers > 0)
    
    return train_loader, val_loader, test_loader


def _build_label_matrix(df: pd.DataFrame) -> np.ndarray:
    """
    Build a binary label matrix of shape (N, NUM_CLASSES) from the DataFrame.
    Each row corresponds to a sample, and each column corresponds to a label.
    """
    label_matrix = np.zeros((len(df), NUM_CLASSES), dtype=np.float32)
    for i, labels in enumerate(df["Finding Labels"]):
        for label in str(labels).split("|"):
            label = label.strip()
            if label in LABELS:
                label_matrix[i, LABELS.index(label)] = 1.0
    return label_matrix

