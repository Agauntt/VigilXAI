import matplotlib.pyplot as plt
import torch
import os, argparse
import pandas as pd
import numpy as np

from torchvision import datasets
from PIL import Image

from data import make_transforms, make_loaders


def visualize_transforms():
    train_tf, eval_tf = make_transforms(224)
    val_ds = datasets.ImageFolder(root=f"data/val", transform=eval_tf)

    for i in range(5):
        img, label = val_ds[i]

        img = img.clone()
        img = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = img * torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        img = img.permute(1, 2, 0).numpy()
        # img = img.clamp(0, 1)

        plt.imshow(img)
        plt.title(f"Label: {label}")
        plt.axis('off')
        plt.show()

    img_path, _, = val_ds.samples[0]

    original = Image.open(img_path).convert('RGB')
    transformed = eval_tf(original)

    transformed = transformed.clone()
    transformed = transformed * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    transformed = transformed * torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    transformed = transformed.permute(1, 2, 0).numpy()

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(original)
    plt.title("Original")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(transformed)
    plt.title("After Transform")
    plt.axis('off')
    plt.show()

def patient_overlap():
    df = pd.read_csv("data/NIH_CXR/Data_Entry_2017.csv")
    train, val, test = make_loaders("data/NIH_CXR", 224, 32, 2)

    # Check for leakage
    train_ids = set(df.loc[train.dataset.df.index, "Patient ID"])
    val_ids   = set(df.loc[val.dataset.df.index,   "Patient ID"])
    print("Overlap:", len(train_ids & val_ids))  # should be 0    


def patient_mapping():
    df = pd.read_csv("data/NIH_CXR/Data_Entry_2017.csv")

    patient_ids = df["Patient ID"].unique()
    rng = np.random.default_rng(seed=42)
    rng.shuffle(patient_ids)

    n = len(patient_ids)
    train_ids = set(patient_ids[:int(0.70 * n)])
    val_ids   = set(patient_ids[int(0.70 * n):int(0.85 * n)])

    train_df = df[df["Patient ID"].isin(train_ids)]
    val_df   = df[df["Patient ID"].isin(val_ids)]

    overlap = set(train_df["Patient ID"]) & set(val_df["Patient ID"])
    print(f"Unique patients total: {len(patient_ids)}")
    print(f"Train patients: {len(train_ids)}")
    print(f"Val patients: {len(val_ids)}")
    print(f"Overlap: {len(overlap)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    args = ap.parse_args()
    if args.run == "transforms":
        visualize_transforms()
    elif args.run == "leakage":
        patient_overlap()
    elif args.run == "mapping":
        patient_mapping()
    else:
        print(f"Unknown run type: {args.run}")