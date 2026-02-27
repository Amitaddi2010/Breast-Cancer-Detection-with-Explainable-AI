"""
dataset.py — Classification Dataset for breast cancer ultrasound images.
Loads images from the 3-class BUSI folder structure.
"""

import os
import re
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from config import (
    DATA_DIR, CLASS_NAMES, CLF_IMG_SIZE,
    BATCH_SIZE, TRAIN_RATIO, VAL_RATIO, TEST_RATIO,
    RANDOM_SEED, NORMALIZE_MEAN, NORMALIZE_STD
)

# ── Augmentation pipelines ────────────────────────────────────────────────────
def get_train_transform():
    return A.Compose([
        A.Resize(CLF_IMG_SIZE, CLF_IMG_SIZE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.Rotate(limit=20, p=0.5),
        A.RandomBrightnessContrast(p=0.4),
        A.ElasticTransform(p=0.3),
        A.CoarseDropout(num_holes_range=(1,8), hole_height_range=(8,16), hole_width_range=(8,16), p=0.3),
        A.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
        ToTensorV2(),
    ])

def get_val_transform():
    return A.Compose([
        A.Resize(CLF_IMG_SIZE, CLF_IMG_SIZE),
        A.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
        ToTensorV2(),
    ])

# ── Mask filename pattern ─────────────────────────────────────────────────────
MASK_RE = re.compile(r'_mask', re.IGNORECASE)

def is_mask_file(filename: str) -> bool:
    return bool(MASK_RE.search(filename))

# ── Dataset class ─────────────────────────────────────────────────────────────
class BUSIDataset(Dataset):
    """Breast Ultrasound Image classification dataset (3 classes)."""

    def __init__(self, samples: list, transform=None):
        """
        Args:
            samples: list of (image_path, label_int)
            transform: albumentations transform pipeline
        """
        self.samples   = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = np.array(Image.open(path).convert("RGB"))

        if self.transform:
            image = self.transform(image=image)["image"]
        else:
            image = torch.tensor(image).permute(2, 0, 1).float() / 255.0

        return image, torch.tensor(label, dtype=torch.long)


# ── Build sample list ─────────────────────────────────────────────────────────
def _collect_samples(data_dir: str, class_names: list) -> list:
    """Collect (image_path, label_int) pairs, excluding mask files."""
    samples = []
    for label_idx, class_name in enumerate(class_names):
        folder = os.path.join(data_dir, class_name)
        if not os.path.isdir(folder):
            continue
        for fname in sorted(os.listdir(folder)):
            if fname.lower().endswith('.png') and not is_mask_file(fname):
                samples.append((os.path.join(folder, fname), label_idx))
    return samples


# ── DataLoaders ───────────────────────────────────────────────────────────────
def get_dataloaders(data_dir=DATA_DIR, class_names=CLASS_NAMES,
                    batch_size=BATCH_SIZE):
    """
    Returns train/val/test DataLoaders with stratified splits.
    """
    all_samples = _collect_samples(data_dir, class_names)
    labels = [s[1] for s in all_samples]

    # Stratified split: train / (val+test)
    train_s, temp_s, _, temp_l = train_test_split(
        all_samples, labels,
        test_size=(VAL_RATIO + TEST_RATIO),
        stratify=labels,
        random_state=RANDOM_SEED
    )
    val_ratio_adjusted = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
    val_s, test_s = train_test_split(
        temp_s,
        test_size=(1 - val_ratio_adjusted),
        stratify=temp_l,
        random_state=RANDOM_SEED
    )

    train_ds = BUSIDataset(train_s, transform=get_train_transform())
    val_ds   = BUSIDataset(val_s,   transform=get_val_transform())
    test_ds  = BUSIDataset(test_s,  transform=get_val_transform())

    print(f"[dataset.py] Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                              shuffle=False, num_workers=0, pin_memory=True)

    return train_loader, val_loader, test_loader


def get_class_weights(data_dir=DATA_DIR, class_names=CLASS_NAMES) -> torch.Tensor:
    """Compute inverse-frequency class weights for imbalanced training."""
    samples = _collect_samples(data_dir, class_names)
    from collections import Counter
    counts = Counter(s[1] for s in samples)
    total  = sum(counts.values())
    weights = torch.tensor(
        [total / (len(class_names) * counts[i]) for i in range(len(class_names))],
        dtype=torch.float
    )
    return weights


if __name__ == "__main__":
    train_l, val_l, test_l = get_dataloaders()
    imgs, labels = next(iter(train_l))
    print(f"Batch shape: {imgs.shape}, Labels: {labels}")
    print(f"Class weights: {get_class_weights()}")
