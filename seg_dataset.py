"""
seg_dataset.py — Segmentation Dataset for U-Net / UNet++ training.
Loads (image, binary_mask) pairs from benign + malignant folders.
Preprocessing aligned with paper Section 3.2: resize 128×128, normalize [0,1].
"""

import os
import re
import glob
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from config import (
    DATA_DIR, SEG_IMG_SIZE, BATCH_SIZE,
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_SEED
)

# Only classes with tumor masks
SEG_CLASSES = ["benign", "malignant"]

MASK_RE    = re.compile(r'_mask(_\d+)?\.png$', re.IGNORECASE)
IMG_RE     = re.compile(r'_mask', re.IGNORECASE)


# ── Augmentation pipelines ─────────────────────────────────────────────────────
def get_seg_train_transform():
    return A.Compose([
        A.Resize(SEG_IMG_SIZE, SEG_IMG_SIZE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.Rotate(limit=20, p=0.5),
        A.RandomBrightnessContrast(p=0.4),
        A.ElasticTransform(p=0.3),
        A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),  # [0,1] per paper
        ToTensorV2(),
    ], additional_targets={"mask": "mask"})

def get_seg_val_transform():
    return A.Compose([
        A.Resize(SEG_IMG_SIZE, SEG_IMG_SIZE),
        A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),
        ToTensorV2(),
    ], additional_targets={"mask": "mask"})


# ── Helper: merge multiple masks (OR) ─────────────────────────────────────────
def _load_merged_mask(image_path: str) -> np.ndarray:
    """
    Load all mask files for an image (incl. _mask_1, _mask_2) and
    merge via logical OR → single binary mask.
    """
    folder   = os.path.dirname(image_path)
    basename = os.path.splitext(os.path.basename(image_path))[0]

    # Primary mask pattern: e.g. "benign (1)_mask.png"
    mask_base = os.path.join(folder, basename + "_mask.png")
    # Additional masks: "...._mask_1.png", "...._mask_2.png", etc.
    extra_pattern = os.path.join(folder, basename + "_mask_*.png")

    all_mask_paths = []
    if os.path.exists(mask_base):
        all_mask_paths.append(mask_base)
    all_mask_paths.extend(glob.glob(extra_pattern))

    if not all_mask_paths:
        # No mask → return zeros
        img = np.array(Image.open(image_path))
        return np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

    merged = None
    for mp in all_mask_paths:
        m = np.array(Image.open(mp).convert("L"))
        binary = (m > 127).astype(np.uint8)
        merged = binary if merged is None else np.logical_or(merged, binary).astype(np.uint8)

    return merged  # HxW, values 0/1


# ── Dataset class ─────────────────────────────────────────────────────────────
class BUSISegDataset(Dataset):
    """
    Segmentation dataset: returns (image_tensor [3,H,W], mask_tensor [1,H,W]).
    """

    def __init__(self, samples: list, transform=None):
        self.samples   = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        image    = np.array(Image.open(img_path).convert("RGB"))    # HxWx3
        mask     = _load_merged_mask(img_path)                       # HxW

        if self.transform:
            result = self.transform(image=image, mask=mask.astype(np.float32))
            image  = result["image"]            # Tensor [3,H,W]
            mask   = result["mask"]             # Tensor [H,W]
        else:
            image = torch.tensor(image).permute(2, 0, 1).float() / 255.0
            mask  = torch.tensor(mask).float()

        mask = mask.unsqueeze(0)  # [1,H,W]
        return image, mask


# ── Collect samples ────────────────────────────────────────────────────────────
def _collect_seg_samples(data_dir: str) -> list:
    samples = []
    for cls in SEG_CLASSES:
        folder = os.path.join(data_dir, cls)
        if not os.path.isdir(folder):
            continue
        for fname in sorted(os.listdir(folder)):
            if fname.lower().endswith('.png') and not IMG_RE.search(fname):
                samples.append(os.path.join(folder, fname))
    return samples


# ── DataLoaders ───────────────────────────────────────────────────────────────
def get_seg_dataloaders(data_dir=DATA_DIR, batch_size=BATCH_SIZE):
    all_samples = _collect_seg_samples(data_dir)

    train_s, temp_s = train_test_split(all_samples,
                                       test_size=(VAL_RATIO + TEST_RATIO),
                                       random_state=RANDOM_SEED)
    val_ratio_adj = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
    val_s, test_s = train_test_split(temp_s,
                                     test_size=(1 - val_ratio_adj),
                                     random_state=RANDOM_SEED)

    train_ds = BUSISegDataset(train_s, transform=get_seg_train_transform())
    val_ds   = BUSISegDataset(val_s,   transform=get_seg_val_transform())
    test_ds  = BUSISegDataset(test_s,  transform=get_seg_val_transform())

    print(f"[seg_dataset.py] Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                              shuffle=False, num_workers=0, pin_memory=True)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    tl, vl, tel = get_seg_dataloaders()
    imgs, masks = next(iter(tl))
    print(f"Image batch: {imgs.shape}, Mask batch: {masks.shape}")
    print(f"Mask unique values: {masks.unique()}")
