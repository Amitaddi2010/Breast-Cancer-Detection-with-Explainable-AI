"""
seg_dataset_v2.py — Enhanced Segmentation Dataset for RepSegNet V2.

Key improvements over V1:
  1. Higher resolution: 512×512 (vs 384 in V1) for finer boundary detail
  2. Copy-Paste augmentation: paste lesions from one image onto another
  3. Progressive resizing support (256 → 384 → 512)
  4. Stratified K-Fold cross-validation support
  5. Stronger spatial + intensity augmentations
  6. Safe GridDropout instead of CoarseDropout
"""

import os
import re
import glob
import random
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split, StratifiedKFold

import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from config import (
    DATA_DIR, BATCH_SIZE, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_SEED
)

# Only classes with tumor masks
SEG_CLASSES = ["benign", "malignant"]

MASK_RE = re.compile(r'_mask(_\d+)?\.png$', re.IGNORECASE)
IMG_RE  = re.compile(r'_mask', re.IGNORECASE)


# ── Augmentation pipelines ─────────────────────────────────────────────────────

def get_v2_train_transform(img_size=512):
    """Enhanced training augmentations for V2."""
    return A.Compose([
        A.Resize(img_size, img_size),

        # === Spatial augmentations ===
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.ShiftScaleRotate(
            shift_limit=0.1, scale_limit=0.2, rotate_limit=45,
            border_mode=0, p=0.7
        ),
        A.ElasticTransform(alpha=150, sigma=150 * 0.05, p=0.3),
        A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.3),
        A.Perspective(scale=(0.05, 0.1), p=0.2),

        # === Intensity augmentations (ultrasound-specific) ===
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.6),
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.5),
        A.GaussNoise(p=0.3),
        A.RandomGamma(gamma_limit=(70, 130), p=0.4),
        A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.3),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),

        # === Regularization (safe for small lesions) ===
        A.GridDropout(ratio=0.3, p=0.2),

        A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),
        ToTensorV2(),
    ], additional_targets={"mask": "mask"})


def get_v2_val_transform(img_size=512):
    """Validation / test transform (no augmentation)."""
    return A.Compose([
        A.Resize(img_size, img_size),
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

    mask_base = os.path.join(folder, basename + "_mask.png")
    extra_pattern = os.path.join(folder, basename + "_mask_*.png")

    all_mask_paths = []
    if os.path.exists(mask_base):
        all_mask_paths.append(mask_base)
    all_mask_paths.extend(glob.glob(extra_pattern))

    if not all_mask_paths:
        img = np.array(Image.open(image_path))
        return np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

    merged = None
    for mp in all_mask_paths:
        m = np.array(Image.open(mp).convert("L"))
        binary = (m > 127).astype(np.uint8)
        merged = binary if merged is None else np.logical_or(merged, binary).astype(np.uint8)

    return merged  # HxW, values 0/1


# ── Copy-Paste Augmentation ───────────────────────────────────────────────────
class CopyPasteAugmentor:
    """
    Copy-paste augmentation: extract lesion from a donor image and paste
    onto the current image at a random location.
    """
    def __init__(self, donor_samples, p=0.3):
        self.donor_samples = donor_samples
        self.p = p

    def __call__(self, image, mask):
        if random.random() > self.p or len(self.donor_samples) == 0:
            return image, mask

        # Pick a random donor
        donor_path = random.choice(self.donor_samples)
        donor_img = np.array(Image.open(donor_path).convert("RGB"))
        donor_mask = _load_merged_mask(donor_path)

        # Check donor has a valid lesion
        if donor_mask.sum() < 100:  # skip tiny masks
            return image, mask

        # Find bounding box of donor lesion
        ys, xs = np.where(donor_mask > 0)
        y1, y2, x1, x2 = ys.min(), ys.max(), xs.min(), xs.max()

        # Extract lesion patch
        lesion_img = donor_img[y1:y2+1, x1:x2+1]
        lesion_mask = donor_mask[y1:y2+1, x1:x2+1]

        # Scale lesion to fit within target image
        lh, lw = lesion_mask.shape
        th, tw = image.shape[:2]

        # Random scale factor (0.5 to 1.5)
        scale = random.uniform(0.5, 1.2)
        new_h = min(int(lh * scale), th // 2)
        new_w = min(int(lw * scale), tw // 2)

        if new_h < 10 or new_w < 10:
            return image, mask

        lesion_img_r = np.array(Image.fromarray(lesion_img).resize((new_w, new_h)))
        lesion_mask_r = np.array(Image.fromarray(lesion_mask * 255).resize(
            (new_w, new_h), Image.NEAREST)) > 127

        # Random paste location
        py = random.randint(0, th - new_h)
        px = random.randint(0, tw - new_w)

        # Paste with blending
        result_img = image.copy()
        result_mask = mask.copy()

        # Alpha blending for smooth edges
        alpha = lesion_mask_r.astype(np.float32)
        for c in range(3):
            result_img[py:py+new_h, px:px+new_w, c] = (
                alpha * lesion_img_r[:, :, c] +
                (1 - alpha) * result_img[py:py+new_h, px:px+new_w, c]
            ).astype(np.uint8)

        result_mask[py:py+new_h, px:px+new_w] = np.logical_or(
            result_mask[py:py+new_h, px:px+new_w],
            lesion_mask_r
        ).astype(np.uint8)

        return result_img, result_mask


# ── Dataset class ─────────────────────────────────────────────────────────────
class BUSISegDatasetV2(Dataset):
    """
    Enhanced segmentation dataset with copy-paste augmentation support.
    Returns (image_tensor [3,H,W], mask_tensor [1,H,W]).
    """

    def __init__(self, samples: list, transform=None,
                 copy_paste: CopyPasteAugmentor = None):
        self.samples    = samples
        self.transform  = transform
        self.copy_paste = copy_paste

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        image    = np.array(Image.open(img_path).convert("RGB"))     # HxWx3
        mask     = _load_merged_mask(img_path)                        # HxW

        # Copy-paste augmentation (before spatial transforms)
        if self.copy_paste is not None:
            image, mask = self.copy_paste(image, mask)

        if self.transform:
            result = self.transform(image=image, mask=mask.astype(np.float32))
            image  = result["image"]           # Tensor [3,H,W]
            mask   = result["mask"]            # Tensor [H,W]
        else:
            image = torch.tensor(image).permute(2, 0, 1).float() / 255.0
            mask  = torch.tensor(mask).float()

        mask = mask.unsqueeze(0)  # [1,H,W]
        return image, mask


# ── Collect samples with class labels ──────────────────────────────────────────
def _collect_seg_samples_with_labels(data_dir: str):
    """Returns (samples_list, labels_list) where label=class index."""
    samples = []
    labels = []
    for cls_idx, cls in enumerate(SEG_CLASSES):
        folder = os.path.join(data_dir, cls)
        if not os.path.isdir(folder):
            continue
        for fname in sorted(os.listdir(folder)):
            if fname.lower().endswith('.png') and not IMG_RE.search(fname):
                samples.append(os.path.join(folder, fname))
                labels.append(cls_idx)
    return samples, labels


# ── DataLoaders (single split — same as V1 for fair comparison) ───────────────
def get_seg_v2_dataloaders(data_dir=DATA_DIR, batch_size=BATCH_SIZE,
                           img_size=512, use_copy_paste=True):
    """
    Create train/val/test loaders with V2 augmentations.
    Uses the SAME split ratios and random seed as V1 for fair comparison.
    """
    samples, labels = _collect_seg_samples_with_labels(data_dir)

    # Same split as V1
    train_s, temp_s, train_l, temp_l = train_test_split(
        samples, labels,
        test_size=(VAL_RATIO + TEST_RATIO),
        random_state=RANDOM_SEED, stratify=labels
    )
    val_ratio_adj = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
    val_s, test_s, val_l, test_l = train_test_split(
        temp_s, temp_l,
        test_size=(1 - val_ratio_adj),
        random_state=RANDOM_SEED, stratify=temp_l
    )

    # Copy-paste augmentor uses training samples as donors
    copy_paste = CopyPasteAugmentor(train_s, p=0.3) if use_copy_paste else None

    train_ds = BUSISegDatasetV2(train_s,
                                 transform=get_v2_train_transform(img_size),
                                 copy_paste=copy_paste)
    val_ds   = BUSISegDatasetV2(val_s,
                                 transform=get_v2_val_transform(img_size))
    test_ds  = BUSISegDatasetV2(test_s,
                                 transform=get_v2_val_transform(img_size))

    print(f"[seg_dataset_v2] Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    print(f"[seg_dataset_v2] Image size: {img_size}×{img_size} | Copy-Paste: {use_copy_paste}")

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=0, pin_memory=True,
                              drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size,
                              shuffle=False, num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=batch_size,
                              shuffle=False, num_workers=0, pin_memory=True)

    return train_loader, val_loader, test_loader


# ── K-Fold DataLoaders ────────────────────────────────────────────────────────
def get_seg_v2_kfold_loaders(fold_idx, n_folds=5, data_dir=DATA_DIR,
                              batch_size=BATCH_SIZE, img_size=512,
                              use_copy_paste=True):
    """
    Create train/val loaders for a specific fold of stratified K-Fold CV.
    Test set is held out separately (same as single-split).
    """
    samples, labels = _collect_seg_samples_with_labels(data_dir)

    # Hold out test set first (same as single split)
    trainval_s, test_s, trainval_l, test_l = train_test_split(
        samples, labels,
        test_size=TEST_RATIO,
        random_state=RANDOM_SEED, stratify=labels
    )

    # K-fold on the remaining trainval
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    folds = list(skf.split(trainval_s, trainval_l))
    train_idx, val_idx = folds[fold_idx]

    train_s = [trainval_s[i] for i in train_idx]
    val_s   = [trainval_s[i] for i in val_idx]

    copy_paste = CopyPasteAugmentor(train_s, p=0.3) if use_copy_paste else None

    train_ds = BUSISegDatasetV2(train_s,
                                 transform=get_v2_train_transform(img_size),
                                 copy_paste=copy_paste)
    val_ds   = BUSISegDatasetV2(val_s,
                                 transform=get_v2_val_transform(img_size))
    test_ds  = BUSISegDatasetV2(test_s,
                                 transform=get_v2_val_transform(img_size))

    print(f"[seg_dataset_v2] Fold {fold_idx+1}/{n_folds}: "
          f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=0, pin_memory=True,
                              drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size,
                              shuffle=False, num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=batch_size,
                              shuffle=False, num_workers=0, pin_memory=True)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    tl, vl, tel = get_seg_v2_dataloaders(img_size=512)
    imgs, masks = next(iter(tl))
    print(f"Image batch: {imgs.shape}, Mask batch: {masks.shape}")
    print(f"Mask unique values: {masks.unique()}")
    print(f"Mask foreground ratio: {masks.mean():.4f}")
