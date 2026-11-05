"""
evaluate_udiat_generalization.py
Evaluates trained segmentation models (U-Net, UNet++, RepSegNet V1, RepSegNet V2)
on the external UDIAT Dataset B (163 images) and In-house dataset (15 images).
Computes exact empirical out-of-distribution generalization metrics:
Accuracy, Precision, Recall/Sensitivity, F1-Score, Dice, IoU.
Generates qualitative comparison figure and saves metrics to JSON.
"""

import os
import glob
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from config import (
    DEVICE, OUTPUT_DIR, SEG_IMG_SIZE, SEG_ENCODER_NAME,
    UNET_CHECKPOINT, UNETPP_CHECKPOINT,
    REPSEGNET_CHECKPOINT, REPSEGNET_V2_CHECKPOINT
)
from unet import get_unet
from unet_plus_plus import get_unetpp
from repsegnet import get_repsegnet
from repsegnet_v2 import get_repsegnet_v2
from postprocess import postprocess_mask


# ── Metric Computation ────────────────────────────────────────────────────────
def compute_single_image_metrics(pred_bin: np.ndarray, target: np.ndarray) -> dict:
    smooth = 1e-6
    p = pred_bin.flatten().astype(int)
    t = target.flatten().astype(int)

    acc  = accuracy_score(t, p) * 100
    prec = precision_score(t, p, zero_division=0) * 100
    rec  = recall_score(t, p, zero_division=0) * 100
    f1   = f1_score(t, p, zero_division=0) * 100

    intersection = np.logical_and(p, t).sum()
    union        = np.logical_or(p, t).sum()
    dice = float(2 * intersection / (p.sum() + t.sum() + smooth)) * 100
    iou  = float((intersection + smooth) / (union + smooth)) * 100

    return {'accuracy': acc, 'precision': prec, 'recall': rec,
            'f1': f1, 'dice': dice, 'iou': iou}


def compute_seg_metrics_per_image(all_preds: list, all_targets: list) -> dict:
    metrics_list = []
    for pred, target in zip(all_preds, all_targets):
        m = compute_single_image_metrics(pred, target)
        metrics_list.append(m)

    avg = {}
    for key in metrics_list[0]:
        avg[key] = float(np.mean([m[key] for m in metrics_list]))
    return avg


# ── External Dataset Class ───────────────────────────────────────────────────
class ExternalDataset(Dataset):
    def __init__(self, image_paths, mask_paths, img_size=SEG_IMG_SIZE):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.img_size = img_size
        self.transform = A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),
            ToTensorV2(),
        ], additional_targets={"mask": "mask"})

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_p = self.image_paths[idx]
        mask_p = self.mask_paths[idx]

        img = cv2.imread(img_p)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_p, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.float32)

        augmented = self.transform(image=img, mask=mask)
        img_t = augmented["image"]
        mask_t = augmented["mask"].unsqueeze(0)
        return img_t, mask_t, img_p


# ── TTA Inference ─────────────────────────────────────────────────────────────
@torch.no_grad()
def tta_predict(model, img_batch):
    preds = []
    p0 = model(img_batch)
    preds.append(p0)

    img_hflip = torch.flip(img_batch, dims=[3])
    p1 = model(img_hflip)
    preds.append(torch.flip(p1, dims=[3]))

    img_vflip = torch.flip(img_batch, dims=[2])
    p2 = model(img_vflip)
    preds.append(torch.flip(p2, dims=[2]))

    img_hvflip = torch.flip(img_batch, dims=[2, 3])
    p3 = model(img_hvflip)
    preds.append(torch.flip(p3, dims=[2, 3]))

    return torch.stack(preds).mean(dim=0)


def evaluate_dataset(dataset_name, dataset, models_dict, device, use_tta=True):
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    results = {}
    saved_samples = {name: [] for name in models_dict}

    print(f"\n=======================================================")
    print(f"  Evaluating External Generalization: {dataset_name} (N = {len(dataset)})")
    print(f"=======================================================")

    for model_name, model in models_dict.items():
        model.eval()
        all_preds = []
        all_targets = []

        for b_idx, (imgs, masks, paths) in enumerate(loader):
            imgs = imgs.to(device)

            with torch.no_grad():
                if use_tta:
                    preds = tta_predict(model, imgs)
                else:
                    preds = model(imgs)

            preds_bin = (preds > 0.5).float()

            for j in range(imgs.size(0)):
                pred_np = preds_bin[j, 0].cpu().numpy()
                pred_clean = postprocess_mask(pred_np, min_size=150)
                target_np = masks[j, 0].numpy()

                all_preds.append(pred_clean)
                all_targets.append(target_np)

                # Save 4 samples for visual comparison
                if len(saved_samples[model_name]) < 4:
                    saved_samples[model_name].append({
                        'img': imgs[j].cpu().numpy(),
                        'gt': target_np,
                        'pred': pred_clean,
                        'path': paths[j]
                    })

        metrics = compute_seg_metrics_per_image(all_preds, all_targets)
        results[model_name] = metrics
        print(f"  [{model_name:<12}] Dice: {metrics['dice']:.2f}% | IoU: {metrics['iou']:.2f}% | Precision: {metrics['precision']:.2f}% | Recall: {metrics['recall']:.2f}%")

    return results, saved_samples


def main():
    udiat_orig = sorted(glob.glob("F:/Amit/Cancer Detection/UDIAT-Dataset/UDIAT_Dataset_B/UDIAT_Dataset_B/original/*.png"))
    udiat_gt   = sorted(glob.glob("F:/Amit/Cancer Detection/UDIAT-Dataset/UDIAT_Dataset_B/UDIAT_Dataset_B/GT/*.png"))

    inhouse_orig = sorted(glob.glob("F:/Amit/Cancer Detection/UDIAT-Dataset/In-house/Tumor with masks/images/*.png"))
    inhouse_gt   = sorted(glob.glob("F:/Amit/Cancer Detection/UDIAT-Dataset/In-house/Tumor with masks/masks/*.png"))

    udiat_ds = ExternalDataset(udiat_orig, udiat_gt)
    inhouse_ds = ExternalDataset(inhouse_orig, inhouse_gt)

    # Load models
    models = {}

    m_unet = get_unet().to(DEVICE)
    m_unet.load_state_dict(torch.load(UNET_CHECKPOINT, map_location=DEVICE))
    models["U-Net"] = m_unet

    m_unetpp = get_unetpp(deep_supervision=True).to(DEVICE)
    m_unetpp.load_state_dict(torch.load(UNETPP_CHECKPOINT, map_location=DEVICE))
    models["UNet++"] = m_unetpp

    m_rep1 = get_repsegnet().to(DEVICE)
    m_rep1.load_state_dict(torch.load(REPSEGNET_CHECKPOINT, map_location=DEVICE))
    models["RepSegNet"] = m_rep1

    m_rep2 = get_repsegnet_v2(encoder_name=SEG_ENCODER_NAME, deep_supervision=True).to(DEVICE)
    m_rep2.load_state_dict(torch.load(REPSEGNET_V2_CHECKPOINT, map_location=DEVICE))
    models["RepSegNetV2"] = m_rep2

    # 1. Run UDIAT Dataset B evaluation
    udiat_results, udiat_samples = evaluate_dataset("UDIAT_Dataset_B", udiat_ds, models, DEVICE, use_tta=True)

    # 2. Run In-house evaluation
    inhouse_results, _ = evaluate_dataset("In-house", inhouse_ds, models, DEVICE, use_tta=True)

    # Save to JSON
    final_output = {
        "UDIAT_Dataset_B": {
            "num_images": len(udiat_ds),
            "metrics": udiat_results
        },
        "In_house_Dataset": {
            "num_images": len(inhouse_ds),
            "metrics": inhouse_results
        }
    }

    out_json = os.path.join(OUTPUT_DIR, "external_dataset_generalization_metrics.json")
    with open(out_json, "w") as f:
        json.dump(final_output, f, indent=2)
    print(f"\nSaved generalization metrics JSON -> {out_json}")

    # Generate qualitative comparison figure
    # Rows: 4 samples
    # Columns: Input, Ground Truth, U-Net, UNet++, RepSegNet V1, RepSegNet V2
    fig, axes = plt.subplots(4, 6, figsize=(18, 12))
    col_titles = ["Input Image", "Ground Truth", "U-Net", "UNet++", "RepSegNet V1", "RepSegNet V2 (Proposed)"]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=13, fontweight='bold', pad=10)

    for r in range(4):
        sample_info = udiat_samples["RepSegNetV2"][r]
        img_np = np.clip(sample_info['img'].transpose(1, 2, 0), 0, 1)
        gt_np = sample_info['gt']

        axes[r, 0].imshow(img_np)
        axes[r, 0].axis('off')

        axes[r, 1].imshow(gt_np, cmap='gray')
        axes[r, 1].axis('off')

        for c_idx, m_name in enumerate(["U-Net", "UNet++", "RepSegNet", "RepSegNetV2"]):
            pred_m = udiat_samples[m_name][r]['pred']
            axes[r, c_idx + 2].imshow(pred_m, cmap='gray')
            axes[r, c_idx + 2].axis('off')

    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, "udiat_qualitative_generalization.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved qualitative generalization figure -> {fig_path}")

if __name__ == "__main__":
    main()

