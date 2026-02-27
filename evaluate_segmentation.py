"""
evaluate_segmentation.py — Evaluation for U-Net and UNet++ segmentation models.
Reproduces paper Table 2 & 3: accuracy, precision, recall, F1-score, IoU, Dice.
Generates visual comparison grids (paper Figures 8 & 9).
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from config import (
    DEVICE, UNET_CHECKPOINT, UNETPP_CHECKPOINT,
    OUTPUT_DIR, SEG_IMG_SIZE
)
from seg_dataset import get_seg_dataloaders
from unet import get_unet
from unet_plus_plus import get_unetpp


# ── Pixel-wise metrics ────────────────────────────────────────────────────────
def compute_seg_metrics(preds_bin: np.ndarray, targets: np.ndarray) -> dict:
    """
    Inputs: flattened binary arrays (0/1)
    Returns: dict with accuracy, precision, recall, F1, Dice, IoU
    """
    acc  = accuracy_score(targets, preds_bin) * 100
    prec = precision_score(targets, preds_bin, zero_division=0) * 100
    rec  = recall_score(targets, preds_bin, zero_division=0) * 100
    f1   = f1_score(targets, preds_bin, zero_division=0) * 100

    intersection = np.logical_and(preds_bin, targets).sum()
    union        = np.logical_or(preds_bin, targets).sum()
    smooth = 1e-6
    dice = float(2 * intersection / (preds_bin.sum() + targets.sum() + smooth)) * 100
    iou  = float((intersection + smooth) / (union + smooth)) * 100

    return {'accuracy': acc, 'precision': prec, 'recall': rec,
            'f1': f1, 'dice': dice, 'iou': iou}


@torch.no_grad()
def run_inference(model, loader, device, is_unetpp=False, threshold=0.5):
    model.eval()
    all_preds, all_targets = [], []
    sample_tuples = []  # (image, mask_gt, mask_pred) for visualization

    for i, (imgs, masks) in enumerate(loader):
        imgs, masks = imgs.to(device), masks.to(device)
        preds = model(imgs)  # [B, 1, H, W] — eval mode always returns single output

        preds_bin = (preds > threshold).float()

        all_preds.append(preds_bin.cpu().numpy().flatten())
        all_targets.append(masks.cpu().numpy().flatten())

        # Collect first batch samples for visualization
        if i == 0:
            for j in range(min(4, imgs.size(0))):
                sample_tuples.append((
                    imgs[j].cpu().numpy(),
                    masks[j, 0].cpu().numpy(),
                    preds_bin[j, 0].cpu().numpy()
                ))

    all_preds   = np.concatenate(all_preds).astype(int)
    all_targets = np.concatenate(all_targets).astype(int)
    return all_preds, all_targets, sample_tuples


def denormalize_img(img_np: np.ndarray) -> np.ndarray:
    """Convert CHW tensor (already 0-1) to HWC uint8."""
    img = np.clip(img_np.transpose(1, 2, 0), 0, 1)
    return (img * 255).astype(np.uint8)


def plot_comparison_grid(samples: list, model_name: str, save_path: str):
    """
    Generates: Input | Ground Truth | Predicted | Overlay
    Reproduces paper Figures 8 (U-Net) and 9 (UNet++).
    """
    n = len(samples)
    fig, axes = plt.subplots(n, 4, figsize=(14, 4 * n))
    if n == 1:
        axes = axes[None, :]

    col_titles = ['Input Image', 'Ground Truth', 'Predicted Mask', 'Overlay']
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=12, fontweight='bold', pad=8)

    for row, (img, gt, pred) in enumerate(samples):
        img_rgb = denormalize_img(img)

        # Overlay (predicted mask with green tint on image)
        overlay = img_rgb.copy().astype(float)
        green_mask = pred[:, :, np.newaxis] * np.array([0, 255, 0])
        overlay = np.clip(overlay * 0.6 + green_mask * 0.4, 0, 255).astype(np.uint8)

        axes[row, 0].imshow(img_rgb)
        axes[row, 1].imshow(gt, cmap='gray')
        axes[row, 2].imshow(pred, cmap='gray')
        axes[row, 3].imshow(overlay)
        for col in range(4):
            axes[row, col].axis('off')

    fig.suptitle(f'{model_name}: Predicted Mask vs Ground Truth Mask',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[evaluate_seg] Grid saved -> {save_path}")


def print_metrics_table(results: dict):
    """Print side-by-side comparison (paper Table 2)."""
    print(f"\n{'='*70}")
    print("  Segmentation Metrics — U-Net vs UNet++ (Paper Table 2 & 3)")
    print(f"{'='*70}")
    header = f"{'Metric':<20}" + "".join(f"{k:>20}" for k in results.keys())
    print(header)
    print('-' * 70)
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'dice', 'iou']
    for m in metrics:
        row = f"{m.capitalize():<20}"
        for model_m in results.values():
            row += f"{model_m[m]:>19.2f}%"
        print(row)
    print('=' * 70)


def evaluate_segmentation():
    _, _, test_l = get_seg_dataloaders()
    all_results  = {}
    all_samples  = {}

    model_configs = [
        ("U-Net",   get_unet(),                UNET_CHECKPOINT,   False),
        ("UNet++",  get_unetpp(deep_supervision=True),
                                               UNETPP_CHECKPOINT, True),
    ]

    for model_name, model, checkpoint, is_unetpp in model_configs:
        if not os.path.exists(checkpoint):
            print(f"[evaluate_seg] Checkpoint not found: {checkpoint} — skipping {model_name}")
            continue
        model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
        model = model.to(DEVICE)
        print(f"\n[evaluate_seg] Evaluating {model_name}...")

        preds, targets, samples = run_inference(model, test_l, DEVICE, is_unetpp)
        metrics = compute_seg_metrics(preds, targets)
        all_results[model_name] = metrics
        all_samples[model_name] = samples

    if not all_results:
        print("[evaluate_seg] No checkpoints found. Train first.")
        return

    print_metrics_table(all_results)

    for model_name, samples in all_samples.items():
        safe_name = model_name.replace('+', 'plus')
        plot_comparison_grid(
            samples, model_name,
            os.path.join(OUTPUT_DIR, f"seg_{safe_name}_comparison.png")
        )

    with open(os.path.join(OUTPUT_DIR, "seg_test_metrics.json"), 'w') as f:
        json.dump(all_results, f, indent=2)

    return all_results


if __name__ == "__main__":
    evaluate_segmentation()
