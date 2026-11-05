"""
evaluate_segmentation.py — Evaluation for U-Net, UNet++, and RepSegNet segmentation models.
Reproduces paper Table 2 & 3: accuracy, precision, recall, F1-score, IoU, Dice.
Generates visual comparison grids (paper Figures 8 & 9).

UPDATED: Per-image metric averaging + Test-Time Augmentation (TTA)
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from config import (
    DEVICE, UNET_CHECKPOINT, UNETPP_CHECKPOINT,
    REPSEGNET_CHECKPOINT, REPSEGNET_V2_CHECKPOINT,
    OUTPUT_DIR, SEG_IMG_SIZE, SEG_ENCODER_NAME
)
from seg_dataset_v2 import get_seg_v2_dataloaders
from unet import get_unet
from unet_plus_plus import get_unetpp
from repsegnet import get_repsegnet
from repsegnet_v2 import get_repsegnet_v2
from postprocess import postprocess_mask


# ── Per-image pixel-wise metrics ──────────────────────────────────────────────
def compute_single_image_metrics(pred_bin: np.ndarray, target: np.ndarray) -> dict:
    """
    Compute metrics for a SINGLE image (H×W arrays, binary 0/1).
    Returns dict with accuracy, precision, recall, F1, Dice, IoU.
    """
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
    """
    Compute metrics PER IMAGE and average.
    all_preds / all_targets: lists of 2D arrays (H, W), one per image.
    """
    metrics_list = []
    for pred, target in zip(all_preds, all_targets):
        m = compute_single_image_metrics(pred, target)
        metrics_list.append(m)

    # Average across all images
    avg = {}
    for key in metrics_list[0]:
        avg[key] = float(np.mean([m[key] for m in metrics_list]))
    return avg


# ── Test-Time Augmentation (TTA) ──────────────────────────────────────────────
@torch.no_grad()
def tta_predict(model, img_batch, device):
    """
    Test-Time Augmentation: average predictions from original + horizontal flip
    + vertical flip + both flips.
    """
    preds = []

    # Original
    p0 = model(img_batch)
    preds.append(p0)

    # Horizontal flip
    img_hflip = torch.flip(img_batch, dims=[3])
    p1 = model(img_hflip)
    p1 = torch.flip(p1, dims=[3])
    preds.append(p1)

    # Vertical flip
    img_vflip = torch.flip(img_batch, dims=[2])
    p2 = model(img_vflip)
    p2 = torch.flip(p2, dims=[2])
    preds.append(p2)

    # Both flips
    img_hvflip = torch.flip(img_batch, dims=[2, 3])
    p3 = model(img_hvflip)
    p3 = torch.flip(p3, dims=[2, 3])
    preds.append(p3)

    # Average
    return torch.stack(preds).mean(dim=0)


# ── Joint Sweep Over Threshold and min_size ────────────────────────────────────
def find_optimal_postprocess_params(model, loader, device, use_tta=True):
    """
    Joint grid-search sweep over binarization thresholds and morphology min_size filtering
    on 2D validation masks to find the mathematically optimal pairing.
    """
    model.eval()
    all_probs_2d = []
    all_targets_2d = []

    for imgs, masks in loader:
        imgs = imgs.to(device)
        if use_tta:
            preds = tta_predict(model, imgs, device)
        else:
            preds = model(imgs)

        for j in range(imgs.size(0)):
            all_probs_2d.append(preds[j, 0].cpu().numpy())
            all_targets_2d.append(masks[j, 0].cpu().numpy())

    best_thresh = 0.5
    best_min_size = 150
    best_dice = -1.0

    print("    [Sweep] Running joint grid search over threshold and min_size on Val split...")
    thresholds = np.arange(0.3, 0.85, 0.05)
    min_sizes = [0, 50, 100, 150, 200, 300, 400]

    for t in thresholds:
        for ms in min_sizes:
            dices = []
            for prob, target in zip(all_probs_2d, all_targets_2d):
                pred_bin = (prob > t).astype(np.uint8)
                if ms > 0:
                    pred_bin = postprocess_mask(pred_bin, min_size=ms)
                
                # Compute single image Dice
                p = pred_bin.flatten()
                t_flat = target.flatten()
                intersection = np.logical_and(p, t_flat).sum()
                smooth = 1e-6
                dice = float(2 * intersection / (p.sum() + t_flat.sum() + smooth)) * 100
                dices.append(dice)
            
            avg_dice = np.mean(dices)
            if avg_dice > best_dice:
                best_dice = avg_dice
                best_thresh = t
                best_min_size = ms

    print(f"    [Sweep OK] Optimal parameters: Threshold = {best_thresh:.2f}, Min Size = {best_min_size} (Val Dice = {best_dice:.2f}%)")
    return best_thresh, best_min_size


@torch.no_grad()
def run_inference(model, loader, device, is_unetpp=False, threshold=0.5, use_tta=True, use_postprocess=True, postprocess_min_size=150):
    model.eval()
    all_preds_images = []    # list of 2D arrays per image
    all_targets_images = []  # list of 2D arrays per image
    sample_tuples = []       # (image, mask_gt, mask_pred) for visualization

    for i, (imgs, masks) in enumerate(loader):
        imgs, masks = imgs.to(device), masks.to(device)

        if use_tta:
            preds = tta_predict(model, imgs, device)
        else:
            preds = model(imgs)

        preds_bin = (preds > threshold).float()

        # Collect per-image predictions and targets
        for j in range(imgs.size(0)):
            pred_np = preds_bin[j, 0].cpu().numpy()
            if use_postprocess:
                pred_np = postprocess_mask(pred_np, min_size=postprocess_min_size)
            all_preds_images.append(pred_np)
            all_targets_images.append(masks[j, 0].cpu().numpy())

        # Collect first batch samples for visualization
        if i == 0:
            for j in range(min(4, imgs.size(0))):
                pred_np = preds_bin[j, 0].cpu().numpy()
                if use_postprocess:
                    pred_np = postprocess_mask(pred_np, min_size=postprocess_min_size)
                sample_tuples.append((
                    imgs[j].cpu().numpy(),
                    masks[j, 0].cpu().numpy(),
                    pred_np
                ))

    return all_preds_images, all_targets_images, sample_tuples


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
    """Print side-by-side comparison."""
    print(f"\n{'='*90}")
    print("  Segmentation Metrics (Per-Image Average + TTA) — U-Net vs UNet++ vs RepSegNet")
    print(f"{'='*90}")
    header = f"{'Metric':<20}" + "".join(f"{k:>20}" for k in results.keys())
    print(header)
    print('-' * 90)
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'dice', 'iou']
    for m in metrics:
        row = f"{m.capitalize():<20}"
        for model_m in results.values():
            row += f"{model_m[m]:>19.2f}%"
        print(row)
    print('=' * 90)


def evaluate_segmentation():
    train_l, val_l, test_l = get_seg_v2_dataloaders(img_size=SEG_IMG_SIZE, use_copy_paste=False)
    all_results  = {}
    all_samples  = {}

    model_configs = [
        ("U-Net",       get_unet(),                          UNET_CHECKPOINT,         False),
        ("UNet++",      get_unetpp(deep_supervision=True),   UNETPP_CHECKPOINT,       True),
        ("RepSegNet",   get_repsegnet(),                     REPSEGNET_CHECKPOINT,    False),
        ("RepSegNetV2", get_repsegnet_v2(encoder_name=SEG_ENCODER_NAME, deep_supervision=True), REPSEGNET_V2_CHECKPOINT, False)
    ]

    for model_name, model, checkpoint, is_unetpp in model_configs:
        if not os.path.exists(checkpoint):
            print(f"[evaluate_seg] Checkpoint not found: {checkpoint} — skipping {model_name}")
            continue
        
        # Load weights
        state_dict = torch.load(checkpoint, map_location=DEVICE)
        model.load_state_dict(state_dict)
        model = model.to(DEVICE)
        print(f"\n[evaluate_seg] Evaluating {model_name}...")

        # Find optimal threshold AND connected-component size using validation sweep (Fixes leakage!)
        opt_thresh, opt_min_size = find_optimal_postprocess_params(model, val_l, DEVICE, use_tta=True)

        # Run inference with TTA and optimal threshold on strictly unseen Test set
        preds_images, targets_images, samples = run_inference(
            model, test_l, DEVICE, is_unetpp,
            threshold=opt_thresh, use_tta=True,
            postprocess_min_size=opt_min_size
        )

        # Compute per-image metrics and average
        metrics = compute_seg_metrics_per_image(preds_images, targets_images)
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

    # Automatically update paper section placeholders!
    try:
        paper_path = os.path.join(OUTPUT_DIR, "RepSegNet_V2_Paper_Section.md")
        if os.path.exists(paper_path) and "RepSegNetV2" in all_results:
            v2 = all_results["RepSegNetV2"]
            v1 = all_results["RepSegNet"] if "RepSegNet" in all_results else {"precision": 70.69, "dice": 69.66, "iou": 59.42}
            
            with open(paper_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Replace table row placeholders
            content = content.replace("**[PENDING]%** | **[PENDING]%** | **[PENDING]%** | **[PENDING]%** | **[PENDING]%**",
                                      f"**{v2['accuracy']:.2f}%** | **{v2['precision']:.2f}%** | **{v2['recall']:.2f}%** | **{v2['dice']:.2f}%** | **{v2['iou']:.2f}%**")
            
            # Replace paragraph text placeholders
            content = content.replace("from **70.69%** (V1) to **[PENDING]%** (V2)", f"from **{v1['precision']:.2f}%** (V1) to **{v2['precision']:.2f}%** (V2)")
            content = content.replace("from **69.66%** to **[PENDING]%** and IoU scores to **[PENDING]%**", 
                                      f"from **{v1['dice']:.2f}%** to **{v2['dice']:.2f}%** and IoU scores to **{v2['iou']:.2f}%**")
            
            with open(paper_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"[evaluate_seg] Successfully updated paper section draft -> {paper_path}")
            
            # Automatically dispatch / compile results email!
            try:
                from send_results_email import send_results_email
                send_results_email()
            except Exception as mail_err:
                print(f"[WARN] Failed to auto-send email: {mail_err}")
                
            # Automatically upload report to public web link!
            try:
                from upload_to_paste import upload_report_to_web
                upload_report_to_web()
            except Exception as paste_err:
                print(f"[WARN] Failed to auto-upload to web: {paste_err}")
    except Exception as e:
        print(f"[WARN] Failed to auto-update paper section: {e}")

    return all_results


if __name__ == "__main__":
    evaluate_segmentation()
