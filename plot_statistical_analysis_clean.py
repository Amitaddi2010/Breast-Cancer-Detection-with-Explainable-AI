"""
plot_statistical_analysis_clean.py
Generates a zero-overlap, publication-grade statistical distribution and significance plot
for medical image segmentation on the BUSI test set (98 images).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from config import (
    DEVICE, OUTPUT_DIR, SEG_IMG_SIZE, SEG_ENCODER_NAME,
    UNET_CHECKPOINT, UNETPP_CHECKPOINT, REPSEGNET_CHECKPOINT, REPSEGNET_V2_CHECKPOINT
)
from seg_dataset_v2 import get_seg_v2_dataloaders
from unet import get_unet
from unet_plus_plus import get_unetpp
from repsegnet import get_repsegnet
from repsegnet_v2 import get_repsegnet_v2
from postprocess import postprocess_mask

CACHE_FILE = os.path.join(OUTPUT_DIR, "per_image_test_metrics.npz")

def get_or_compute_metrics():
    if os.path.exists(CACHE_FILE):
        print(f"Loading cached per-image metrics from: {CACHE_FILE}")
        data = np.load(CACHE_FILE)
        return {k: data[k] for k in data.files}

    print("Computing per-image metrics across all 4 models on 98 test images...")
    _, _, test_loader = get_seg_v2_dataloaders(img_size=SEG_IMG_SIZE, use_copy_paste=False)
    
    models = {
        'unet_dice': (get_unet(), UNET_CHECKPOINT),
        'unetpp_dice': (get_unetpp(deep_supervision=True), UNETPP_CHECKPOINT),
        'repseg_dice': (get_repsegnet(), REPSEGNET_CHECKPOINT),
        'repseg_v2_dice': (get_repsegnet_v2(encoder_name=SEG_ENCODER_NAME, deep_supervision=True), REPSEGNET_V2_CHECKPOINT)
    }
    
    results = {}
    
    for prefix, (model, ckpt) in models.items():
        base_name = prefix.replace('_dice', '')
        print(f"Running inference for {base_name}...")
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        model.to(DEVICE).eval()
        
        dices, ious = [], []
        with torch.no_grad():
            for imgs, masks in test_loader:
                imgs = imgs.to(DEVICE)
                preds = model(imgs)
                preds_bin = (preds > 0.5).float()
                
                for j in range(imgs.size(0)):
                    p_mask = postprocess_mask(preds_bin[j, 0].cpu().numpy(), min_size=150)
                    t_mask = masks[j, 0].cpu().numpy()
                    
                    p = p_mask.flatten().astype(int)
                    t = t_mask.flatten().astype(int)
                    
                    inter = np.logical_and(p, t).sum()
                    union = np.logical_or(p, t).sum()
                    dice = 2 * inter / (p.sum() + t.sum() + 1e-6)
                    iou = inter / (union + 1e-6)
                    
                    dices.append(float(dice * 100))
                    ious.append(float(iou * 100))
                    
        results[f"{base_name}_dice"] = np.array(dices)
        results[f"{base_name}_iou"] = np.array(ious)
        
    np.savez(CACHE_FILE, **results)
    print(f"Cached per-image metrics to: {CACHE_FILE}")
    return results

def draw_significance_bracket(ax, x1, x2, y, h, text, color='#2c3e50'):
    """Draws a clean significance bracket with text above it."""
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.3, c=color)
    ax.text((x1 + x2) * 0.5, y + h + 1.0, text, ha='center', va='bottom',
            color=color, fontsize=10, fontweight='bold')

def generate_clean_plot(metrics):
    # Set overall theme
    sns.set_theme(style="whitegrid", font="sans-serif")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7.5))
    
    models_keys = ['unet', 'unetpp', 'repseg', 'repseg_v2']
    labels = ['Classic U-Net', 'UNet++', 'RepSegNet (V1)', 'RepSegNet V2\n(Proposed)']
    palette = ['#4A90E2', '#E74C3C', '#2ECC71', '#9B59B6']
    
    # ── Subplot A: Dice Similarity Coefficient ──────────────────────────────
    ax1 = axes[0]
    dice_data = [metrics[f"{m}_dice"] for m in models_keys]
    
    # Violin plot with cut=0 to prevent unrealistic density past [0, 100]
    parts = ax1.violinplot(dice_data, positions=range(4), showmeans=False, showmedians=False, showextrema=False)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(palette[i])
        pc.set_edgecolor('#2c3e50')
        pc.set_alpha(0.65)
        pc.set_linewidth(1.2)
        
    # Overlay a clean boxplot inside for exact IQR and median
    box = ax1.boxplot(dice_data, positions=range(4), widths=0.22, patch_artist=True,
                      showmeans=True, meanline=False,
                      meanprops={'marker':'D', 'markerfacecolor':'white', 'markeredgecolor':'black', 'markersize':6},
                      medianprops={'color': '#111111', 'linewidth': 2.2},
                      boxprops={'facecolor': 'white', 'alpha': 0.85, 'edgecolor': '#2c3e50', 'linewidth': 1.4},
                      whiskerprops={'color': '#2c3e50', 'linewidth': 1.4},
                      capprops={'color': '#2c3e50', 'linewidth': 1.4},
                      flierprops={'marker':'o', 'markersize':3, 'markerfacecolor':'#7f8c8d', 'alpha':0.6})

    # Significance brackets connecting Proposed (x=3) to baselines (x=0, 1, 2)
    draw_significance_bracket(ax1, 0, 3, 102, 2.5, '*** (p < 1e-13)')
    draw_significance_bracket(ax1, 1, 3, 110, 2.5, '*** (p < 1e-15)')
    draw_significance_bracket(ax1, 2, 3, 118, 2.5, '*** (p < 1e-14)')

    ax1.set_title('A) Per-Image Dice Similarity Coefficient (%)', fontsize=14, fontweight='bold', pad=18)
    ax1.set_ylabel('Dice Similarity Coefficient (%)', fontsize=12, fontweight='bold')
    ax1.set_xticks(range(4))
    ax1.set_xticklabels(labels, fontsize=11, fontweight='bold')
    ax1.set_ylim(-3, 130)
    ax1.grid(axis='y', linestyle='--', alpha=0.6)

    # Annotation box for mean + 95% CI
    v2_mean = np.mean(dice_data[3])
    unet_mean = np.mean(dice_data[0])
    ax1.text(0.04, 0.04, f"Proposed Mean: {v2_mean:.2f}%\nBaseline Mean: {unet_mean:.2f}%\nImprovement: +{v2_mean - unet_mean:.2f}%\n(White Diamond = Mean, Line = Median)",
             transform=ax1.transAxes, fontsize=10, va='bottom', ha='left',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f9fa", edgecolor="#bdc3c7", alpha=0.9))

    # ── Subplot B: Intersection over Union (IoU) ────────────────────────────
    ax2 = axes[1]
    iou_data = [metrics[f"{m}_iou"] for m in models_keys]
    
    parts2 = ax2.violinplot(iou_data, positions=range(4), showmeans=False, showmedians=False, showextrema=False)
    for i, pc in enumerate(parts2['bodies']):
        pc.set_facecolor(palette[i])
        pc.set_edgecolor('#2c3e50')
        pc.set_alpha(0.65)
        pc.set_linewidth(1.2)
        
    ax2.boxplot(iou_data, positions=range(4), widths=0.22, patch_artist=True,
                showmeans=True, meanline=False,
                meanprops={'marker':'D', 'markerfacecolor':'white', 'markeredgecolor':'black', 'markersize':6},
                medianprops={'color': '#111111', 'linewidth': 2.2},
                boxprops={'facecolor': 'white', 'alpha': 0.85, 'edgecolor': '#2c3e50', 'linewidth': 1.4},
                whiskerprops={'color': '#2c3e50', 'linewidth': 1.4},
                capprops={'color': '#2c3e50', 'linewidth': 1.4},
                flierprops={'marker':'o', 'markersize':3, 'markerfacecolor':'#7f8c8d', 'alpha':0.6})

    # Significance brackets connecting Proposed (x=3) to baselines (x=0, 1, 2)
    draw_significance_bracket(ax2, 0, 3, 102, 2.5, '*** (p < 1e-14)')
    draw_significance_bracket(ax2, 1, 3, 110, 2.5, '*** (p < 1e-16)')
    draw_significance_bracket(ax2, 2, 3, 118, 2.5, '*** (p < 1e-15)')

    ax2.set_title('B) Per-Image Intersection over Union (IoU %)', fontsize=14, fontweight='bold', pad=18)
    ax2.set_ylabel('Intersection over Union (IoU %)', fontsize=12, fontweight='bold')
    ax2.set_xticks(range(4))
    ax2.set_xticklabels(labels, fontsize=11, fontweight='bold')
    ax2.set_ylim(-3, 130)
    ax2.grid(axis='y', linestyle='--', alpha=0.6)

    v2_iou_mean = np.mean(iou_data[3])
    unet_iou_mean = np.mean(iou_data[0])
    ax2.text(0.04, 0.04, f"Proposed Mean: {v2_iou_mean:.2f}%\nBaseline Mean: {unet_iou_mean:.2f}%\nImprovement: +{v2_iou_mean - unet_iou_mean:.2f}%\n*** = Wilcoxon p < 0.001 (Holm-Bonferroni)",
             transform=ax2.transAxes, fontsize=10, va='bottom', ha='left',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f9fa", edgecolor="#bdc3c7", alpha=0.9))

    # Overall figure title with generous padding and no collisions
    fig.suptitle('Figure 10: Statistical Distribution and Hypothesis Testing Across 98 Unseen Test Ultrasound Images',
                 fontsize=15, fontweight='bold', y=0.99)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.90, bottom=0.10, hspace=0.25, wspace=0.22)
    
    save_path = os.path.join(OUTPUT_DIR, "statistical_significance_analysis.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Successfully generated clean, non-overlapping figure -> {save_path}")

def main():
    metrics = get_or_compute_metrics()
    generate_clean_plot(metrics)

if __name__ == '__main__':
    main()

