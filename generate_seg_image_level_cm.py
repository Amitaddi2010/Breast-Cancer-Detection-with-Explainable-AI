"""
generate_seg_image_level_cm.py
Generates:
1. Image-Level Clinical Segmentation Confusion Matrix (98 test images).
2. Side-by-side Comprehensive Confusion Matrix (Pixel-level + Image-level).
3. Comparative image-level localization accuracy across U-Net, UNet++, RepSegNet, and RepSegNet V2.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix

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

def evaluate_models_image_level(test_l, device):
    test_ds = test_l.dataset
    is_benign = np.array(['benign' in s.lower() for s in test_ds.samples])
    
    models = {
        'U-Net': (get_unet(), UNET_CHECKPOINT, False),
        'UNet++': (get_unetpp(deep_supervision=True), UNETPP_CHECKPOINT, True),
        'RepSegNet': (get_repsegnet(), REPSEGNET_CHECKPOINT, False),
        'RepSegNet V2 (Ours)': (get_repsegnet_v2(encoder_name=SEG_ENCODER_NAME, deep_supervision=True), REPSEGNET_V2_CHECKPOINT, False)
    }
    
    results = {}
    pixel_cm_v2 = None
    
    for name, (m, ckpt, is_upp) in models.items():
        if not os.path.exists(ckpt):
            continue
        print(f"Running inference for {name} on 98 test images...")
        m.load_state_dict(torch.load(ckpt, map_location=device))
        m = m.to(device).eval()
        
        ious, dices = [], []
        all_preds_pix, all_targets_pix = [], []
        
        with torch.no_grad():
            for imgs, masks in test_l:
                imgs = imgs.to(device)
                preds = m(imgs)
                preds_bin = (preds > 0.5).float()
                
                for j in range(imgs.size(0)):
                    p_mask = postprocess_mask(preds_bin[j, 0].cpu().numpy(), min_size=150)
                    t_mask = masks[j, 0].cpu().numpy()
                    
                    p_flat = p_mask.flatten()
                    t_flat = t_mask.flatten().astype(int)
                    
                    inter = np.logical_and(p_flat, t_flat).sum()
                    union = np.logical_or(p_flat, t_flat).sum()
                    iou = inter / (union + 1e-6)
                    dice = 2 * inter / (p_flat.sum() + t_flat.sum() + 1e-6)
                    
                    ious.append(iou)
                    dices.append(dice)
                    
                    if 'V2' in name:
                        all_preds_pix.append(p_flat)
                        all_targets_pix.append(t_flat)
                        
        ious = np.array(ious)
        dices = np.array(dices)
        
        results[name] = {
            'ious': ious,
            'dices': dices,
            'success_all': int((ious >= 0.5).sum()),
            'success_benign': int((ious[is_benign] >= 0.5).sum()),
            'success_malignant': int((ious[~is_benign] >= 0.5).sum()),
            'total_all': len(ious),
            'total_benign': int(is_benign.sum()),
            'total_malignant': int((~is_benign).sum()),
        }
        
        if 'V2' in name:
            y_pred_all = np.concatenate(all_preds_pix).astype(int)
            y_true_all = np.concatenate(all_targets_pix).astype(int)
            pixel_cm_v2 = confusion_matrix(y_true_all, y_pred_all)
            
    return results, is_benign, pixel_cm_v2


def plot_image_level_confusion_matrix(results, is_benign, save_path):
    v2 = results.get('RepSegNet V2 (Ours)')
    if not v2:
        return
        
    # Table dimensions:
    # Rows: Benign (66), Malignant (32)
    # Cols: Localized / Overlap >= 0.50, Sub-optimal / Overlap < 0.50
    matrix = np.array([
        [v2['success_benign'], v2['total_benign'] - v2['success_benign']],
        [v2['success_malignant'], v2['total_malignant'] - v2['success_malignant']]
    ])
    
    total_imgs = v2['total_all']
    matrix_pct = matrix / total_imgs * 100
    
    plt.figure(figsize=(9, 7))
    labels = np.array([
        [f"{matrix[0,0]} images\n({matrix[0,0]/v2['total_benign']*100:.1f}% of Benign)", 
         f"{matrix[0,1]} images\n({matrix[0,1]/v2['total_benign']*100:.1f}% of Benign)"],
        [f"{matrix[1,0]} images\n({matrix[1,0]/v2['total_malignant']*100:.1f}% of Malignant)", 
         f"{matrix[1,1]} images\n({matrix[1,1]/v2['total_malignant']*100:.1f}% of Malignant)"]
    ])
    
    sns.heatmap(matrix, annot=labels, fmt="", cmap='Blues', cbar=False,
                annot_kws={"size": 13, "weight": "bold"},
                xticklabels=['Localized (IoU \u2265 0.50)', 'Sub-optimal (IoU < 0.50)'],
                yticklabels=['Benign Lesion\n(66 Test Images)', 'Malignant Lesion\n(32 Test Images)'])
    
    plt.title(f'Image-Level Clinical Localization Matrix \u2014 RepSegNet V2\n(Total Test Images = {total_imgs})',
              fontsize=14, fontweight='bold', pad=14)
    plt.xlabel('Clinical Segmentation Outcome', fontsize=12, fontweight='bold', labelpad=10)
    plt.ylabel('Ground Truth Lesion Type', fontsize=12, fontweight='bold', labelpad=10)
    
    # Text summary at bottom
    acc_text = f"Overall Tumor Localization Rate: {v2['success_all']}/{total_imgs} images ({v2['success_all']/total_imgs*100:.1f}%)"
    plt.figtext(0.5, 0.01, acc_text, ha="center", fontsize=12, 
                bbox={"facecolor":"#e3f2fd", "edgecolor":"#1976d2", "boxstyle":"round,pad=0.5"})
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.14)
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[CM] Saved Image-Level Matrix -> {save_path}")


def plot_side_by_side_cm(results, pixel_cm, save_path):
    v2 = results.get('RepSegNet V2 (Ours)')
    if not v2 or pixel_cm is None:
        return
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7.5))
    
    # Left: Pixel-wise Confusion Matrix
    cm_norm = pixel_cm.astype('float') / pixel_cm.sum() * 100
    pix_labels = np.array([
        [f"{pixel_cm[0,0]:,}\n({cm_norm[0,0]:.2f}%)", f"{pixel_cm[0,1]:,}\n({cm_norm[0,1]:.2f}%)"],
        [f"{pixel_cm[1,0]:,}\n({cm_norm[1,0]:.2f}%)", f"{pixel_cm[1,1]:,}\n({cm_norm[1,1]:.2f}%)"]
    ])
    
    sns.heatmap(pixel_cm, annot=pix_labels, fmt="", cmap='Greens', ax=ax1, cbar=False,
                annot_kws={"size": 12, "weight": "bold"},
                xticklabels=['Background', 'Tumor'],
                yticklabels=['Background', 'Tumor'])
    ax1.set_title(f'A) Pixel-Wise Confusion Matrix\nTotal Pixels = {pixel_cm.sum():,} (98 \u00d7 512 \u00d7 512)',
                  fontsize=13, fontweight='bold', pad=10)
    ax1.set_xlabel('Predicted Pixel Class', fontsize=11, fontweight='bold')
    ax1.set_ylabel('True Ground-Truth Class', fontsize=11, fontweight='bold')
    
    # Right: Image-level Detection Matrix
    img_matrix = np.array([
        [v2['success_benign'], v2['total_benign'] - v2['success_benign']],
        [v2['success_malignant'], v2['total_malignant'] - v2['success_malignant']]
    ])
    img_labels = np.array([
        [f"{img_matrix[0,0]} images\n({img_matrix[0,0]/v2['total_benign']*100:.1f}%)", 
         f"{img_matrix[0,1]} images\n({img_matrix[0,1]/v2['total_benign']*100:.1f}%)"],
        [f"{img_matrix[1,0]} images\n({img_matrix[1,0]/v2['total_malignant']*100:.1f}%)", 
         f"{img_matrix[1,1]} images\n({img_matrix[1,1]/v2['total_malignant']*100:.1f}%)"]
    ])
    
    sns.heatmap(img_matrix, annot=img_labels, fmt="", cmap='Blues', ax=ax2, cbar=False,
                annot_kws={"size": 12, "weight": "bold"},
                xticklabels=['Success (IoU \u2265 0.50)', 'Sub-optimal (IoU < 0.50)'],
                yticklabels=['Benign (66 Images)', 'Malignant (32 Images)'])
    ax2.set_title(f'B) Image-Level Detection Matrix\nTotal Test Images = {v2["total_all"]} (66 Benign, 32 Malignant)',
                  fontsize=13, fontweight='bold', pad=10)
    ax2.set_xlabel('Clinical Localization Outcome', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Ground Truth Lesion Type', fontsize=11, fontweight='bold')
    
    fig.suptitle('RepSegNet V2: Comprehensive Pixel-Level vs Image-Level Evaluation',
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[CM] Saved Side-by-Side Matrix -> {save_path}")


def plot_comparative_model_bars(results, save_path):
    models = list(results.keys())
    success_rates = [results[m]['success_all'] / results[m]['total_all'] * 100 for m in models]
    benign_rates = [results[m]['success_benign'] / results[m]['total_benign'] * 100 for m in models]
    mal_rates = [results[m]['success_malignant'] / results[m]['total_malignant'] * 100 for m in models]
    
    x = np.arange(len(models))
    width = 0.26
    
    plt.figure(figsize=(11, 6.5))
    rects1 = plt.bar(x - width, success_rates, width, label='Overall (98 Images)', color='#1976D2')
    rects2 = plt.bar(x, benign_rates, width, label='Benign (66 Images)', color='#43A047')
    rects3 = plt.bar(x + width, mal_rates, width, label='Malignant (32 Images)', color='#E53935')
    
    plt.ylabel('Localization Success Rate (%) [IoU \u2265 0.50]', fontsize=12, fontweight='bold')
    plt.title('Benchmark on 98 Unseen Test Images: Tumor Localization Success (IoU \u2265 0.50)', 
              fontsize=14, fontweight='bold', pad=12)
    plt.xticks(x, models, fontsize=11, fontweight='bold')
    plt.ylim(0, 105)
    plt.legend(frameon=True, fontsize=11, loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Add value labels on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            plt.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
                        
    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[CM] Saved Comparative Model Bar Chart -> {save_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    _, _, test_l = get_seg_v2_dataloaders(img_size=SEG_IMG_SIZE, use_copy_paste=False)
    
    results, is_benign, pixel_cm_v2 = evaluate_models_image_level(test_l, DEVICE)
    
    p1 = os.path.join(OUTPUT_DIR, "seg_image_level_confusion_matrix.png")
    plot_image_level_confusion_matrix(results, is_benign, p1)
    
    p2 = os.path.join(OUTPUT_DIR, "repsegnet_v2_comprehensive_cm.png")
    plot_side_by_side_cm(results, pixel_cm_v2, p2)
    
    p3 = os.path.join(OUTPUT_DIR, "seg_models_localization_comparison.png")
    plot_comparative_model_bars(results, p3)
    
    import json
    out_summary = {m: {k: v for k, v in data.items() if k not in ['ious', 'dices']} 
                   for m, data in results.items()}
    with open(os.path.join(OUTPUT_DIR, "seg_image_level_metrics.json"), 'w') as f:
        json.dump(out_summary, f, indent=2)
    print(f"[CM] Saved metrics JSON -> {os.path.join(OUTPUT_DIR, 'seg_image_level_metrics.json')}")

if __name__ == "__main__":
    main()
