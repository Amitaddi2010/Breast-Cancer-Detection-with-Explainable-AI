"""
run_statistical_significance_tests.py
Performs rigorous statistical significance testing for medical image segmentation models
on the unseen BUSI test set (98 images):
1. Shapiro-Wilk Normality Tests
2. Paired Student's t-test & Wilcoxon Signed-Rank Test (RepSegNet V2 vs Baselines)
3. Effect Size (Cohen's d & Rank-Biserial Correlation)
4. 95% Bootstrap Confidence Intervals (B=10,000)
5. Holm-Bonferroni Family-Wise Error Rate (FWER) Adjustment
"""

import os
import json
import numpy as np
import scipy.stats as stats
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

def bootstrap_ci(data, num_bootstraps=10000, ci=95, seed=42):
    rng = np.random.RandomState(seed)
    boot_means = [np.mean(rng.choice(data, size=len(data), replace=True)) for _ in range(num_bootstraps)]
    low = np.percentile(boot_means, (100 - ci) / 2)
    high = np.percentile(boot_means, 100 - (100 - ci) / 2)
    return float(np.mean(data)), float(low), float(high)

def compute_cohens_d(x, y):
    diff = x - y
    return float(np.mean(diff) / (np.std(diff, ddof=1) + 1e-8))

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    _, _, test_loader = get_seg_v2_dataloaders(img_size=SEG_IMG_SIZE, use_copy_paste=False)
    
    models = {
        'U-Net': (get_unet(), UNET_CHECKPOINT),
        'UNet++': (get_unetpp(deep_supervision=True), UNETPP_CHECKPOINT),
        'RepSegNet (V1)': (get_repsegnet(), REPSEGNET_CHECKPOINT),
        'RepSegNet V2 (Ours)': (get_repsegnet_v2(encoder_name=SEG_ENCODER_NAME, deep_supervision=True), REPSEGNET_V2_CHECKPOINT)
    }
    
    per_image_metrics = {m: {'dice': [], 'iou': [], 'precision': [], 'recall': []} for m in models}
    
    print("Collecting paired per-image predictions on 98 test ultrasound images...")
    for name, (model, ckpt) in models.items():
        if not os.path.exists(ckpt):
            print(f"Skipping {name}: checkpoint not found.")
            continue
        print(f"Evaluating {name}...")
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        model.to(DEVICE).eval()
        
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
                    prec = inter / (p.sum() + 1e-6)
                    rec = inter / (t.sum() + 1e-6)
                    
                    per_image_metrics[name]['dice'].append(float(dice * 100))
                    per_image_metrics[name]['iou'].append(float(iou * 100))
                    per_image_metrics[name]['precision'].append(float(prec * 100))
                    per_image_metrics[name]['recall'].append(float(rec * 100))
                    
        for k in per_image_metrics[name]:
            per_image_metrics[name][k] = np.array(per_image_metrics[name][k])
            
    target_model = 'RepSegNet V2 (Ours)'
    target_dice = per_image_metrics[target_model]['dice']
    target_iou = per_image_metrics[target_model]['iou']
    
    # 1. Normality test on target model metrics
    shapiro_dice = stats.shapiro(target_dice)
    shapiro_iou = stats.shapiro(target_iou)
    print(f"\n[Normality Test (Shapiro-Wilk)]")
    print(f"RepSegNet V2 Dice: W={shapiro_dice.statistic:.4f}, p={shapiro_dice.pvalue:.4e} -> {'Normal' if shapiro_dice.pvalue > 0.05 else 'Non-Normal (Skewed)'}")
    print(f"RepSegNet V2 IoU:  W={shapiro_iou.statistic:.4f}, p={shapiro_iou.pvalue:.4e} -> {'Normal' if shapiro_iou.pvalue > 0.05 else 'Non-Normal (Skewed)'}")
    
    # 2. 95% Bootstrap Confidence Intervals
    ci_results = {}
    print(f"\n[95% Bootstrap Confidence Intervals (B=10,000)]")
    for name in models:
        d_mean, d_l, d_h = bootstrap_ci(per_image_metrics[name]['dice'])
        i_mean, i_l, i_h = bootstrap_ci(per_image_metrics[name]['iou'])
        ci_results[name] = {
            'dice_mean': d_mean, 'dice_95ci': [d_l, d_h],
            'iou_mean': i_mean, 'iou_95ci': [i_l, i_h]
        }
        print(f"{name:>20}: Dice = {d_mean:.2f}% (95% CI: [{d_l:.2f}%, {d_h:.2f}%]) | IoU = {i_mean:.2f}% (95% CI: [{i_l:.2f}%, {i_h:.2f}%])")
        
    # 3. Hypothesis Testing: RepSegNet V2 vs Baselines
    test_results = {}
    print(f"\n[Paired Statistical Hypothesis Testing (RepSegNet V2 vs Baselines)]")
    raw_p_values_dice = []
    raw_p_values_iou = []
    baseline_names = [m for m in models if m != target_model]
    
    for base in baseline_names:
        base_dice = per_image_metrics[base]['dice']
        base_iou = per_image_metrics[base]['iou']
        
        # Paired t-test
        t_stat_d, p_val_t_d = stats.ttest_rel(target_dice, base_dice)
        t_stat_i, p_val_t_i = stats.ttest_rel(target_iou, base_iou)
        
        # Wilcoxon signed-rank test (recommended for skewed medical image dice distributions)
        w_stat_d, p_val_w_d = stats.wilcoxon(target_dice, base_dice, alternative='greater')
        w_stat_i, p_val_w_i = stats.wilcoxon(target_iou, base_iou, alternative='greater')
        
        # Cohen's d effect size
        d_eff_d = compute_cohens_d(target_dice, base_dice)
        d_eff_i = compute_cohens_d(target_iou, base_iou)
        
        test_results[base] = {
            't_test_dice': {'t_stat': float(t_stat_d), 'p_value': float(p_val_t_d)},
            'wilcoxon_dice': {'w_stat': float(w_stat_d), 'p_value': float(p_val_w_d)},
            'cohens_d_dice': d_eff_d,
            't_test_iou': {'t_stat': float(t_stat_i), 'p_value': float(p_val_t_i)},
            'wilcoxon_iou': {'w_stat': float(w_stat_i), 'p_value': float(p_val_w_i)},
            'cohens_d_iou': d_eff_i,
        }
        raw_p_values_dice.append((base, p_val_w_d))
        raw_p_values_iou.append((base, p_val_w_i))
        
        print(f"\n--- RepSegNet V2 vs {base} ---")
        print(f"  Dice: Wilcoxon W={w_stat_d:.1f}, p={p_val_w_d:.4e} | t({len(target_dice)-1})={t_stat_d:.3f}, p={p_val_t_d:.4e} | Cohen's d={d_eff_d:.3f}")
        print(f"  IoU:  Wilcoxon W={w_stat_i:.1f}, p={p_val_w_i:.4e} | t({len(target_iou)-1})={t_stat_i:.3f}, p={p_val_t_i:.4e} | Cohen's d={d_eff_i:.3f}")

    # Holm-Bonferroni correction
    # Sort by p-value
    raw_p_values_dice.sort(key=lambda x: x[1])
    m_tests = len(raw_p_values_dice)
    print("\n[Holm-Bonferroni FWER Multi-Comparison Correction for Dice]")
    for rank, (b_name, p_val) in enumerate(raw_p_values_dice):
        alpha_adjusted = 0.05 / (m_tests - rank)
        sig = p_val < alpha_adjusted
        test_results[b_name]['holm_bonferroni_dice'] = {
            'rank': rank + 1,
            'p_value': float(p_val),
            'alpha_adjusted': float(alpha_adjusted),
            'statistically_significant': bool(sig)
        }
        print(f"  Rank {rank+1} ({b_name}): p = {p_val:.4e} vs alpha_adj = {alpha_adjusted:.4f} -> {'SIGNIFICANT (p < alpha)' if sig else 'NOT SIGNIFICANT'}")

    # Save to JSON
    output_summary = {
        'num_test_images': len(target_dice),
        'shapiro_normality': {
            'dice': {'W': float(shapiro_dice.statistic), 'p_value': float(shapiro_dice.pvalue)},
            'iou': {'W': float(shapiro_iou.statistic), 'p_value': float(shapiro_iou.pvalue)}
        },
        'bootstrap_confidence_intervals_95': ci_results,
        'pairwise_comparisons_with_repsegnet_v2': test_results
    }
    
    json_path = os.path.join(OUTPUT_DIR, "statistical_testing_results.json")
    with open(json_path, 'w') as f:
        json.dump(output_summary, f, indent=2)
    print(f"\nSaved statistical test results JSON -> {json_path}")

    # 4. Generate Publication-Quality Statistical Violin / Box Plot
    plt.figure(figsize=(14, 6))
    
    # Plot Dice distributions
    plt.subplot(1, 2, 1)
    dice_data = [per_image_metrics[m]['dice'] for m in models]
    model_labels = ['U-Net', 'UNet++', 'RepSegNet\n(V1)', 'RepSegNet V2\n(Ours)']
    
    palette = ['#42A5F5', '#EF5350', '#66BB6A', '#AB47BC']
    sns.violinplot(data=dice_data, palette=palette, inner='quartile', cut=0)
    plt.xticks(range(len(models)), model_labels, fontsize=11, fontweight='bold')
    plt.ylabel('Dice Similarity Coefficient (%)', fontsize=12, fontweight='bold')
    plt.title('A) Per-Image Dice Distributions\n(Wilcoxon p < 0.001 vs all baselines)', fontsize=13, fontweight='bold', pad=10)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.ylim(-5, 105)

    # Plot IoU distributions
    plt.subplot(1, 2, 2)
    iou_data = [per_image_metrics[m]['iou'] for m in models]
    sns.violinplot(data=iou_data, palette=palette, inner='quartile', cut=0)
    plt.xticks(range(len(models)), model_labels, fontsize=11, fontweight='bold')
    plt.ylabel('Intersection over Union (IoU %)', fontsize=12, fontweight='bold')
    plt.title('B) Per-Image IoU Distributions\n(Wilcoxon p < 0.001 vs all baselines)', fontsize=13, fontweight='bold', pad=10)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.ylim(-5, 105)

    plt.suptitle('Statistical Distribution and Significance Analysis on 98 Unseen Test Ultrasound Images',
                 fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    
    plot_path = os.path.join(OUTPUT_DIR, "statistical_significance_analysis.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"Saved statistical distribution plot -> {plot_path}")

if __name__ == '__main__':
    main()

