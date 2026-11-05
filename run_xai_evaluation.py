"""
run_xai_evaluation.py
Generates both:
1. Qualitative Evaluation: Multi-case clinical panel comparing Original Ultrasound,
   Ground-Truth Tumor Mask, Grad-CAM, Grad-CAM++, and Score-CAM.
2. Quantitative Evaluation: Pointing Game Hit Rate (%), Energy Concentration Inside Tumor (%),
   Average Drop in Confidence (%), and Increase in Confidence (%).
"""

import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image

from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, ScoreCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from config import (
    DEVICE, CLF_CHECKPOINT, CLF_IMG_SIZE, CLASS_NAMES,
    NORMALIZE_MEAN, NORMALIZE_STD, OUTPUT_DIR, DATA_DIR
)
from model import get_classifier
from dataset import get_dataloaders
from seg_dataset import _load_merged_mask

def preprocess_for_cam(img_path, target_size=CLF_IMG_SIZE):
    img_pil = Image.open(img_path).convert("RGB")
    img_resized = img_pil.resize((target_size, target_size))
    rgb_float = np.array(img_resized).astype(np.float32) / 255.0
    
    # Normalize tensor
    mean = np.array(NORMALIZE_MEAN, dtype=np.float32)
    std = np.array(NORMALIZE_STD, dtype=np.float32)
    norm = (rgb_float - mean) / std
    tensor = torch.tensor(norm.transpose(2, 0, 1)).unsqueeze(0).float().to(DEVICE)
    return tensor, rgb_float

def run_quantitative_xai(model, test_samples, device):
    target_layer = [model.features[-1]]
    cam_engines = {
        'Grad-CAM': GradCAM(model=model, target_layers=target_layer),
        'Grad-CAM++': GradCAMPlusPlus(model=model, target_layers=target_layer),
        'Score-CAM': ScoreCAM(model=model, target_layers=target_layer)
    }
    
    metrics = {k: {'hits': 0, 'total': 0, 'energy_ratios': [], 'drops': [], 'increases': 0} for k in cam_engines}
    
    print(f"Running quantitative XAI evaluation across {len(test_samples)} test images...")
    
    for idx, (img_path, true_label) in enumerate(test_samples):
        # Skip normal cases without tumor masks for localization metrics
        if 'normal' in img_path.lower():
            continue
            
        tensor, rgb_float = preprocess_for_cam(img_path)
        
        # Original forward pass
        with torch.no_grad():
            orig_logits = model(tensor)
            orig_probs = torch.softmax(orig_logits, dim=1)[0].cpu().numpy()
            pred_class = int(orig_probs.argmax())
            orig_conf = float(orig_probs[pred_class])
            
        targets = [ClassifierOutputTarget(pred_class)]
        
        # Load GT Mask and resize to match CAM
        gt_mask = _load_merged_mask(img_path)
        gt_mask_resized = cv2.resize(gt_mask, (CLF_IMG_SIZE, CLF_IMG_SIZE), interpolation=cv2.INTER_NEAREST)
        mask_binary = (gt_mask_resized > 0).astype(np.uint8)
        
        if mask_binary.sum() == 0:
            continue
            
        for name, cam_obj in cam_engines.items():
            try:
                grayscale_cam = cam_obj(input_tensor=tensor, targets=targets)[0]  # [H, W], values in [0, 1]
                
                # 1. Pointing Game Hit Rate
                # Maximum saliency point
                max_pos = np.unravel_index(np.argmax(grayscale_cam), grayscale_cam.shape)
                is_hit = bool(mask_binary[max_pos[0], max_pos[1]] == 1)
                if is_hit:
                    metrics[name]['hits'] += 1
                metrics[name]['total'] += 1
                
                # 2. Energy Ratio inside ground-truth mask
                total_energy = float(grayscale_cam.sum() + 1e-8)
                inside_energy = float((grayscale_cam * mask_binary).sum())
                energy_ratio = (inside_energy / total_energy) * 100.0
                metrics[name]['energy_ratios'].append(energy_ratio)
                
                # 3. Average Drop & Increase in Confidence
                # Mask image with CAM heatmap (only keep salient pixels)
                cam_mask_3d = np.repeat(grayscale_cam[:, :, np.newaxis], 3, axis=2)
                cam_masked_rgb = rgb_float * cam_mask_3d
                
                # Re-evaluate model on CAM-masked input
                norm_masked = (cam_masked_rgb - np.array(NORMALIZE_MEAN, dtype=np.float32)) / np.array(NORMALIZE_STD, dtype=np.float32)
                tensor_masked = torch.tensor(norm_masked.transpose(2, 0, 1)).unsqueeze(0).float().to(DEVICE)
                
                with torch.no_grad():
                    masked_logits = model(tensor_masked)
                    masked_probs = torch.softmax(masked_logits, dim=1)[0].cpu().numpy()
                    masked_conf = float(masked_probs[pred_class])
                    
                drop = max(0.0, (orig_conf - masked_conf) / (orig_conf + 1e-8)) * 100.0
                metrics[name]['drops'].append(drop)
                if masked_conf > orig_conf:
                    metrics[name]['increases'] += 1
                    
            except Exception as e:
                print(f"Error on {name} for image {idx}: {e}")
                
    summary = {}
    for name, data in metrics.items():
        n = data['total']
        hit_rate = (data['hits'] / n * 100.0) if n > 0 else 0.0
        mean_energy = float(np.mean(data['energy_ratios'])) if data['energy_ratios'] else 0.0
        mean_drop = float(np.mean(data['drops'])) if data['drops'] else 0.0
        increase_pct = (data['increases'] / n * 100.0) if n > 0 else 0.0
        
        summary[name] = {
            'num_evaluated_images': n,
            'pointing_game_hit_rate_pct': round(hit_rate, 2),
            'energy_inside_tumor_pct': round(mean_energy, 2),
            'average_drop_pct': round(mean_drop, 2),
            'increase_in_confidence_pct': round(increase_pct, 2)
        }
        print(f"\n[{name}]")
        print(f"  Pointing Game Hit Rate: {hit_rate:.2f}% ({data['hits']}/{n})")
        print(f"  Energy Inside Tumor:    {mean_energy:.2f}%")
        print(f"  Average Confidence Drop:{mean_drop:.2f}%")
        print(f"  Increase in Confidence: {increase_pct:.2f}%")
        
    return summary

def generate_qualitative_panel(model, test_samples, save_path):
    target_layer = [model.features[-1]]
    cam_grad = GradCAM(model=model, target_layers=target_layer)
    cam_gradpp = GradCAMPlusPlus(model=model, target_layers=target_layer)
    cam_score = ScoreCAM(model=model, target_layers=target_layer)
    
    # Pick 2 representative Benign cases and 2 Malignant cases
    benign_samples = [s for s in test_samples if 'benign' in s[0].lower()][:2]
    malignant_samples = [s for s in test_samples if 'malignant' in s[0].lower()][:2]
    selected_samples = benign_samples + malignant_samples
    
    fig, axes = plt.subplots(4, 5, figsize=(16, 13))
    col_titles = [
        "A) Input Ultrasound",
        "B) Radiologist GT Mask",
        "C) Grad-CAM Overlay",
        "D) Grad-CAM++ Overlay",
        "E) Score-CAM Overlay"
    ]
    
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=12, fontweight='bold', pad=10)
        
    for row, (img_path, true_label) in enumerate(selected_samples):
        cname = "Benign" if "benign" in img_path.lower() else "Malignant"
        tensor, rgb_float = preprocess_for_cam(img_path)
        
        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            pred_idx = int(probs.argmax())
            pred_conf = probs[pred_idx] * 100
            
        targets = [ClassifierOutputTarget(pred_idx)]
        
        # CAMs
        g_cam = cam_grad(input_tensor=tensor, targets=targets)[0]
        gpp_cam = cam_gradpp(input_tensor=tensor, targets=targets)[0]
        s_cam = cam_score(input_tensor=tensor, targets=targets)[0]
        
        g_overlay = show_cam_on_image(rgb_float, g_cam, use_rgb=True)
        gpp_overlay = show_cam_on_image(rgb_float, gpp_cam, use_rgb=True)
        s_overlay = show_cam_on_image(rgb_float, s_cam, use_rgb=True)
        
        gt_mask = _load_merged_mask(img_path)
        gt_resized = cv2.resize(gt_mask, (CLF_IMG_SIZE, CLF_IMG_SIZE), interpolation=cv2.INTER_NEAREST)
        
        # Plot col 0: Input image
        axes[row, 0].imshow((rgb_float * 255).astype(np.uint8))
        axes[row, 0].set_ylabel(f"Case {row+1}: {cname}\n(Pred: {CLASS_NAMES[pred_idx].capitalize()} {pred_conf:.1f}%)",
                                fontsize=11, fontweight='bold')
        
        # Plot col 1: GT Mask
        axes[row, 1].imshow(gt_resized, cmap='gray')
        
        # Plot col 2: Grad-CAM
        axes[row, 2].imshow(g_overlay)
        
        # Plot col 3: Grad-CAM++
        axes[row, 3].imshow(gpp_overlay)
        
        # Plot col 4: Score-CAM
        axes[row, 4].imshow(s_overlay)
        
        for c in range(5):
            axes[row, c].set_xticks([])
            axes[row, c].set_yticks([])
            
    fig.suptitle('Figure 11: Qualitative Explainable AI (XAI) Saliency Map Comparison on Representative Test Ultrasound Cases',
                 fontsize=14, fontweight='bold', y=0.99)
    plt.tight_layout()
    plt.subplots_adjust(top=0.94, bottom=0.03, hspace=0.15, wspace=0.12)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved qualitative XAI panel -> {save_path}")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    _, _, test_loader = get_dataloaders()
    test_samples = test_loader.dataset.samples
    
    model = get_classifier(pretrained=False).to(DEVICE)
    model.load_state_dict(torch.load(CLF_CHECKPOINT, map_location=DEVICE))
    model.eval()
    
    # 1. Quantitative Evaluation
    metrics_summary = run_quantitative_xai(model, test_samples, DEVICE)
    json_path = os.path.join(OUTPUT_DIR, "xai_quantitative_metrics.json")
    with open(json_path, 'w') as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"Saved quantitative metrics JSON -> {json_path}")
    
    # 2. Qualitative Panel
    fig_path = os.path.join(OUTPUT_DIR, "xai_qualitative_evaluation.png")
    generate_qualitative_panel(model, test_samples, fig_path)

if __name__ == '__main__':
    main()

