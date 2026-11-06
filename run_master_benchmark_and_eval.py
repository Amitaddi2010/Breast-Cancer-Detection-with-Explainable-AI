"""
run_master_benchmark_and_eval.py
Unified, leak-proof Single Source of Truth for:
1. Master Comparative Benchmark (BUSI Test Set: Acc, Prec, Sens/Recall, Spec, Dice/F1, IoU, HD95, ASSD)
2. Resolution Scaling Analysis (RepSegNet V2 at 512x512, 256x256, 128x128)
3. External Generalization Benchmark (UDIAT Dataset B, N=163)
4. Comprehensive Ablation Analysis
5. Complete Computational Complexity Profiling
Outputs: outputs/master_source_of_truth.json
"""

import os
import glob
import json
import time
import cv2
import numpy as np
import scipy.ndimage
from scipy.ndimage import distance_transform_edt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from thop import profile

from config import (
    BASE_DIR, DEVICE, OUTPUT_DIR, SEG_IMG_SIZE, SEG_ENCODER_NAME,
    UNET_CHECKPOINT, UNETPP_CHECKPOINT,
    REPSEGNET_CHECKPOINT, REPSEGNET_V2_CHECKPOINT
)
from unet import get_unet
from unet_plus_plus import get_unetpp
from repsegnet import get_repsegnet
from repsegnet_v2 import get_repsegnet_v2
from seg_dataset_v2 import get_seg_v2_dataloaders
from postprocess import postprocess_mask

# ── Surface Distance Computation ─────────────────────────────────────────────
def compute_surface_metrics(pred_bin: np.ndarray, target_bin: np.ndarray):
    """Compute HD95 and ASSD in pixels."""
    p_sum = pred_bin.sum()
    t_sum = target_bin.sum()
    if p_sum == 0 and t_sum == 0:
        return 0.0, 0.0
    if p_sum == 0 or t_sum == 0:
        # Penalty for empty prediction/target
        return 50.0, 25.0

    pred_b = pred_bin.astype(bool)
    target_b = target_bin.astype(bool)

    # Extract single-pixel contours
    p_border = pred_b ^ (scipy.ndimage.binary_erosion(pred_b))
    t_border = target_b ^ (scipy.ndimage.binary_erosion(target_b))

    if not p_border.any() or not t_border.any():
        return 0.0, 0.0

    dt_p = distance_transform_edt(~p_border)
    dt_t = distance_transform_edt(~t_border)

    d_p2t = dt_t[p_border]
    d_t2p = dt_p[t_border]
    all_d = np.concatenate([d_p2t, d_t2p])

    hd95 = float(np.percentile(all_d, 95))
    assd = float(np.mean(all_d))
    return hd95, assd

# ── Comprehensive Single-Image Metrics ───────────────────────────────────────
def compute_image_metrics(pred_bin: np.ndarray, target_bin: np.ndarray) -> dict:
    p = pred_bin.flatten().astype(int)
    t = target_bin.flatten().astype(int)
    smooth = 1e-6

    tp = np.logical_and(p == 1, t == 1).sum()
    fp = np.logical_and(p == 1, t == 0).sum()
    fn = np.logical_and(p == 0, t == 1).sum()
    tn = np.logical_and(p == 0, t == 0).sum()

    acc  = float((tp + tn) / (tp + tn + fp + fn + smooth)) * 100.0
    prec = float(tp / (tp + fp + smooth)) * 100.0
    sens = float(tp / (tp + fn + smooth)) * 100.0  # Recall / Sensitivity
    spec = float(tn / (tn + fp + smooth)) * 100.0  # Specificity

    # Mathematically exact Dice = F1
    dice = float((2 * tp) / (2 * tp + fp + fn + smooth)) * 100.0
    f1 = dice
    iou  = float((tp + smooth) / (tp + fp + fn + smooth)) * 100.0

    hd95, assd = compute_surface_metrics(pred_bin, target_bin)

    return {
        'accuracy': acc,
        'precision': prec,
        'sensitivity': sens,
        'specificity': spec,
        'dice': dice,
        'f1': f1,
        'iou': iou,
        'hd95': hd95,
        'assd': assd
    }

def aggregate_metrics(metrics_list: list) -> dict:
    avg = {}
    for k in metrics_list[0]:
        avg[k] = float(np.mean([m[k] for m in metrics_list]))
    return avg

# ── TTA Inference ─────────────────────────────────────────────────────────────
@torch.no_grad()
def tta_predict(model, img_batch):
    preds = []
    p0 = model(img_batch)
    if isinstance(p0, (tuple, list)):
        p0 = p0[0]
    preds.append(p0)

    p1 = model(torch.flip(img_batch, dims=[3]))
    if isinstance(p1, (tuple, list)): p1 = p1[0]
    preds.append(torch.flip(p1, dims=[3]))

    p2 = model(torch.flip(img_batch, dims=[2]))
    if isinstance(p2, (tuple, list)): p2 = p2[0]
    preds.append(torch.flip(p2, dims=[2]))

    p3 = model(torch.flip(img_batch, dims=[2, 3]))
    if isinstance(p3, (tuple, list)): p3 = p3[0]
    preds.append(torch.flip(p3, dims=[2, 3]))

    return torch.stack(preds).mean(dim=0)

def evaluate_loader(model, loader, device, target_size=512, use_postprocess=True, min_size=150):
    model.eval()
    per_image_results = []
    
    for imgs, masks in loader:
        imgs = imgs.to(device)
        if imgs.shape[-1] != target_size:
            imgs = F.interpolate(imgs, size=(target_size, target_size), mode='bilinear', align_corners=True)

        preds = tta_predict(model, imgs)
        if preds.shape[-1] != 512:
            preds = F.interpolate(preds, size=(512, 512), mode='bilinear', align_corners=True)

        preds_bin = (preds > 0.5).float()

        for j in range(imgs.size(0)):
            p_np = preds_bin[j, 0].cpu().numpy()
            if use_postprocess:
                p_np = postprocess_mask(p_np, min_size=min_size)
            t_np = masks[j, 0].numpy()
            if t_np.shape != (512, 512):
                t_np = cv2.resize(t_np, (512, 512), interpolation=cv2.INTER_NEAREST)

            # Evaluate on lesion-bearing images
            if t_np.sum() > 0:
                m = compute_image_metrics(p_np, t_np)
                per_image_results.append(m)

    return aggregate_metrics(per_image_results)

def main():
    print("="*80, flush=True)
    print("  EXECUTING MASTER BENCHMARK & MULTI-METRIC PROFILE", flush=True)
    print("="*80, flush=True)

    _, _, test_loader = get_seg_v2_dataloaders(img_size=512, use_copy_paste=False)

    # 1. Load trained models
    print("\n[1/5] Loading Checkpoints...", flush=True)
    m_unet = get_unet().to(DEVICE)
    m_unet.load_state_dict(torch.load(UNET_CHECKPOINT, map_location=DEVICE))

    m_unetpp = get_unetpp(deep_supervision=True).to(DEVICE)
    m_unetpp.load_state_dict(torch.load(UNETPP_CHECKPOINT, map_location=DEVICE))

    m_rep1 = get_repsegnet().to(DEVICE)
    m_rep1.load_state_dict(torch.load(REPSEGNET_CHECKPOINT, map_location=DEVICE))

    m_rep2 = get_repsegnet_v2(encoder_name=SEG_ENCODER_NAME, deep_supervision=True).to(DEVICE)
    m_rep2.load_state_dict(torch.load(REPSEGNET_V2_CHECKPOINT, map_location=DEVICE))

    # 2. Benchmark on BUSI Test Set (512x512)
    print("\n[2/5] Evaluating Models on BUSI Test Cohort (512x512)...", flush=True)
    busi_results = {
        'Classic U-Net': evaluate_loader(m_unet, test_loader, DEVICE, 512),
        'UNet++': evaluate_loader(m_unetpp, test_loader, DEVICE, 512),
        'RepSegNet V1': evaluate_loader(m_rep1, test_loader, DEVICE, 512),
        'RepSegNet V2 (Proposed)': evaluate_loader(m_rep2, test_loader, DEVICE, 512)
    }

    # 3. Resolution Scaling Analysis for RepSegNet V2 (128, 256, 512)
    print("\n[3/5] Evaluating Resolution Scaling for RepSegNet V2...", flush=True)
    res_scaling_results = {
        'RepSegNet V2 (128x128)': evaluate_loader(m_rep2, test_loader, DEVICE, target_size=128),
        'RepSegNet V2 (256x256)': evaluate_loader(m_rep2, test_loader, DEVICE, target_size=256),
        'RepSegNet V2 (512x512)': busi_results['RepSegNet V2 (Proposed)']
    }

    # 4. External Generalization Benchmark on UDIAT Dataset B (N=163)
    print("\n[4/5] Evaluating External Generalization on UDIAT Dataset B...", flush=True)
    udiat_files = sorted(glob.glob(os.path.join(BASE_DIR, "UDIAT-Dataset", "UDIAT_Dataset_B", "UDIAT_Dataset_B", "original", "*.png")))
    udiat_gt = sorted(glob.glob(os.path.join(BASE_DIR, "UDIAT-Dataset", "UDIAT_Dataset_B", "UDIAT_Dataset_B", "GT", "*.png")))

    udiat_metrics = {k: [] for k in ['Classic U-Net', 'UNet++', 'RepSegNet V1', 'RepSegNet V2 (Proposed)']}
    
    transform_512 = A.Compose([
        A.Resize(512, 512),
        A.Normalize(mean=[0,0,0], std=[1,1,1]),
        ToTensorV2()
    ])

    for img_p, zip_gt in zip(udiat_files, udiat_gt):
        img = cv2.cvtColor(cv2.imread(img_p), cv2.COLOR_BGR2RGB)
        gt = (cv2.imread(zip_gt, cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
        gt_512 = cv2.resize(gt, (512, 512), interpolation=cv2.INTER_NEAREST)

        t_res = transform_512(image=img)
        tensor = t_res['image'].unsqueeze(0).to(DEVICE)

        for name, m in [('Classic U-Net', m_unet), ('UNet++', m_unetpp), 
                        ('RepSegNet V1', m_rep1), ('RepSegNet V2 (Proposed)', m_rep2)]:
            pred = tta_predict(m, tensor)
            pred_bin = (pred[0, 0].cpu().numpy() > 0.5).astype(np.uint8)
            pred_clean = postprocess_mask(pred_bin, min_size=150)
            udiat_metrics[name].append(compute_image_metrics(pred_clean, gt_512))

    udiat_final = {k: aggregate_metrics(v) for k, v in udiat_metrics.items()}

    # 5. Full Computational Profile
    print("\n[5/5] Extracting Computational Profile...", flush=True)
    with open('outputs/computational_complexity_metrics.json') as f:
        comp_metrics = json.load(f)

    # 6. Real Empirical Ablation Breakdown (Evaluated directly from real trained checkpoints)
    print("\n[6/6] Evaluating Real Empirical Ablation Variants...", flush=True)
    ablation_base_path = os.path.join(OUTPUT_DIR, 'ablation_baseline.pth')
    ablation_scse_path = os.path.join(OUTPUT_DIR, 'ablation_scse.pth')

    m_base = get_repsegnet_v2(encoder_name=SEG_ENCODER_NAME, deep_supervision=False, use_scse=False).to(DEVICE)
    m_base.load_state_dict(torch.load(ablation_base_path, map_location=DEVICE))
    res_base = evaluate_loader(m_base, test_loader, DEVICE, 512, use_postprocess=False)

    m_scse = get_repsegnet_v2(encoder_name=SEG_ENCODER_NAME, deep_supervision=False, use_scse=True).to(DEVICE)
    m_scse.load_state_dict(torch.load(ablation_scse_path, map_location=DEVICE))
    res_scse = evaluate_loader(m_scse, test_loader, DEVICE, 512, use_postprocess=False)

    # Variant C is RepSegNet V2 before postprocessing
    res_deepsup = evaluate_loader(m_rep2, test_loader, DEVICE, 512, use_postprocess=False)

    # Variant D is RepSegNet V2 with postprocessing
    res_full_v2 = busi_results['RepSegNet V2 (Proposed)']

    dummy_input = torch.randn(1, 3, 512, 512).to(DEVICE)
    f_b, p_b = profile(m_base, inputs=(dummy_input,), verbose=False)
    f_s, p_s = profile(m_scse, inputs=(dummy_input,), verbose=False)
    f_v, p_v = profile(m_rep2, inputs=(dummy_input,), verbose=False)

    ablation_study = {
        'Baseline (EfficientNet-B3 + Standard Decoder)': {
            'dice': res_base['dice'], 'iou': res_base['iou'], 'hd95': res_base['hd95'], 'assd': res_base['assd'],
            'params_M': round(p_b / 1e6, 2), 'gflops': round(f_b / 1e9, 2),
            'estimated': False
        },
        '+ Concurrent sCSE Attention on Skips': {
            'dice': res_scse['dice'], 'iou': res_scse['iou'], 'hd95': res_scse['hd95'], 'assd': res_scse['assd'],
            'params_M': round(p_s / 1e6, 2), 'gflops': round(f_s / 1e9, 2),
            'estimated': False
        },
        '+ Multi-Scale Deep Supervision': {
            'dice': res_deepsup['dice'], 'iou': res_deepsup['iou'], 'hd95': res_deepsup['hd95'], 'assd': res_deepsup['assd'],
            'params_M': round(p_v / 1e6, 2), 'gflops': round(f_v / 1e9, 2),
            'estimated': False
        },
        '+ Connected-Component Post-Processing (Full V2)': {
            'dice': res_full_v2['dice'], 'iou': res_full_v2['iou'], 'hd95': res_full_v2['hd95'], 'assd': res_full_v2['assd'],
            'params_M': round(p_v / 1e6, 2), 'gflops': round(f_v / 1e9, 2),
            'estimated': False
        }
    }

    # Compile Single Master JSON
    master_data = {
        'busi_comparative_benchmark': busi_results,
        'resolution_scaling_analysis': res_scaling_results,
        'udiat_external_generalization': udiat_final,
        'computational_profile': comp_metrics,
        'ablation_study': ablation_study
    }

    out_file = os.path.join(OUTPUT_DIR, 'master_source_of_truth.json')
    with open(out_file, 'w') as f:
        json.dump(master_data, f, indent=2)

    print(f"\n[OK] Master Source of Truth successfully generated -> {out_file}", flush=True)

if __name__ == '__main__':
    main()

