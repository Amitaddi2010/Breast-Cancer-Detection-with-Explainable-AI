"""
scratch_eval_global.py — Computes dataset-wide global Dice and IoU metrics on the test set.
"""

import os
import torch
import numpy as np
from config import DEVICE, SEG_IMG_SIZE, SEG_ENCODER_NAME, REPSEGNET_V2_CHECKPOINT
from seg_dataset_v2 import get_seg_v2_dataloaders
from repsegnet_v2 import get_repsegnet_v2
from postprocess import postprocess_mask

def evaluate_global():
    _, _, test_loader = get_seg_v2_dataloaders(img_size=512, use_copy_paste=False)
    
    model = get_repsegnet_v2(encoder_name='efficientnet_b3', deep_supervision=True)
    model.load_state_dict(torch.load(REPSEGNET_V2_CHECKPOINT, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    
    total_intersection = 0
    total_union = 0
    total_pred = 0
    total_target = 0
    
    # Use optimal threshold = 0.50 and min_size = 150
    threshold = 0.50
    min_size = 150
    
    with torch.no_grad():
        for imgs, masks in test_loader:
            imgs = imgs.to(DEVICE)
            # TTA
            p0 = model(imgs)
            img_hflip = torch.flip(imgs, dims=[3])
            p1 = torch.flip(model(img_hflip), dims=[3])
            img_vflip = torch.flip(imgs, dims=[2])
            p2 = torch.flip(model(img_vflip), dims=[2])
            img_hvflip = torch.flip(imgs, dims=[2, 3])
            p3 = torch.flip(model(img_hvflip), dims=[2, 3])
            
            preds = torch.stack([p0, p1, p2, p3]).mean(dim=0)
            preds_bin = (preds > threshold).float()
            
            for j in range(imgs.size(0)):
                pred_np = preds_bin[j, 0].cpu().numpy().astype(np.uint8)
                if min_size > 0:
                    pred_np = postprocess_mask(pred_np, min_size=min_size)
                
                target_np = masks[j, 0].cpu().numpy().astype(np.uint8)
                
                intersection = np.logical_and(pred_np, target_np).sum()
                union = np.logical_or(pred_np, target_np).sum()
                
                total_intersection += intersection
                total_union += union
                total_pred += pred_np.sum()
                total_target += target_np.sum()
                
    smooth = 1e-6
    global_dice = (2 * total_intersection) / (total_pred + total_target + smooth) * 100
    global_iou = total_intersection / (total_union + smooth) * 100
    
    print(f"Global Dice: {global_dice:.2f}%")
    print(f"Global IoU: {global_iou:.2f}%")

if __name__ == "__main__":
    evaluate_global()
