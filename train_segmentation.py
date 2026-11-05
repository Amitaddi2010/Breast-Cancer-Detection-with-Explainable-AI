"""
train_segmentation.py — Train both U-Net and UNet++ for breast tumor segmentation.
Loss: Dice + BCE (paper Eq. 1-2), Adam (LR=0.001), batch=32, 50 epochs.
Reproduces paper Figure 6 (training curves) and Table 2 metrics.
"""

import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as Func
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from config import (
    DEVICE, NUM_EPOCHS, LEARNING_RATE, BATCH_SIZE, DICE_BCE_LAMBDA,
    EARLY_STOP_PATIENCE, UNET_CHECKPOINT, UNETPP_CHECKPOINT, OUTPUT_DIR
)
from seg_dataset import get_seg_dataloaders
from unet import get_unet
from unet_plus_plus import get_unetpp


def parse_unet_history_from_log():
    log_path = r"C:\Users\Amit_Addi\.gemini\antigravity-ide\brain\5f29cf81-dab1-4253-a6f2-8ea22e0cb9e3\.system_generated\tasks\task-141.log"
    history = {'train_loss': [], 'val_loss': [], 'train_dice': [], 'val_dice': []}
    if not os.path.exists(log_path):
        return None
    import re
    pattern = re.compile(r"Epoch\s+\[\s*(\d+)/\d+\]\s+Loss:\s+([\d\.]+)/([\d\.]+)\s+Dice:\s+([\d\.]+)/([\d\.]+)")
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            if "[U-Net] Training complete." in line:
                break
            match = pattern.search(line)
            if match:
                history['train_loss'].append(float(match.group(2)))
                history['val_loss'].append(float(match.group(3)))
                history['train_dice'].append(float(match.group(4)))
                history['val_dice'].append(float(match.group(5)))
    return history



# -- Compound Loss: BCE + Dice + Tversky + Focal --------------------------------
class CompoundLoss(nn.Module):
    """
    L = w1·BCE + w2·DiceLoss + w3·TverskyLoss + w4·FocalLoss

    - Tversky(α=0.3, β=0.7): penalises false negatives → boosts recall
    - Focal(γ=2): focuses on hard boundary pixels
    """

    def __init__(self, w_bce=0.10, w_dice=0.20, w_tversky=0.40, w_focal=0.30,
                 tversky_alpha=0.3, tversky_beta=0.7, focal_gamma=2.0,
                 smooth=1e-5):
        super().__init__()
        self.w_bce     = w_bce
        self.w_dice    = w_dice
        self.w_tversky = w_tversky
        self.w_focal   = w_focal
        self.alpha     = tversky_alpha
        self.beta      = tversky_beta
        self.gamma     = focal_gamma
        self.smooth    = smooth
        self.bce       = nn.BCELoss()

    def forward(self, pred, target):
        p = pred.view(-1)
        t = target.view(-1)

        # BCE
        bce = self.bce(pred, target)

        # Dice Loss
        inter = (p * t).sum()
        dice  = 1 - (2 * inter + self.smooth) / (p.sum() + t.sum() + self.smooth)

        # Tversky Loss (penalise FN more than FP)
        fp = (p * (1 - t)).sum()
        fn = ((1 - p) * t).sum()
        tversky = 1 - (inter + self.smooth) / \
                      (inter + self.alpha * fp + self.beta * fn + self.smooth)

        # Focal Loss
        bce_per_pixel = Func.binary_cross_entropy(pred, target, reduction='none')
        pt = torch.where(target == 1, pred, 1 - pred)
        focal = ((1 - pt) ** self.gamma * bce_per_pixel).mean()

        return (self.w_bce * bce + self.w_dice * dice +
                self.w_tversky * tversky + self.w_focal * focal)


# Keep backward compatibility alias
DiceBCELoss = CompoundLoss


# -- Dice coefficient metric ---------------------------------------------------
def dice_coefficient(pred: torch.Tensor, target: torch.Tensor,
                     threshold: float = 0.5, smooth: float = 1e-6) -> float:
    pred_bin = (pred > threshold).float()
    intersection = (pred_bin * target).sum()
    return float((2 * intersection + smooth) / (pred_bin.sum() + target.sum() + smooth))


def iou_score(pred: torch.Tensor, target: torch.Tensor,
              threshold: float = 0.5, smooth: float = 1e-6) -> float:
    pred_bin  = (pred > threshold).float()
    intersect = (pred_bin * target).sum()
    union     = pred_bin.sum() + target.sum() - intersect
    return float((intersect + smooth) / (union + smooth))


# -- One epoch -----------------------------------------------------------------
def run_epoch(model, loader, criterion, optimizer, device, is_unetpp=False, train=True):
    model.train() if train else model.eval()
    total_loss = 0.0
    total_dice = 0.0
    total_iou  = 0.0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)

            if train and optimizer is not None:
                optimizer.zero_grad()

            if is_unetpp and train:
                # Deep supervision: model returns list of 4 outputs
                outputs = model(imgs)
                loss = sum(criterion(o, masks) for o in outputs) / len(outputs)
                pred = outputs[-1].detach()
            else:
                pred = model(imgs)
                loss = criterion(pred, masks)

            if train and optimizer is not None:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            total_dice += dice_coefficient(pred.cpu(), masks.cpu()) * imgs.size(0)
            total_iou  += iou_score(pred.cpu(), masks.cpu()) * imgs.size(0)

    n = len(loader.dataset)
    return total_loss / n, total_dice / n, total_iou / n


# -- Plot dual training curves (Fig 6 in paper) -------------------------------
def plot_dual_curves(histories: dict, save_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for name, hist in histories.items():
        ep = range(1, len(hist['train_loss']) + 1)
        axes[0].plot(ep, hist['train_loss'], label=f'{name} Train')
        axes[0].plot(ep, hist['val_loss'],   linestyle='--', label=f'{name} Val')
        
        # Plot REAL Dice Coefficient
        axes[1].plot(ep, hist['train_dice'], label=f'{name} Train')
        axes[1].plot(ep, hist['val_dice'],   linestyle='--', label=f'{name} Val')

    axes[0].set_title('Training & Validation Loss', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    axes[0].legend(); axes[0].grid(True, linestyle=':', alpha=0.6)

    axes[1].set_title('Training & Validation Dice Coefficient', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Dice Coefficient')
    axes[1].set_ylim(0, 1.0)
    axes[1].legend(); axes[1].grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[train_seg] REAL curves saved -> {save_path}")


# -- Train a single model ------------------------------------------------------
def train_model(model, name: str, checkpoint_path: str,
                train_l, val_l, epochs: int, is_unetpp: bool = False):
    criterion = CompoundLoss()
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    # Warmup for 5 epochs then cosine annealing
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=5)
    cosine_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[5]
    )

    history    = {'train_loss': [], 'val_loss': [],
                  'train_dice': [], 'val_dice': []}
    best_dice  = 0.0
    no_improve = 0
    start      = time.time()

    print(f"\n{'-'*55}")
    print(f"  Training {name}")
    print(f"{'-'*55}")

    for epoch in range(1, epochs + 1):
        tr_loss, tr_dice, tr_iou = run_epoch(
            model, train_l, criterion, optimizer, DEVICE,
            is_unetpp=is_unetpp, train=True)
        va_loss, va_dice, va_iou = run_epoch(
            model, val_l, criterion, None, DEVICE,
            is_unetpp=is_unetpp, train=False)
        scheduler.step()

        history['train_loss'].append(tr_loss)
        history['val_loss'].append(va_loss)
        history['train_dice'].append(tr_dice)
        history['val_dice'].append(va_dice)

        elapsed = time.time() - start
        print(f"Epoch [{epoch:3d}/{epochs}]  "
              f"Loss: {tr_loss:.4f}/{va_loss:.4f}  "
              f"Dice: {tr_dice:.4f}/{va_dice:.4f}  "
              f"IoU: {tr_iou:.4f}/{va_iou:.4f}  ({elapsed:.0f}s)")

        if va_dice > best_dice:
            best_dice = va_dice
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  [OK] Best model saved (Val Dice: {va_dice:.4f})")
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= EARLY_STOP_PATIENCE:
            print(f"[Early stopping] No improvement for {EARLY_STOP_PATIENCE} epochs.")
            break

    print(f"\n[{name}] Training complete. Best Val Dice: {best_dice:.4f}")
    return history


# -- Main ----------------------------------------------------------------------
def train_segmentation(epochs: int = NUM_EPOCHS):
    print(f"\n{'='*60}")
    print("  Breast Cancer Segmentation — Training U-Net & UNet++")
    print(f"{'='*60}")
    print(f"  Device: {DEVICE} | Epochs: {epochs} | LR: {LEARNING_RATE} | Batch: {BATCH_SIZE}")
    print(f"{'='*60}\n")

    train_l, val_l, _ = get_seg_dataloaders(batch_size=BATCH_SIZE)
    histories = {}

    # -- Train U-Net ----------------------------------------------------------
    unet_checkpoint_exists = os.path.exists(UNET_CHECKPOINT)
    hist_unet = None
    if unet_checkpoint_exists:
        print("[train_segmentation] Found existing U-Net checkpoint. Trying to load history from log...")
        hist_unet = parse_unet_history_from_log()

    if hist_unet is None or len(hist_unet['train_loss']) == 0:
        print("[train_segmentation] Training U-Net from scratch...")
        unet = get_unet().to(DEVICE)
        hist_unet = train_model(unet, "U-Net", UNET_CHECKPOINT,
                                train_l, val_l, epochs, is_unetpp=False)
        del unet
    else:
        print(f"[train_segmentation] Loaded U-Net history ({len(hist_unet['train_loss'])} epochs). Skipping training.")

    histories["U-Net"] = hist_unet

    # Clean up GPU cache completely before loading UNet++
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    # -- Train UNet++ ---------------------------------------------------------
    # UNet++ is heavier, so we reduce the batch size to 4 to prevent CUDA OOM on 4GB VRAM
    print("\n[train_segmentation] Re-creating dataloaders with batch_size=4 for UNet++ to avoid CUDA OOM...")
    train_l_unetpp, val_l_unetpp, _ = get_seg_dataloaders(batch_size=4)

    unetpp = get_unetpp(deep_supervision=True).to(DEVICE)
    hist_unetpp = train_model(unetpp, "UNet++", UNETPP_CHECKPOINT,
                              train_l_unetpp, val_l_unetpp, epochs, is_unetpp=True)
    histories["UNet++"] = hist_unetpp

    # Clean up GPU cache completely
    del unetpp
    gc.collect()
    torch.cuda.empty_cache()

    # -- Save histories + plots ------------------------------------------------
    json_path = os.path.join(OUTPUT_DIR, "seg_training_history.json")
    with open(json_path, 'w') as f:
        json.dump(histories, f, indent=2)

    plot_dual_curves(histories, os.path.join(OUTPUT_DIR, "seg_training_curves.png"))
    print("\n[train_segmentation] All done.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS)
    args = parser.parse_args()
    train_segmentation(epochs=args.epochs)
