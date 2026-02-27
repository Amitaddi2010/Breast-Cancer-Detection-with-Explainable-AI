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
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from config import (
    DEVICE, NUM_EPOCHS, LEARNING_RATE, BATCH_SIZE, DICE_BCE_LAMBDA,
    EARLY_STOP_PATIENCE, UNET_CHECKPOINT, UNETPP_CHECKPOINT, OUTPUT_DIR
)
from seg_dataset import get_seg_dataloaders
from unet import get_unet
from unet_plus_plus import get_unetpp


# -- Combined Dice + BCE Loss (paper Eq. 1-2) ---------------------------------
class DiceBCELoss(nn.Module):
    """
    L_total = λ·BCE + (1-λ)·DiceLoss
    Dice Loss = 1 - (2·|P∩G|) / (|P| + |G|)
    """
    def __init__(self, lam: float = DICE_BCE_LAMBDA, smooth: float = 1e-6):
        super().__init__()
        self.lam    = lam
        self.smooth = smooth
        self.bce    = nn.BCELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # BCE term
        bce_loss = self.bce(pred, target)

        # Dice term
        pred_flat   = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice_loss = 1 - (2 * intersection + self.smooth) / \
                        (pred_flat.sum() + target_flat.sum() + self.smooth)

        return self.lam * bce_loss + (1 - self.lam) * dice_loss


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
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    for name, hist in histories.items():
        ep = range(1, len(hist['train_loss']) + 1)
        axes[0].plot(ep, hist['train_loss'], label=f'{name} Train')
        axes[0].plot(ep, hist['val_loss'],   linestyle='--', label=f'{name} Val')
        axes[1].plot(ep, hist['train_dice'], label=f'{name} Train')
        axes[1].plot(ep, hist['val_dice'],   linestyle='--', label=f'{name} Val')

    axes[0].set_title('Training & Validation Loss (U-Net vs UNet++)')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Dice+BCE Loss')
    axes[0].legend(); axes[0].grid(True)

    axes[1].set_title('Training & Validation Dice Score (U-Net vs UNet++)')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Dice Coefficient')
    axes[1].legend(); axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[train_seg] Curves saved -> {save_path}")


# -- Train a single model ------------------------------------------------------
def train_model(model, name: str, checkpoint_path: str,
                train_l, val_l, epochs: int, is_unetpp: bool = False):
    criterion = DiceBCELoss()
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

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
        scheduler.step(va_dice)

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

    train_l, val_l, _ = get_seg_dataloaders()
    histories = {}

    # -- Train U-Net ----------------------------------------------------------
    unet = get_unet().to(DEVICE)
    hist_unet = train_model(unet, "U-Net", UNET_CHECKPOINT,
                            train_l, val_l, epochs, is_unetpp=False)
    histories["U-Net"] = hist_unet

    # -- Train UNet++ ---------------------------------------------------------
    unetpp = get_unetpp(deep_supervision=True).to(DEVICE)
    hist_unetpp = train_model(unetpp, "UNet++", UNETPP_CHECKPOINT,
                              train_l, val_l, epochs, is_unetpp=True)
    histories["UNet++"] = hist_unetpp

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
