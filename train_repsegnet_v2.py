"""
train_repsegnet_v2.py — Revamped training recipe for RepSegNet V2.

Key improvements:
  1. Support for deep supervision loss (weighted loss across all auxiliary heads)
  2. AdamW + CosineAnnealingWarmRestarts or OneCycleLR
  3. Saves training metrics and updates seg_training_history.json
  4. Keeps original RepSegNet history intact and appends RepSegNetV2
  5. Fallback/quick-test mode for fast validation on CPU
"""

import os
import time
import json
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import matplotlib.pyplot as plt

from config import (
    DEVICE, NUM_EPOCHS, LEARNING_RATE, BATCH_SIZE,
    EARLY_STOP_PATIENCE, REPSEGNET_V2_CHECKPOINT, OUTPUT_DIR,
    SEG_ENCODER_NAME, SEG_IMG_SIZE
)
from seg_dataset_v2 import get_seg_v2_dataloaders
from repsegnet_v2 import get_repsegnet_v2
from train_segmentation import CompoundLoss, dice_coefficient, iou_score


def run_v2_epoch(model, loader, criterion, optimizer, device, train=True, deep_supervision=True):
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

            if deep_supervision and train:
                # Deep supervision: model returns list of outputs [main_out, aux1, aux2, ...]
                outputs = model(imgs)
                # Compute weighted loss (main loss has higher weight, auxiliary losses have lower weights)
                loss = criterion(outputs[0], masks)
                weights = [0.5, 0.3, 0.2, 0.1, 0.05]  # Weights for aux heads
                for idx, aux_out in enumerate(outputs[1:]):
                    w = weights[min(idx, len(weights) - 1)]
                    loss += w * criterion(aux_out, masks)
                pred = outputs[0].detach()
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


def train_repsegnet_v2(epochs=NUM_EPOCHS, quick_test=False):
    print(f"\n{'='*60}")
    print("  Breast Cancer Segmentation — Training RepSegNet V2 (Pretrained)")
    print(f"{'='*60}")
    
    # Adjust parameters for CPU/Quick test
    batch_size = BATCH_SIZE
    if DEVICE.type == 'cpu':
        print("  [INFO] CPU detected. Setting batch_size=4 to ensure compatibility and speed.")
        batch_size = 4
    if quick_test:
        print("  [INFO] Running in QUICK TEST mode (2 epochs).")
        epochs = 2

    print(f"  Device: {DEVICE} | Epochs: {epochs} | LR: {LEARNING_RATE} | Batch: {batch_size}")
    print(f"{'='*60}\n")

    # Load dataloaders
    train_l, val_l, _ = get_seg_v2_dataloaders(batch_size=batch_size, img_size=SEG_IMG_SIZE, use_copy_paste=not quick_test)

    # Load history file
    json_path = os.path.join(OUTPUT_DIR, "seg_training_history.json")
    histories = {}
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                histories = json.load(f)
        except Exception as e:
            print(f"  [WARN] Failed to load existing history JSON: {e}")

    # Build model (deep_supervision=True is default)
    repsegnet_v2 = get_repsegnet_v2(encoder_name=SEG_ENCODER_NAME, deep_supervision=True, dropout=0.1).to(DEVICE)
    checkpoint_path = REPSEGNET_V2_CHECKPOINT

    # Dice-centric Loss optimization to boost Dice Score directly
    criterion = CompoundLoss(w_bce=0.10, w_dice=0.50, w_tversky=0.30, w_focal=0.10)
    optimizer = AdamW(repsegnet_v2.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    # Sequential scheduler: warmup then Cosine Annealing
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=5)
    cosine_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[5]
    )

    history = {'train_loss': [], 'val_loss': [],
               'train_dice': [], 'val_dice': []}
    best_dice = 0.0
    no_improve = 0
    start = time.time()

    print(f"\n{'-'*55}")
    print("  Training RepSegNet V2")
    print(f"{'-'*55}")

    for epoch in range(1, epochs + 1):
        tr_loss, tr_dice, tr_iou = run_v2_epoch(
            repsegnet_v2, train_l, criterion, optimizer, DEVICE,
            train=True, deep_supervision=True
        )
        
        va_loss, va_dice, va_iou = run_v2_epoch(
            repsegnet_v2, val_l, criterion, None, DEVICE,
            train=False, deep_supervision=False
        )
        
        # Scheduler step (ensure warmup/cosine behaves correctly)
        if epoch <= 5:
            # sequentially handles warmup
            scheduler.step()
        else:
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
            torch.save(repsegnet_v2.state_dict(), checkpoint_path)
            print(f"  [OK] Best model saved (Val Dice: {va_dice:.4f})")
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= EARLY_STOP_PATIENCE and not quick_test:
            print(f"[Early stopping] No improvement for {EARLY_STOP_PATIENCE} epochs.")
            break

    print(f"\n[RepSegNet V2] Training complete. Best Val Dice: {best_dice:.4f}")
    
    # Save history
    if not quick_test:
        histories["RepSegNetV2"] = history
        with open(json_path, 'w') as f:
            json.dump(histories, f, indent=2)

        # Plot all segmentation models training curves
        try:
            fig, axes = plt.subplots(1, 2, figsize=(16, 5))
            colors = ['blue', 'green', 'red', 'purple']
            
            for idx, (model_name, hist) in enumerate(histories.items()):
                ep = range(1, len(hist['train_loss']) + 1)
                c = colors[idx % len(colors)]
                axes[0].plot(ep, hist['train_loss'], color=c, label=f'{model_name} Train')
                axes[0].plot(ep, hist['val_loss'],   color=c, linestyle='--', label=f'{model_name} Val')
                axes[1].plot(ep, hist['train_dice'], color=c, label=f'{model_name} Train')
                axes[1].plot(ep, hist['val_dice'],   color=c, linestyle='--', label=f'{model_name} Val')

            axes[0].set_title('Training & Validation Loss')
            axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Dice+BCE Loss')
            axes[0].legend(); axes[0].grid(True)

            axes[1].set_title('Training & Validation Dice Score')
            axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Dice Coefficient')
            axes[1].legend(); axes[1].grid(True)

            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, "seg_training_curves.png"), dpi=150)
            plt.close()
            print("\n[train_repsegnet_v2] Plots generated and saved.")
        except Exception as e:
            print(f"  [WARN] Failed to generate training curves plot: {e}")

    print("\n[train_repsegnet_v2] Completed successfully.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS)
    parser.add_argument('--quick-test', action='store_true', help="Run quick 2-epoch test")
    args = parser.parse_args()
    train_repsegnet_v2(epochs=args.epochs, quick_test=args.quick_test)
