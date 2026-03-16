import os
import time
import json
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt

from config import (
    DEVICE, NUM_EPOCHS, LEARNING_RATE, BATCH_SIZE, DICE_BCE_LAMBDA,
    EARLY_STOP_PATIENCE, REPSEGNET_CHECKPOINT, OUTPUT_DIR
)
from seg_dataset import get_seg_dataloaders
from repsegnet import get_repsegnet
from train_segmentation import DiceBCELoss, dice_coefficient, iou_score, run_epoch, plot_dual_curves

def train_repsegnet(epochs=NUM_EPOCHS):
    print(f"\n{'='*60}")
    print("  Breast Cancer Segmentation - Training RepSegNet (2026)")
    print(f"{'='*60}")
    print(f"  Device: {DEVICE} | Epochs: {epochs} | LR: {LEARNING_RATE} | Batch: {BATCH_SIZE}")
    print(f"{'='*60}\n")

    train_l, val_l, _ = get_seg_dataloaders()
    
    # Check if history exists to append
    json_path = os.path.join(OUTPUT_DIR, "seg_training_history.json")
    histories = {}
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            histories = json.load(f)

    # Train RepSegNet
    repsegnet = get_repsegnet().to(DEVICE)
    name = "RepSegNet"
    checkpoint_path = REPSEGNET_CHECKPOINT
    is_unetpp = False

    criterion = DiceBCELoss()
    optimizer = Adam(repsegnet.parameters(), lr=LEARNING_RATE)
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
            repsegnet, train_l, criterion, optimizer, DEVICE,
            is_unetpp=is_unetpp, train=True)
        va_loss, va_dice, va_iou = run_epoch(
            repsegnet, val_l, criterion, None, DEVICE,
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
            torch.save(repsegnet.state_dict(), checkpoint_path)
            print(f"  [OK] Best model saved (Val Dice: {va_dice:.4f})")
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= EARLY_STOP_PATIENCE:
            print(f"[Early stopping] No improvement for {EARLY_STOP_PATIENCE} epochs.")
            break

    print(f"\n[{name}] Training complete. Best Val Dice: {best_dice:.4f}")
    histories["RepSegNet"] = history
    
    # Save combined history
    with open(json_path, 'w') as f:
        json.dump(histories, f, indent=2)

    # Note: Using custom plot for 3 models now
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    colors = ['blue', 'green', 'red']
    
    for idx, (model_name, hist) in enumerate(histories.items()):
        ep = range(1, len(hist['train_loss']) + 1)
        axes[0].plot(ep, hist['train_loss'], color=colors[idx % 3], label=f'{model_name} Train')
        axes[0].plot(ep, hist['val_loss'],   color=colors[idx % 3], linestyle='--', label=f'{model_name} Val')
        axes[1].plot(ep, hist['train_dice'], color=colors[idx % 3], label=f'{model_name} Train')
        axes[1].plot(ep, hist['val_dice'],   color=colors[idx % 3], linestyle='--', label=f'{model_name} Val')

    axes[0].set_title('Training & Validation Loss')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Dice+BCE Loss')
    axes[0].legend(); axes[0].grid(True)

    axes[1].set_title('Training & Validation Dice Score')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Dice Coefficient')
    axes[1].legend(); axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "seg_training_curves.png"), dpi=150)
    plt.close()
    print("\n[train_repsegnet] All done.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS)
    args = parser.parse_args()
    train_repsegnet(epochs=args.epochs)
