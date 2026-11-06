"""
train_ablations.py
Empirically trains the intermediate ablation models:
  Variant A: Baseline (EfficientNet-B3 + Standard Decoder without sCSE, without DeepSup)
  Variant B: + Concurrent sCSE Attention on Skips (without DeepSup)
Saves:
  outputs/ablation_baseline.pth
  outputs/ablation_scse.pth
"""

import os
import time
import json
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from config import (
    DEVICE, NUM_EPOCHS, LEARNING_RATE, BATCH_SIZE,
    EARLY_STOP_PATIENCE, OUTPUT_DIR, SEG_ENCODER_NAME, SEG_IMG_SIZE
)
from seg_dataset_v2 import get_seg_v2_dataloaders
from repsegnet_v2 import get_repsegnet_v2
from train_segmentation import CompoundLoss, dice_coefficient, iou_score


def train_ablation_variant(variant_name, use_scse, deep_supervision, checkpoint_path, epochs=NUM_EPOCHS):
    print(f"\n{'='*70}", flush=True)
    print(f"  TRAINING ABLATION: {variant_name}", flush=True)
    print(f"  use_scse={use_scse} | deep_supervision={deep_supervision} | Device={DEVICE}", flush=True)
    print(f"{'='*70}\n", flush=True)

    train_l, val_l, _ = get_seg_v2_dataloaders(batch_size=BATCH_SIZE, img_size=SEG_IMG_SIZE, use_copy_paste=True)

    model = get_repsegnet_v2(
        encoder_name=SEG_ENCODER_NAME,
        deep_supervision=deep_supervision,
        use_scse=use_scse,
        dropout=0.1
    ).to(DEVICE)

    criterion = CompoundLoss(w_bce=0.10, w_dice=0.50, w_tversky=0.30, w_focal=0.10)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=5)
    cosine_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[5]
    )

    best_dice = 0.0
    no_improve = 0
    start = time.time()

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        tr_loss, tr_dice, tr_iou = 0.0, 0.0, 0.0
        for imgs, masks in train_l:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()
            if deep_supervision:
                outputs = model(imgs)
                loss = criterion(outputs[0], masks)
                weights = [0.5, 0.3, 0.2, 0.1, 0.05]
                for idx, aux_out in enumerate(outputs[1:]):
                    w = weights[min(idx, len(weights) - 1)]
                    loss += w * criterion(aux_out, masks)
                pred = outputs[0].detach()
            else:
                pred = model(imgs)
                loss = criterion(pred, masks)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            tr_loss += loss.item() * imgs.size(0)
            tr_dice += dice_coefficient(pred.cpu(), masks.cpu()) * imgs.size(0)
            tr_iou  += iou_score(pred.cpu(), masks.cpu()) * imgs.size(0)

        n_tr = len(train_l.dataset)
        tr_loss, tr_dice, tr_iou = tr_loss / n_tr, tr_dice / n_tr, tr_iou / n_tr

        # Validation
        model.eval()
        va_loss, va_dice, va_iou = 0.0, 0.0, 0.0
        with torch.no_grad():
            for imgs, masks in val_l:
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                pred = model(imgs)
                loss = criterion(pred, masks)
                va_loss += loss.item() * imgs.size(0)
                va_dice += dice_coefficient(pred.cpu(), masks.cpu()) * imgs.size(0)
                va_iou  += iou_score(pred.cpu(), masks.cpu()) * imgs.size(0)

        n_va = len(val_l.dataset)
        va_loss, va_dice, va_iou = va_loss / n_va, va_dice / n_va, va_iou / n_va

        scheduler.step()

        elapsed = time.time() - start
        print(f"[{variant_name}] Epoch [{epoch:2d}/{epochs}] "
              f"Loss: {tr_loss:.4f}/{va_loss:.4f} | "
              f"Dice: {tr_dice:.4f}/{va_dice:.4f} | "
              f"IoU: {tr_iou:.4f}/{va_iou:.4f} ({elapsed:.0f}s)", flush=True)

        if va_dice > best_dice:
            best_dice = va_dice
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  [OK] Saved new best checkpoint (Val Dice: {va_dice:.4f})", flush=True)
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= EARLY_STOP_PATIENCE:
            print(f"[{variant_name}] Early stopping triggered after {EARLY_STOP_PATIENCE} epochs without improvement.", flush=True)
            break

    print(f"\nFinished {variant_name}. Best Val Dice: {best_dice:.4f} saved to {checkpoint_path}\n", flush=True)
    return best_dice


def main():
    base_ckpt = os.path.join(OUTPUT_DIR, "ablation_baseline.pth")
    scse_ckpt = os.path.join(OUTPUT_DIR, "ablation_scse.pth")

    print("STARTING EMPIRICAL ABLATION EXPERIMENTS (RTX 5060 Ti)...", flush=True)

    # 1. Variant A: Baseline
    train_ablation_variant("Variant A (Baseline: No sCSE, No DeepSup)",
                           use_scse=False, deep_supervision=False,
                           checkpoint_path=base_ckpt, epochs=60)

    # 2. Variant B: + Concurrent sCSE
    train_ablation_variant("Variant B (+ Concurrent sCSE, No DeepSup)",
                           use_scse=True, deep_supervision=False,
                           checkpoint_path=scse_ckpt, epochs=60)

    print("ALL EMPIRICAL ABLATION CHECKPOINTS TRAINED SUCCESSFULLY!", flush=True)


if __name__ == "__main__":
    main()

