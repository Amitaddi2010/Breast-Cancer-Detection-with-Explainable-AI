"""
train.py — Training loop for the EfficientNet-B0 breast cancer classifier.
Uses class-weighted CrossEntropyLoss, AdamW, CosineAnnealingLR, early stopping.
"""

import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score, f1_score

from config import (
    DEVICE, NUM_EPOCHS, LEARNING_RATE, BATCH_SIZE,
    EARLY_STOP_PATIENCE, CLF_CHECKPOINT, OUTPUT_DIR, CLASS_NAMES
)
from dataset import get_dataloaders, get_class_weights
from model import get_classifier


# ── Loss ─────────────────────────────────────────────────────────────────────
def get_criterion(device=DEVICE):
    weights = get_class_weights().to(device)
    return nn.CrossEntropyLoss(weight=weights)


# ── Training one epoch ───────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc  = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc


# ── Validation epoch ─────────────────────────────────────────────────────────
@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc  = accuracy_score(all_labels, all_preds)
    epoch_f1   = f1_score(all_labels, all_preds, average='weighted')
    return epoch_loss, epoch_acc, epoch_f1


# ── Plot training curves ─────────────────────────────────────────────────────
def plot_curves(history: dict, save_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(history['train_loss']) + 1)

    axes[0].plot(epochs, history['train_loss'], 'b-o', label='Train')
    axes[0].plot(epochs, history['val_loss'],   'r-o', label='Val')
    axes[0].set_title('Loss Curve')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    axes[0].legend(); axes[0].grid(True)

    axes[1].plot(epochs, history['train_acc'], 'b-o', label='Train')
    axes[1].plot(epochs, history['val_acc'],   'r-o', label='Val')
    axes[1].set_title('Accuracy Curve')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy')
    axes[1].legend(); axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[train.py] Training curves saved -> {save_path}")


# ── Main training loop ────────────────────────────────────────────────────────
def train(epochs: int = NUM_EPOCHS):
    print(f"\n{'='*60}")
    print("  Breast Cancer Classifier — Training")
    print(f"{'='*60}")
    print(f"  Device : {DEVICE}")
    print(f"  Epochs : {epochs} | Batch: {BATCH_SIZE} | LR: {LEARNING_RATE}")
    print(f"{'='*60}\n")

    train_l, val_l, _ = get_dataloaders()
    model     = get_classifier().to(DEVICE)
    criterion = get_criterion()
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                      lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    history    = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_f1    = 0.0
    no_improve = 0
    start      = time.time()

    for epoch in range(1, epochs + 1):
        t_loss, t_acc = train_one_epoch(model, train_l, optimizer, criterion, DEVICE)
        v_loss, v_acc, v_f1 = validate(model, val_l, criterion, DEVICE)
        scheduler.step()

        history['train_loss'].append(t_loss)
        history['val_loss'].append(v_loss)
        history['train_acc'].append(t_acc)
        history['val_acc'].append(v_acc)

        elapsed = time.time() - start
        print(f"Epoch [{epoch:3d}/{epochs}]  "
              f"Train Loss: {t_loss:.4f}  Acc: {t_acc:.4f}  |  "
              f"Val Loss: {v_loss:.4f}  Acc: {v_acc:.4f}  F1: {v_f1:.4f}  "
              f"({elapsed:.0f}s)")

        # Save best model
        if v_f1 > best_f1:
            best_f1 = v_f1
            torch.save(model.state_dict(), CLF_CHECKPOINT)
            print(f"  [OK] Best model saved (Val F1: {v_f1:.4f})")
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= EARLY_STOP_PATIENCE:
            print(f"[Early stopping] No improvement for {EARLY_STOP_PATIENCE} epochs.")
            break

    # Save history and plot
    json_path = os.path.join(OUTPUT_DIR, "clf_training_history.json")
    with open(json_path, 'w') as f:
        json.dump(history, f, indent=2)
    plot_curves(history, os.path.join(OUTPUT_DIR, "clf_training_curves.png"))
    print(f"\n[train.py] Training complete. Best Val F1: {best_f1:.4f}")
    print(f"[train.py] Model saved -> {CLF_CHECKPOINT}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS)
    args = parser.parse_args()
    train(epochs=args.epochs)
