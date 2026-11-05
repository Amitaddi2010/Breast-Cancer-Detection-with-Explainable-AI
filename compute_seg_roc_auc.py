"""
compute_seg_roc_auc.py — Pixel-level ROC AUC for segmentation models.

Computes the Receiver Operating Characteristic Area Under the Curve (ROC AUC)
for U-Net, UNet++, and RepSegNet on the BUSI test set.

The ROC is computed at the *pixel level*: each pixel's predicted probability
(raw sigmoid output) is compared against the binary ground-truth mask.

Outputs:
    - outputs/seg_roc_curves.png   : Comparative ROC curve plot
    - outputs/seg_roc_auc.json     : Per-model AUC values
    - Console table with results
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import roc_curve, auc

from config import (
    DEVICE, UNET_CHECKPOINT, UNETPP_CHECKPOINT,
    REPSEGNET_CHECKPOINT, OUTPUT_DIR
)
from seg_dataset import get_seg_dataloaders
from unet import get_unet
from unet_plus_plus import get_unetpp
from repsegnet import get_repsegnet


# ── Collect raw probabilities ────────────────────────────────────────────────
@torch.no_grad()
def collect_probabilities(model, loader, device, max_pixels=5_000_000):
    """
    Run inference and collect raw sigmoid probabilities + ground-truth labels.
    
    To keep memory manageable on large test sets, we subsample pixels
    uniformly if the total exceeds `max_pixels`.
    """
    model.eval()
    all_probs = []
    all_targets = []

    for imgs, masks in loader:
        imgs = imgs.to(device)
        preds = model(imgs)                        # [B, 1, H, W], sigmoid output
        all_probs.append(preds.cpu().numpy().flatten())
        all_targets.append(masks.cpu().numpy().flatten())

    all_probs   = np.concatenate(all_probs)
    all_targets = np.concatenate(all_targets).astype(int)

    # Subsample if needed to avoid sklearn memory issues
    if len(all_probs) > max_pixels:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(all_probs), max_pixels, replace=False)
        all_probs   = all_probs[idx]
        all_targets = all_targets[idx]

    return all_probs, all_targets


# ── Plot ─────────────────────────────────────────────────────────────────────
def plot_roc_curves(results: dict, save_path: str):
    """
    Plot comparative ROC curves for all evaluated models on a single figure.
    """
    colors = {
        "U-Net":     "#3498db",
        "UNet++":    "#e74c3c",
        "RepSegNet": "#2ecc71",
    }

    plt.figure(figsize=(9, 7))
    plt.rcParams.update({'font.size': 12})

    for model_name, data in results.items():
        color = colors.get(model_name, "#888888")
        plt.plot(
            data["fpr"], data["tpr"],
            color=color, linewidth=2.2,
            label=f'{model_name}  (AUC = {data["auc"]:.4f})'
        )

    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random Chance')

    plt.xlabel('False Positive Rate', fontsize=13)
    plt.ylabel('True Positive Rate', fontsize=13)
    plt.title('Pixel-Level ROC Curves — Segmentation Models (BUSI Test Set)',
              fontsize=14, fontweight='bold', pad=12)
    plt.legend(loc='lower right', fontsize=11, framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[ROC AUC] Comparative ROC curve saved -> {save_path}")


# ── Main ─────────────────────────────────────────────────────────────────────
def compute_roc_auc():
    _, _, test_loader = get_seg_dataloaders()

    model_configs = [
        ("U-Net",     get_unet(),                              UNET_CHECKPOINT,      False),
        ("UNet++",    get_unetpp(deep_supervision=True),       UNETPP_CHECKPOINT,    True),
        ("RepSegNet", get_repsegnet(),                         REPSEGNET_CHECKPOINT, False),
    ]

    results = {}

    for model_name, model, checkpoint, is_unetpp in model_configs:
        if not os.path.exists(checkpoint):
            print(f"[ROC AUC] Checkpoint not found: {checkpoint} — skipping {model_name}")
            continue

        model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
        model = model.to(DEVICE)
        print(f"\n[ROC AUC] Computing ROC AUC for {model_name}...")

        probs, targets = collect_probabilities(model, test_loader, DEVICE)
        fpr, tpr, thresholds = roc_curve(targets, probs)
        roc_auc = auc(fpr, tpr)

        results[model_name] = {
            "auc": float(roc_auc),
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
        }
        print(f"[ROC AUC] {model_name}: AUC = {roc_auc:.4f}")

    if not results:
        print("[ROC AUC] No checkpoints found. Please train models first.")
        return

    # ── Print summary table ──────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print("  Pixel-Level ROC AUC — Segmentation Models")
    print(f"{'='*55}")
    print(f"  {'Model':<15} {'ROC AUC':>10}")
    print(f"  {'-'*25}")
    for name, data in results.items():
        print(f"  {name:<15} {data['auc']:>10.4f}")
    print(f"{'='*55}\n")

    # ── Save plot ────────────────────────────────────────────────────────────
    plot_roc_curves(results, os.path.join(OUTPUT_DIR, "seg_roc_curves.png"))

    # ── Save JSON (AUC values only, no giant FPR/TPR arrays) ─────────────────
    auc_summary = {name: round(data["auc"] * 100, 2) for name, data in results.items()}
    json_path = os.path.join(OUTPUT_DIR, "seg_roc_auc.json")
    with open(json_path, 'w') as f:
        json.dump(auc_summary, f, indent=2)
    print(f"[ROC AUC] AUC values saved -> {json_path}")

    return results


if __name__ == "__main__":
    compute_roc_auc()
