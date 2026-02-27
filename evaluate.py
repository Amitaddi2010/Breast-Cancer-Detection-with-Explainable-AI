"""
evaluate.py — Evaluation for the EfficientNet-B0 breast cancer classifier.
Generates: confusion matrix, ROC curves, per-class metrics, sample prediction grid.
Reproduces paper Table 3 metrics (accuracy, precision, recall, F1).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
from sklearn.preprocessing import label_binarize

from config import DEVICE, CLF_CHECKPOINT, OUTPUT_DIR, CLASS_NAMES, NUM_CLASSES
from dataset import get_dataloaders
from model import get_classifier


@torch.no_grad()
def get_predictions(model, loader, device):
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        outputs = model(imgs)
        probs   = torch.softmax(outputs, dim=1).cpu().numpy()
        preds   = outputs.argmax(dim=1).cpu().numpy()
        all_probs.extend(probs)
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix — EfficientNet-B0 Classifier', fontsize=14)
    plt.ylabel('True Label'); plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[evaluate] Confusion matrix -> {save_path}")


def plot_roc_curves(y_true, y_probs, class_names, save_path):
    y_bin   = label_binarize(y_true, classes=list(range(len(class_names))))
    colors  = ['#2196F3', '#F44336', '#4CAF50']
    plt.figure(figsize=(9, 7))
    for i, (cname, color) in enumerate(zip(class_names, colors)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, lw=2,
                 label=f'{cname.capitalize()} (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title('ROC Curves — Per Class (One-vs-Rest)')
    plt.legend(loc='lower right'); plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[evaluate] ROC curves -> {save_path}")


def print_metrics_table(y_true, y_pred, y_probs, class_names):
    acc  = accuracy_score(y_true, y_pred) * 100
    prec = precision_score(y_true, y_pred, average='weighted') * 100
    rec  = recall_score(y_true, y_pred, average='weighted') * 100
    f1   = f1_score(y_true, y_pred, average='weighted') * 100

    y_bin = label_binarize(y_true, classes=list(range(len(class_names))))
    auc_vals = []
    for i in range(len(class_names)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_probs[:, i])
        auc_vals.append(auc(fpr, tpr) * 100)
    macro_auc = np.mean(auc_vals)

    print(f"\n{'='*55}")
    print("  EfficientNet-B0 Classification Metrics")
    print(f"{'='*55}")
    print(f"  Accuracy  : {acc:.2f}%")
    print(f"  Precision : {prec:.2f}%")
    print(f"  Recall    : {rec:.2f}%")
    print(f"  F1-Score  : {f1:.2f}%")
    print(f"  AUC (macro): {macro_auc:.2f}%")
    print(f"{'='*55}")
    print("\n  Per-Class Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    return {'accuracy': acc, 'precision': prec, 'recall': rec,
            'f1': f1, 'auc': macro_auc}


def evaluate_classifier():
    _, _, test_l = get_dataloaders()
    model = get_classifier(pretrained=False).to(DEVICE)

    if not os.path.exists(CLF_CHECKPOINT):
        print(f"[evaluate] Checkpoint not found: {CLF_CHECKPOINT}")
        print("[evaluate] Please run train.py first.")
        return

    model.load_state_dict(torch.load(CLF_CHECKPOINT, map_location=DEVICE))
    print(f"[evaluate] Loaded classifier from: {CLF_CHECKPOINT}")

    y_true, y_pred, y_probs = get_predictions(model, test_l, DEVICE)

    metrics = print_metrics_table(y_true, y_pred, y_probs, CLASS_NAMES)
    plot_confusion_matrix(y_true, y_pred, CLASS_NAMES,
                          os.path.join(OUTPUT_DIR, "clf_confusion_matrix.png"))
    plot_roc_curves(y_true, y_probs, CLASS_NAMES,
                    os.path.join(OUTPUT_DIR, "clf_roc_curves.png"))

    import json
    with open(os.path.join(OUTPUT_DIR, "clf_test_metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=2)

    return metrics


if __name__ == "__main__":
    try:
        import seaborn
    except ImportError:
        import subprocess, sys
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'seaborn', '-q'])
    evaluate_classifier()
