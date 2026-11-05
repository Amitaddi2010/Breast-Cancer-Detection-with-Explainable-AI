"""
plot_individual_curves.py — Generates individual high-quality publication-grade training and validation diagrams for U-Net, UNet++, RepSegNet V1, and RepSegNet V2.
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np

# Set standard publication-ready styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'figure.dpi': 300
})

def plot_model_curves(model_key, display_name, save_filename, history_dict):
    data = history_dict[model_key]
    epochs = np.arange(1, len(data['train_loss']) + 1)
    
    # Extract metrics
    train_loss = data['train_loss']
    val_loss = data['val_loss']
    train_dice = [d * 100 if max(data['train_dice']) <= 1.0 else d for d in data['train_dice']]
    val_dice = [d * 100 if max(data['val_dice']) <= 1.0 else d for d in data['val_dice']]
    
    # Find best validation Dice epoch
    best_idx = np.argmax(val_dice)
    best_epoch = epochs[best_idx]
    best_val_dice = val_dice[best_idx]
    
    # Create side-by-side subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    
    # ── Left Plot: Loss Curves ────────────────────────────────────────────────
    ax1.plot(epochs, train_loss, label='Training Loss', color='#1A365D', linewidth=2.0)
    ax1.plot(epochs, val_loss, label='Validation Loss', color='#DD6B20', linewidth=2.0, linestyle='--')
    ax1.set_xlabel('Epochs', fontweight='bold')
    ax1.set_ylabel('Loss', fontweight='bold')
    ax1.set_title('Training & Validation Loss', pad=10)
    ax1.legend(frameon=True, facecolor='white', edgecolor='none')
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    # ── Right Plot: Dice Coefficient Curves ────────────────────────────────────
    ax2.plot(epochs, train_dice, label='Training Dice', color='#1A365D', linewidth=2.0)
    ax2.plot(epochs, val_dice, label='Validation Dice', color='#DD6B20', linewidth=2.0, linestyle='--')
    
    # Mark the best epoch
    ax2.scatter(best_epoch, best_val_dice, color='#E53E3E', s=80, zorder=5, 
                label=f'Best Epoch ({best_epoch}): {best_val_dice:.2f}%')
    ax2.axvline(best_epoch, color='#E53E3E', linestyle=':', alpha=0.7)
    
    ax2.set_xlabel('Epochs', fontweight='bold')
    ax2.set_ylabel('Dice Coefficient (%)', fontweight='bold')
    ax2.set_title('Training & Validation Dice Score', pad=10)
    ax2.legend(frameon=True, facecolor='white', edgecolor='none')
    ax2.grid(True, linestyle=':', alpha=0.6)
    
    # Main title
    fig.suptitle(f'{display_name}: Training and Validation Performance Curves', fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    # Ensure outputs folder exists
    os.makedirs("outputs", exist_ok=True)
    save_path = os.path.join("outputs", save_filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Plot OK] Saved diagram for {display_name} -> {save_path}")

def generate_all_diagrams():
    history_path = os.path.join("outputs", "seg_training_history.json")
    if not os.path.exists(history_path):
        print(f"[Plot Error] Training history file not found: {history_path}")
        return
        
    with open(history_path, "r", encoding="utf-8") as f:
        history = json.load(f)
        
    models_to_plot = [
        ("U-Net", "Classic U-Net", "seg_curve_unet.png"),
        ("UNet++", "UNet++ (Deep Supervision)", "seg_curve_unetpp.png"),
        ("RepSegNet", "RepSegNet V1", "seg_curve_repsegnet.png"),
        ("RepSegNetV2", "RepSegNet V2 (Ours)", "seg_curve_repsegnet_v2.png")
    ]
    
    for key, display, filename in models_to_plot:
        if key in history:
            plot_model_curves(key, display, filename, history)
        else:
            print(f"[Plot Warning] Model {key} not found in history.")

if __name__ == "__main__":
    generate_all_diagrams()
