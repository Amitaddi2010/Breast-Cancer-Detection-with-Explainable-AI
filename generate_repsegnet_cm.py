import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix
from config import DEVICE, REPSEGNET_CHECKPOINT, OUTPUT_DIR
from seg_dataset import get_seg_dataloaders
from repsegnet import get_repsegnet

@torch.no_grad()
def get_pixel_predictions(model, loader, device, threshold=0.5):
    model.eval()
    all_preds, all_targets = [], []
    print("Running inference on test set for RepSegNet...")
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        preds = model(imgs)
        preds_bin = (preds > threshold).float()
        
        all_preds.append(preds_bin.cpu().numpy().flatten())
        all_targets.append(masks.cpu().numpy().flatten())
        
    all_preds = np.concatenate(all_preds).astype(int)
    all_targets = np.concatenate(all_targets).astype(int)
    return all_targets, all_preds

def plot_segmentation_cm(y_true, y_pred, save_path):
    print("Computing pixel-wise confusion matrix...")
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize by total pixels for percentage visualization
    cm_norm = cm.astype('float') / cm.sum() * 100
    
    plt.figure(figsize=(10, 8))
    # Use labels for the heatmap
    labels = np.array([[f"{cm[0,0]}\n({cm_norm[0,0]:.2f}%)", f"{cm[0,1]}\n({cm_norm[0,1]:.2f}%)"],
                       [f"{cm[1,0]}\n({cm_norm[1,0]:.2f}%)", f"{cm[1,1]}\n({cm_norm[1,1]:.2f}%)"]])
    
    sns.heatmap(cm, annot=labels, fmt="", cmap='Greens',
                xticklabels=['Background', 'Tumor'], 
                yticklabels=['Background', 'Tumor'])
    
    plt.title('Pixel-wise Confusion Matrix — RepSegNet (Novel SOTA)', fontsize=16, fontweight='bold')
    plt.ylabel('True Class (Ground Truth)', fontsize=12)
    plt.xlabel('Predicted Class', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to: {save_path}")

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    _, _, test_loader = get_seg_dataloaders()
    model = get_repsegnet().to(DEVICE)
    
    if not os.path.exists(REPSEGNET_CHECKPOINT):
        print(f"Checkpoint not found: {REPSEGNET_CHECKPOINT}")
        return
        
    model.load_state_dict(torch.load(REPSEGNET_CHECKPOINT, map_location=DEVICE))
    
    y_true, y_pred = get_pixel_predictions(model, test_loader, DEVICE)
    
    save_path = os.path.join(OUTPUT_DIR, "repsegnet_confusion_matrix.png")
    plot_segmentation_cm(y_true, y_pred, save_path)

if __name__ == "__main__":
    main()
