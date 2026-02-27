"""
config.py — Centralized configuration for the Breast Cancer Detection pipeline.
Hyperparameters aligned with: "Advancements in Breast Cancer Detection:
Deep Learning and Ultrasound Imaging Techniques"
"""

import os
import torch

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "Dataset_BUSI_with_GT")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Class folders
CLASS_NAMES = ["benign", "malignant", "normal"]   # alphabetical = 0,1,2
NUM_CLASSES = 3

# ─── Image Sizes ─────────────────────────────────────────────────────────────
SEG_IMG_SIZE  = 128   # Segmentation (paper Section 3.2): resize to 128×128
CLF_IMG_SIZE  = 224   # Classification: EfficientNet standard input

# ─── Training Hyperparameters (Paper Section 3.5.2 & Table 4) ────────────────
BATCH_SIZE    = 32    # optimal per paper Table 4
LEARNING_RATE = 0.001 # paper optimal LR
NUM_EPOCHS    = 50    # paper training protocol
EARLY_STOP_PATIENCE = 10

# Dice + BCE combined loss weight factor λ (paper Eq. 1)
# L = λ·BCE + (1-λ)·Dice
DICE_BCE_LAMBDA = 0.5

# ─── Data Splits ─────────────────────────────────────────────────────────────
TRAIN_RATIO   = 0.70
VAL_RATIO     = 0.15
TEST_RATIO    = 0.15
RANDOM_SEED   = 42

# ─── Device ──────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── Model Checkpoint Paths ──────────────────────────────────────────────────
CLF_CHECKPOINT  = os.path.join(OUTPUT_DIR, "best_classifier.pth")
UNET_CHECKPOINT = os.path.join(OUTPUT_DIR, "best_unet.pth")
UNETPP_CHECKPOINT = os.path.join(OUTPUT_DIR, "best_unetpp.pth")

# ─── Normalization (ImageNet stats for classifier) ───────────────────────────
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD  = [0.229, 0.224, 0.225]

# ─── XAI ─────────────────────────────────────────────────────────────────────
XAI_N_LIME_SAMPLES   = 1000   # LIME perturbation samples
XAI_SHAP_BACKGROUND  = 50    # SHAP background set size

print(f"[config.py] Device: {DEVICE}")
print(f"[config.py] Data dir: {DATA_DIR}")
print(f"[config.py] Output dir: {OUTPUT_DIR}")
