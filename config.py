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
SEG_IMG_SIZE  = 512   # Segmentation: resize to 512x512 for full boundary preservation
CLF_IMG_SIZE  = 224   # Classification: EfficientNet standard input
SEG_ENCODER_NAME = 'efficientnet_b3'  # High capacity pretrained backbone

# ─── Training Hyperparameters (Paper Section 3.5.2 & Table 4) ────────────────
BATCH_SIZE    = 8     # reduced for 384×384 resolution memory
LEARNING_RATE = 3e-4  # tuned for AdamW + cosine warmup
NUM_EPOCHS    = 100   # extended for cosine annealing convergence
EARLY_STOP_PATIENCE = 20

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
REPSEGNET_CHECKPOINT = os.path.join(OUTPUT_DIR, "best_repsegnet.pth")
REPSEGNET_V2_CHECKPOINT = os.path.join(OUTPUT_DIR, "best_repsegnet_v2.pth")

# ─── Normalization (ImageNet stats for classifier) ───────────────────────────
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD  = [0.229, 0.224, 0.225]

# ─── XAI ─────────────────────────────────────────────────────────────────────
XAI_N_LIME_SAMPLES   = 1000   # LIME perturbation samples
XAI_SHAP_BACKGROUND  = 50    # SHAP background set size

print(f"[config.py] Device: {DEVICE}")
print(f"[config.py] Data dir: {DATA_DIR}")
print(f"[config.py] Output dir: {OUTPUT_DIR}")
