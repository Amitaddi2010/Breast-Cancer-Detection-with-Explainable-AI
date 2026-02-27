"""
xai_shap.py — SHAP GradientExplainer pixel-level attributions for classification XAI.
Auto-installs `shap` if not present.
"""

import os
import sys
import subprocess
import numpy as np
from PIL import Image
import torch

try:
    import shap
except ImportError:
    print("[xai_shap] Installing shap...")
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'shap', '-q'])
    import shap

import albumentations as A
from albumentations.pytorch import ToTensorV2

from config import (
    DEVICE, CLF_CHECKPOINT, CLF_IMG_SIZE,
    NORMALIZE_MEAN, NORMALIZE_STD, XAI_SHAP_BACKGROUND
)
from model import get_classifier
from dataset import get_dataloaders

_transform = A.Compose([
    A.Resize(CLF_IMG_SIZE, CLF_IMG_SIZE),
    A.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
    ToTensorV2(),
])

_model      = None
_background = None


def _load_model():
    global _model
    if _model is None:
        m = get_classifier(pretrained=False).to(DEVICE)
        if os.path.exists(CLF_CHECKPOINT):
            m.load_state_dict(torch.load(CLF_CHECKPOINT, map_location=DEVICE))
        m.eval()
        _model = m
    return _model


def _get_background(n: int = XAI_SHAP_BACKGROUND):
    """Sample n random images from training set as SHAP background."""
    global _background
    if _background is None:
        train_l, _, _ = get_dataloaders()
        imgs_list = []
        for imgs, _ in train_l:
            imgs_list.append(imgs)
            if sum(x.size(0) for x in imgs_list) >= n:
                break
        bg = torch.cat(imgs_list, dim=0)[:n].to(DEVICE)
        _background = bg
    return _background


def generate_shap(image_path: str) -> np.ndarray:
    """
    Generate SHAP pixel attribution map.
    Returns: HxWx3 uint8 image with SHAP heatmap overlaid.
    """
    model = _load_model()
    img_arr = np.array(Image.open(image_path).convert("RGB"))
    tensor  = _transform(image=img_arr)["image"].unsqueeze(0).to(DEVICE)
    background = _get_background()

    explainer = shap.GradientExplainer(model, background)
    shap_values = explainer.shap_values(tensor)  # list of arrays, one per class

    # Select channel for predicted class
    with torch.no_grad():
        pred_idx = int(model(tensor).argmax(dim=1).item())

    sv = shap_values[pred_idx][0]           # [3, H, W]
    sv_mean = np.abs(sv).mean(axis=0)       # [H, W] mean across channels

    # Normalize to [0, 1] and colorize
    sv_norm = (sv_mean - sv_mean.min()) / (sv_mean.max() - sv_mean.min() + 1e-8)

    # Create colormap overlay on original image
    import cv2
    heatmap = cv2.applyColorMap(
        (sv_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    img_resized = np.array(Image.fromarray(img_arr).resize((CLF_IMG_SIZE, CLF_IMG_SIZE)))
    overlay     = cv2.addWeighted(img_resized, 0.5, heatmap_rgb, 0.5, 0)
    return overlay.astype(np.uint8)


def shap_to_base64(overlay_rgb: np.ndarray) -> str:
    import base64, io
    img = Image.fromarray(overlay_rgb)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('utf-8')


if __name__ == "__main__":
    import glob
    imgs = [f for f in glob.glob("Dataset_BUSI_with_GT/benign/*.png") if "_mask" not in f]
    print(f"Testing SHAP on: {imgs[0]}")
    result = generate_shap(imgs[0])
    out_path = os.path.join("outputs", "xai_shap.png")
    os.makedirs("outputs", exist_ok=True)
    Image.fromarray(result).save(out_path)
    print(f"SHAP saved → {out_path}")
