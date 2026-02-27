"""
xai_lime.py — LIME superpixel explanations for classification XAI.
Installs `lime` if not present, then generates annotated explanation image.
"""

import os
import sys
import subprocess
import numpy as np
from PIL import Image
import torch

# Auto-install lime if needed
try:
    from lime import lime_image
    from lime.wrappers.scikit_image import SegmentationAlgorithm
    from skimage.segmentation import mark_boundaries
except ImportError:
    print("[xai_lime] Installing lime...")
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'lime', '-q'])
    from lime import lime_image
    from lime.wrappers.scikit_image import SegmentationAlgorithm
    from skimage.segmentation import mark_boundaries

import albumentations as A
from albumentations.pytorch import ToTensorV2

from config import (DEVICE, CLF_CHECKPOINT, CLF_IMG_SIZE,
                    CLASS_NAMES, NORMALIZE_MEAN, NORMALIZE_STD,
                    XAI_N_LIME_SAMPLES)
from model import get_classifier

# ── Preprocessing ─────────────────────────────────────────────────────────────
_transform = A.Compose([
    A.Resize(CLF_IMG_SIZE, CLF_IMG_SIZE),
    A.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
    ToTensorV2(),
])

_model = None

def _load_model():
    global _model
    if _model is None:
        m = get_classifier(pretrained=False).to(DEVICE)
        if os.path.exists(CLF_CHECKPOINT):
            m.load_state_dict(torch.load(CLF_CHECKPOINT, map_location=DEVICE))
        m.eval()
        _model = m
    return _model


def _predict_fn(images: np.ndarray) -> np.ndarray:
    """
    LIME calls this with N×H×W×3 uint8 arrays.
    Returns N×num_classes probability arrays.
    """
    model = _load_model()
    probs_list = []
    for img in images:
        tensor = _transform(image=img)["image"].unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = model(tensor)
            p = torch.softmax(logits, dim=1)[0].cpu().numpy()
        probs_list.append(p)
    return np.array(probs_list)


def generate_lime(image_path: str, n_samples: int = XAI_N_LIME_SAMPLES) -> np.ndarray:
    """
    Generate LIME explanation.
    Returns: HxWx3 uint8 annotated image with top superpixel boundaries.
    """
    img_arr = np.array(Image.open(image_path).convert("RGB").resize(
        (CLF_IMG_SIZE, CLF_IMG_SIZE)))

    explainer   = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        img_arr,
        _predict_fn,
        top_labels=1,
        hide_color=0,
        num_samples=n_samples,
        batch_size=32,
    )

    top_label = explanation.top_labels[0]
    temp, mask = explanation.get_image_and_mask(
        top_label,
        positive_only=True,
        num_features=10,
        hide_rest=False,
    )

    # Draw superpixel boundaries on original image
    overlay = mark_boundaries(temp / 255.0, mask, color=(1, 0.5, 0))
    overlay = np.clip(overlay * 255, 0, 255).astype(np.uint8)
    return overlay


def lime_to_base64(overlay_rgb: np.ndarray) -> str:
    import base64, io
    img = Image.fromarray(overlay_rgb)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('utf-8')


if __name__ == "__main__":
    import glob
    imgs = [f for f in glob.glob("Dataset_BUSI_with_GT/benign/*.png") if "_mask" not in f]
    print(f"Testing LIME on: {imgs[0]}")
    result = generate_lime(imgs[0], n_samples=200)
    out_path = os.path.join("outputs", "xai_lime.png")
    os.makedirs("outputs", exist_ok=True)
    Image.fromarray(result).save(out_path)
    print(f"LIME saved → {out_path}")
