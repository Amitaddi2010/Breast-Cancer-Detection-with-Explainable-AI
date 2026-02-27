"""
xai.py — Grad-CAM, Grad-CAM++, and Score-CAM heatmaps for classification XAI.
Uses the pytorch-grad-cam library (already installed).
"""

import os
import cv2
import numpy as np
import torch
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, ScoreCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from config import DEVICE, CLF_CHECKPOINT, CLF_IMG_SIZE, CLASS_NAMES, NORMALIZE_MEAN, NORMALIZE_STD
from model import get_classifier


# ── Image preprocessing ───────────────────────────────────────────────────────
_transform = A.Compose([
    A.Resize(CLF_IMG_SIZE, CLF_IMG_SIZE),
    A.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
    ToTensorV2(),
])

def preprocess_image(image_path: str, device=DEVICE):
    """Return (tensor [1,3,H,W], rgb_float [H,W,3]) tuple."""
    img_rgb = np.array(Image.open(image_path).convert("RGB"))
    img_resized = cv2.resize(img_rgb, (CLF_IMG_SIZE, CLF_IMG_SIZE))
    rgb_float   = img_resized.astype(np.float32) / 255.0

    tensor = _transform(image=img_rgb)["image"].unsqueeze(0).to(device)
    return tensor, rgb_float


# ── Load model ────────────────────────────────────────────────────────────────
_cached_model = None

def load_model(device=DEVICE) -> torch.nn.Module:
    global _cached_model
    if _cached_model is None:
        model = get_classifier(pretrained=False).to(device)
        if os.path.exists(CLF_CHECKPOINT):
            model.load_state_dict(torch.load(CLF_CHECKPOINT, map_location=device))
        model.eval()
        _cached_model = model
    return _cached_model


# ── Predict class ─────────────────────────────────────────────────────────────
def predict(image_path: str, device=DEVICE):
    """Returns (class_name, confidence_pct, probs_array)."""
    model = load_model(device)
    tensor, _ = preprocess_image(image_path, device)
    with torch.no_grad():
        logits = model(tensor)
    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    pred_idx  = int(probs.argmax())
    return CLASS_NAMES[pred_idx], float(probs[pred_idx]) * 100, probs


# ── CAM methods ───────────────────────────────────────────────────────────────
CAM_METHODS = {
    "Grad-CAM":     GradCAM,
    "Grad-CAM++":   GradCAMPlusPlus,
    "Score-CAM":    ScoreCAM,
}


def generate_all_cams(image_path: str, device=DEVICE) -> dict:
    """
    Generate heatmap overlays for all three CAM methods.
    Returns dict: {method_name: overlay_RGB_uint8}
    """
    model = load_model(device)
    tensor, rgb_float = preprocess_image(image_path, device)

    # Target: predicted class
    with torch.no_grad():
        logits = model(tensor)
    pred_idx = int(logits.argmax(dim=1).item())
    targets  = [ClassifierOutputTarget(pred_idx)]

    # Target layer: last feature block before avgpool
    target_layer = [model.features[-1]]

    results = {}
    for method_name, CamClass in CAM_METHODS.items():
        try:
            with CamClass(model=model, target_layers=target_layer) as cam:
                grayscale_cam = cam(input_tensor=tensor, targets=targets)
                grayscale_cam = grayscale_cam[0, :]          # [H, W]
                overlay = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)
            results[method_name] = overlay  # HxWx3 uint8
        except Exception as e:
            print(f"[xai.py] Warning: {method_name} failed: {e}")
            results[method_name] = (rgb_float * 255).astype(np.uint8)

    return results


def cam_to_base64(overlay_rgb: np.ndarray) -> str:
    """Convert HxWx3 uint8 array to base64 PNG string for Flask."""
    import base64, io
    img = Image.fromarray(overlay_rgb)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('utf-8')


if __name__ == "__main__":
    import glob
    test_img = glob.glob("Dataset_BUSI_with_GT/benign/*.png")[0]
    # Filter out masks
    while "_mask" in test_img:
        test_img = glob.glob("Dataset_BUSI_with_GT/benign/*.png")[1]
    print(f"Testing XAI on: {test_img}")
    results = generate_all_cams(test_img)
    for name, overlay in results.items():
        out_path = os.path.join("outputs", f"xai_{name.replace('-','').replace('+','plus').replace(' ','_')}.png")
        os.makedirs("outputs", exist_ok=True)
        Image.fromarray(overlay).save(out_path)
        print(f"Saved {name} -> {out_path}")
