"""
app.py — Flask web application for breast cancer detection.
Runs classification (EfficientNet) + segmentation (UNet++) + XAI (Grad-CAM/LIME/SHAP).
"""

import os
import io
import base64
import numpy as np
import torch
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from PIL import Image

from config import DEVICE, CLF_CHECKPOINT, UNETPP_CHECKPOINT, CLASS_NAMES, SEG_IMG_SIZE
from model import get_classifier
from unet_plus_plus import get_unetpp
from xai import predict as clf_predict, generate_all_cams, cam_to_base64
import cv2

app = Flask(__name__)
CORS(app)

# ── Load models at startup ────────────────────────────────────────────────────
print("[app.py] Loading models...")

clf_model = get_classifier(pretrained=False).to(DEVICE)
if os.path.exists(CLF_CHECKPOINT):
    clf_model.load_state_dict(torch.load(CLF_CHECKPOINT, map_location=DEVICE))
    print(f"[app.py] Classifier loaded from {CLF_CHECKPOINT}")
else:
    print(f"[app.py] WARNING: No classifier checkpoint at {CLF_CHECKPOINT}")
clf_model.eval()

seg_model = get_unetpp(deep_supervision=True).to(DEVICE)
if os.path.exists(UNETPP_CHECKPOINT):
    seg_model.load_state_dict(torch.load(UNETPP_CHECKPOINT, map_location=DEVICE))
    print(f"[app.py] UNet++ loaded from {UNETPP_CHECKPOINT}")
else:
    print(f"[app.py] WARNING: No segmentation checkpoint at {UNETPP_CHECKPOINT}")
seg_model.eval()

print("[app.py] Models ready.\n")


# ── Helpers ──────────────────────────────────────────────────────────────────
def array_to_base64(img_rgb: np.ndarray) -> str:
    img = Image.fromarray(img_rgb.astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def run_segmentation(image: Image.Image) -> str:
    """Run UNet++, return base64 overlay PNG."""
    img_arr = np.array(image.convert("RGB").resize((SEG_IMG_SIZE, SEG_IMG_SIZE)))
    img_norm = img_arr.astype(np.float32) / 255.0
    tensor = torch.tensor(img_norm.transpose(2, 0, 1)).unsqueeze(0).float().to(DEVICE)

    with torch.no_grad():
        pred = seg_model(tensor)[0, 0].cpu().numpy()  # [H, W]

    pred_bin = (pred > 0.5).astype(np.uint8) * 255
    pred_rgb = cv2.cvtColor(pred_bin, cv2.COLOR_GRAY2RGB)

    # Green overlay on original image
    overlay = img_arr.copy().astype(float)
    green   = np.zeros_like(img_arr)
    green[:, :, 1] = pred_bin
    overlay = np.clip(overlay * 0.6 + green * 0.4, 0, 255).astype(np.uint8)

    # Return both mask and overlay as combined side-by-side
    combined = np.concatenate([img_arr, pred_rgb, overlay], axis=1)
    return array_to_base64(combined)


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file  = request.files['image']
    image = Image.open(file.stream).convert("RGB")

    # Save temp file for XAI (needs file path)
    tmp_path = os.path.join("outputs", "_tmp_upload.png")
    os.makedirs("outputs", exist_ok=True)
    image.save(tmp_path)

    # ── Classification ──────────────────────────────────────────────────────
    try:
        class_name, confidence, probs = clf_predict(tmp_path, DEVICE)
        prob_dict = {CLASS_NAMES[i]: float(probs[i] * 100) for i in range(len(CLASS_NAMES))}
    except Exception as e:
        return jsonify({'error': f'Classification failed: {str(e)}'}), 500

    # ── Segmentation ────────────────────────────────────────────────────────
    seg_b64 = None
    if os.path.exists(UNETPP_CHECKPOINT):
        try:
            seg_b64 = run_segmentation(image)
        except Exception as e:
            print(f"[app.py] Segmentation error: {e}")

    # ── XAI: Grad-CAM, Grad-CAM++, Score-CAM ───────────────────────────────
    cam_results = {}
    try:
        cams = generate_all_cams(tmp_path, DEVICE)
        cam_results = {name: cam_to_base64(overlay) for name, overlay in cams.items()}
    except Exception as e:
        print(f"[app.py] CAM error: {e}")

    # ── LIME ─────────────────────────────────────────────────────────────────
    lime_b64 = None
    try:
        from xai_lime import generate_lime, lime_to_base64
        lime_overlay = generate_lime(tmp_path, n_samples=500)  # faster for web
        lime_b64 = lime_to_base64(lime_overlay)
    except Exception as e:
        print(f"[app.py] LIME error: {e}")

    # ── SHAP ──────────────────────────────────────────────────────────────────
    shap_b64 = None
    try:
        from xai_shap import generate_shap, shap_to_base64
        shap_overlay = generate_shap(tmp_path)
        shap_b64 = shap_to_base64(shap_overlay)
    except Exception as e:
        print(f"[app.py] SHAP error: {e}")

    return jsonify({
        'class':       class_name,
        'confidence':  round(confidence, 2),
        'probabilities': prob_dict,
        'segmentation':  seg_b64,
        'gradcam':       cam_results.get('Grad-CAM'),
        'gradcampp':     cam_results.get('Grad-CAM++'),
        'scorecam':      cam_results.get('Score-CAM'),
        'lime':          lime_b64,
        'shap':          shap_b64,
    })


@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'device': str(DEVICE)})


if __name__ == '__main__':
    print("\n[app.py] Starting Flask server on http://localhost:5000\n")
    app.run(debug=False, host='0.0.0.0', port=5000)
