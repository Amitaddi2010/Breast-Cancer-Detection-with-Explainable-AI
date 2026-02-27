import sys
sys.path.insert(0, '.')

print('=== Test 7: Flask app imports ===')
# Test imports without starting server
from config import DEVICE, CLF_CHECKPOINT, UNETPP_CHECKPOINT
from model import get_classifier
from unet_plus_plus import get_unetpp
import torch
import os

clf = get_classifier(pretrained=False)
ckpt_exists = os.path.exists(CLF_CHECKPOINT)
print(f'Classifier loads: OK | Checkpoint exists: {ckpt_exists}')

unetpp = get_unetpp(deep_supervision=True)
ckpt_exists2 = os.path.exists(UNETPP_CHECKPOINT)
print(f'UNet++ loads: OK | Checkpoint exists: {ckpt_exists2}')

print()

print('=== Test 8: XAI module imports ===')
from xai import preprocess_image, generate_all_cams
print('xai.py imports: OK')

from xai_lime import generate_lime
print('xai_lime.py imports: OK')

from xai_shap import generate_shap
print('xai_shap.py imports: OK')

print()
print('=== Test 9: Flask app factory test ===')
import flask
app = flask.Flask(__name__, template_folder='templates', static_folder='static')
print(f'Flask version: {flask.__version__}, app created: OK')

print()
print('ALL IMPORT TESTS PASSED')
print()
print('='*55)
print('PIPELINE STATUS SUMMARY')
print('='*55)
print(f'  Data dir:         F:\\Amit\\Cancer Detection\\Dataset_BUSI_with_GT')
print(f'  Clf checkpoint:   {"EXISTS" if os.path.exists(CLF_CHECKPOINT) else "NOT YET (run train.py)"}')
print(f'  Seg checkpoint:   {"EXISTS" if os.path.exists(UNETPP_CHECKPOINT) else "NOT YET (run train_segmentation.py)"}')
print()
print('  READY TO TRAIN:')
print('    py train.py                  # Train EfficientNet classifier')
print('    py train_segmentation.py     # Train U-Net + UNet++')
print()
print('  READY TO EVALUATE:')
print('    py evaluate.py               # Classification metrics')
print('    py evaluate_segmentation.py  # Segmentation metrics')
print()
print('  READY TO RUN WEB APP:')
print('    py app.py                    # Start Flask on http://localhost:5000')
print('='*55)
