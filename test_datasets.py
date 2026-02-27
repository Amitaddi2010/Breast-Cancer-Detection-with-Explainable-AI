import sys
sys.path.insert(0, '.')

print('=== Test 1: Config ===')
import config
import os
print(f'Device: {config.DEVICE}')
print(f'Data dir exists: {os.path.isdir(config.DATA_DIR)}')
print(f'Classes: {config.CLASS_NAMES}')
print()

print('=== Test 2: Classification dataset ===')
from dataset import get_dataloaders, get_class_weights
tl, vl, tel = get_dataloaders()
imgs, labels = next(iter(tl))
print(f'Batch shape: {imgs.shape}, Labels sample: {labels[:4].tolist()}')
weights = get_class_weights()
print(f'Class weights: {weights}')
print()

print('=== Test 3: Segmentation dataset ===')
from seg_dataset import get_seg_dataloaders
stl, svl, stel = get_seg_dataloaders()
imgs2, masks = next(iter(stl))
print(f'Seg image shape: {imgs2.shape}, Mask shape: {masks.shape}')
print(f'Mask unique values: {masks.unique().tolist()}')
print()
print('ALL DATASET TESTS PASSED')
