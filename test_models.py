import sys
sys.path.insert(0, '.')

print('=== Test 4: EfficientNet-B0 Forward Pass ===')
import torch
from config import DEVICE, CLF_IMG_SIZE
from model import get_classifier
clf = get_classifier(pretrained=True).to(DEVICE)
x = torch.randn(2, 3, CLF_IMG_SIZE, CLF_IMG_SIZE).to(DEVICE)
out = clf(x)
print(f'EfficientNet output: {out.shape}  expected [2, 3]')
assert out.shape == (2, 3), 'FAIL: wrong output shape!'
trainable = sum(p.numel() for p in clf.parameters() if p.requires_grad)
total = sum(p.numel() for p in clf.parameters())
print(f'Trainable: {trainable:,} / {total:,}')
print()

print('=== Test 5: U-Net Forward Pass ===')
from config import SEG_IMG_SIZE
from unet import get_unet
unet = get_unet().to(DEVICE)
x2 = torch.randn(2, 3, SEG_IMG_SIZE, SEG_IMG_SIZE).to(DEVICE)
out2 = unet(x2)
print(f'U-Net output: {out2.shape}  expected [2, 1, 128, 128]')
assert out2.shape == (2, 1, SEG_IMG_SIZE, SEG_IMG_SIZE), 'FAIL!'
print(f'U-Net params: {sum(p.numel() for p in unet.parameters()):,}')
print()

print('=== Test 6: UNet++ Forward Pass ===')
from unet_plus_plus import get_unetpp
unetpp = get_unetpp(deep_supervision=True).to(DEVICE)

unetpp.train()
outs = unetpp(x2)
print(f'UNet++ TRAIN outputs: {len(outs)} heads, each {outs[0].shape}')
assert len(outs) == 4, 'FAIL: expected 4 deep supervision outputs!'

unetpp.eval()
with torch.no_grad():
    out3 = unetpp(x2)
print(f'UNet++ EVAL output: {out3.shape}  expected [2, 1, 128, 128]')
assert out3.shape == (2, 1, SEG_IMG_SIZE, SEG_IMG_SIZE), 'FAIL!'
print(f'UNet++ params: {sum(p.numel() for p in unetpp.parameters()):,}')
print()

print('ALL MODEL FORWARD PASS TESTS PASSED')
