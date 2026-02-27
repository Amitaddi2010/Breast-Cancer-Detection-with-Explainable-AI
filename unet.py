"""
unet.py — Standard U-Net baseline for breast ultrasound segmentation.
Architecture per paper Section 3.4:
  Encoder: 3x3 conv + ReLU + 2x2 max-pool
  Decoder: 2x2 transposed conv + skip connections
  Output:  1x1 conv → sigmoid
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Building blocks ───────────────────────────────────────────────────────────
class DoubleConv(nn.Module):
    """Two consecutive 3×3 conv → BN → ReLU."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    """Encoder block: MaxPool 2×2 then DoubleConv."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.pool(x))


class Up(nn.Module):
    """Decoder block: Transposed conv upsample + skip concat + DoubleConv."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        # Handle size mismatches via center-crop
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


# ── U-Net ─────────────────────────────────────────────────────────────────────
class UNet(nn.Module):
    """
    Standard U-Net for binary segmentation.
    Input:  [B, 3, 128, 128]
    Output: [B, 1, 128, 128] (sigmoid probability map)
    """

    def __init__(self, in_channels: int = 3, features: list = None):
        super().__init__()
        if features is None:
            features = [64, 128, 256, 512]

        self.enc1  = DoubleConv(in_channels, features[0])
        self.enc2  = Down(features[0], features[1])
        self.enc3  = Down(features[1], features[2])
        self.enc4  = Down(features[2], features[3])

        self.bottleneck = Down(features[3], features[3] * 2)  # 512 → 1024

        self.dec4  = Up(features[3] * 2, features[3])
        self.dec3  = Up(features[3],     features[2])
        self.dec2  = Up(features[2],     features[1])
        self.dec1  = Up(features[1],     features[0])

        self.out   = nn.Conv2d(features[0], 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        s1 = self.enc1(x)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)

        # Bottleneck
        b  = self.bottleneck(s4)

        # Decoder
        x  = self.dec4(b,  s4)
        x  = self.dec3(x,  s3)
        x  = self.dec2(x,  s2)
        x  = self.dec1(x,  s1)

        return torch.sigmoid(self.out(x))


def get_unet() -> UNet:
    return UNet()


if __name__ == "__main__":
    from config import DEVICE, SEG_IMG_SIZE
    model = get_unet().to(DEVICE)
    x = torch.randn(2, 3, SEG_IMG_SIZE, SEG_IMG_SIZE).to(DEVICE)
    out = model(x)
    print(f"U-Net output shape: {out.shape}")   # [2, 1, 128, 128]
    total = sum(p.numel() for p in model.parameters())
    print(f"U-Net total params: {total:,}")
