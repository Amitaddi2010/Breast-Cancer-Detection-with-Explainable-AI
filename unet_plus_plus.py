"""
unet_plus_plus.py — UNet++ with dense skip connections and deep supervision.
Architecture per paper Section 3.5 and Zhou et al. (2018):
  Encoder: 5-level conv blocks with max-pooling
  Dense skip pathways: nested intermediate nodes X^{i,j}
  Decoder: transposed conv + bilinear alignment
  Deep supervision: loss from 4 intermediate outputs
  Output: 1×1 conv → sigmoid (binary segmentation)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Building block ────────────────────────────────────────────────────────────
class ConvBlock(nn.Module):
    """Two 3×3 Conv → BN → ReLU."""
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


# ── UNet++ ────────────────────────────────────────────────────────────────────
class UNetPlusPlus(nn.Module):
    """
    UNet++ with nested dense skip connections and deep supervision.

    Node naming convention: X^{i,j}
      i = encoder scale/depth (0 = shallowest / highest resolution)
      j = number of dense refinement steps on the skip path

    The key insight: self.up[i] maps features[i+1] → features[i]
    So skip node X^{i,j} receives:
      - j previous nodes at level i (each with features[i] channels)
      - 1 upsampled result from level i+1 (features[i] channels, after transposed conv)
    Total input channels = features[i] * (j + 1)

    Input:  [B, 3, 128, 128]
    Output: [B, 1, 128, 128] sigmoid
    """

    def __init__(self, in_channels: int = 3, features: list = None,
                 deep_supervision: bool = True):
        super().__init__()
        if features is None:
            features = [64, 128, 256, 512, 1024]   # 5-level encoder

        self.deep_supervision = deep_supervision
        self.pool = nn.MaxPool2d(2, 2)

        # ── Encoder nodes X^{i,0} ──────────────────────────────────────────
        self.enc = nn.ModuleList([
            ConvBlock(in_channels,  features[0]),   # X(0,0): output 64ch
            ConvBlock(features[0],  features[1]),   # X(1,0): output 128ch
            ConvBlock(features[1],  features[2]),   # X(2,0): output 256ch
            ConvBlock(features[2],  features[3]),   # X(3,0): output 512ch
            ConvBlock(features[3],  features[4]),   # X(4,0): output 1024ch (bottleneck)
        ])

        # ── Transposed conv: up[i] maps features[i+1] → features[i] ───────
        self.up = nn.ModuleList([
            nn.ConvTranspose2d(features[k + 1], features[k], kernel_size=2, stride=2)
            for k in range(4)
        ])
        # self.up[0]: 128→64, self.up[1]: 256→128, self.up[2]: 512→256, self.up[3]: 1024→512

        # ── Dense skip nodes X^{i,j}, j >= 1 ─────────────────────────────
        # X^{i,j} input: j previous nodes (features[i] each) + 1 upsampled (features[i]) = (j+1)*features[i]
        self.skip = nn.ModuleList()
        for i in range(4):     # level depth i = 0..3
            row = nn.ModuleList()
            for j in range(1, 5 - i):   # j = 1..(4-i)  →  4,3,2,1 nodes
                in_ch = features[i] * (j + 1)   # j same-level + 1 upsampled
                row.append(ConvBlock(in_ch, features[i]))
            self.skip.append(row)

        # ── Output heads (deep supervision: 4 heads, one per decoder level) ─
        if deep_supervision:
            self.out_convs = nn.ModuleList([
                nn.Conv2d(features[0], 1, kernel_size=1) for _ in range(4)
            ])
        else:
            self.out_convs = nn.ModuleList([
                nn.Conv2d(features[0], 1, kernel_size=1)
            ])

    def _up_align(self, x: torch.Tensor, up_layer: nn.Module,
                  target: torch.Tensor) -> torch.Tensor:
        """Upsample x and align to target's spatial size."""
        x = up_layer(x)
        if x.shape[2:] != target.shape[2:]:
            x = F.interpolate(x, size=target.shape[2:],
                               mode='bilinear', align_corners=True)
        return x

    def forward(self, x: torch.Tensor):
        # ── Encoder ───────────────────────────────────────────────────────
        x0_0 = self.enc[0](x)
        x1_0 = self.enc[1](self.pool(x0_0))
        x2_0 = self.enc[2](self.pool(x1_0))
        x3_0 = self.enc[3](self.pool(x2_0))
        x4_0 = self.enc[4](self.pool(x3_0))

        # ── Dense skip connections ─────────────────────────────────────────
        # Level 0 nodes (j=1..4)
        # X(0,1): x0_0(64) + up(x1_0)(64) = 128ch
        x0_1 = self.skip[0][0](torch.cat([x0_0,
                    self._up_align(x1_0, self.up[0], x0_0)], dim=1))

        # Level 1 nodes (j=1..3)
        # X(1,1): x1_0(128) + up(x2_0)(128) = 256ch
        x1_1 = self.skip[1][0](torch.cat([x1_0,
                    self._up_align(x2_0, self.up[1], x1_0)], dim=1))

        # X(0,2): x0_0(64) + x0_1(64) + up(x1_1)(64) = 192ch
        x0_2 = self.skip[0][1](torch.cat([x0_0, x0_1,
                    self._up_align(x1_1, self.up[0], x0_0)], dim=1))

        # Level 2 nodes (j=1..2)
        # X(2,1): x2_0(256) + up(x3_0)(256) = 512ch
        x2_1 = self.skip[2][0](torch.cat([x2_0,
                    self._up_align(x3_0, self.up[2], x2_0)], dim=1))

        # X(1,2): x1_0(128) + x1_1(128) + up(x2_1)(128) = 384ch
        x1_2 = self.skip[1][1](torch.cat([x1_0, x1_1,
                    self._up_align(x2_1, self.up[1], x1_0)], dim=1))

        # X(0,3): x0_0(64) + x0_1(64) + x0_2(64) + up(x1_2)(64) = 256ch
        x0_3 = self.skip[0][2](torch.cat([x0_0, x0_1, x0_2,
                    self._up_align(x1_2, self.up[0], x0_0)], dim=1))

        # Level 3 nodes (j=1)
        # X(3,1): x3_0(512) + up(x4_0)(512) = 1024ch
        x3_1 = self.skip[3][0](torch.cat([x3_0,
                    self._up_align(x4_0, self.up[3], x3_0)], dim=1))

        # X(2,2): x2_0(256) + x2_1(256) + up(x3_1)(256) = 768ch
        x2_2 = self.skip[2][1](torch.cat([x2_0, x2_1,
                    self._up_align(x3_1, self.up[2], x2_0)], dim=1))

        # X(1,3): x1_0(128) + x1_1(128) + x1_2(128) + up(x2_2)(128) = 512ch
        x1_3 = self.skip[1][2](torch.cat([x1_0, x1_1, x1_2,
                    self._up_align(x2_2, self.up[1], x1_0)], dim=1))

        # X(0,4): x0_0(64)*4 + up(x1_3)(64) = 320ch → final node
        x0_4 = self.skip[0][3](torch.cat([x0_0, x0_1, x0_2, x0_3,
                    self._up_align(x1_3, self.up[0], x0_0)], dim=1))

        # ── Output heads ──────────────────────────────────────────────────
        if self.deep_supervision:
            out1 = torch.sigmoid(self.out_convs[0](x0_1))
            out2 = torch.sigmoid(self.out_convs[1](x0_2))
            out3 = torch.sigmoid(self.out_convs[2](x0_3))
            out4 = torch.sigmoid(self.out_convs[3](x0_4))
            if self.training:
                return [out1, out2, out3, out4]   # all heads for training loss
            else:
                return out4                        # final output at eval
        else:
            return torch.sigmoid(self.out_convs[0](x0_4))


def get_unetpp(deep_supervision: bool = True) -> UNetPlusPlus:
    return UNetPlusPlus(deep_supervision=deep_supervision)


if __name__ == "__main__":
    from config import DEVICE, SEG_IMG_SIZE
    model = get_unetpp().to(DEVICE)
    x     = torch.randn(2, 3, SEG_IMG_SIZE, SEG_IMG_SIZE).to(DEVICE)

    model.train()
    outs = model(x)
    print(f"UNet++ TRAIN: {len(outs)} heads, shape: {outs[0].shape}")

    model.eval()
    with torch.no_grad():
        out = model(x)
    print(f"UNet++ EVAL output: {out.shape}")

    total = sum(p.numel() for p in model.parameters())
    print(f"UNet++ params: {total:,}")
