"""
repsegnet_v2.py — RepSegNet V2: Pretrained Encoder + SCSE Attention + Deep Supervision

Key improvements over V1:
  1. Pretrained EfficientNet-B3 encoder (ImageNet features → far better on small datasets)
  2. SCSE (Spatial + Channel Squeeze-Excitation) on skip connections
  3. Deep supervision with auxiliary heads at each decoder level
  4. 5 encoder levels (vs 4 in V1) for finer multi-scale features
  5. Configurable dropout (default 0.1 vs 0.2 in V1)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


# ── SCSE: Concurrent Spatial and Channel Squeeze & Excitation ─────────────────
class ChannelSE(nn.Module):
    """Channel Squeeze-and-Excitation (cSE)."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


class SpatialSE(nn.Module):
    """Spatial Squeeze-and-Excitation (sSE)."""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        w = self.sigmoid(self.conv(x))
        return x * w


class SCSE(nn.Module):
    """Concurrent Spatial + Channel SE (scSE)."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.cse = ChannelSE(channels, reduction)
        self.sse = SpatialSE(channels)

    def forward(self, x):
        return self.cse(x) + self.sse(x)


# ── Decoder Block ─────────────────────────────────────────────────────────────
class DecoderBlock(nn.Module):
    """
    Decoder block: Upsample → Concat skip → Conv-BN-ReLU × 2 → (Optional SCSE) → Dropout
    """
    def __init__(self, in_channels, skip_channels, out_channels, dropout=0.1, use_scse=True):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        self.scse_skip = SCSE(skip_channels) if use_scse else nn.Identity()

        cat_channels = in_channels + skip_channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(cat_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.scse_out = SCSE(out_channels) if use_scse else nn.Identity()
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x, skip):
        x = self.upsample(x)
        # Handle size mismatch
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)

        skip = self.scse_skip(skip)
        x = torch.cat([skip, x], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.scse_out(x)
        x = self.dropout(x)
        return x


# ── RepSegNet V2 ──────────────────────────────────────────────────────────────
class RepSegNetV2(nn.Module):
    """
    RepSegNet V2 with pretrained EfficientNet encoder.

    Architecture:
      Encoder: EfficientNet-B3 (pretrained on ImageNet)
      Skip:    SCSE attention on each skip connection
      Decoder: Conv-BN-ReLU × 2 + SCSE + Dropout
      Output:  Deep supervision (auxiliary heads at each decoder level)

    Args:
        encoder_name: timm model name (default: 'efficientnet_b3')
        in_channels: input channels (default: 3)
        out_channels: output channels (default: 1 for binary segmentation)
        decoder_channels: decoder feature sizes at each level
        dropout: dropout rate in decoder blocks
        deep_supervision: if True, return list of outputs during training
    """
    def __init__(self, encoder_name='efficientnet_b3', in_channels=3,
                 out_channels=1, decoder_channels=(256, 128, 64, 32, 16),
                 dropout=0.1, deep_supervision=True, use_scse=True):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.use_scse = use_scse

        # ── Pretrained Encoder ──
        self.encoder = timm.create_model(
            encoder_name, pretrained=True, features_only=True,
            in_chans=in_channels
        )

        # Get encoder channel sizes from the model
        # EfficientNet-B3: [24, 32, 48, 136, 384] for 5 feature levels
        encoder_channels = self.encoder.feature_info.channels()
        self.num_levels = len(encoder_channels)

        # ── Bottleneck ──
        self.bottleneck = nn.Sequential(
            nn.Conv2d(encoder_channels[-1], decoder_channels[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_channels[0], decoder_channels[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels[0]),
            nn.ReLU(inplace=True),
        )

        # ── Decoder ──
        self.decoder_blocks = nn.ModuleList()
        prev_channels = decoder_channels[0]

        for i in range(self.num_levels):
            skip_ch = encoder_channels[-(i + 1)]  # reverse order
            out_ch = decoder_channels[min(i, len(decoder_channels) - 1)]
            # For first decoder block, input comes from bottleneck
            # For subsequent blocks, input comes from previous decoder
            if i == 0:
                in_ch = decoder_channels[0]
            else:
                in_ch = prev_channels
            self.decoder_blocks.append(
                DecoderBlock(in_ch, skip_ch, out_ch, dropout=dropout, use_scse=use_scse)
            )
            prev_channels = out_ch

        # ── Final Segmentation Head ──
        self.final_conv = nn.Conv2d(prev_channels, out_channels, kernel_size=1)

        # ── Deep Supervision Heads ──
        if deep_supervision:
            self.aux_heads = nn.ModuleList()
            for i in range(self.num_levels - 1):  # all but last level
                ch = decoder_channels[min(i, len(decoder_channels) - 1)]
                self.aux_heads.append(nn.Conv2d(ch, out_channels, kernel_size=1))

        self._init_decoder_weights()

    def _init_decoder_weights(self):
        """Initialize decoder weights with Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) and m not in self.encoder.modules():
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d) and m not in self.encoder.modules():
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        input_size = x.shape[2:]

        # ── Encoder ──
        features = self.encoder(x)  # list of 5 feature maps

        # ── Bottleneck ──
        x = self.bottleneck(features[-1])

        # ── Decoder ──
        aux_outputs = []
        skips = features[::-1]  # reverse: deepest first

        for i, (decoder_block, skip) in enumerate(zip(self.decoder_blocks, skips)):
            x = decoder_block(x, skip)

            # Collect auxiliary outputs for deep supervision
            if self.deep_supervision and self.training and i < len(self.aux_heads):
                aux = self.aux_heads[i](x)
                aux = F.interpolate(aux, size=input_size, mode='bilinear', align_corners=True)
                aux_outputs.append(torch.sigmoid(aux))

        # ── Final output ──
        out = torch.sigmoid(self.final_conv(x))

        # Ensure output matches input spatial size
        if out.shape[2:] != input_size:
            out = F.interpolate(out, size=input_size, mode='bilinear', align_corners=True)

        if self.deep_supervision and self.training:
            return [out] + aux_outputs  # main + auxiliary outputs
        return out


def get_repsegnet_v2(encoder_name='efficientnet_b3', in_channels=3,
                     out_channels=1, deep_supervision=True, dropout=0.1,
                     use_scse=True):
    """Factory function for RepSegNet V2."""
    return RepSegNetV2(
        encoder_name=encoder_name,
        in_channels=in_channels,
        out_channels=out_channels,
        deep_supervision=deep_supervision,
        dropout=dropout,
        use_scse=use_scse
    )


if __name__ == "__main__":
    net = get_repsegnet_v2()
    print(f"RepSegNet V2 created")
    print(f"Parameters: {sum(p.numel() for p in net.parameters()):,}")

    # Test training mode (deep supervision)
    net.train()
    x = torch.randn(2, 3, 256, 256)
    outputs = net(x)
    if isinstance(outputs, list):
        print(f"Training mode: {len(outputs)} outputs")
        for i, o in enumerate(outputs):
            print(f"  Output {i}: {o.shape}")
    else:
        print(f"Training output: {outputs.shape}")

    # Test eval mode (single output)
    net.eval()
    with torch.no_grad():
        y = net(x)
    print(f"Eval output: {y.shape}")
    print(f"Output range: [{y.min():.4f}, {y.max():.4f}]")
