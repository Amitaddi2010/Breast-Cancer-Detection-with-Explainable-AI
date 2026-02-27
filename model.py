"""
model.py — EfficientNet-B0 based 3-class breast cancer classifier.
Fine-tunes a pretrained EfficientNet-B0 for Normal / Benign / Malignant.
"""

import torch
import torch.nn as nn
from torchvision import models
from config import NUM_CLASSES


class BreastCancerClassifier(nn.Module):
    """
    EfficientNet-B0 (ImageNet pretrained) with custom classification head.
    The last 2 feature blocks and classifier are unfrozen for fine-tuning.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, pretrained: bool = True):
        super().__init__()

        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.efficientnet_b0(weights=weights)

        # Freeze all feature layers
        for param in backbone.parameters():
            param.requires_grad = False

        # Unfreeze last 2 feature blocks (blocks 6 & 7) for fine-tuning
        for block in [backbone.features[6], backbone.features[7],
                      backbone.features[8]]:
            for param in block.parameters():
                param.requires_grad = True

        # Feature extractor (used for XAI hook targets)
        self.features   = backbone.features
        self.avgpool    = backbone.avgpool

        # Custom classification head: 1280 → num_classes
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(1280, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, num_classes),
        )
        # Unfreeze classifier head
        for param in self.classifier.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)        # → [B, 1280, 7, 7]
        x = self.avgpool(x)         # → [B, 1280, 1, 1]
        x = torch.flatten(x, 1)    # → [B, 1280]
        x = self.classifier(x)     # → [B, num_classes]
        return x

    def get_cam_target_layer(self):
        """Return the last conv layer for Grad-CAM."""
        return self.features[-1]


def get_classifier(pretrained: bool = True) -> BreastCancerClassifier:
    return BreastCancerClassifier(pretrained=pretrained)


if __name__ == "__main__":
    from config import DEVICE, CLF_IMG_SIZE
    model = get_classifier().to(DEVICE)
    x = torch.randn(2, 3, CLF_IMG_SIZE, CLF_IMG_SIZE).to(DEVICE)
    out = model(x)
    print(f"Classifier output shape: {out.shape}")  # [2, 3]
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,}")
