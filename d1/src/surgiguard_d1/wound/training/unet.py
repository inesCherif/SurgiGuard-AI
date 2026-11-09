"""U-Net with a ResNet18 encoder (ImageNet-pretrained) for binary wound segmentation.

Encoder: torchvision ResNet18 feature maps at strides 2, 4, 8, 16, 32 (64, 64, 128, 256, 512 channels).
Decoder: at each level, upsample x2, concatenate the encoder skip, two 3x3 conv-BN-ReLU (the classic U-Net
"skip connection" that brings back fine detail - essential for thin incisions). Output: 1-channel logits at the
input resolution. About 14 M parameters; fits a 4 GB GPU at 320 x 320, batch 8, mixed precision.
"""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


class _Up(nn.Module):
    def __init__(self, cin: int, cskip: int, cout: int):
        super().__init__()
        self.conv = _block(cin + cskip, cout)

    def forward(self, x, skip=None):
        x = nn.functional.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class ResNetUNet(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        enc = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.stem = nn.Sequential(enc.conv1, enc.bn1, enc.relu)          # /2,  64
        self.pool = enc.maxpool
        self.l1, self.l2, self.l3, self.l4 = enc.layer1, enc.layer2, enc.layer3, enc.layer4   # /4 /8 /16 /32
        self.u4 = _Up(512, 256, 256)
        self.u3 = _Up(256, 128, 128)
        self.u2 = _Up(128, 64, 64)
        self.u1 = _Up(64, 64, 64)
        self.u0 = _Up(64, 0, 32)
        self.head = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        s0 = self.stem(x)                  # /2
        s1 = self.l1(self.pool(s0))        # /4
        s2 = self.l2(s1)                   # /8
        s3 = self.l3(s2)                   # /16
        s4 = self.l4(s3)                   # /32
        d = self.u4(s4, s3)
        d = self.u3(d, s2)
        d = self.u2(d, s1)
        d = self.u1(d, s0)
        d = self.u0(d)
        return self.head(d)


def dice_bce_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    """BCE (pixel accuracy) + soft Dice (overlap). Dice keeps a 1 %-of-the-image incision from being ignored."""
    bce = nn.functional.binary_cross_entropy_with_logits(logits, target)
    p = torch.sigmoid(logits)
    inter = (p * target).sum(dim=(1, 2, 3))
    dice = (2 * inter + eps) / (p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + eps)
    return bce + (1 - dice).mean()
