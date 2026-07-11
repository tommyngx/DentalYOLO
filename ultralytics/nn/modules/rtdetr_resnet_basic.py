# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""ResNet BasicBlock modules for RT-DETR with ResNet18/34 backbones.

ResNet18/34 use BasicBlock (expansion=1) with 2 conv layers, unlike ResNet50/101 which use
BottleNeck (expansion=4) with 3 conv layers. This module provides BasicBlock-based layers
compatible with the ultralytics YAML config system.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv


class ResNetBasicBlock(nn.Module):
    """ResNet BasicBlock with 2 convolution layers (for ResNet18/34).

    Unlike ResNetBlock (BottleNeck, expansion=4), this block uses expansion=1 with
    two 3x3 convolutions, matching the standard BasicBlock architecture.

    Attributes:
        cv1 (Conv): First 3x3 convolution (with stride).
        cv2 (Conv): Second 3x3 convolution.
        shortcut (nn.Module): Shortcut connection (identity or 1x1 conv).
    """

    def __init__(self, c1: int, c2: int, s: int = 1):
        """Initialize ResNet BasicBlock.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            s (int): Stride for the first convolution.
        """
        super().__init__()
        self.cv1 = Conv(c1, c2, k=3, s=s, p=1, act=True)
        self.cv2 = Conv(c2, c2, k=3, s=1, p=1, act=False)
        self.shortcut = nn.Sequential(Conv(c1, c2, k=1, s=s, act=False)) if s != 1 or c1 != c2 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the ResNet BasicBlock."""
        return F.relu(self.cv2(self.cv1(x)) + self.shortcut(x))


class ResNetBasicLayer(nn.Module):
    """ResNet layer with multiple BasicBlocks (for ResNet18/34).

    This layer stacks multiple ResNetBasicBlock modules. The first block uses the given
    stride for downsampling, while subsequent blocks use stride=1.

    Attributes:
        layer (nn.Sequential): Sequential container of BasicBlocks.
    """

    def __init__(self, c1: int, c2: int, s: int = 1, n: int = 1):
        """Initialize ResNet BasicBlock layer.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            s (int): Stride for the first block (downsampling).
            n (int): Number of BasicBlocks in this layer.
        """
        super().__init__()
        blocks = [ResNetBasicBlock(c1, c2, s)]
        blocks.extend([ResNetBasicBlock(c2, c2, 1) for _ in range(n - 1)])
        self.layer = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the ResNet BasicBlock layer."""
        return self.layer(x)
