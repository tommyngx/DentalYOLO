"""DentalYOLO26 SSL modules.

This file intentionally stays separate from ``dental_modules.py`` so the
reconstruction pretraining path is easy to debug and can be removed without
touching the detection model.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log2

from ultralytics.nn.modules.conv import Conv


class DentalReconstructionDecoder(nn.Module):
    """Lightweight multi-scale decoder for masked image reconstruction.

    Args:
        channels (list[int] | tuple[int, ...] | int): Input feature channels.
        out_ch (int): Reconstructed image channels, usually 1 for OPG or 3 for RGB-compatible training.
        hidden (int): Internal decoder width.
        p2_stride (int): Stride of the highest-resolution input feature. YOLO26-P2 layer 19 uses stride 4.
    """

    def __init__(self, channels, out_ch=1, hidden=128, p2_stride=4):
        super().__init__()
        if isinstance(channels, int):
            channels = [channels]
        self.channels = list(channels)
        self.out_ch = out_ch
        self.p2_stride = p2_stride

        self.proj = nn.ModuleList(Conv(c, hidden, 1, 1) for c in self.channels)
        self.fuse = nn.Sequential(
            Conv(hidden * len(self.channels), hidden, 3, 1),
            Conv(hidden, hidden, 3, 1),
        )
        stages = []
        c = hidden
        for _ in range(int(log2(p2_stride))):
            c2 = max(c // 2, 32)
            stages.extend([nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False), Conv(c, c2, 3, 1)])
            c = c2
        self.up = nn.Sequential(*stages)
        self.out = nn.Conv2d(c, out_ch, 1)

    def forward(self, x):
        feats = x if isinstance(x, (list, tuple)) else [x]
        target_hw = feats[0].shape[-2:]
        ys = []
        for feat, proj in zip(feats, self.proj):
            y = proj(feat)
            if y.shape[-2:] != target_hw:
                y = F.interpolate(y, size=target_hw, mode="bilinear", align_corners=False)
            ys.append(y)
        return torch.sigmoid(self.out(self.up(self.fuse(torch.cat(ys, dim=1)))))


def random_patch_mask(x, mask_ratio=0.5, patch_size=32, mask_value=0.0):
    """Apply SimMIM-style random patch masking.

    Returns:
        masked_x (Tensor): Image with selected patches replaced by ``mask_value``.
        mask (Tensor): Binary mask with 1 on masked pixels, shape ``B,1,H,W``.
    """
    b, _, h, w = x.shape
    gh = max(h // patch_size, 1)
    gw = max(w // patch_size, 1)
    lowres = (torch.rand(b, 1, gh, gw, device=x.device) < mask_ratio).float()
    mask = F.interpolate(lowres, size=(h, w), mode="nearest")
    masked_x = x * (1.0 - mask) + mask_value * mask
    return masked_x, mask


def ssim_loss(pred, target, window_size=11, eps=1e-6, reduction="mean"):
    """Fast differentiable SSIM loss using average pooling."""
    pad = window_size // 2
    mu_x = F.avg_pool2d(pred, window_size, stride=1, padding=pad)
    mu_y = F.avg_pool2d(target, window_size, stride=1, padding=pad)
    sigma_x = F.avg_pool2d(pred * pred, window_size, stride=1, padding=pad) - mu_x * mu_x
    sigma_y = F.avg_pool2d(target * target, window_size, stride=1, padding=pad) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(pred * target, window_size, stride=1, padding=pad) - mu_x * mu_y

    c1 = 0.01**2
    c2 = 0.03**2
    ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x.square() + mu_y.square() + c1) * (sigma_x + sigma_y + c2) + eps
    )
    loss = (1.0 - ssim).clamp(0.0, 2.0)
    if reduction == "none":
        return loss
    return loss.mean() if reduction == "mean" else loss.sum()


class MaskedReconstructionLoss(nn.Module):
    """Hybrid L1 + SSIM reconstruction loss for masked X-ray SSL."""

    def __init__(self, l1_weight=0.8, ssim_weight=0.2, masked_only=True, eps=1e-6):
        super().__init__()
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.masked_only = masked_only
        self.eps = eps

    def forward(self, pred, target, mask=None):
        if pred.shape[-2:] != target.shape[-2:]:
            target = F.interpolate(target, size=pred.shape[-2:], mode="bilinear", align_corners=False)
            if mask is not None:
                mask = F.interpolate(mask, size=pred.shape[-2:], mode="nearest")
        if pred.shape[1] != target.shape[1]:
            target = target.mean(1, keepdim=True) if pred.shape[1] == 1 else target.repeat(1, pred.shape[1], 1, 1)

        l1_map = (pred - target).abs()
        ssim_map = ssim_loss(pred, target, reduction="none")

        if self.masked_only and mask is not None:
            denom = mask.sum().clamp_min(self.eps)
            l1 = (l1_map * mask).sum() / denom
            ssim = (ssim_map * mask).sum() / denom
        else:
            l1 = l1_map.mean()
            ssim = ssim_map.mean()
        return self.l1_weight * l1 + self.ssim_weight * ssim, torch.stack((l1.detach(), ssim.detach()))
