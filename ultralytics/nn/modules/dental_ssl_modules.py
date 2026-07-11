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


# ---------------------------------------------------------------------------
# SSL1: Feature-level Reconstruction (Sobel edge target)
# ---------------------------------------------------------------------------


class SobelEdgeExtractor(nn.Module):
    """GPU-accelerated Sobel edge extractor (non-trainable).

    Converts a grayscale or single-channel image into a 2-channel edge map
    (horizontal + vertical gradients) that serves as the reconstruction target
    for SSL1 feature-level pretraining.
    """

    def __init__(self):
        super().__init__()
        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("kx", kx)
        self.register_buffer("ky", ky)

    @torch.no_grad()
    def forward(self, x):
        """Return 2-channel edge map from 1-channel input ``x``."""
        if x.shape[1] > 1:
            x = x.mean(dim=1, keepdim=True)
        gx = F.conv2d(x, self.kx, padding=1)
        gy = F.conv2d(x, self.ky, padding=1)
        return torch.cat([gx, gy], dim=1)  # B, 2, H, W


class FeatureReconstructionLoss(nn.Module):
    """L1 + SSIM loss computed on Sobel edge maps instead of raw pixels.

    This forces the model to learn structural edges (tooth boundaries, bone
    contours) rather than wasting capacity on X-ray film noise.

    Args:
        l1_weight: Weight for L1 component.
        ssim_weight: Weight for SSIM component.
        masked_only: If True, only compute loss on masked regions.
    """

    def __init__(self, l1_weight=0.8, ssim_weight=0.2, masked_only=True, eps=1e-6):
        super().__init__()
        self.sobel = SobelEdgeExtractor()
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.masked_only = masked_only
        self.eps = eps

    def forward(self, pred, target, mask=None):
        """Compute feature reconstruction loss.

        Args:
            pred: Decoder output, shape ``B, 2, H, W`` (predicted edge map).
            target: Original image, shape ``B, 1, H, W``.
            mask: Binary mask, shape ``B, 1, H, W``.
        """
        edge_target = self.sobel(target)
        if pred.shape[-2:] != edge_target.shape[-2:]:
            edge_target = F.interpolate(edge_target, size=pred.shape[-2:], mode="bilinear", align_corners=False)
            if mask is not None:
                mask = F.interpolate(mask, size=pred.shape[-2:], mode="nearest")
        # Expand mask to 2 channels to match edge maps
        if mask is not None and mask.shape[1] == 1:
            mask = mask.expand_as(pred)

        l1_map = (pred - edge_target).abs()
        ssim_map = ssim_loss(pred, edge_target, reduction="none")

        if self.masked_only and mask is not None:
            denom = mask.sum().clamp_min(self.eps)
            l1 = (l1_map * mask).sum() / denom
            ssim_val = (ssim_map * mask).sum() / denom
        else:
            l1 = l1_map.mean()
            ssim_val = ssim_map.mean()
        return self.l1_weight * l1 + self.ssim_weight * ssim_val, torch.stack((l1.detach(), ssim_val.detach()))


# ---------------------------------------------------------------------------
# SSL3: DINOv2 Knowledge Distillation
# ---------------------------------------------------------------------------


class DINOv2DistillHead(nn.Module):
    """Projection head for SSL3 (DINOv2 Distillation) pretraining.

    Projects multiscale features P3, P4, P5 to a unified channel dimension
    matching the DINOv2 teacher's embedding dimension (e.g., 384 for ViT-S).
    """

    def __init__(self, channels, target_dim=384):
        """Initialize projection layers for each input channel dimension."""
        super().__init__()
        if isinstance(channels, int):
            channels = [channels]
        self.channels = list(channels)
        self.target_dim = target_dim
        self.proj = nn.ModuleList(nn.Conv2d(c, target_dim, 1) for c in self.channels)

    def forward(self, x):
        """Project input list of features to the target dimension."""
        feats = x if isinstance(x, (list, tuple)) else [x]
        return [proj(f) for f, proj in zip(feats, self.proj)]


class DistillationLoss(nn.Module):
    """Cosine similarity loss between student projected features and DINOv2 teacher features."""

    def __init__(self, target_dim=384):
        """Initialize the distillation loss module."""
        super().__init__()
        self.teacher = None
        self.target_dim = target_dim

    def forward(self, pred, target, mask=None):
        """Compute distillation loss between student projected features and DINOv2.

        Args:
            pred: Projected student features (list of 3 Tensors: P3, P4, P5).
            target: Original input image, shape ``B, 1, H, W`` or ``B, 3, H, W``.
            mask: Unused here, included for interface compatibility with trainer.
        """
        if self.teacher is None:
            # Dynamically import and load DINOv2 via timm to avoid urllib HTTP/2.0 bugs
            device = target.device
            LOGGER = None
            try:
                from ultralytics.utils import LOGGER
                LOGGER.info("Loading DINOv2 (ViT-S/14) via timm...")
            except Exception:
                print("Loading DINOv2 (ViT-S/14) via timm...")
            import timm
            self.teacher = timm.create_model("vit_small_patch14_dinov2", pretrained=True).to(device).eval()
            for p in self.teacher.parameters():
                p.requires_grad = False

        b, c, h, w = target.shape
        h_d = ((h + 13) // 14) * 14
        w_d = ((w + 13) // 14) * 14
        if h != h_d or w != w_d:
            dinov_in = F.interpolate(target, size=(h_d, w_d), mode="bilinear", align_corners=False)
        else:
            dinov_in = target

        if c == 1:
            dinov_in = dinov_in.repeat(1, 3, 1, 1)

        with torch.no_grad():
            grid_h, grid_w = h_d // 14, w_d // 14
            # Extract patch tokens: skip class token (index 0) from forward_features output
            feats = self.teacher.forward_features(dinov_in)  # shape (B, 1 + N, 384)
            patch_feats = feats[:, 1:, :]  # shape (B, N, 384)
            teacher_feat = patch_feats.permute(0, 2, 1).reshape(b, self.target_dim, grid_h, grid_w)

        loss = 0.0
        student_feats = pred if isinstance(pred, (list, tuple)) else [pred]
        for f in student_feats:
            target_feat = F.interpolate(teacher_feat, size=f.shape[-2:], mode="bilinear", align_corners=False)
            cos_sim = F.cosine_similarity(f, target_feat, dim=1)
            loss += (1.0 - cos_sim).mean()

        return loss, torch.stack((loss.detach(), torch.tensor(0.0, device=target.device)))


