"""SSL4: I-JEPA (Image-based Joint-Embedding Predictive Architecture) modules.

Developed for training abstract representation matching in latent space.
"""

from __future__ import annotations

from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentPredictor(nn.Module):
    """Predictor MLP for SSL4 (I-JEPA) pretraining.

    Predicts target latent representations from context features using a lightweight MLP.

    Args:
        in_dim (int): Channel dimension of the input feature maps (e.g. 512 for YOLO26 P5).
        hidden_dim (int): Channel dimension of the MLP hidden layer.
        out_dim (int): Channel dimension of the target latent representation (e.g. 256).
    """

    def __init__(self, in_dim=512, hidden_dim=1024, out_dim=256):
        """Initialize LatentPredictor layer with MLP layers."""
        super().__init__()
        if isinstance(in_dim, (list, tuple)):
            # If multiple feature maps are passed, project each to out_dim and average
            self.proj = nn.ModuleList(
                nn.Sequential(
                    nn.Conv2d(c, hidden_dim, 1, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    nn.GELU(),
                    nn.Conv2d(hidden_dim, out_dim, 1),
                )
                for c in in_dim
            )
            self.multiscale = True
        else:
            self.mlp = nn.Sequential(
                nn.Conv2d(in_dim, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                nn.Conv2d(hidden_dim, out_dim, 1),
            )
            self.multiscale = False

    def forward(self, x):
        """Forward pass projecting features to latent dimension."""
        if self.multiscale:
            feats = x if isinstance(x, (list, tuple)) else [x]
            outs = []
            target_hw = feats[0].shape[-2:]
            for f, proj in zip(feats, self.proj):
                y = proj(f)
                if y.shape[-2:] != target_hw:
                    y = F.interpolate(y, size=target_hw, mode="bilinear", align_corners=False)
                outs.append(y)
            return torch.mean(torch.stack(outs), dim=0)
        else:
            return self.mlp(x)


class EMATargetEncoder:
    """Exponential Moving Average (EMA) teacher copy of the student encoder for SSL4 (I-JEPA).

    This target encoder is frozen (requires_grad = False) and updated using momentum.
    """

    def __init__(self, student_model, momentum=0.996):
        """Initialize the EMA Target Encoder with a copy of student model."""
        self.target = deepcopy(student_model)
        for p in self.target.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update(self, student_model, momentum=0.996):
        """Update target weights using Exponential Moving Average."""
        for t_p, s_p in zip(self.target.parameters(), student_model.parameters()):
            t_p.data.mul_(momentum).add_(s_p.data, alpha=1.0 - momentum)

    def __call__(self, x):
        """Forward pass through target model."""
        return self.target(x)


class JEPALoss(nn.Module):
    """L1 / Smooth-L1 loss in the latent space between context predictions and target features."""

    def __init__(self, masked_only=True, eps=1e-6):
        """Initialize JEPALoss module."""
        super().__init__()
        self.masked_only = masked_only
        self.eps = eps

    def forward(self, pred, target, mask=None):
        """Compute Smooth L1 loss in latent space.

        Args:
            pred (Tensor): Predicted latent features, shape (B, out_dim, H, W).
            target (Tensor): Target latent features from Target Encoder, shape (B, out_dim, H, W).
            mask (Tensor, optional): Binary mask, shape (B, 1, H, W). 1 = masked region, 0 = visible.

        Returns:
            loss (Tensor): Scalar loss value.
            items (Tensor): Log items [loss, 0.0].
        """
        if pred.shape[-2:] != target.shape[-2:]:
            target = F.interpolate(target, size=pred.shape[-2:], mode="bilinear", align_corners=False)
            if mask is not None:
                mask = F.interpolate(mask, size=pred.shape[-2:], mode="nearest")

        loss_map = F.smooth_l1_loss(pred, target, reduction="none")

        if self.masked_only and mask is not None:
            # We want to predict target features of masked regions (so we compute loss on mask == 1)
            if mask.shape[1] == 1:
                mask = mask.expand_as(pred)
            denom = mask.sum().clamp_min(self.eps)
            loss = (loss_map * mask).sum() / denom
        else:
            loss = loss_map.mean()

        return loss, torch.stack((loss.detach(), torch.tensor(0.0, device=pred.device)))
