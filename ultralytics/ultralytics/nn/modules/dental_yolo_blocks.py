"""Lightweight DentalYOLO blocks for OPG X-ray models."""

import torch
import torch.nn as nn

from .block import C3k2
from .conv import Conv


class ECALayer(nn.Module):
    """Efficient channel attention with negligible inference overhead."""

    def __init__(self, c, k=3):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        y = self.pool(x).squeeze(-1).transpose(-1, -2)
        y = self.act(self.conv(y)).transpose(-1, -2).unsqueeze(-1)
        return x * y


class C3k2ECA(C3k2):
    """C3k2 with a small ECA channel attention gate."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, attn=False, g=1, shortcut=True, eca_k=3):
        super().__init__(c1, c2, n, c3k, e, attn, g, shortcut)
        self.eca = ECALayer(c2, eca_k)

    def forward(self, x):
        return self.eca(super().forward(x))


class LargeKernelDWContext(nn.Module):
    """Large receptive-field context using cheap depthwise and strip convolutions."""

    def __init__(self, c1, c2, k=7):
        super().__init__()
        if c1 != c2:
            raise ValueError(f"LargeKernelDWContext requires c1 == c2 (got {c1} != {c2})")
        p = k // 2
        self.dw = nn.Conv2d(c1, c1, k, padding=p, groups=c1, bias=False)
        self.dw_h = nn.Conv2d(c1, c1, (1, k), padding=(0, p), groups=c1, bias=False)
        self.dw_v = nn.Conv2d(c1, c1, (k, 1), padding=(p, 0), groups=c1, bias=False)
        self.bn = nn.BatchNorm2d(c1)
        self.act = nn.SiLU(inplace=True)
        self.pw = Conv(c1, c1, 1, 1)
        self.eca = ECALayer(c1)

    def forward(self, x):
        y = self.dw(x) + self.dw_h(x) + self.dw_v(x)
        y = self.pw(self.act(self.bn(y)))
        return self.eca(y) + x


class BiFPNLite(nn.Module):
    """Weighted-add feature fusion with per-input projection to a shared channel count."""

    def __init__(self, channels, c2):
        super().__init__()
        self.proj = nn.ModuleList(Conv(c, c2, 1, 1) for c in channels)
        self.w = nn.Parameter(torch.ones(len(channels), dtype=torch.float32))
        self.eps = 1e-4
        self.out = Conv(c2, c2, 3, 1)

    def forward(self, xs):
        w = torch.relu(self.w)
        w = w / (w.sum() + self.eps)
        y = sum(wi * proj(x) for wi, proj, x in zip(w, self.proj, xs))
        return self.out(y)
