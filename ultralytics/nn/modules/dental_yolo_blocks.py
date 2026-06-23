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


class _ChannelPool(nn.Module):
    """Channel pooling for Triplet Attention."""

    def forward(self, x):
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        return torch.cat((max_pool, avg_pool), dim=1)


class _TripletSpatialGate(nn.Module):
    """Spatial gate using max/average pooled channel maps."""

    def __init__(self, kernel_size=7):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd, got {kernel_size}")
        padding = kernel_size // 2
        self.compress = _ChannelPool()
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        attention = self.spatial(self.compress(x))
        return x * attention


class TripletAttention(nn.Module):
    """Lightweight Triplet Attention with optional identity-preserving residual blend."""

    def __init__(self, c1, c2, kernel_size=7, no_spatial=False, residual=True, init_gamma=0.1):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError(f"TripletAttention requires an odd kernel_size, got {kernel_size}")
        self.proj = Conv(c1, c2, 1, 1, act=False) if c1 != c2 else nn.Identity()
        self.cw_gate = _TripletSpatialGate(kernel_size)
        self.hc_gate = _TripletSpatialGate(kernel_size)
        self.hw_gate = None if no_spatial else _TripletSpatialGate(kernel_size)
        self.residual = residual
        if residual:
            self.gamma = nn.Parameter(torch.tensor(float(init_gamma)))
        else:
            self.register_parameter("gamma", None)

    def forward(self, x):
        x = self.proj(x)

        x_cw = x.permute(0, 2, 1, 3).contiguous()
        x_cw = self.cw_gate(x_cw).permute(0, 2, 1, 3).contiguous()

        x_hc = x.permute(0, 3, 2, 1).contiguous()
        x_hc = self.hc_gate(x_hc).permute(0, 3, 2, 1).contiguous()

        if self.hw_gate is None:
            y = 0.5 * (x_cw + x_hc)
        else:
            y = (x_cw + x_hc + self.hw_gate(x)) / 3.0

        if self.residual:
            return x + self.gamma * (y - x)

        return y


class ArchLSKA(nn.Module):
    """Dental arch-oriented large separable kernel attention."""

    def __init__(
        self,
        c1,
        c2,
        horizontal_kernel=31,
        vertical_kernel=15,
        local_kernel=5,
        vertical_scale=0.5,
        residual=True,
        init_gamma=0.1,
    ):
        super().__init__()
        for name, kernel_size in {
            "horizontal_kernel": horizontal_kernel,
            "vertical_kernel": vertical_kernel,
            "local_kernel": local_kernel,
        }.items():
            if kernel_size % 2 == 0:
                raise ValueError(f"ArchLSKA requires odd kernels, got {name}={kernel_size}")

        self.proj = Conv(c1, c2, 1, 1, act=False) if c1 != c2 else nn.Identity()
        self.local = nn.Conv2d(
            c2,
            c2,
            local_kernel,
            padding=local_kernel // 2,
            groups=c2,
            bias=False,
        )
        self.horizontal = nn.Conv2d(
            c2,
            c2,
            (1, horizontal_kernel),
            padding=(0, horizontal_kernel // 2),
            groups=c2,
            bias=False,
        )
        self.vertical = nn.Conv2d(
            c2,
            c2,
            (vertical_kernel, 1),
            padding=(vertical_kernel // 2, 0),
            groups=c2,
            bias=False,
        )
        self.mix = Conv(c2, c2, 1, 1, act=False)
        self.gate = nn.Sigmoid()
        self.vertical_scale = float(vertical_scale)
        self.residual = residual
        if residual:
            self.gamma = nn.Parameter(torch.tensor(float(init_gamma)))
        else:
            self.register_parameter("gamma", None)

    def forward(self, x):
        y = self.proj(x)

        context = self.local(y)
        context = self.horizontal(context) + self.vertical_scale * self.vertical(context)

        attended = y * self.gate(self.mix(context))

        if self.residual:
            return y + self.gamma * (attended - y)

        return attended


class DAABLite(nn.Module):
    """Dental Arch Attention Block Lite: ArchLSKA + Triplet Attention."""

    def __init__(
        self,
        c1,
        c2,
        horizontal_kernel=31,
        vertical_kernel=15,
        vertical_scale=0.5,
        triplet_kernel=7,
        local_kernel=5,
        init_gamma=0.1,
    ):
        super().__init__()
        self.shortcut = Conv(c1, c2, 1, 1, act=False) if c1 != c2 else nn.Identity()
        self.arch = ArchLSKA(
            c1,
            c2,
            horizontal_kernel=horizontal_kernel,
            vertical_kernel=vertical_kernel,
            local_kernel=local_kernel,
            vertical_scale=vertical_scale,
            residual=False,
        )
        self.triplet = TripletAttention(c2, c2, kernel_size=triplet_kernel, residual=False)
        self.gamma = nn.Parameter(torch.tensor(float(init_gamma)))

    def forward(self, x):
        identity = self.shortcut(x)

        y = self.arch(x)
        y = self.triplet(y)

        return identity + self.gamma * (y - identity)


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


class WeightedAddFusion(nn.Module):
    """Minimal weighted-add fusion: project inputs to c2 and skip the output 3x3 conv."""

    def __init__(self, channels, c2):
        super().__init__()
        self.proj = nn.ModuleList(nn.Identity() if c == c2 else Conv(c, c2, 1, 1) for c in channels)
        self.w = nn.Parameter(torch.ones(len(channels), dtype=torch.float32))
        self.eps = 1e-4

    def forward(self, xs):
        w = torch.relu(self.w)
        w = w / (w.sum() + self.eps)
        return sum(wi * proj(x) for wi, proj, x in zip(w, self.proj, xs))
