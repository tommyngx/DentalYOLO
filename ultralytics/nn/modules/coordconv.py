"""Coordinate-aware convolution modules for spatially structured images."""

from __future__ import annotations

import torch
import torch.nn as nn

from .conv import Conv


class AddCoords(nn.Module):
    """Append normalized x/y coordinates and an optional radial channel to a BCHW tensor."""

    def __init__(self, with_r: bool = False):
        super().__init__()
        self.with_r = bool(with_r)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``x`` concatenated with coordinate maps on the same device and dtype."""
        if x.ndim != 4:
            raise ValueError(f"AddCoords expects a BCHW tensor, got shape {tuple(x.shape)}")

        b, _, h, w = x.shape
        yy = torch.arange(h, device=x.device, dtype=torch.float32)
        xx = torch.arange(w, device=x.device, dtype=torch.float32)

        # The explicit singleton cases avoid division by zero while keeping
        # normal feature maps in the conventional [-1, 1] coordinate range.
        yy = yy.mul(2.0 / (h - 1)).sub(1.0) if h > 1 else yy.zero_()
        xx = xx.mul(2.0 / (w - 1)).sub(1.0) if w > 1 else xx.zero_()

        yy = yy.view(1, 1, h, 1).expand(b, 1, h, w).to(dtype=x.dtype)
        xx = xx.view(1, 1, 1, w).expand(b, 1, h, w).to(dtype=x.dtype)
        coordinates = [xx, yy]
        if self.with_r:
            r = (xx.square() + yy.square()).clamp_min(0).sqrt()
            coordinates.append(r)
        return torch.cat((x, *coordinates), dim=1)


class CoordConv(nn.Module):
    """Ultralytics-compatible convolution preceded by coordinate injection."""

    def __init__(
        self,
        c1: int,
        c2: int,
        k: int | tuple[int, int] = 1,
        s: int | tuple[int, int] = 1,
        p: int | tuple[int, int] | None = None,
        g: int = 1,
        d: int = 1,
        act: bool | nn.Module = True,
        with_r: bool = False,
    ):
        super().__init__()
        extra_channels = 3 if with_r else 2
        if (c1 + extra_channels) % g:
            raise ValueError(
                f"CoordConv input channels ({c1} + {extra_channels}) must be divisible by groups={g}"
            )
        self.addcoords = AddCoords(with_r=with_r)
        self.conv = Conv(c1 + extra_channels, c2, k, s, p, g, d, act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Inject coordinates and apply the wrapped Ultralytics convolution."""
        return self.conv(self.addcoords(x))


__all__ = ("AddCoords", "CoordConv")
