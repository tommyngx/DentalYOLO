"""DentalYOLO26 v16 blocks: the three v15 ideas, re-engineered for fine-tuning from YOLO26 checkpoints.

Design rules shared by every block in this file:

1. Same layer index as the YOLO26 layer it replaces. Each block *wraps* a stock YOLO26 module instead of being
   inserted as an extra layer, so pretrained ``model.N.*`` keys line up and ``model.load("yolo26s.pt")`` transfers
   100% of the backbone/neck/head weights. (v15 inserted layers and shifted indices, leaving ~37% of the parameters
   randomly initialised, including the whole Detect head.)
2. Exact identity at initialisation. Every new branch is gated by a zero-initialised scale, so a freshly built v16
   model with YOLO26 weights produces bit-identical predictions to YOLO26. Fine-tuning therefore starts from the
   baseline optimum and the new branches only have to learn a residual.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .block import C3k2
from .conv import Conv
from .coordconv import AddCoords
from .head import Detect
from .stable_slot_attention import C2StableSlot


class ECAGate(nn.Module):
    """Efficient channel attention whose gate is exactly 1 at initialisation.

    The gate is ``2 * sigmoid(conv1d(gap(x)))`` with a zero-initialised conv, so channels start un-scaled and can
    later be attenuated or amplified in ``(0, 2)``. Classic ECA uses ``sigmoid`` and halves every channel at init.
    """

    def __init__(self, c: int, k: int = 3):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        nn.init.zeros_(self.conv.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.pool(x).squeeze(-1).transpose(-1, -2)  # [B, 1, C]
        y = (2.0 * torch.sigmoid(self.conv(y))).transpose(-1, -2).unsqueeze(-1)  # [B, C, 1, 1]
        return x * y


class C3k2ECAv2(C3k2):
    """C3k2 followed by an identity-initialised ECA gate. State-dict keys of the C3k2 part match stock C3k2."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, attn=False, g=1, shortcut=True, eca_k=3):
        super().__init__(c1, c2, n, c3k, e, attn, g, shortcut)
        self.eca = ECAGate(c2, eca_k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.eca(super().forward(x))


class C3k2Slot(C3k2):
    """C3k2 followed by a stable slot-attention branch that contributes exactly zero at initialisation.

    Equivalent to v15's ``C3k2 -> C2StableSlot`` pair, but as one layer so the Detect index does not move. The
    slot branch's final GroupNorm gamma is zeroed and its residual gate starts at 0.5 instead of 0.1 so the branch
    is silent at init yet receives a usable gradient once gamma grows.
    """

    def __init__(
        self, c1, c2, n=1, c3k=False, e=0.5, attn=False, g=1, shortcut=True, num_slots=4, iters=2, attn_ratio=0.25
    ):
        super().__init__(c1, c2, n, c3k, e, attn, g, shortcut)
        self.slot = C2StableSlot(c2, c2, 1, num_slots, iters, attn_ratio)
        nn.init.zeros_(self.slot.cv2.norm.weight)  # branch output == 0 -> block == C3k2 at init
        nn.init.zeros_(self.slot.cv2.norm.bias)
        with torch.no_grad():
            self.slot.branch_logit.fill_(0.0)  # sigmoid(0) == 0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.slot(super().forward(x))


class CoordInject(nn.Module):
    """Residual coordinate injection that is an exact identity at initialisation.

    Default (``full=False``): ``x + BN(conv1x1([xx, yy]))`` - a learned per-channel positional bias, ~2c params.
    A 1x1 CoordConv over ``[x, xx, yy]`` is linear, so it equals exactly this positional term plus a channel
    mixing ``W x`` that the Detect head already provides; dropping the mixing removes >99% of the parameters
    (347k -> 2k at the s scale) with no loss of positional expressiveness. ``full=True`` keeps the v15-style conv
    over ``[x, xx, yy]`` for comparison. The conv is an Ultralytics ``Conv`` with zero-initialised BN gamma, so
    ``model.fuse()`` folds it into a single conv at inference.
    """

    def __init__(self, c: int, with_r: bool = False, full: bool = False):
        super().__init__()
        self.addcoords = AddCoords(with_r=with_r)
        self.full = bool(full)
        c_coord = 3 if with_r else 2
        self.conv = Conv((c + c_coord) if self.full else c_coord, c, 1, act=False)
        nn.init.zeros_(self.conv.bn.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        coords = self.addcoords(x) if self.full else self.addcoords.maps(x)
        return x + self.conv(coords)


class DetectCoord(Detect):
    """YOLO26 Detect head that injects normalised x/y coordinates into each input scale before prediction.

    Subclassing keeps the ``cv2``/``cv3``/``one2one_*`` key names, so YOLO26 head weights load unchanged; only the
    ``coord.*`` tensors are new, and they start as an exact identity.
    """

    def __init__(self, nc: int = 80, reg_max=16, end2end=False, ch: tuple = (), with_r: bool = False, full: bool = False):
        super().__init__(nc, reg_max, end2end, ch)
        self.coord = nn.ModuleList(CoordInject(c, with_r, full) for c in ch)

    def forward(self, x: list[torch.Tensor]):
        return super().forward([m(xi) for m, xi in zip(self.coord, x)])


__all__ = ("C3k2ECAv2", "C3k2Slot", "CoordInject", "DetectCoord", "ECAGate")
