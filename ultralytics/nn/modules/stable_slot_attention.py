"""Numerically stable slot-inspired attention blocks for DentalYOLO."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


_INITIAL_GATE_LOGIT = -2.197224577  # sigmoid(logit) == 0.1


class _GroupNormConv(nn.Module):
    """Convolution followed by per-sample normalization and an optional activation."""

    def __init__(self, c1: int, c2: int, act: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, kernel_size=1, bias=False)
        self.norm = nn.GroupNorm(1, c2)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project and normalize an image feature map."""
        return self.act(self.norm(self.conv(x)))


class StableSlimSlotAttention(nn.Module):
    """Deterministic slot-inspired attention with FP32 routing and bounded updates.

    This variant is intentionally separate from ``SlimSlotAttention`` so existing
    checkpoints and model YAML files retain their original behavior.
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        num_slots: int = 4,
        iters: int = 2,
        attn_ratio: float = 0.5,
        eps: float = 1e-4,
    ):
        super().__init__()
        if c1 != c2:
            raise ValueError(f"StableSlimSlotAttention requires c1 == c2, got {c1} != {c2}")
        if num_slots < 1 or iters < 1 or attn_ratio <= 0 or eps <= 0:
            raise ValueError("num_slots, iters, attn_ratio, and eps must all be positive")

        self.c = c1
        self.attn_c = max(int(c1 * attn_ratio), 16)
        self.num_slots = int(num_slots)
        self.iters = int(iters)
        self.eps = float(eps)
        self.scale = self.attn_c**-0.5

        # Learned but deterministic slot seeds avoid train-time sampling noise.
        self.slots_mu = nn.Parameter(torch.empty(1, self.num_slots, self.attn_c))

        self.norm_inputs = nn.GroupNorm(1, c1)
        self.to_k = nn.Conv2d(c1, self.attn_c, kernel_size=1, bias=False)
        self.to_v = nn.Conv2d(c1, self.attn_c, kernel_size=1, bias=False)
        self.to_q = nn.Linear(self.attn_c, self.attn_c, bias=False)

        self.norm_slots = nn.LayerNorm(self.attn_c)
        self.norm_updates = nn.LayerNorm(self.attn_c)
        self.update_mlp = nn.Sequential(
            nn.Linear(self.attn_c, self.attn_c),
            nn.GELU(),
            nn.Linear(self.attn_c, self.attn_c),
        )

        # Sigmoid gates remain in (0, 1), preventing unbounded residual scales.
        self.slot_update_logit = nn.Parameter(torch.tensor(_INITIAL_GATE_LOGIT))
        self.output_logit = nn.Parameter(torch.tensor(_INITIAL_GATE_LOGIT))
        self.to_out = nn.Conv2d(self.attn_c, c1, kernel_size=1, bias=False)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize routing layers conservatively around an identity mapping."""
        nn.init.normal_(self.slots_mu, std=0.02)
        for module in (self.to_k, self.to_v):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        nn.init.xavier_uniform_(self.to_q.weight)
        for module in self.update_mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.zeros_(self.to_out.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply stable slot routing while preserving the input shape and dtype."""
        b, _, h, w = x.shape
        x_norm = self.norm_inputs(x)
        k = self.to_k(x_norm).flatten(2)  # [B, D, HW]
        v = self.to_v(x_norm).flatten(2).transpose(1, 2)  # [B, HW, D]
        slots = self.slots_mu.expand(b, -1, -1).to(dtype=k.dtype)

        attn = None
        for _ in range(self.iters):
            q = self.to_q(self.norm_slots(slots))

            # Casting alone is insufficient inside AMP because autocast may cast
            # bmm back to FP16. Disable autocast for the sensitive routing region.
            with torch.autocast(device_type=x.device.type, enabled=False):
                dots = torch.bmm(q.float(), k.float()) * self.scale
                attn = F.softmax(dots, dim=1)
                denominator = attn.sum(dim=2, keepdim=True).clamp_min(self.eps)
                updates_fp32 = torch.bmm(attn / denominator, v.float())

            updates = self.norm_updates(updates_fp32.to(dtype=slots.dtype))
            delta = self.update_mlp(updates)
            update_scale = self.slot_update_logit.sigmoid().to(dtype=delta.dtype)
            slots = self.norm_slots(slots + update_scale * delta)

        with torch.autocast(device_type=x.device.type, enabled=False):
            out_spatial = torch.bmm(attn.transpose(1, 2), slots.float())
        out = out_spatial.reshape(b, h, w, self.attn_c).permute(0, 3, 1, 2).contiguous()
        out = self.to_out(out.to(dtype=x.dtype))
        output_scale = self.output_logit.sigmoid().to(dtype=out.dtype)
        return x + output_scale * out


class C2StableSlot(nn.Module):
    """YOLO-compatible wrapper around one or more stable slot-attention blocks."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        num_slots: int = 4,
        iters: int = 2,
        attn_ratio: float = 0.5,
    ):
        super().__init__()
        repeats = max(int(n), 1)
        self.cv1 = _GroupNormConv(c1, c2)
        self.slot = nn.Sequential(
            *(
                StableSlimSlotAttention(
                    c2,
                    c2,
                    num_slots=num_slots,
                    iters=iters,
                    attn_ratio=attn_ratio,
                )
                for _ in range(repeats)
            )
        )
        self.cv2 = _GroupNormConv(c2, c2, act=False)
        self.shortcut = nn.Identity() if c1 == c2 else _GroupNormConv(c1, c2, act=False)
        self.branch_logit = nn.Parameter(torch.tensor(_INITIAL_GATE_LOGIT))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the bounded attention branch and add its shortcut."""
        identity = self.shortcut(x)
        branch = self.cv2(self.slot(self.cv1(x)))
        branch_scale = self.branch_logit.sigmoid().to(dtype=branch.dtype)
        return identity + branch_scale * branch


__all__ = ("C2StableSlot", "StableSlimSlotAttention")
