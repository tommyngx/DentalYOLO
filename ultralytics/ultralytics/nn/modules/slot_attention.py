"""Slot attention modules for DentalYOLO."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SlotAttention(nn.Module):
    """Lightweight slot attention for low-resolution YOLO feature maps.

    Signature compatible with ultralytics base_modules: (c1, c2, num_slots=4, iters=2, attn_ratio=1.0).
    The block preserves channel count, so c1 must equal c2.
    """

    def __init__(self, c1, c2, num_slots=4, iters=2, attn_ratio=1.0, eps=1e-8):
        super().__init__()
        if c1 != c2:
            raise ValueError(f"SlotAttention requires c1 == c2 (got {c1} != {c2})")
        self.c = c1
        self.attn_c = max(int(c1 * attn_ratio), 16)
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = self.attn_c**-0.5

        self.slots_mu = nn.Parameter(torch.randn(1, num_slots, self.attn_c))
        self.slots_sigma = nn.Parameter(torch.rand(1, num_slots, self.attn_c))

        self.to_q = nn.Linear(self.attn_c, self.attn_c, bias=False)
        self.to_k = nn.Linear(c1, self.attn_c, bias=False)
        self.to_v = nn.Linear(c1, self.attn_c, bias=False)
        self.mlp_update = nn.Sequential(
            nn.Linear(self.attn_c, self.attn_c * 2),
            nn.GELU(),
            nn.Linear(self.attn_c * 2, self.attn_c),
        )
        self.norm_inputs = nn.LayerNorm(c1)
        self.norm_slots = nn.LayerNorm(self.attn_c)
        self.to_out = nn.Linear(self.attn_c, c1, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape
        inputs = x.permute(0, 2, 3, 1).reshape(b, h * w, c)
        inputs = self.norm_inputs(inputs)

        slots = self.slots_mu.expand(b, -1, -1)
        if self.training:
            slots = slots + self.slots_sigma.expand(b, -1, -1) * torch.randn_like(slots)

        k = self.to_k(inputs)
        v = self.to_v(inputs)
        attn = None
        for _ in range(self.iters):
            slots_prev = slots
            q = self.to_q(self.norm_slots(slots))
            dots = torch.bmm(q, k.transpose(1, 2)) * self.scale
            attn = F.softmax(dots, dim=1)
            attn_norm = attn / (attn.sum(dim=-1, keepdim=True) + self.eps)
            updates = torch.bmm(attn_norm, v)
            slots = slots_prev + self.mlp_update(updates)

        out = self.to_out(torch.bmm(attn.transpose(1, 2), slots))
        out = out.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        return out + x
