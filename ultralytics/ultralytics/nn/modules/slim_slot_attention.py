"""Slim slot attention modules for DentalYOLO."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SlimSlotAttention(nn.Module):
    """Optimized slot attention block for YOLO feature maps.

    The attention tensor is split into two roles:
    - ``attn`` uses softmax over slots so slots compete for each pixel.
    - ``attn_weights`` is normalized over spatial positions for slot updates.
    """

    def __init__(self, c1, c2, num_slots=4, iters=2, attn_ratio=0.5, eps=1e-8):
        super().__init__()
        if c1 != c2:
            raise ValueError(
                f"SlimSlotAttention requires c1 == c2 (got {c1} != {c2}). "
                "Use C2Slot to project channels before applying slot attention."
            )

        self.c = c1
        self.attn_c = max(int(c1 * attn_ratio), 16)
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = self.attn_c**-0.5

        self.slots_mu = nn.Parameter(torch.randn(1, num_slots, self.attn_c))
        self.slots_sigma = nn.Parameter(torch.rand(1, num_slots, self.attn_c))

        self.to_k = nn.Conv2d(c1, self.attn_c, 1, bias=False)
        self.to_v = nn.Conv2d(c1, self.attn_c, 1, bias=False)
        self.to_q = nn.Linear(self.attn_c, self.attn_c, bias=False)

        # BatchNorm2d uses running stats in eval mode; call model.eval() for inference.
        self.norm_inputs = nn.BatchNorm2d(c1)
        self.norm_slots = nn.LayerNorm(self.attn_c)

        self.mlp_update = nn.Sequential(
            nn.Linear(self.attn_c, self.attn_c),
            nn.GELU(),
            nn.Linear(self.attn_c, self.attn_c),
        )
        self.to_out = nn.Conv2d(self.attn_c, c1, 1, bias=False)

        self._init_weights()

    def _init_weights(self):
        """Initialize projections with an identity-preserving output path."""
        for m in (self.to_k, self.to_v):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        nn.init.zeros_(self.to_out.weight)
        for m in self.mlp_update:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        b, _, h, w = x.shape
        x_norm = self.norm_inputs(x)

        k = self.to_k(x_norm).flatten(2)
        v = self.to_v(x_norm).flatten(2).transpose(1, 2)

        slots = self.slots_mu.expand(b, -1, -1)
        if self.training:
            slots = slots + self.slots_sigma.expand(b, -1, -1) * torch.randn_like(slots)

        attn = None
        for _ in range(self.iters):
            slots_prev = slots
            q = self.to_q(self.norm_slots(slots))
            dots = torch.bmm(q, k) * self.scale

            attn = F.softmax(dots, dim=1) + self.eps
            attn_weights = attn / attn.sum(dim=2, keepdim=True)
            updates = torch.bmm(attn_weights, v)

            slots = slots_prev + self.mlp_update(updates)

        out_spatial = torch.bmm(attn.transpose(1, 2), slots)
        out = out_spatial.transpose(1, 2).reshape(b, self.attn_c, h, w).contiguous()
        out = self.to_out(out)
        return x + out


class C2Slot(nn.Module):
    """YOLO-compatible wrapper around SlimSlotAttention."""

    def __init__(self, c1, c2, n=1, num_slots=4, iters=2, attn_ratio=0.5):
        super().__init__()
        self.cv1 = nn.Conv2d(c1, c2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(c2)
        self.act1 = nn.SiLU()

        self.slot = SlimSlotAttention(c2, c2, num_slots=num_slots, iters=iters, attn_ratio=attn_ratio)

        self.cv2 = nn.Conv2d(c2, c2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c2)
        self.act2 = nn.SiLU()

        self.shortcut = nn.Identity() if c1 == c2 else nn.Conv2d(c1, c2, 1, bias=False)

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.act1(self.bn1(self.cv1(x)))
        out = self.slot(out)
        out = self.act2(self.bn2(self.cv2(out)))
        return out + identity


def register_slot_modules():
    """Register SlimSlotAttention and C2Slot in Ultralytics namespaces."""
    try:
        from ultralytics.nn import tasks
        import ultralytics.nn.modules as modules_pkg

        modules_pkg.SlimSlotAttention = SlimSlotAttention
        modules_pkg.C2Slot = C2Slot
        tasks.SlimSlotAttention = SlimSlotAttention
        tasks.C2Slot = C2Slot
    except ImportError:
        return
