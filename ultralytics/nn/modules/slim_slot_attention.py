"""SlimSlotAttention - lightweight slot-inspired spatial attention for DentalYOLO / YOLO26."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SlimSlotAttention(nn.Module):
    """
    Slim Slot Attention for real-time YOLO-style models.

    Notes:
        - Slots compete per pixel using softmax over slot dimension.
        - Slot updates aggregate spatial features using normalized attention.
        - Output projection is zero-initialized, so the module starts as near-identity.
        - This is a lightweight slot-inspired module, not the full original Slot Attention with GRU.
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        num_slots: int = 4,
        iters: int = 2,
        attn_ratio: float = 0.5,
        eps: float = 1e-8,
    ):
        super().__init__()

        if c1 != c2:
            raise ValueError(
                f"SlimSlotAttention requires c1 == c2, got {c1} != {c2}. "
                "Use C2Slot wrapper to handle channel projection."
            )

        self.c = c1
        self.attn_c = max(int(c1 * attn_ratio), 16)
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = self.attn_c**-0.5

        # Learnable slot initialization
        self.slots_mu = nn.Parameter(torch.randn(1, num_slots, self.attn_c))
        self.slots_log_sigma = nn.Parameter(torch.zeros(1, num_slots, self.attn_c))

        # Input projections
        self.to_k = nn.Conv2d(c1, self.attn_c, kernel_size=1, bias=False)
        self.to_v = nn.Conv2d(c1, self.attn_c, kernel_size=1, bias=False)
        self.to_q = nn.Linear(self.attn_c, self.attn_c, bias=False)

        # Normalization
        self.norm_inputs = nn.BatchNorm2d(c1)
        self.norm_slots = nn.LayerNorm(self.attn_c)

        # Lightweight slot update MLP
        self.mlp_update = nn.Sequential(
            nn.Linear(self.attn_c, self.attn_c),
            nn.GELU(),
            nn.Linear(self.attn_c, self.attn_c),
        )

        # Output projection
        self.to_out = nn.Conv2d(self.attn_c, c1, kernel_size=1, bias=False)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights for stable YOLO fine-tuning."""
        for m in (self.to_k, self.to_v):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

        # Zero-init output projection so the block starts as near identity
        nn.init.zeros_(self.to_out.weight)

        for m in self.mlp_update:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape

        # Normalize input features
        x_norm = self.norm_inputs(x)

        # Project to key and value
        k = self.to_k(x_norm).flatten(2)  # [B, attn_c, HW]
        v = self.to_v(x_norm).flatten(2).transpose(1, 2)  # [B, HW, attn_c]

        # Initialize slots
        slots = self.slots_mu.expand(b, -1, -1)  # [B, S, attn_c]

        if self.training:
            sigma = F.softplus(self.slots_log_sigma) + self.eps
            slots = slots + sigma.expand(b, -1, -1) * torch.randn_like(slots)

        attn = None

        # Iterative slot routing
        for _ in range(self.iters):
            slots_prev = slots

            q = self.to_q(self.norm_slots(slots))  # [B, S, attn_c]
            dots = torch.bmm(q, k) * self.scale  # [B, S, HW]

            # Slots compete for each pixel
            attn = F.softmax(dots, dim=1)  # [B, S, HW]

            # Normalize over spatial dimension for weighted aggregation
            attn_sum = attn.sum(dim=2, keepdim=True).clamp_min(self.eps)
            attn_weights = attn / attn_sum  # [B, S, HW]

            # Aggregate values into slots
            updates = torch.bmm(attn_weights, v)  # [B, S, attn_c]

            # Lightweight residual update
            slots = slots_prev + self.mlp_update(updates)  # [B, S, attn_c]

        # Project slots back to spatial feature map
        out_spatial = torch.bmm(attn.transpose(1, 2), slots)  # [B, HW, attn_c]

        # Restore spatial layout safely: [B, HW, C] -> [B, C, H, W]
        out = out_spatial.reshape(b, h, w, self.attn_c)
        out = out.permute(0, 3, 1, 2).contiguous()

        out = self.to_out(out)

        return x + out


class C2Slot(nn.Module):
    """
    YOLO-compatible wrapper for SlimSlotAttention.

    This wrapper:
        - Projects input channels from c1 to c2.
        - Applies SlimSlotAttention.
        - Applies a final 1x1 projection.
        - Adds shortcut connection.
    """

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

        self.cv1 = nn.Conv2d(c1, c2, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c2)
        self.act1 = nn.SiLU(inplace=True)

        self.slot = SlimSlotAttention(
            c1=c2,
            c2=c2,
            num_slots=num_slots,
            iters=iters,
            attn_ratio=attn_ratio,
        )

        self.cv2 = nn.Conv2d(c2, c2, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c2)
        self.act2 = nn.SiLU(inplace=True)

        if c1 == c2:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(c1, c2, kernel_size=1, bias=False),
                nn.BatchNorm2d(c2),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)

        out = self.cv1(x)
        out = self.bn1(out)
        out = self.act1(out)

        out = self.slot(out)

        out = self.cv2(out)
        out = self.bn2(out)
        out = self.act2(out)

        return out + identity
