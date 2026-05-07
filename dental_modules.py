"""DentalYOLO26 custom modules.

Lives in a .py file (not a notebook cell) so DDP subprocesses can import it
and pickle/unpickle the model.

On import, this module:
  1. Defines the custom nn.Module classes (XRayEnhanceConv, LATDAA, DRELAN, ...)
  2. Registers them in ultralytics.nn.tasks globals
  3. Source-patches ultralytics.nn.tasks.parse_model so our custom modules are
     in `base_modules`, which makes parse_model auto-inject c_in from `ch[f]`
     and track c_out for downstream layers — i.e. they behave just like Conv,
     C3k2, etc. when used in a yaml.

After import, you can write a normal yaml (dental-yolo26.yaml) that uses
XRayEnhanceConv / LATDAA / DRELAN as if they were stock Ultralytics layers.

Module signatures are designed to match Ultralytics' base_modules convention:
    XRayEnhanceConv(c1, c2, stride=1)
    LATDAA(c1, c2, num_heads=4, num_anchors=32)        # c1 must equal c2
    DRELAN(c1, c2, n=2, e=0.5)                          # n = inner blocks
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# 1. XRayEnhanceConv (multi-scale + deformable + SE)
# ============================================================================
class SEBlock(nn.Module):
    def __init__(self, c, r=16):
        super().__init__()
        c_h = max(c // r, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(c, c_h, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(c_h, c, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(self.pool(x))


class _DSConv(nn.Module):
    def __init__(self, c_in, c_out, k):
        super().__init__()
        self.dw = nn.Conv2d(c_in, c_in, k, padding=k // 2, groups=c_in, bias=False)
        self.pw = nn.Conv2d(c_in, c_out, 1, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.pw(self.dw(x))))


class _DeformableConv2d(nn.Module):
    def __init__(self, c_in, c_out, k=3, stride=1):
        super().__init__()
        from torchvision.ops import deform_conv2d
        self.deform = deform_conv2d
        self.k, self.stride, self.pad = k, stride, k // 2
        self.offset = nn.Conv2d(c_in, 2 * k * k, k, stride=stride, padding=self.pad)
        self.mask   = nn.Conv2d(c_in,     k * k, k, stride=stride, padding=self.pad)
        self.weight = nn.Parameter(torch.empty(c_out, c_in, k, k))
        self.bias   = nn.Parameter(torch.zeros(c_out))
        nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="relu")
        for m in (self.offset, self.mask):
            nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        off = self.offset(x)
        msk = torch.sigmoid(self.mask(x))
        out = self.deform(x, off, self.weight, self.bias,
                          stride=self.stride, padding=self.pad, mask=msk)
        return self.act(self.bn(out))


class XRayEnhanceConv(nn.Module):
    """Multi-scale (3/5/7) DSConv + deformable conv + SE fusion.

    Signature compatible with ultralytics base_modules: (c1, c2, stride=1).
    parse_model passes c1 = ch[f]; we never see it written in yaml.
    """
    def __init__(self, c1, c2, stride=1):
        super().__init__()
        self.stride = stride
        c_branch = c2 // 4
        c_dcn    = c2 - 3 * c_branch
        if stride > 1:
            self.down = nn.Sequential(
                nn.Conv2d(c1, c1, 3, stride=stride, padding=1, groups=c1, bias=False),
                nn.BatchNorm2d(c1),
                nn.SiLU(inplace=True),
            )
        self.b3 = _DSConv(c1, c_branch, 3)
        self.b5 = _DSConv(c1, c_branch, 5)
        self.b7 = _DSConv(c1, c_branch, 7)
        self.bd = _DeformableConv2d(c1, c_dcn, k=3, stride=1)
        self.fuse = nn.Conv2d(c2, c2, 1, bias=False)
        self.bn   = nn.BatchNorm2d(c2)
        self.act  = nn.SiLU(inplace=True)
        self.se   = SEBlock(c2)

    def forward(self, x):
        if self.stride > 1:
            x = self.down(x)
        out = torch.cat([self.b3(x), self.b5(x), self.b7(x), self.bd(x)], dim=1)
        return self.se(self.act(self.bn(self.fuse(out))))


# ============================================================================
# 2. LATDAA (learnable anatomical-anchor attention)
# ============================================================================
class ArchPriorLoss(nn.Module):
    """Pulls K anchors onto two learnable parabolic arches (upper + lower jaw)."""
    def __init__(self, K=32, lambda_arch=0.05):
        super().__init__()
        self.K = K
        self.lam = lambda_arch
        self.a_up = nn.Parameter(torch.full((), -1.5))
        self.h_up = nn.Parameter(torch.full((), 0.40))
        self.a_lo = nn.Parameter(torch.full((),  1.5))
        self.h_lo = nn.Parameter(torch.full((), 0.60))

    def forward(self, anchors_xy):
        if anchors_xy is None:
            return torch.tensor(0.0, device=self.a_up.device)
        K_half = anchors_xy.shape[1] // 2
        x_u, y_u = anchors_xy[:, :K_half, 0], anchors_xy[:, :K_half, 1]
        x_l, y_l = anchors_xy[:,  K_half:, 0], anchors_xy[:,  K_half:, 1]
        y_u_pred = self.a_up * (x_u - 0.5).pow(2) + self.h_up
        y_l_pred = self.a_lo * (x_l - 0.5).pow(2) + self.h_lo
        return self.lam * ((y_u - y_u_pred).pow(2).mean() + (y_l - y_l_pred).pow(2).mean())


class LATDAA(nn.Module):
    """Local quadrant attention + global anchor cross-attention.

    Signature: (c1, c2, num_heads=4, num_anchors=32). c1 must equal c2 — LATDAA
    does not change channel count. parse_model auto-fills c1 from ch[f] and you
    must write the same value as c2 in the yaml.
    """
    def __init__(self, c1, c2, num_heads=4, num_anchors=32):
        super().__init__()
        if c1 != c2:
            raise ValueError(f"LATDAA requires c1 == c2 (got {c1} != {c2})")
        c = c1
        # Auto-fix num_heads if it doesn't divide c
        while c % num_heads != 0 and num_heads > 1:
            num_heads -= 1
        self.c = c
        self.h = num_heads
        self.d = c // num_heads
        self.scale = self.d ** -0.5
        self.K = num_anchors

        self.qkv_local = nn.Conv2d(c, c * 3, 1, bias=False)
        self.anchor_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(8),
            nn.Conv2d(c, c, 1), nn.SiLU(inplace=True),
            nn.Flatten(),
            nn.Linear(c * 64, num_anchors * 2),
            nn.Sigmoid(),
        )
        self.q_proj   = nn.Conv2d(c, c, 1, bias=False)
        self.k_proj   = nn.Linear(c, c, bias=False)
        self.v_proj   = nn.Linear(c, c, bias=False)
        self.out_proj = nn.Conv2d(c * 2, c, 1, bias=False)
        self.norm     = nn.GroupNorm(1, c)
        self.arch_prior = ArchPriorLoss(K=num_anchors)
        self.last_anchors = None
        self.last_arch_loss = None

    def _quadrant_attn(self, x):
        B, C, H, W = x.shape
        Hp = (H // 2) * 2; Wp = (W // 2) * 2
        xc = x[:, :, :Hp, :Wp]
        rh, rw = Hp // 2, Wp // 2
        qkv = self.qkv_local(xc)
        q, k, v = qkv.chunk(3, dim=1)

        def regions(t):
            return t.reshape(B, C, 2, rh, 2, rw).permute(0, 2, 4, 1, 3, 5).reshape(B*4, C, rh*rw)

        q_r, k_r, v_r = regions(q), regions(k), regions(v)
        BR, _, N = q_r.shape
        q_r = q_r.reshape(BR, self.h, self.d, N)
        k_r = k_r.reshape(BR, self.h, self.d, N)
        v_r = v_r.reshape(BR, self.h, self.d, N)
        attn = (q_r.transpose(-2, -1) @ k_r) * self.scale
        attn = attn.softmax(dim=-1)
        out = (v_r @ attn.transpose(-2, -1)).reshape(BR, C, N)
        out = out.reshape(B, 2, 2, C, rh, rw).permute(0, 3, 1, 4, 2, 5).reshape(B, C, Hp, Wp)
        if Hp != H or Wp != W:
            out = F.pad(out, (0, W - Wp, 0, H - Hp))
        return out

    def _anchor_attn(self, x):
        B, C, H, W = x.shape
        anchors = self.anchor_head(x).reshape(B, self.K, 2)
        self.last_anchors = anchors
        self.last_arch_loss = self.arch_prior(anchors)
        grid = (anchors * 2 - 1).reshape(B, 1, self.K, 2)
        anchor_feats = F.grid_sample(x, grid, mode="bilinear", align_corners=False)\
                          .squeeze(2).transpose(1, 2)
        Q  = self.q_proj(x).reshape(B, self.h, self.d, H * W).permute(0, 1, 3, 2)
        K_ = self.k_proj(anchor_feats).reshape(B, self.K, self.h, self.d).permute(0, 2, 1, 3)
        V_ = self.v_proj(anchor_feats).reshape(B, self.K, self.h, self.d).permute(0, 2, 1, 3)
        attn = (Q @ K_.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        return (attn @ V_).permute(0, 1, 3, 2).reshape(B, C, H, W)

    def forward(self, x):
        local = self._quadrant_attn(x)
        glob  = self._anchor_attn(x)
        out   = self.out_proj(torch.cat([local, glob], dim=1))
        return self.norm(out + x)

    def __deepcopy__(self, memo):
        # Ultralytics' ModelEMA does deepcopy(model) at training start. If
        # last_anchors / last_arch_loss already hold non-leaf intermediate
        # tensors from a prior forward (AMP check, stride probe, etc.),
        # deepcopy raises "Only Tensors created explicitly by the user
        # support the deepcopy protocol". Solution: temporarily clear them,
        # let the default deepcopy run, then restore on the original.
        saved_anchors, saved_arch = self.last_anchors, self.last_arch_loss
        self.last_anchors, self.last_arch_loss = None, None
        try:
            cls = self.__class__
            new = cls.__new__(cls)
            memo[id(self)] = new
            from copy import deepcopy as _dc
            new.__dict__.update({k: _dc(v, memo) for k, v in self.__dict__.items()})
        finally:
            self.last_anchors, self.last_arch_loss = saved_anchors, saved_arch
        # The clone starts with empty caches — correct.
        new.last_anchors = None
        new.last_arch_loss = None
        return new

    def __getstate__(self):
        # Same defence for pickle (torch.save of best.pt, DDP transfer, etc.).
        state = self.__dict__.copy()
        state["last_anchors"] = None
        state["last_arch_loss"] = None
        return state


# ============================================================================
# 3. DRELAN (residual ELAN)
# ============================================================================
class DRELAN(nn.Module):
    """Residual ELAN-style block.

    Signature: (c1, c2, n=2, e=0.5). n is inner-block count (NOT layer-repeat).
    parse_model auto-fills c1 from ch[f].
    """
    def __init__(self, c1, c2, n=2, e=0.5):
        super().__init__()
        c_h = max(int(c2 * e), 1)
        self.cv1 = nn.Conv2d(c1, 2 * c_h, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(2 * c_h)
        self.act1 = nn.SiLU(inplace=True)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c_h, c_h, 3, padding=1, groups=c_h, bias=False),
                nn.Conv2d(c_h, c_h, 1, bias=False),
                nn.BatchNorm2d(c_h), nn.SiLU(inplace=True),
            ) for _ in range(n)
        ])
        self.lambdas = nn.ParameterList([nn.Parameter(torch.full((), 0.01)) for _ in range(n)])
        self.cv2 = nn.Conv2d((2 + n) * c_h, c2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c2)
        self.act2 = nn.SiLU(inplace=True)

    def forward(self, x):
        y = self.act1(self.bn1(self.cv1(x)))
        a, b = y.chunk(2, dim=1)
        outs = [a, b]
        h = b
        for blk, lam in zip(self.blocks, self.lambdas):
            h = h + lam * blk(h)
            outs.append(h)
        return self.act2(self.bn2(self.cv2(torch.cat(outs, dim=1))))


# ============================================================================
# 4. Scale-Adaptive NWD-CIoU loss (used by the loss patcher)
# ============================================================================
def normalized_wasserstein_similarity(pred_xywh, tgt_xywh, C=12.8):
    pcx, pcy, pw, ph = pred_xywh.unbind(-1)
    tcx, tcy, tw, th = tgt_xywh.unbind(-1)
    center = (pcx - tcx).pow(2) + (pcy - tcy).pow(2)
    size   = ((pw - tw).pow(2) + (ph - th).pow(2)) / 4.0
    return torch.exp(-torch.sqrt((center + size).clamp(min=1e-7)) / C)


def ciou_loss_xywh(pred_xywh, tgt_xywh, eps=1e-7):
    pcx, pcy, pw, ph = pred_xywh.unbind(-1)
    tcx, tcy, tw, th = tgt_xywh.unbind(-1)
    px1, py1, px2, py2 = pcx - pw/2, pcy - ph/2, pcx + pw/2, pcy + ph/2
    tx1, ty1, tx2, ty2 = tcx - tw/2, tcy - th/2, tcx + tw/2, tcy + th/2
    iw = (torch.min(px2, tx2) - torch.max(px1, tx1)).clamp(0)
    ih = (torch.min(py2, ty2) - torch.max(py1, ty1)).clamp(0)
    inter = iw * ih
    union = pw*ph + tw*th - inter + eps
    iou = inter / union
    cw = torch.max(px2, tx2) - torch.min(px1, tx1)
    ch_ = torch.max(py2, ty2) - torch.min(py1, ty1)
    c2 = cw*cw + ch_*ch_ + eps
    rho2 = (pcx - tcx).pow(2) + (pcy - tcy).pow(2)
    v = (4 / math.pi**2) * (torch.atan(tw / (th + eps)) - torch.atan(pw / (ph + eps))).pow(2)
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)
    return 1 - iou + rho2 / c2 + alpha * v


class ScaleAdaptiveNWDCIoU(nn.Module):
    """alpha(s) = sigmoid(-k * log(s/s0)).  Small -> NWD, large -> CIoU."""
    def __init__(self, s0=24.0, k=2.0, C=12.8):
        super().__init__()
        self.s0, self.k, self.C = s0, k, C

    def alpha(self, tgt_xywh):
        s = torch.sqrt(tgt_xywh[..., 2] * tgt_xywh[..., 3]).clamp(min=1.0)
        return torch.sigmoid(-self.k * torch.log(s / self.s0))

    def forward(self, pred_xywh, tgt_xywh):
        l_nwd  = 1.0 - normalized_wasserstein_similarity(pred_xywh, tgt_xywh, C=self.C)
        l_ciou = ciou_loss_xywh(pred_xywh, tgt_xywh)
        a = self.alpha(tgt_xywh)
        return a * l_nwd + (1 - a) * l_ciou


# ============================================================================
# 5. Loss patcher (must run in EVERY process: main + DDP children)
# ============================================================================
def install_dental_loss(s0=24.0, k=2.0, C=12.8, arch_weight=1.0, verbose=True):
    """Patch ultralytics' BboxLoss.forward + v8DetectionLoss.__call__."""
    from ultralytics.utils import loss as ul_loss
    if not hasattr(ul_loss, "BboxLoss"):
        raise RuntimeError("BboxLoss not found.")
    if getattr(ul_loss.BboxLoss, "_dental_patched", False):
        if verbose:
            print("  Loss already patched, skipping.")
        return None

    BboxLoss = ul_loss.BboxLoss
    DetLoss  = ul_loss.v8DetectionLoss

    sa_loss_fn = ScaleAdaptiveNWDCIoU(s0=s0, k=k, C=C)
    original_bbox_forward = BboxLoss.forward
    original_det_call     = DetLoss.__call__

    def patched_bbox(self, pred_dist, pred_bboxes, anchor_points,
                     target_bboxes, target_scores, target_scores_sum, fg_mask,
                     *args, **kwargs):
        if fg_mask.sum() == 0:
            return original_bbox_forward(
                self, pred_dist, pred_bboxes, anchor_points,
                target_bboxes, target_scores, target_scores_sum, fg_mask)
        pb = pred_bboxes[fg_mask]
        tb = target_bboxes[fg_mask]
        pb_x = torch.stack([(pb[:,0]+pb[:,2])/2, (pb[:,1]+pb[:,3])/2,
                            (pb[:,2]-pb[:,0]),   (pb[:,3]-pb[:,1])], dim=-1)
        tb_x = torch.stack([(tb[:,0]+tb[:,2])/2, (tb[:,1]+tb[:,3])/2,
                            (tb[:,2]-tb[:,0]),   (tb[:,3]-tb[:,1])], dim=-1)
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        loss_iou = (sa_loss_fn(pb_x, tb_x).unsqueeze(-1) * weight).sum() / target_scores_sum
        loss_dfl = torch.tensor(0.0, device=pb.device)
        return loss_iou, loss_dfl

    def patched_call(self, preds, batch):
        loss, loss_items = original_det_call(self, preds, batch)
        model = getattr(self, "model", None) or getattr(self, "module", None)
        if model is not None:
            arch_l = sum(
                m.last_arch_loss for m in model.modules()
                if isinstance(m, LATDAA) and m.last_arch_loss is not None
            )
            if torch.is_tensor(arch_l):
                loss = loss + arch_weight * arch_l
        return loss, loss_items

    BboxLoss.forward = patched_bbox
    DetLoss.__call__ = patched_call
    BboxLoss._dental_patched = True
    BboxLoss._dental_original_forward = original_bbox_forward
    DetLoss._dental_original_call = original_det_call
    if verbose:
        print(f"  Loss patches installed (s0={s0}, k={k}, C={C}, arch_w={arch_weight})")
    return {"bbox": original_bbox_forward, "det": original_det_call}


def restore_dental_loss(handle=None, verbose=True):
    from ultralytics.utils import loss as ul_loss
    if handle is None:
        if hasattr(ul_loss.BboxLoss, "_dental_original_forward"):
            ul_loss.BboxLoss.forward = ul_loss.BboxLoss._dental_original_forward
            ul_loss.v8DetectionLoss.__call__ = ul_loss.v8DetectionLoss._dental_original_call
            del ul_loss.BboxLoss._dental_original_forward
            del ul_loss.v8DetectionLoss._dental_original_call
    else:
        ul_loss.BboxLoss.forward = handle["bbox"]
        ul_loss.v8DetectionLoss.__call__ = handle["det"]
    if hasattr(ul_loss.BboxLoss, "_dental_patched"):
        delattr(ul_loss.BboxLoss, "_dental_patched")
    if verbose:
        print("  Loss patches reverted.")


# ============================================================================
# 6. Register custom modules with ultralytics' parse_model.
#
# parse_model treats unknown modules as identity for channel tracking
# (output channels = input channels) and never auto-injects c1. We need our
# modules to behave like base_modules: parse_model should auto-inject c1=ch[f]
# and track c2=args[0] for downstream layers.
#
# Implementation: source-patch parse_model so our classes are added to the
# `base_modules` frozenset. Done once per process; idempotent.
# ============================================================================
def _patch_parse_model(_t):
    if getattr(_t, "_dental_parse_model_patched", False):
        return
    import inspect, textwrap
    src = inspect.getsource(_t.parse_model)
    src = textwrap.dedent(src)

    needle = "base_modules = frozenset("
    if needle not in src:
        raise RuntimeError("Could not find base_modules in parse_model source")

    # Find the '{' opening the set literal after `frozenset(`
    pos = src.index(needle) + len(needle)
    while pos < len(src) and src[pos] in " \t\n":
        pos += 1
    if src[pos] != "{":
        raise RuntimeError(f"Expected '{{' after frozenset(, got {src[pos:pos+30]!r}")
    insert_at = pos + 1
    injection = "\n            XRayEnhanceConv,\n            LATDAA,\n            DRELAN,"
    new_src = src[:insert_at] + injection + src[insert_at:]

    # Exec in the module's namespace so it sees Conv, C2PSA, etc.
    exec(compile(new_src, "<dental_patched_parse_model>", "exec"), _t.__dict__)
    _t._dental_parse_model_patched = True


def _register_with_ultralytics():
    import ultralytics.nn.tasks as _t
    # 1. Make classes resolvable by name in parse_model's globals lookup
    _t.XRayEnhanceConv = XRayEnhanceConv
    _t.LATDAA = LATDAA
    _t.DRELAN = DRELAN
    # 2. Patch parse_model to recognize them as channel-changing modules
    _patch_parse_model(_t)


_register_with_ultralytics()


# ============================================================================
# 7. DDP support: patch generate_ddp_file so subprocess temp files import
#    this module at startup. Without this, DDP children don't know about our
#    custom classes and parse_model fails when reading dental yaml.
# ============================================================================
def patch_ddp_for_dental(module_dir, module_name="dental_modules",
                        loss_kwargs=None, verbose=True):
    """Make DDP subprocesses import `module_name` from `module_dir` at startup,
    and (optionally) call install_dental_loss in each child too.

    Args:
        module_dir (str | Path): directory containing dental_modules.py.
        module_name (str): module name to import (default: dental_modules).
        loss_kwargs (dict | None): kwargs for install_dental_loss; if not None,
            children will install the loss patch too. Pass None to skip.
    """
    import ultralytics.utils.dist as _dist
    if getattr(_dist, "_dental_ddp_patched", False):
        if verbose:
            print("  DDP already patched, skipping.")
        return

    _orig = _dist.generate_ddp_file
    module_dir = str(module_dir)

    def patched_generate_ddp_file(trainer):
        path = _orig(trainer)
        # Inject our imports right after the `if __name__ == "__main__":` line.
        with open(path, "r") as f:
            content = f.read()

        # Build the snippet to inject. Must be indented to match the if-block.
        loss_line = ""
        if loss_kwargs is not None:
            loss_line = (
                f"    {module_name}.install_dental_loss(**{loss_kwargs!r})\n"
            )
        injection = (
            f'    import sys\n'
            f'    sys.path.insert(0, {module_dir!r})\n'
            f'    import {module_name}\n'
            f'{loss_line}'
        )

        marker = 'if __name__ == "__main__":\n'
        if marker not in content:
            raise RuntimeError("DDP file format unrecognised; cannot inject.")
        idx = content.index(marker) + len(marker)
        new_content = content[:idx] + injection + content[idx:]

        with open(path, "w") as f:
            f.write(new_content)
        return path

    _dist.generate_ddp_file = patched_generate_ddp_file
    _dist._dental_ddp_patched = True
    if verbose:
        print(f"  DDP file generator patched (will import {module_name} from "
              f"{module_dir} in each subprocess).")
