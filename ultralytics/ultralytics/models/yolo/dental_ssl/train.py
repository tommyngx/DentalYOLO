"""Standalone DentalYOLO26 masked-reconstruction SSL trainer.

Example:
    python -m ultralytics.models.yolo.dental_ssl.train \
        --data /path/to/unlabeled_opg/images \
        --model ultralytics/cfg/models/dentalssl/dental-yolo26m_v2ssl.yaml \
        --epochs 100 --imgsz 640 --batch 8 --device 0
"""

from __future__ import annotations

import argparse
import os
import re
import time
from copy import deepcopy
from pathlib import Path

import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from ultralytics.nn.modules import MaskedReconstructionLoss, random_patch_mask
from ultralytics.nn.tasks import BaseModel, parse_model, yaml_model_load
from ultralytics.utils import LOGGER, YAML
from ultralytics.utils.patches import torch_load
from ultralytics.utils.torch_utils import autocast, init_seeds, initialize_weights, intersect_dicts, select_device

IMG_SUFFIXES = {".bmp", ".dcm", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
DEFAULT_MODEL = Path(__file__).resolve().parents[3] / "cfg/models/dentalssl/dental-yolo26m_v2ssl.yaml"


def dental_ssl_yaml_load(path):
    """Load dentalssl YAML while preserving Ultralytics-style scale aliases.

    This loader fixes scale detection for filenames ending in ``ssl``. The
    generic Ultralytics regex can read ``yolo26ssl`` as scale ``s``.
    """
    path = Path(path)
    if path.exists():
        d = yaml_model_load(path)
        stem = path.stem
        match = re.search(r"yolo26([nsmxl])(?:_|-)?[^/]*ssl$", stem)
        if match:
            d["scale"] = match.group(1)
        elif stem.endswith("ssl"):
            d["scale"] = "m"
    else:
        scale = ""
        stem = path.stem
        if stem.endswith(("n", "s", "m", "l", "x")) and "yolo26" in stem:
            scale = stem[-1]
            base = path.with_name(stem[:-1] + path.suffix)
            if not base.exists():
                raise FileNotFoundError(f"Neither '{path}' nor base YAML '{base}' exists")
            d = YAML.load(base)
            d["scale"] = scale
            d["yaml_file"] = str(path)
        else:
            d = yaml_model_load(path)
    return d


class UnlabeledOPGDataset(Dataset):
    """Minimal image-folder dataset for SSL pretraining."""

    def __init__(self, root, imgsz=640, channels=1):
        self.root = resolve_ssl_data(root)
        self.imgsz = imgsz
        self.channels = channels
        if self.root.is_file():
            self.files = [Path(x.strip()) for x in self.root.read_text().splitlines() if x.strip()]
        else:
            self.files = sorted(p for p in self.root.rglob("*") if p.suffix.lower() in IMG_SUFFIXES)
        if not self.files:
            raise FileNotFoundError(f"No images found in {self.root}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path = self.files[index]
        flag = cv2.IMREAD_GRAYSCALE if self.channels == 1 else cv2.IMREAD_COLOR
        im = cv2.imread(str(path), flag)
        if im is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        if self.channels == 1:
            if im.ndim == 3:
                im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        else:
            if im.ndim == 2:
                im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
            else:
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = cv2.resize(im, (self.imgsz, self.imgsz), interpolation=cv2.INTER_AREA)
        if im.ndim == 2:
            im = im[:, :, None]
        im = torch.from_numpy(im).permute(2, 0, 1).contiguous().float() / 255.0
        return im, str(path)


def resolve_ssl_data(data):
    """Resolve an image folder, txt image list, or YOLO dataset YAML to SSL image input."""
    path = Path(data)
    if path.suffix.lower() in {".yaml", ".yml"}:
        cfg = YAML.load(path)
        root = Path(cfg.get("path", path.parent))
        train = cfg.get("train")
        if train is None:
            raise KeyError(f"Dataset YAML '{path}' has no 'train' key for SSL image discovery")
        train_path = Path(train)
        return train_path if train_path.is_absolute() else root / train_path
    return path


class DentalSSLModel(BaseModel):
    """YOLO26-P2 feature extractor with reconstruction decoder."""

    def __init__(self, cfg, ch=1, verbose=True):
        super().__init__()
        self.yaml = cfg if isinstance(cfg, dict) else dental_ssl_yaml_load(cfg)
        self.yaml["channels"] = ch
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=ch, verbose=verbose)
        self.stride = torch.tensor([4])
        initialize_weights(self)
        if verbose:
            self.info()
            LOGGER.info("")

    def init_criterion(self):
        return MaskedReconstructionLoss()

    def load(self, weights, verbose=True):
        model = weights["model"] if isinstance(weights, dict) and "model" in weights else weights
        csd = model.float().state_dict() if isinstance(model, nn.Module) else weights.get("state_dict", weights)
        updated_csd = intersect_dicts(csd, self.state_dict())
        self.load_state_dict(updated_csd, strict=False)
        if verbose:
            LOGGER.info(f"Transferred {len(updated_csd)}/{len(self.state_dict())} items from pretrained weights")


def save_checkpoint(path, model, optimizer, epoch, best_loss, args):
    ckpt = {
        "epoch": epoch,
        "best_loss": best_loss,
        "model": deepcopy(model).half(),
        "optimizer": optimizer.state_dict(),
        "train_args": vars(args),
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    torch.save(ckpt, path)


def save_encoder_state(path, model, decoder_index=29):
    """Save a lightweight state_dict without the reconstruction decoder."""
    prefix = f"model.{decoder_index}."
    state = {k: v.cpu() for k, v in model.float().state_dict().items() if not k.startswith(prefix)}
    torch.save({"state_dict": state, "decoder_removed": True}, path)


def train(args):
    if not args.data:
        raise ValueError("Dental SSL training requires --data or train_dental_ssl(data=...).")
    if args.seed is not None:
        init_seeds(args.seed, deterministic=args.deterministic)
    device = select_device(args.device)
    channels = args.channels
    dataset = UnlabeledOPGDataset(args.data, imgsz=args.imgsz, channels=channels)
    workers = min(args.workers, max((os.cpu_count() or 1) - 1, 0))
    if workers != args.workers:
        LOGGER.warning(f"Reducing workers from {args.workers} to {workers} for this runtime.")
    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=workers,
        pin_memory=device.type != "cpu",
        drop_last=False,
    )
    model = DentalSSLModel(args.model, ch=channels, verbose=True).to(device)
    if args.weights:
        ckpt = torch_load(args.weights, map_location="cpu")
        model.load(ckpt)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    save_dir = Path(args.project) / args.name
    save_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")

    for epoch in range(args.epochs):
        model.train()
        use_ssim = epoch >= args.ssim_start
        criterion = MaskedReconstructionLoss(
            l1_weight=0.8 if use_ssim else 1.0,
            ssim_weight=0.2 if use_ssim else 0.0,
            masked_only=True,
        )
        total = 0.0
        for i, (imgs, _) in enumerate(loader):
            imgs = imgs.to(device, non_blocking=True)
            masked, mask = random_patch_mask(imgs, mask_ratio=args.mask_ratio, patch_size=args.patch)
            with autocast(enabled=args.amp and device.type != "cpu", device=device.type):
                pred = model(masked)
                loss, items = criterion(pred, imgs, mask)
            scaler.scale(loss).backward()
            if (i + 1) % args.accumulate == 0 or (i + 1) == len(loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            total += float(loss.detach())

        mean_loss = total / max(len(loader), 1)
        LOGGER.info(
            f"epoch {epoch + 1}/{args.epochs} loss={mean_loss:.5f} "
            f"l1={float(items[0]):.5f} ssim={float(items[1]):.5f}"
        )
        save_checkpoint(save_dir / "last.pt", model, optimizer, epoch, best_loss, args)
        if mean_loss < best_loss:
            best_loss = mean_loss
            save_checkpoint(save_dir / "best.pt", model, optimizer, epoch, best_loss, args)
            save_encoder_state(save_dir / "best_encoder_state.pt", model, decoder_index=args.decoder_index)

    LOGGER.info(f"SSL training complete. Outputs saved to {save_dir}")


def train_dental_ssl(**kwargs):
    """Notebook-friendly entrypoint for DentalYOLO26 SSL pretraining."""
    args = parse_args(args=[])
    for k, v in kwargs.items():
        if not hasattr(args, k):
            raise AttributeError(f"Unknown dental SSL training argument: {k}")
        setattr(args, k, v)
    return train(args)


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="", help="Folder of unlabeled OPG images or a txt file with image paths.")
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--weights", default="", help="Optional SSL/detection checkpoint to initialize matching layers.")
    parser.add_argument("--project", default="runs/dentalssl")
    parser.add_argument("--name", default="pretrain")
    parser.add_argument("--device", default="")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--channels", type=int, default=1, choices=(1, 3))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--mask-ratio", type=float, default=0.5)
    parser.add_argument("--patch", type=int, default=32)
    parser.add_argument("--ssim-start", type=int, default=20)
    parser.add_argument("--accumulate", type=int, default=1)
    parser.add_argument("--decoder-index", type=int, default=29)
    parser.add_argument("--amp", action="store_true")
    return parser.parse_args(args=args)


if __name__ == "__main__":
    train(parse_args())
