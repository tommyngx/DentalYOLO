#!/usr/bin/env python3
"""Smoke-test DentalYOLO26 CoordConv model construction, inference, training, and ONNX export."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO  # noqa: E402
from ultralytics.nn.modules import CoordConv  # noqa: E402
from ultralytics.utils.torch_utils import select_device  # noqa: E402


DEFAULT_MODEL = ROOT / "ultralytics/cfg/models/dental26/dental-yolo26_v14.yaml"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="CoordConv YAML or checkpoint.")
    parser.add_argument("--imgsz", type=int, default=640, help="Square dummy/training image size.")
    parser.add_argument("--batch", type=int, default=1, help="Dummy/training batch size.")
    parser.add_argument("--device", default="cpu", help="Device accepted by Ultralytics, e.g. cpu, 0, or mps.")
    parser.add_argument("--data", help="Dataset YAML used only with --train-smoke.")
    parser.add_argument("--train-smoke", action="store_true", help="Run one real training epoch using --data.")
    parser.add_argument("--project", default="runs/coordconv-smoke", help="Smoke-training output directory.")
    parser.add_argument("--name", default="train", help="Smoke-training run name.")
    parser.add_argument("--export-onnx", action="store_true", help="Attempt an Ultralytics ONNX export.")
    return parser.parse_args()


def tensor_shapes(value):
    """Recursively collect tensor shapes from an Ultralytics model output."""
    if isinstance(value, torch.Tensor):
        return [tuple(value.shape)]
    if isinstance(value, dict):
        return [shape for item in value.values() for shape in tensor_shapes(item)]
    if isinstance(value, (list, tuple)):
        return [shape for item in value for shape in tensor_shapes(item)]
    return []


def main():
    args = parse_args()
    model = YOLO(args.model)
    coord_layers = [module for module in model.model.modules() if isinstance(module, CoordConv)]
    if len(coord_layers) != 3:
        raise AssertionError(f"Expected 3 CoordConv layers, found {len(coord_layers)}")

    device = select_device(args.device)
    net = model.model.to(device).eval()
    dummy = torch.zeros(args.batch, 3, args.imgsz, args.imgsz, device=device)
    with torch.inference_mode():
        output = net(dummy)
    shapes = tensor_shapes(output)
    if not shapes:
        raise AssertionError(f"Model returned no tensor outputs: {type(output).__name__}")
    print(f"build=ok coordconv_layers={len(coord_layers)} output_shapes={shapes}")

    if args.train_smoke:
        if not args.data:
            raise ValueError("--train-smoke requires --data /path/to/dataset.yaml")
        model.train(
            data=args.data,
            epochs=1,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=0,
            plots=False,
            project=args.project,
            name=args.name,
            exist_ok=True,
        )

    if args.export_onnx:
        exported = model.export(format="onnx", imgsz=args.imgsz, device=args.device, dynamic=True)
        print(f"onnx_export={exported}")


if __name__ == "__main__":
    main()
