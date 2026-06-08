#!/usr/bin/env python3
"""Benchmark DentalYOLO26 architecture variants."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
ULTRALYTICS_SRC = ROOT / "ultralytics"
if str(ULTRALYTICS_SRC) not in sys.path:
    sys.path.insert(0, str(ULTRALYTICS_SRC))

from ultralytics import YOLO  # noqa: E402
from ultralytics.utils.torch_utils import get_flops, get_num_params  # noqa: E402


DEFAULT_MODELS = [
    "ultralytics/ultralytics/cfg/models/26/yolo26.yaml",
    "ultralytics/ultralytics/cfg/models/dental26/dental-yolo26_v9.yaml",
    "ultralytics/ultralytics/cfg/models/dental26/dental-yolo26_v10.yaml",
    "ultralytics/ultralytics/cfg/models/dental26/dental-yolo26_v11.yaml",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="YAML configs or checkpoint paths.")
    parser.add_argument("--weights", nargs="*", default=None, help="Optional trained checkpoints, one per model.")
    parser.add_argument("--pretrained", default=None, help="Optional checkpoint partially loaded into each YAML model.")
    parser.add_argument("--data", default=None, help="Optional dataset YAML for validation metrics.")
    parser.add_argument("--imgsz", type=int, default=640, help="Square image size for info, latency, and validation.")
    parser.add_argument("--batch", type=int, default=1, help="Batch size for latency and validation.")
    parser.add_argument("--warmup", type=int, default=20, help="Warmup forward passes before timing.")
    parser.add_argument("--iters", type=int, default=100, help="Timed forward passes.")
    parser.add_argument("--device", default="", help="Device string: cpu, cuda:0, mps, or empty for auto.")
    parser.add_argument("--half", action="store_true", help="Use FP16 latency timing on CUDA.")
    parser.add_argument("--threads", type=int, default=0, help="CPU thread count override.")
    parser.add_argument("--split", default="val", help="Dataset split for validation.")
    parser.add_argument("--verbose-info", action="store_true", help="Print full model.info() output.")
    return parser.parse_args()


def select_device(device_arg):
    if device_arg:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def load_model(model_path, weight_path=None, pretrained=None):
    source = weight_path or model_path
    model = YOLO(str(source))
    if pretrained and not weight_path:
        model.load(pretrained)
    return model


def benchmark_latency(model, device, imgsz, batch, warmup, iters, half):
    net = model.model.to(device).eval()
    use_half = half and device.type == "cuda"
    if use_half:
        net.half()
    dtype = torch.float16 if use_half else torch.float32
    x = torch.zeros(batch, 3, imgsz, imgsz, device=device, dtype=dtype)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    with torch.inference_mode():
        for _ in range(warmup):
            net(x)
        synchronize(device)
        start = time.perf_counter()
        for _ in range(iters):
            net(x)
        synchronize(device)
        elapsed = time.perf_counter() - start

    latency_ms = elapsed * 1000.0 / max(iters, 1)
    fps = batch * 1000.0 / latency_ms if latency_ms > 0 else 0.0
    memory_mb = torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else 0.0
    return latency_ms, fps, memory_mb


def validate_model(model, data, device, imgsz, batch, split):
    metrics = model.val(
        data=data,
        imgsz=imgsz,
        batch=batch,
        device=str(device),
        split=split,
        plots=False,
        verbose=False,
    )
    if hasattr(metrics, "box"):
        return metrics.box.map50, metrics.box.map, metrics.box.mp, metrics.box.mr
    results = getattr(metrics, "results_dict", {})
    return (
        results.get("metrics/mAP50(B)", 0.0),
        results.get("metrics/mAP50-95(B)", 0.0),
        results.get("metrics/precision(B)", 0.0),
        results.get("metrics/recall(B)", 0.0),
    )


def format_value(value, precision=3):
    if value is None:
        return "-"
    return f"{value:.{precision}f}"


def print_table(rows):
    headers = [
        "model",
        "params(M)",
        "GFLOPs",
        "latency(ms)",
        "FPS",
        "GPU_mem(MB)",
        "mAP50",
        "mAP50-95",
        "precision",
        "recall",
    ]
    widths = [len(h) for h in headers]
    table_rows = []
    for row in rows:
        values = [
            row["name"],
            format_value(row["params_m"]),
            format_value(row["gflops"]),
            format_value(row["latency_ms"]),
            format_value(row["fps"], 2),
            format_value(row["gpu_mem_mb"], 1),
            format_value(row["map50"]),
            format_value(row["map"]),
            format_value(row["precision"]),
            format_value(row["recall"]),
        ]
        table_rows.append(values)
        widths = [max(width, len(value)) for width, value in zip(widths, values)]

    header_line = " | ".join(h.ljust(width) for h, width in zip(headers, widths))
    sep_line = " | ".join("-" * width for width in widths)
    print(header_line)
    print(sep_line)
    for values in table_rows:
        print(" | ".join(value.ljust(width) for value, width in zip(values, widths)))


def main():
    args = parse_args()
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    if args.weights and len(args.weights) not in {0, len(args.models)}:
        raise ValueError("--weights must be omitted or contain exactly one checkpoint per model")

    device = select_device(args.device)
    rows = []
    for i, model_path in enumerate(args.models):
        weight_path = args.weights[i] if args.weights else None
        model = load_model(model_path, weight_path=weight_path, pretrained=args.pretrained)
        model.model.to(device)
        model.model.info(verbose=args.verbose_info, imgsz=args.imgsz)

        params_m = get_num_params(model.model) / 1e6
        gflops = get_flops(model.model, args.imgsz)
        latency_ms, fps, gpu_mem_mb = benchmark_latency(
            model,
            device=device,
            imgsz=args.imgsz,
            batch=args.batch,
            warmup=args.warmup,
            iters=args.iters,
            half=args.half,
        )

        map50 = map5095 = precision = recall = None
        if args.data:
            map50, map5095, precision, recall = validate_model(
                model,
                data=args.data,
                device=device,
                imgsz=args.imgsz,
                batch=args.batch,
                split=args.split,
            )

        rows.append(
            {
                "name": Path(model_path).stem,
                "params_m": params_m,
                "gflops": gflops,
                "latency_ms": latency_ms,
                "fps": fps,
                "gpu_mem_mb": gpu_mem_mb,
                "map50": map50,
                "map": map5095,
                "precision": precision,
                "recall": recall,
            }
        )

    print_table(rows)


if __name__ == "__main__":
    main()
