#!/usr/bin/env python3
"""Benchmark inference latency for trained object detection models."""

from __future__ import annotations

# Standard Library
import argparse
import re
import sys
import time
from pathlib import Path

# Third-party Packages
import pandas as pd
import torch

from ultralytics import YOLO
from ultralytics.utils.torch_utils import (
    get_flops,
    get_num_params,
)

# ARGUMENT PARSER
def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark inference latency.")
    parser.add_argument("--project",type=str,required=True,help="Training output directory.",)
    parser.add_argument("--name",type=str,required=True,help="Experiment name.",)
    parser.add_argument("--imgsz",type=int,default=640,help="Inference image size.",)
    parser.add_argument("--batch",type=int,default=1,help="Inference batch size.",)
    parser.add_argument("--device",default="",help="Device string. Default: auto.",)
    parser.add_argument("--warmup",type=int,default=20,help="Warmup iterations.",)
    parser.add_argument("--iters",type=int,default=100,help="Benchmark iterations.",)
    parser.add_argument("--half",action="store_true",help="Use FP16 on CUDA.",)
    return parser.parse_args()

# VALIDATE CONFIG
def validate_config(args):
    # Project
    if args.project.strip() == "":
        print("\nERROR: Project directory is empty")
        sys.exit(1)
    args.project = Path(args.project)
    # Experiment
    if args.name.strip() == "":
        print("\nERROR: Experiment name is empty")
        sys.exit(1)
    args.run_dir = args.project / args.name
    if not args.run_dir.exists():
        print("\nERROR: Experiment directory not found\n")
        print(args.run_dir)
        sys.exit(1)

    # Weight
    args.weight = args.run_dir / "weights" / "best.pt"
    if not args.weight.exists():
        print("\nERROR: best.pt not found\n")
        print(args.weight)
        sys.exit(1)

    # Model Summary
    args.summary_file = args.run_dir / "model_summary.txt"
    if not args.summary_file.exists():
        print("\nERROR: model_summary.txt not found\n")
        print(args.summary_file)
        sys.exit(1)

    # Validation Directory
    args.validation_dir = args.run_dir / "validation"
    args.validation_dir.mkdir(parents=True,exist_ok=True,)

    # Output CSV
    args.output_csv = (
        args.validation_dir
        / "benchmark.csv"
    )
    
# SELECT DEVICE
def select_device(device_arg):
    if device_arg:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if (hasattr(torch.backends, "mps")and torch.backends.mps.is_available()):
        return torch.device("mps")
    return torch.device("cpu")

# SYNCHRONIZE DEVICE
def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif (device.type == "mps"and hasattr(torch, "mps")):
        torch.mps.synchronize()

# LOAD MODEL
def load_model(args):
    model = YOLO(str(args.weight))
    model.model.to(args.device)
    model.model.eval()
    return model

# BENCHMARK LATENCY
def benchmark_latency(model, args):
    net = model.model
    use_half = (args.half and args.device.type == "cuda")
    if use_half:
        net.half()
    dtype = (torch.float16n if use_half else torch.float32)
    x = torch.zeros(
        args.batch,
        3,
        args.imgsz,
        args.imgsz,
        device=args.device,
        dtype=dtype,
    )

    # Warmup
    with torch.inference_mode():
        for _ in range(args.warmup):
            net(x)
        synchronize(args.device)
        # Benchmark
        start = time.perf_counter()
        for _ in range(args.iters):
            net(x)
        synchronize(args.device)
        elapsed = time.perf_counter() - start

    latency_ms = (elapsed * 1000 / args.iter)
    fps = (args.batch * 1000 / latency_ms)
    gpu = (
        torch.cuda.get_device_name(args.device)
        if args.device.type == "cuda"
        else str(args.device).upper()
    )

    return {
        "Latency (ms)": round(latency_ms, 3),
        "FPS": round(fps, 2),
        "GPU": gpu,
    }
# SAVE BENCHMARK
def save_benchmark(row, args):
    df = pd.DataFrame([row])
    df.to_csv(args.output_csv,index=False,)
    print(f"Saved benchmark: {args.output_csv}")


# MAIN
def main():
    args = parse_args()
    validate_config(args)
    args.device = select_device(args.device)
    model = load_model(args)
    row = benchmark_latency(model=model,args=args,)
    row = {
        "Model": args.run_dir.parent.name,
        "Experiment": args.run_dir.name,
        **row,
    }
    save_benchmark(row=row,args=args,)
    print()
    print("Benchmark completed.")
    print(f"Results saved to: {args.output_csv}")


if __name__ == "__main__":
    main()