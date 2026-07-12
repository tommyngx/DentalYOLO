#!/usr/bin/env python3
"""Validation Script"""

from __future__ import annotations

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import sys
from pathlib import Path
import torch
from ultralytics import YOLO

# ARGUMENT PARSER
def parse_args():
    parser = argparse.ArgumentParser(description="Validation Script")
    parser.add_argument("--data",type=str,required=True,help="YOLO dataset yaml.")
    parser.add_argument("--project",type=str,required=True,help="Training output directory.")
    parser.add_argument("--name",type=str,required=True,help="Experiment name.")
    parser.add_argument("--batch",type=int,default=16,help="Batch size.")
    parser.add_argument("--imgsz",type=int,default=640,help="Image size.")
    parser.add_argument("--device",type=str,default="0",help="Device (0, cuda:0 or cpu).")
    return parser.parse_args()

# VALIDATE CONFIG
def validate_config(args):
    # Dataset
    dataset_path = Path(args.data)
    if not dataset_path.exists():
        print("\nERROR: Dataset yaml not found\n")
        print(dataset_path)
        sys.exit(1)
    # Project
    if args.project.strip() == "":
        print("\nERROR: project directory is empty")
        sys.exit(1)
    # Experiment
    if args.name.strip() == "":
        print("\nERROR: experiment name is empty")
        sys.exit(1)
    # Best Weight
    args.weight = (Path(args.project) / args.name / "weights" / "best.pt")
    if not args.weight.exists():
        print("\nERROR: best.pt not found\n")
        print(args.weight)
        sys.exit(1)
    # Batch
    if args.batch <= 0:
        print("\nERROR: batch must be > 0")
        sys.exit(1)
    # Image Size
    if args.imgsz <= 0:
        print("\nERROR: imgsz must be > 0")
        sys.exit(1)
    # Device
    if args.device != "cpu":
        if args.device.startswith("cuda:"):
            device_index = int(args.device.split(":")[1])
        else:
            device_index = int(args.device)
        if not torch.cuda.is_available():
            print("\nERROR: CUDA is not available")
            sys.exit(1)
        if device_index >= torch.cuda.device_count():
            print(f"\nERROR: CUDA device {device_index} does not exist")
            sys.exit(1)
            
# VALIDATION
def validate(args):
    print(f"Model      : {args.weight}")
    print(f"Dataset    : {args.data}")
    print(f"Experiment : {args.name}")
    print()
    model = YOLO(args.weight)
    results = model.val(
        data=args.data,
        split="test",
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=f"{args.name}/validation",
        save_json=True,
        plots=True,
        exist_ok=True,)
    save_dir = getattr(results, "save_dir", None)
    if save_dir is None and hasattr(model, "validator"):
        save_dir = getattr(model.validator, "save_dir", None)
    args.result_dir = str(save_dir)
    return results

# SHOW METRICS
def show_metrics(results):
    print()
    print("Validation Results")
    print("-" * 40)
    metrics = [
        ("mAP50", results.box.map50),
        ("mAP50-95", results.box.map),
        ("Precision", results.box.mp),
        ("Recall", results.box.mr),
    ]
    for name, value in metrics:
        print(f"{name:<12}: {value:.4f}")

# MAIN
def main():
    args = parse_args()
    validate_config(args)
    results = validate(args)
    show_metrics(results)
    print()
    print(f"Results saved to: {args.result_dir}")

if __name__ == "__main__":
    main()