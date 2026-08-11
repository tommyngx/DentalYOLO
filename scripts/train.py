#!/usr/bin/env python3
"""
DentalYOLO26 Training Script
"""

from __future__ import annotations

import argparse
import contextlib
import platform
import sys
import time
from datetime import datetime

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
    
    
import pandas as pd
import torch
import ultralytics
import yaml
from ultralytics import YOLO, RTDETR

MODEL_REGISTRY = {

    # YOLOv8
    "yolov8n": "yolov8n.pt",
    "yolov8s": "yolov8s.pt",
    "yolov8m": "yolov8m.pt",
    "yolov8l": "yolov8l.pt",
    "yolov8x": "yolov8x.pt",

    # YOLO12
    "yolo12n": "yolo12n.pt",
    "yolo12s": "yolo12s.pt",
    "yolo12m": "yolo12m.pt",
    "yolo12l": "yolo12l.pt",
    "yolo12x": "yolo12x.pt",
    
    
    # YOLO26
    "yolo26n": "yolo26n.pt",
    "yolo26s": "yolo26s.pt",
    "yolo26m": "yolo26m.pt",
    "yolo26l": "yolo26l.pt",
    "yolo26x": "yolo26x.pt",
    
    # RT-DETR
    # "rtdetr-l": "rtdetr-l.pt",
    # "rtdetr-x": "rtdetr-x.pt",
    "rtdetr-r18": "ultralytics/cfg/models/rt-detr/rtdetr-r18.yaml",
    "rtdetr-r34": "ultralytics/cfg/models/rt-detr/rtdetr-r34.yaml",
    "rtdetr-resnet50": "ultralytics/cfg/models/rt-detr/rtdetr-resnet50.yaml",
    "rtdetr-resnet101": "ultralytics/cfg/models/rt-detr/rtdetr-resnet101.yaml",
    "rtdetr-l": "ultralytics/cfg/models/rt-detr/rtdetr-l.yaml",
    "rtdetr-x": "ultralytics/cfg/models/rt-detr/rtdetr-x.yaml",

}
DENTAL_YOLO26_DIR = "ultralytics/cfg/models/dental26"
for scale in ["n", "s", "m", "l", "x"]:
    MODEL_REGISTRY[f"dental-yolo26{scale}_v15"] = {
        "model": f"{DENTAL_YOLO26_DIR}/dental-yolo26{scale}_v15.yaml",
        "pretrained": f"yolo26{scale}.pt",
}
    
# ARGUMENT PARSER
def parse_args():
    parser = argparse.ArgumentParser(description="Training Script")
    parser.add_argument("--model",type=str,required=True,help="Model name in MODEL_REGISTRY.")
    parser.add_argument("--data",type=str,required=True,help="YOLO dataset yaml.")
    parser.add_argument("--epochs",type=int,required=True)
    parser.add_argument("--batch",type=int,required=True)
    parser.add_argument("--imgsz",type=int,default=640)
    parser.add_argument("--seed",type=int,required=True)
    parser.add_argument("--project",type=str,required=True)
    parser.add_argument("--name",type=str,required=True)
    parser.add_argument("--device",type=str,default="0")
    parser.add_argument("--show-best",action="store_true",help="Show best validation metrics after training.")
    return parser.parse_args()


# VALIDATE CONFIG
def validate_config(args):
    # Model
    if args.model not in MODEL_REGISTRY:
        print(f"\nERROR: Unknown model '{args.model}'\n")
        print("Available models:")
        for name in MODEL_REGISTRY:
            print(f"  - {name}")
        sys.exit(1)
    model_source = MODEL_REGISTRY[args.model]
    # DentalYOLO26
    if isinstance(model_source, dict):
        model_path = Path(model_source["model"])
        if not model_path.exists():
            print("\nERROR: Model yaml not found\n")
            print(model_path)
            sys.exit(1)
    # Custom yaml models (RT-DETR, ...)
    elif model_source.endswith(".yaml"):
        yaml_path = Path(model_source)
        if not yaml_path.exists():
            print("\nERROR: Model yaml not found\n")
            print(yaml_path)
            sys.exit(1)
    # Dataset
    if not Path(args.data).exists():
        print("\nERROR: Dataset yaml not found\n")
        print(args.data)
        sys.exit(1)
    # Epoch
    if args.epochs <= 0:
        print("\nERROR: epochs must be > 0")
        sys.exit(1)
    # Batch
    if args.batch <= 0:
        print("\nERROR: batch must be > 0")
        sys.exit(1)
    # Image Size
    if args.imgsz <= 0:
        print("\nERROR: imgsz must be > 0")
        sys.exit(1)
    # Seed
    if args.seed is None:
        print("\nERROR: seed is required")
        sys.exit(1)
    # Project
    if args.project.strip() == "":
        print("\nERROR: project directory is empty")
        sys.exit(1)
    # Experiment
    if args.name.strip() == "":
        print("\nERROR: experiment name is empty")
        sys.exit(1)
    # Device
    if args.device != "cpu":
        if args.device.startswith("cuda:"):
            index = int(args.device.split(":")[1])
        else:
            index = int(args.device)
        if not torch.cuda.is_available():
            print("\nERROR: CUDA is not available")
            sys.exit(1)
        if index >= torch.cuda.device_count():
            print(f"\nERROR: CUDA device {index} does not exist")
            sys.exit(1)
            
def build_model(model_name):
    model_info = MODEL_REGISTRY[model_name]
    if isinstance(model_info, dict):
        model = YOLO(model_info["model"])
        model.load(model_info["pretrained"])
        print(f"Loaded pretrained weight: {model_info['pretrained']}")
        return model
    if isinstance(model_info, str):
        if model_name.startswith("rtdetr"):
            return RTDETR(model_info)
        return YOLO(model_info)
    raise TypeError(f"Unsupported MODEL_REGISTRY entry for '{model_name}'")

# TRAIN
def train(args):
    print(f"Model      : {args.model}")
    print(f"Dataset    : {args.data}")
    print(f"Experiment : {args.name}")
    print()
    model = build_model(args.model)
    
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        seed=args.seed,
        device=args.device,
        project=args.project,
        name=args.name,
        exist_ok=True,
        )
    save_dir = getattr(results, "save_dir", None)
    if save_dir is None and hasattr(model, "trainer"):
        save_dir = getattr(model.trainer, "save_dir", None)
    args.result_dir = str(save_dir)
    print(f"\nResult directory : {args.result_dir}")
    return model, results

BEST_METRIC = "metrics/mAP50-95(B)"

# SAVE CONFIG
def save_config(args):
    config_path = Path(args.result_dir) / "config.yaml"
    with open(config_path, "w") as f:
        yaml.safe_dump(vars(args),f,sort_keys=False,)
    print(f"Saved config: {config_path}")

def save_model_summary(model, args):
    summary_path = Path(args.result_dir) / "model_summary.txt"
    layers, params, grads, gflops = model.info(detailed=True)
    with open(summary_path, "w") as f:
        f.write(f"{layers} layers\n")       
        f.write(f"{params:,} parameters\n")
        f.write(f"{grads:,} gradients\n")
        f.write(f"{gflops:.2f} GFLOPs\n")
    print(f"Saved model summary: {summary_path}")
    
# SAVE ENVIRONMENT
def save_environment(args):
    env_path = Path(args.result_dir) / "environment.txt"
    with open(env_path, "w") as f:
        f.write(f"Date          : {datetime.now()}\n")
        f.write(f"Platform      : {platform.platform()}\n")
        f.write(f"Python        : {platform.python_version()}\n")
        f.write(f"Torch         : {torch.__version__}\n")
        f.write(f"CUDA          : {torch.version.cuda}\n")
        f.write(f"Ultralytics   : {ultralytics.__version__}\n")
        if torch.cuda.is_available():
            if args.device == "cpu":
                device_index = 0
            elif str(args.device).startswith("cuda:"):
                device_index = int(str(args.device).split(":")[1])
            else:
                device_index = int(args.device)
            f.write(f"GPU           : {torch.cuda.get_device_name(device_index)}\n")
    print(f"Saved environment: {env_path}")

# SAVE TRAIN COMMAND
def save_train_command(args):
    command_path = Path(args.result_dir) / "train_command.txt"
    command = "python " + " ".join(sys.argv)
    with open(command_path, "w") as f:
        f.write(command)
    print(f"Saved command: {command_path}")

# SHOW BEST METRICS
def show_best(args):
    csv_path = Path(args.result_dir) / "results.csv"
    if not csv_path.exists():
        return
    df = pd.read_csv(csv_path)
    if BEST_METRIC not in df.columns:
        return
    best = df.loc[df[BEST_METRIC].idxmax()]
    print("\nBest Validation")
    print("-" * 40)
    metrics = [
        ("Epoch", "epoch"),
        ("mAP50", "metrics/mAP50(B)"),
        ("mAP50-95", "metrics/mAP50-95(B)"),
        ("Precision", "metrics/precision(B)"),
        ("Recall", "metrics/recall(B)")
    ]
    for title, column in metrics:
        if column not in df.columns:
            continue
        value = best[column]
        if title == "Epoch":
            print(f"{title:<12}: {int(value)}")
        else:
            print(f"{title:<12}: {value:.4f}")


def main():
    start_time = time.perf_counter()
    args = parse_args()
    validate_config(args)
    model, results = train(args)
    save_config(args)
    save_model_summary(model, args)
    save_environment(args)
    save_train_command(args)
    if args.show_best:
        show_best(args)
    elapsed = time.perf_counter() - start_time
    print(f"\nTraining finished in {elapsed / 60:.2f} minutes")
    print(f"Results saved to: {args.result_dir}")

if __name__ == "__main__":
    main()