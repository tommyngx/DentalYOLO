#!/usr/bin/env python3
"""COCO Evaluation Script"""

from __future__ import annotations



import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd
import yaml
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


# ARGUMENT PARSER
def parse_args():
    parser = argparse.ArgumentParser(description="COCO Evaluation Script")
    parser.add_argument("--data",type=str,required=True,help="YOLO dataset yaml.")
    parser.add_argument("--project",type=str,required=True,help="Training output directory.")
    parser.add_argument("--name",type=str,required=True,help="Experiment name.")
    return parser.parse_args()


# VALIDATE CONFIG
def validate_config(args):
    # Dataset yaml
    args.data = Path(args.data)
    if not args.data.exists():
        print("\nERROR: Dataset yaml not found\n")
        print(args.data)
        sys.exit(1)
        
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
        
    # Validation directory
    args.validation_dir = args.run_dir / "validation"
    if not args.validation_dir.exists():
        print("\nERROR: Validation directory not found\n")
        print(args.validation_dir)
        sys.exit(1)

    # predictions.json
    args.pred_json = args.validation_dir / "predictions.json"
    if not args.pred_json.exists():
        print("\nERROR: predictions.json not found\n")
        print(args.pred_json)
        sys.exit(1)

    # model_summary.txt
    args.summary_file = args.run_dir / "model_summary.txt"
    if not args.summary_file.exists():
        print("\nERROR: model_summary.txt not found\n")
        print(args.summary_file)
        sys.exit(1)

    # Output files
    args.gt_json = args.validation_dir / "test_coco_gt.json"
    args.metrics_csv = args.validation_dir / "coco_metrics.csv"
    
# YOLO TO COCO GROUND TRUTH
def yolo_to_coco_gt(cfg, split="test"):
    root = Path(cfg["path"])
    img_dir = root / cfg[split]
    lbl_dir = root / cfg[split].replace("images", "labels")
    images = []
    annotations = []
    categories = []
    for cid, cname in cfg["names"].items():
        categories.append({
            "id": int(cid),
            "name": str(cname)
        })

    ann_id = 1
    img_files = sorted(
        p for p in img_dir.glob("*")
        if p.suffix.lower() in {".jpg",".jpeg",".png",".bmp"}
    )

    for p in img_files:
        with Image.open(p) as im:
            W, H = im.size
        # Keep image_id as string to match Ultralytics predictions.json
        img_id = str(p.stem)
        images.append({
            "id": img_id,
            "file_name": p.name,
            "width": W,
            "height": H,
        })

        lp = lbl_dir / f"{p.stem}.txt"
        if not lp.exists():
            continue

        for ln in lp.read_text().strip().splitlines():
            parts = ln.split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            cx, cy, bw, bh = map(float, parts[1:5])
            x = (cx - bw / 2) * W
            y = (cy - bh / 2) * H
            w = bw * W
            h = bh * H
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": cls,
                "bbox": [x, y, w, h],
                "area": w * h,
                "iscrowd": 0,
            })
            ann_id += 1
    return {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }


# ENSURE TEST COCO GROUND TRUTH
def ensure_test_coco_gt(args, cfg):
    if args.gt_json.exists():
        try:
            with open(args.gt_json, "r") as f:
                gt = json.load(f)
            required_keys = {
                "images",
                "annotations",
                "categories"
            }
            if required_keys.issubset(gt.keys()):
                print(f"Using existing COCO GT: {args.gt_json}")
                return args.gt_json
            print("Existing COCO GT is invalid. Regenerating...")
        except Exception:
            print("Cannot read existing COCO GT. Regenerating...")
    else:
        print("Generating test_coco_gt.json...")
    coco_gt = yolo_to_coco_gt(cfg,split="test",)
    with open(args.gt_json, "w") as f:
        json.dump(coco_gt, f)
    print(
        f"Saved COCO GT: {args.gt_json}\n"
        f"Images      : {len(coco_gt['images'])}\n"
        f"Annotations : {len(coco_gt['annotations'])}\n"
        f"Categories  : {len(coco_gt['categories'])}"
    )

    if len(coco_gt["images"]) > 0:
        print(
            f"Sample image_id: "
            f"{coco_gt['images'][0]['id']!r} "
            f"(type: {type(coco_gt['images'][0]['id']).__name__})"
        )
    return args.gt_json

# EXTRACT MODEL STATS
def extract_model_stats(summary_file):
    if not summary_file.exists():
        return None, None
    text = summary_file.read_text()
    try:
        param_match = re.search(r"([\d,]+)\s+parameters",text,re.IGNORECASE,)
        gflops_match = re.search(r"([\d.]+)\s+GFLOPs",text,re.IGNORECASE,)

        params = None
        gflops = None
        if param_match:
            params = int(param_match.group(1).replace(",", ""))
        if gflops_match:
            gflops = float(gflops_match.group(1))
        return params, gflops

    except Exception as e:
        print(f"Cannot parse {summary_file}: {e}")
        return None, None


# COCO EVALUATION
def coco_evaluate(gt_json, pred_json, label):
    with open(gt_json) as f:
        gt_data = json.load(f)
    with open(pred_json) as f:
        pred_data = json.load(f)
    # Match image_id with COCO GT
    for p in pred_data:
        p["image_id"] = str(p["image_id"])
    # Filter low confidence predictions
    pred_data = [
        p for p in pred_data
        if p["score"] > 0.01
    ]

    # Convert category_id
    # Prediction : 1-based
    # Ground Truth : 0-based
    for p in pred_data:
        p["category_id"] -= 1
    print(f"[{label}] Predictions: {len(pred_data)}")
    tmp_pred = (Path(gt_json).parent/ f"_tmp_pred_{label}.json")
    tmp_pred.write_text(json.dumps(pred_data))
    coco_gt = COCO(str(gt_json))
    coco_dt = coco_gt.loadRes(str(tmp_pred))
    evaluator = COCOeval(
        coco_gt,
        coco_dt,
        iouType="bbox",
    )

    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    tmp_pred.unlink()
    stats = evaluator.stats
    return {
        "AP": stats[0],
        "AP50": stats[1],
        "AP75": stats[2],
        "AP_small": stats[3],
        "AP_medium": stats[4],
        "AP_large": stats[5],
        "AR_1": stats[6],
        "AR_10": stats[7],
        "AR_100": stats[8],
        "AR_small": stats[9],
        "AR_medium": stats[10],
        "AR_large": stats[11],
    }
    
# EVALUATE EXPERIMENT
def evaluate(args):
    metrics = coco_evaluate(
        gt_json=args.gt_json,
        pred_json=args.pred_json,
        label=args.name,
    )
    params, gflops = extract_model_stats(args.summary_file)
    row = {
        "Model": args.run_dir.parent.name,
        "Experiment": args.run_dir.name,
        **metrics,
        "Params": params,
        "GFLOPs": gflops,
    }
    return row


# SAVE METRICS
def save_metrics(row, args):
    df = pd.DataFrame([row])
    df.to_csv(args.metrics_csv,index=False,)
    print()
    print(f"Saved COCO metrics: {args.metrics_csv}")
    return df


# MAIN
def main():
    args = parse_args()
    validate_config(args)
    # Read dataset yaml 
    with open(args.data, "r") as f:
        cfg = yaml.safe_load(f)
    # Build COCO Ground Truth
    ensure_test_coco_gt(args,cfg,)
    # COCO Evaluation
    row = evaluate(args)
    save_metrics(row,args,)
    print()
    print("COCO evaluation completed.")
    print(f"Results saved to: {args.metrics_csv}")

if __name__ == "__main__":
    main()