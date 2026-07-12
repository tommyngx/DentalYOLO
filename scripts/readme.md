# Training Pipeline

## General Purpose

This repository provides a lightweight and reproducible training pipeline for object detection models based on the Ultralytics framework. The workflow is divided into three independent stages: model training, test-set validation, and COCO evaluation. Each stage produces standardized artifacts that are used by the following stage, making experiments easy to reproduce, benchmark, and compare.

```text
                 data.yaml
                     │
                     ▼
              train.py
                     │
                     ▼
         best.pt + training artifacts
                     │
                     ▼
             validate.py
                     │
                     ▼
          predictions.json
                     │
                     ▼
          coco_evaluate.py
                     │
                     ▼
      test_coco_gt.json
      coco_metrics.csv
```

---

# 1. train.py

## General Purpose

Train an object detection model and save all training artifacts for later evaluation.

## Short Description

### Input

- Model name (`--model`)
- Dataset configuration (`data.yaml`)
- Training settings (epochs, batch size, image size, seed)
- Output project directory
- Experiment name

### Output

```text
output/
└── <experiment>/
    ├── weights/
    │   ├── best.pt
    │   └── last.pt
    ├── results.csv
    ├── results.png
    ├── model_summary.txt
    ├── config.yaml
    ├── environment.txt
    └── train_command.txt
```

## Usage

```bash
python scripts/train.py \
    --model yolov8m \
    --data /path/to/data.yaml \
    --epochs 100 \
    --batch 16 \
    --imgsz 640 \
    --seed 2026 \
    --project output \
    --name exp1
```

Optional:

```bash
--show-best
```

Display the best validation metrics after training.

---

# 2. validate.py

## General Purpose

Evaluate the trained model on the test dataset and generate prediction results.

## Short Description

### Input

- Dataset configuration (`data.yaml`)
- Project directory
- Experiment name

The script automatically loads

```text
weights/best.pt
```

from the experiment folder.

### Output

```text
output/
└── <experiment>/
    └── validation/
        ├── predictions.json
        ├── confusion_matrix.png
        ├── PR_curve.png
        ├── F1_curve.png
        ├── P_curve.png
        └── R_curve.png
```

## Usage

```bash
python scripts/validate.py \
    --data /path/to/data.yaml \
    --project output \
    --name exp1 \
    --batch 16 \
    --imgsz 640
```

---

# 3. coco_evaluate.py

## General Purpose

Compute COCO evaluation metrics from the validation predictions.

## Short Description

### Input

- Dataset configuration (`data.yaml`)
- Project directory
- Experiment name

The script automatically loads:

- `validation/predictions.json`
- `model_summary.txt`

If `validation/test_coco_gt.json` does not exist, it will be generated automatically from the YOLO annotations.

### Output

```text
output/
└── <experiment>/
    └── validation/
        ├── test_coco_gt.json
        └── coco_metrics.csv
```

The generated CSV contains:

- Model
- Experiment
- AP
- AP50
- AP75
- AP Small / Medium / Large
- AR
- Parameters
- GFLOPs

## Usage

```bash
python scripts/coco_evaluate.py \
    --data /path/to/data.yaml \
    --project output \
    --name exp1
```

---

# Complete Workflow

```bash
# Step 1: Train
python scripts/train.py ...

# Step 2: Validate
python scripts/validate.py ...

# Step 3: COCO Evaluation
python scripts/coco_evaluate.py ...
```