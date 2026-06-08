# DentalYOLO

**DentalYOLO** is a customized object detection framework for dental X-ray images.  
This repository is built on a forked Ultralytics source codebase and focuses on improving detection performance for small and subtle dental abnormalities.

## Repository Structure

```text
DentalYOLO/
├── ultralytics/        # Forked Ultralytics source code
├── configs/            # Model and dataset configuration files
├── scripts/            # Training, validation, prediction, and export scripts
├── experiments/        # Experiment outputs, logs, and results
└── README.md
```

## DentalYOLO26 v9-v11 Attention Variants

These variants are dental-specific, selectively placed attention adaptations of YOLO26 for panoramic radiograph analysis. They do not claim new attention mechanisms; they reuse lightweight attention ideas in positions motivated by small, dense, low-contrast OPG findings and long dental arch context.

- `ultralytics/ultralytics/cfg/models/dental26/dental-yolo26_v9.yaml`: Triplet Attention is inserted only in the neck after the P3 `C3k2` fusion block. This targets small-object recall with minimal parameter growth.
- `ultralytics/ultralytics/cfg/models/dental26/dental-yolo26_v10.yaml`: one `ArchLSKA` block is inserted in the late backbone after the deepest `C3k2` stage and before `SPPF`. It uses horizontal-biased asymmetric depthwise kernels for dental arch context.
- `ultralytics/ultralytics/cfg/models/dental26/dental-yolo26_v11.yaml`: one `DAABLite` block is inserted in the late backbone, and one extra `TripletAttention` block is inserted at the P3 neck output for small-object fusion.

The baseline `ultralytics/ultralytics/cfg/models/26/yolo26.yaml` remains unchanged. All v9-v11 configs keep the YOLO26 `Detect` module and three-scale detection outputs unchanged.

Run architecture and latency checks:

```bash
python scripts/benchmark_dental_yolo26.py --imgsz 640 --batch 1
```

Run validation metrics when trained checkpoints and a dataset YAML are available:

```bash
python scripts/benchmark_dental_yolo26.py \
  --models ultralytics/ultralytics/cfg/models/dental26/dental-yolo26_v9.yaml \
  --weights runs/detect/train/weights/best.pt \
  --data path/to/dental-opg.yaml
```
