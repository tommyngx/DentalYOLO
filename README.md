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

## DentalYOLO26 v16 (recommended)

v16 keeps the three v15 ideas (ECA channel gate at P3, stable slot attention at P4, coordinate injection before
Detect) but re-engineers them so that fine-tuning really starts from the YOLO26 checkpoint:

| | v15 | v16 |
|---|---|---|
| How modules are added | inserted as extra layers, so Detect moves from index 23 to 27 | wrap the YOLO26 layer they replace: `C3k2ECAv2`@16, `C3k2Slot`@19, `DetectCoord`@23 |
| Weights that load from `yolo26{scale}.pt` | ~63% of parameters; the whole Detect head, the P5 `C3k2` and the P4→P5 `Conv` start random | 100% of YOLO26 tensors (708/708 for n/s, 768/768 for m) |
| Behaviour at initialisation | features entering Detect differ from YOLO26 by 2-8x | bit-identical predictions to YOLO26; every new branch is zero-gated |
| Extra parameters (s scale) | +0.54M (0.35M of it in the three CoordConv 1x1 convs) | +0.20M: ECA 0, slot +0.19M, coord +3.6k (see below) |
| Inference after `fuse()` | CoordConv conv+BN fused | coord conv+BN fused; slot/ECA unchanged. CPU latency within noise of YOLO26 |

Module notes:

- `C3k2ECAv2` (P3): ECA gate `2·sigmoid(conv1d(GAP(x)))` with a zero-initialised conv, so the gate is exactly 1 at
  init (classic ECA halves every channel). Adds 3 parameters.
- `C3k2Slot` (P4): stock `C3k2` followed by the v13/v15 `C2StableSlot` branch; the branch's final GroupNorm gamma is
  zeroed so the block equals `C3k2` at init.
- `DetectCoord` (head): `x + BN₀(conv1x1([xx, yy]))` on each scale, i.e. a learned per-channel positional bias. A
  1x1 CoordConv over `[x, xx, yy]` is linear, so it equals this positional term plus a channel mixing the Detect
  head already performs; dropping the mixing removes >99% of the parameters with the same positional
  expressiveness. `DetectCoord(..., full=True)` restores the v15-style conv over `[x, xx, yy]` for comparison.

Configs: `ultralytics/cfg/models/dental26/dental-yolo26_v16.yaml` plus ablations `v16a` (ECA), `v16b` (slot),
`v16c` (coord), `v16d` (ECA+slot), `v16e` (ECA+coord), `v16f` (slot+coord). Modules live in
`ultralytics/nn/modules/dental_v16.py`. Put the scale letter in the name you pass:

```python
from ultralytics import YOLO
model = YOLO("dental-yolo26s_v16.yaml").load("yolo26s.pt")  # Transferred 708/708 items
model.train(data="data.yaml", epochs=100, imgsz=640, batch=8, seed=2026)
```

or `python scripts/train.py --model dental-yolo26s_v16 ...`, which loads the matching `yolo26s.pt` itself.

### Evaluation protocol

The OPG val and test splits have 23 images each; mAP50-95 of one run moves by 3-5 points between neighbouring
epochs, so a single run cannot separate two models. Train the baseline and each variant with the same recipe over
at least 3 seeds and compare mean ± std:

```bash
for M in yolo26s dental-yolo26s_v16 dental-yolo26s_v16a dental-yolo26s_v16b dental-yolo26s_v16c; do
  for SEED in 1 2 3; do
    python scripts/train.py --model $M --data data.yaml --epochs 100 --batch 8 --imgsz 640 --seed $SEED \
        --project output/dental_opg_2024/$M --name seed$SEED
    python scripts/validate.py --data data.yaml --project output/dental_opg_2024/$M --name seed$SEED
    python scripts/coco_evaluate.py --data data.yaml --project output/dental_opg_2024/$M --name seed$SEED
  done
done
python scripts/aggregate_results.py --root output/dental_opg_2024 --baseline yolo26s
```

Keep a module only if its ablation beats the baseline mean by more than the baseline's std.

## DentalYOLO26 v9-v11 Attention Variants

These variants are dental-specific, selectively placed attention adaptations of YOLO26 for panoramic radiograph analysis. They do not claim new attention mechanisms; they reuse lightweight attention ideas in positions motivated by small, dense, low-contrast OPG findings and long dental arch context.

- `ultralytics/ultralytics/cfg/models/dental26/dental-yolo26_v9.yaml`: one `ArchLSKA` block is inserted in the late backbone after the deepest `C3k2` stage and before `SPPF`. It uses horizontal-biased asymmetric depthwise kernels for dental arch context.
- `ultralytics/ultralytics/cfg/models/dental26/dental-yolo26_v10.yaml`: one `DAABLite` block is inserted in the late backbone, and one extra `TripletAttention` block is inserted at the P3 neck output for small-object fusion.
- `ultralytics/ultralytics/cfg/models/dental26/dental-yolo26_v11.yaml`: fast DAAB-lite removes the extra P3 neck Triplet block and uses smaller DAAB-lite strip kernels for lower latency.

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
