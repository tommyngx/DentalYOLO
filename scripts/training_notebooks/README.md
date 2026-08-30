# DentalYOLO26 Research Code

This repository contains the research implementation and experimental notebooks for dental image object detection.

The repository includes three main notebooks covering different model families:

* `dental-yolo26.ipynb` — DentalYOLO26 models
* `yolo_family.ipynb` — YOLO model family and RT-DETR models
* `rfdetr.ipynb` — RF-DETR models

The notebooks preserve the experimental workflow used in the research study. The code structure and processing pipeline are not intended to be modified for basic usage.

---

## 1. Overview

The notebooks follow a similar experimental workflow:

```text
Environment Setup
        ↓
Configuration
        ↓
Dataset Preparation
        ↓
Model Setup
        ↓
Training
        ↓
Validation / Evaluation
        ↓
Benchmark / Results
```

Each notebook contains its own stages and tasks.

For normal use, users only need to:

1. Prepare the dataset.
2. Set the project path.
3. Set the dataset name.
4. Set the dataset ZIP path if required.
5. Select the model.
6. Configure the training parameters.
7. Run the notebook sequentially.

> **Important:** The notebook structure and research implementation should be kept unchanged when reproducing the reported experiments.

---

# 2. Environment

The notebooks are designed to run in a Python environment with GPU support.

Before running an experiment, make sure that:

* Python and the required packages are installed.
* The required model libraries are available.
* CUDA/GPU is available when GPU training is intended.
* The dataset is prepared in the expected YOLO-compatible format.

The notebooks include environment and dependency checks where required.

---

# 3. Project Directory

The project directory is the main location used to store datasets, experiment outputs, and results.

The configuration follows this structure:

```text
Your-Project/
├── datasets/
│   ├── dataset_name.zip
│   └── ...
│
└── output/
    ├── dataset_name/
    │   ├── model_name/
    │   │   └── experiment_name/
    │   └── ...
    └── ...
```

The project path is configured using:

```python
project_dir = "/content/drive/MyDrive/Your-Project"
```

### Change this path

Replace:

```text
/content/drive/MyDrive/Your-Project
```

with the directory where you want to store your project.

For example:

```python
project_dir = "/content/drive/MyDrive/DentalYOLO26"
```

The path should point to the **root directory of the project**, not directly to the dataset ZIP file.

---

# 4. Dataset Path

The dataset configuration uses the dataset name to construct the ZIP path automatically.

Example:

```python
dataset_name = "dataset_name"

zip_dataset = os.path.join(
    project_dir,
    "datasets",
    f"{dataset_name}.zip"
)
```

Therefore, if:

```python
dataset_name = "ADLD"
```

the expected ZIP file is:

```text
/content/drive/MyDrive/DentalYOLO26/datasets/ADLD.zip
```

The general structure is:

```text
project_dir/
└── datasets/
    ├── ADLD.zip
    ├── TUFTS.zip
    ├── Dental_OPG.zip
    └── ...
```

### Important

The following names should be consistent:

```text
dataset_name
        ↓
dataset_name.zip
```

For example:

```python
dataset_name = "ADLD"
```

expects:

```text
datasets/ADLD.zip
```

Avoid manually changing the generated `zip_dataset` path unless your dataset uses a different file naming convention.

---

# 5. Extracted Dataset Path

The notebooks use:

```python
data_dir = os.path.join("/content", dataset_name)
```

Therefore:

```python
dataset_name = "ADLD"
```

results in:

```text
/content/ADLD/
```

The dataset YAML is expected at:

```text
/content/ADLD/data.yaml
```

The resulting structure should be similar to:

```text
/content/
└── ADLD/
    ├── data.yaml
    ├── train/
    │   ├── images/
    │   └── labels/
    ├── val/
    │   ├── images/
    │   └── labels/
    └── test/
        ├── images/
        └── labels/
```

The exact internal dataset structure depends on the dataset preparation used in the experiment.

---

# 6. Configuration

The main configuration is defined using the `Config` class.

A typical configuration is:

```python
@dataclass
class Config:
    # PROJECT
    project_dir: str = "/content/drive/MyDrive/Your-Project"
    experiment_name: str = "experiment_name"

    # DATA
    # Change this to the dataset you want to use.
    dataset_name: str = "dataset_name"

    # Change this to your dataset ZIP file.
    zip_dataset: str = os.path.join(
        project_dir,
        "datasets",
        f"{dataset_name}.zip"
    )

    # MODEL
    # Change this to select the model used in the experiment.
    model_name: str = "dental-yolo26n_v15"

    # TRAINING
    epochs: int = 100
    imgsz: int = 640
    batchsize: int = 8
    seed: int = 2026
```

For a basic experiment, the main parameters to check are:

| Parameter         | Purpose                   | Usually change?             |
| ----------------- | ------------------------- | --------------------------- |
| `project_dir`     | Project root directory    | Yes                         |
| `experiment_name` | Name of the experiment    | Yes                         |
| `dataset_name`    | Dataset identifier        | Yes                         |
| `zip_dataset`     | Dataset ZIP path          | Automatically generated     |
| `model_name`      | Model to train            | Yes                         |
| `epochs`          | Number of training epochs | Yes                         |
| `imgsz`           | Input image size          | Only if required            |
| `batchsize`       | Training batch size       | Depending on GPU            |
| `seed`            | Random seed               | Keep fixed for reproduction |

---

# 7. Output Paths

Output paths are generated automatically from the configuration.

```python
output_root = os.path.join(
    project_dir,
    "output"
)

output_dir = os.path.join(
    output_root,
    dataset_name,
    model_name,
)

result_dir = os.path.join(
    output_dir,
    experiment_name,
)
```

For example:

```python
project_dir = "/content/drive/MyDrive/DentalYOLO26"
dataset_name = "ADLD"
model_name = "dental-yolo26m_v15"
experiment_name = "exp1_e200_b8"
```

will produce:

```text
DentalYOLO26/
└── output/
    └── ADLD/
        └── dental-yolo26m_v15/
            └── exp1_e200_b8/
```

This allows results from different datasets, models, and experiments to remain separated.

> **Do not manually create or rename these output directories unless required.** They are generated from the configuration.

---

# 8. Selecting a Model

Each notebook supports a specific group of models.

The model is selected through:

```python
model_name = "..."
```

Only use model names supported by the corresponding notebook.

---

## 8.1 `dental-yolo26.ipynb`

This notebook is used for the DentalYOLO26 models.

Available models:

```text
dental-yolo26n_v15
dental-yolo26s_v15
dental-yolo26m_v15
dental-yolo26l_v15
```

Example:

```python
model_name = "dental-yolo26m_v15"
```

To train another DentalYOLO26 variant, change only the model name:

```python
model_name = "dental-yolo26l_v15"
```

---

## 8.2 `yolo_family.ipynb`

This notebook supports the YOLO family and RT-DETR models defined in its model registry.

### YOLOv8

```text
yolov8n
yolov8s
yolov8m
yolov8l
yolov8x
```

### YOLO12

```text
yolo12n
yolo12s
yolo12m
yolo12l
yolo12x
```

### YOLO26

```text
yolo26n
yolo26s
yolo26m
yolo26l
yolo26x
```

### RT-DETR

```text
rtdetr-r18
rtdetr-r34
rtdetr-resnet50
rtdetr-resnet101
rtdetr-l
rtdetr-x
```

For example:

```python
model_name = "yolo26m"
```

or:

```python
model_name = "yolo12l"
```

or:

```python
model_name = "rtdetr-l"
```

The available model names should match the model registry defined in the notebook.

---

## 8.3 `rfdetr.ipynb`

This notebook is used for RF-DETR models.

Available models:

```text
rfdetr-nano
rfdetr-small
rfdetr-medium
rfdetr-large
```

Example:

```python
model_name = "rfdetr-medium"
```

To use another RF-DETR variant:

```python
model_name = "rfdetr-large"
```

---

# 9. Training Configuration

The main training parameters are:

```python
epochs = 100
imgsz = 640
batchsize = 8
seed = 2026
```

### Epochs

```python
epochs = 100
```

Controls the maximum number of training epochs.

Use the value specified by the research experiment when reproducing reported results.

### Image size

```python
imgsz = 640
```

Controls the input image resolution used during training/evaluation.

### Batch size

```python
batchsize = 8
```

The batch size may need to be adjusted according to available GPU memory.

However, when reproducing a reported experiment, use the original research value.

### Random seed

```python
seed = 2026
```

The seed should normally remain unchanged when reproducing the reported experiment.

---

# 10. Basic Usage

## Step 1 — Open the appropriate notebook

Choose the notebook according to the model family:

```text
DentalYOLO26
    → dental-yolo26.ipynb

YOLO / RT-DETR
    → yolo_family.ipynb

RF-DETR
    → rfdetr.ipynb
```

---

## Step 2 — Configure the project path

Set:

```python
project_dir = "/content/drive/MyDrive/DentalYOLO26"
```

The project directory should contain the `datasets` folder.

---

## Step 3 — Prepare the dataset

Place the dataset ZIP file inside:

```text
project_dir/
└── datasets/
    └── dataset_name.zip
```

For example:

```text
DentalYOLO26/
└── datasets/
    └── ADLD.zip
```

---

## Step 4 — Set the dataset name

Change:

```python
dataset_name = "dataset_name"
```

to:

```python
dataset_name = "ADLD"
```

The notebook will then construct:

```text
project_dir/datasets/ADLD.zip
```

and:

```text
/content/ADLD/
```

for the dataset paths used by the pipeline.

---

## Step 5 — Select the model

Choose a model supported by the notebook.

For example, in `dental-yolo26.ipynb`:

```python
model_name = "dental-yolo26m_v15"
```

In `yolo_family.ipynb`:

```python
model_name = "yolo26m"
```

In `rfdetr.ipynb`:

```python
model_name = "rfdetr-medium"
```

---

## Step 6 — Configure training

Set the required experiment parameters:

```python
epochs = 100
imgsz = 640
batchsize = 8
seed = 2026
```

For research reproduction, use the values reported for the corresponding experiment.

---

## Step 7 — Run the notebook sequentially

Run the notebook from the first cell to the final cell.

Do not skip preparation or verification cells because later stages depend on the paths, dataset configuration, and outputs generated earlier in the notebook.

---

# 11. Choosing the Correct Notebook

Use the following guide:

| Goal               | Notebook              | Model                |
| ------------------ | --------------------- | -------------------- |
| Train DentalYOLO26 | `dental-yolo26.ipynb` | `dental-yolo26*_v15` |
| Train YOLOv8       | `yolo_family.ipynb`   | `yolov8*`            |
| Train YOLO12       | `yolo_family.ipynb`   | `yolo12*`            |
| Train YOLO26       | `yolo_family.ipynb`   | `yolo26*`            |
| Train RT-DETR      | `yolo_family.ipynb`   | `rtdetr-*`           |
| Train RF-DETR      | `rfdetr.ipynb`        | `rfdetr-*`           |

The `*` represents the model scale such as `n`, `s`, `m`, `l`, or `x`.

---

# 12. Reproducibility

To reproduce an existing research experiment, keep the following consistent:

```text
Dataset
Model
Epochs
Batch size
Image size
Random seed
Software/library versions
Hardware
Evaluation procedure
```

In particular, do not change the model, dataset, training parameters, or evaluation settings if the goal is to reproduce the reported results.

For a new experiment, update the configuration and use a new `experiment_name`.

Example:

```python
dataset_name = "ADLD"
model_name = "dental-yolo26m_v15"
epochs = 200
batchsize = 8
seed = 2026

experiment_name = "exp_new_e200_b8"
```

---

# 13. Common Path Issues

### Dataset ZIP not found

Check that:

```text
project_dir/
└── datasets/
    └── dataset_name.zip
```

matches:

```python
dataset_name = "dataset_name"
```

For example:

```python
dataset_name = "ADLD"
```

requires:

```text
datasets/ADLD.zip
```

---

### `data.yaml` not found

The extracted dataset should contain:

```text
/content/dataset_name/data.yaml
```

For example:

```text
/content/ADLD/data.yaml
```

---

### Output saved to the wrong location

Check:

```python
project_dir
dataset_name
model_name
experiment_name
```

The output path is generated as:

```text
project_dir/
└── output/
    └── dataset_name/
        └── model_name/
            └── experiment_name/
```

---

# 14. Recommended Configuration Comments

The notebooks use comments to identify parameters that users may need to modify.

For dataset selection:

```python
# Change this to the dataset you want to use.
dataset_name = "dataset_name"
```

For the dataset archive:

```python
# Change this to your dataset ZIP file.
```

For model selection:

```python
# Change this to select the model used in the experiment.
model_name = "..."
```

For research settings:

```python
# Keep unchanged to reproduce the reported experiment.
```

For fair model comparison:

```python
# Keep this setting consistent across models for a fair comparison.
```

For reproducibility:

```python
# Random seed used for reproducibility.
```

---

# 15. Summary

For a standard experiment, the required workflow is:

```text
1. Select notebook
        ↓
2. Set project_dir
        ↓
3. Place dataset ZIP in project_dir/datasets/
        ↓
4. Set dataset_name
        ↓
5. Select model_name
        ↓
6. Set training parameters
        ↓
7. Run notebook sequentially
        ↓
8. Check output/project results
```

The most important configuration is:

```python
project_dir = "/content/drive/MyDrive/Your-Project"
dataset_name = "dataset_name"
model_name = "your-model"
epochs = 100
imgsz = 640
batchsize = 8
seed = 2026
```

For basic usage, users should normally only need to modify:

```text
project_dir
dataset_name
model_name
experiment_name
```

Training parameters should be changed only when intentionally running a different experiment.

---

## Model Reference

### DentalYOLO26

```text
dental-yolo26n_v15
dental-yolo26s_v15
dental-yolo26m_v15
dental-yolo26l_v15
```

### YOLOv8

```text
yolov8n
yolov8s
yolov8m
yolov8l
yolov8x
```

### YOLO12

```text
yolo12n
yolo12s
yolo12m
yolo12l
yolo12x
```

### YOLO26

```text
yolo26n
yolo26s
yolo26m
yolo26l
yolo26x
```

### RT-DETR

```text
rtdetr-r18
rtdetr-r34
rtdetr-resnet50
rtdetr-resnet101
rtdetr-l
rtdetr-x
```

### RF-DETR

```text
rfdetr-nano
rfdetr-small
rfdetr-medium
rfdetr-large
```
