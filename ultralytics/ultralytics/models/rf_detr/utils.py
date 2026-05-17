# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Utilities for the RF-DETR model-family wrapper."""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import Any

from ultralytics.utils import YAML

RFDETR_SRC = Path(__file__).resolve().parent
RFDETR_CFG_DIR = Path(__file__).resolve().parents[2] / "cfg" / "models" / "rf-detr"

VARIANT_ALIASES = {
    "base": "base",
    "nano": "nano",
    "small": "small",
    "medium": "medium",
    "large": "large",
    "rfdetr-base": "base",
    "rfdetr-nano": "nano",
    "rfdetr-small": "small",
    "rfdetr-medium": "medium",
    "rfdetr-large": "large",
    "rf-detr-base": "base",
    "rf-detr-nano": "nano",
    "rf-detr-small": "small",
    "rf-detr-medium": "medium",
    "rf-detr-large": "large",
}

VARIANT_CLASSES = {
    "base": "RFDETRBase",
    "nano": "RFDETRNano",
    "small": "RFDETRSmall",
    "medium": "RFDETRMedium",
    "large": "RFDETRLarge",
}

TRAIN_FIELD_ALIASES = {
    "data": "dataset_dir",
    "batch": "batch_size",
    "imgsz": "resolution",
    "workers": "num_workers",
    "lr0": "lr",
}

TRAIN_FIELDS = {
    "accelerator",
    "aug_config",
    "augmentation_backend",
    "auto_batch_ema_headroom",
    "auto_batch_max_targets_per_image",
    "auto_batch_target_effective",
    "batch_size",
    "checkpoint_interval",
    "class_names",
    "clearml",
    "clip_max_norm",
    "compute_test_loss",
    "compute_val_loss",
    "dataset_dir",
    "dataset_file",
    "devices",
    "do_random_resize_via_padding",
    "dont_save_weights",
    "drop_path",
    "early_stopping",
    "early_stopping_min_delta",
    "early_stopping_patience",
    "early_stopping_use_ema",
    "ema_decay",
    "ema_tau",
    "ema_update_interval",
    "epochs",
    "eval_interval",
    "eval_max_dets",
    "expanded_scales",
    "fp16_eval",
    "grad_accum_steps",
    "ia_bce_loss",
    "log_per_class_metrics",
    "lr",
    "lr_component_decay",
    "lr_drop",
    "lr_encoder",
    "lr_min_factor",
    "lr_scheduler",
    "lr_vit_layer_decay",
    "mlflow",
    "multi_scale",
    "notes",
    "num_nodes",
    "num_select",
    "num_workers",
    "output_dir",
    "persistent_workers",
    "pin_memory",
    "prefetch_factor",
    "progress_bar",
    "project",
    "resume",
    "run",
    "run_test",
    "save_dataset_grids",
    "seed",
    "segmentation_head",
    "skip_best_epochs",
    "square_resize_div_64",
    "strategy",
    "sync_bn",
    "tensorboard",
    "train_log_on_step",
    "train_log_sync_dist",
    "use_ema",
    "wandb",
    "warmup_epochs",
    "weight_decay",
}


def add_rfdetr_path() -> None:
    """Make the bundled RF-DETR package importable without requiring installation."""
    src = str(RFDETR_SRC)
    if (RFDETR_SRC / "rfdetr").exists() and src not in sys.path:
        sys.path.insert(0, src)


def import_rfdetr():
    """Import vendored RF-DETR lazily and report missing optional dependencies clearly."""
    add_rfdetr_path()
    try:
        return importlib.import_module("rfdetr")
    except ModuleNotFoundError as exc:
        missing = exc.name or "an RF-DETR dependency"
        raise ImportError(
            f"RF-DETR is bundled inside Ultralytics, but dependency '{missing}' is not installed. "
            "Install the RF-DETR runtime/training dependencies before using `RFDETR` "
            "(for example: pydantic, transformers, pytorch-lightning, pycocotools, albumentations). "
            "YOLO and other Ultralytics model families are unaffected."
        ) from exc


def _load_yaml(path: Path) -> dict[str, Any]:
    data = YAML.load(str(path))
    if not isinstance(data, dict):
        raise ValueError(f"Expected RF-DETR YAML to contain a mapping, got {type(data).__name__}: {path}")
    return data


def resolve_model_config(model: str | Path) -> tuple[Path | None, dict[str, Any]]:
    """Resolve an RF-DETR model alias or YAML path into wrapper config data."""
    model_str = str(model)
    path = Path(model_str).expanduser()
    if path.exists():
        return path, _load_yaml(path)

    key = model_str.lower().replace("_", "-").replace(".yaml", "").replace(".yml", "")
    if key in VARIANT_ALIASES:
        variant = VARIANT_ALIASES[key]
        cfg_path = RFDETR_CFG_DIR / f"rfdetr-{variant}.yaml"
        if cfg_path.exists():
            return cfg_path, _load_yaml(cfg_path)
        return None, {"family": "rf-detr", "variant": variant, "model": {}, "train": {}}

    raise FileNotFoundError(
        f"Could not resolve RF-DETR model '{model}'. Use a YAML path or one of: {', '.join(sorted(VARIANT_ALIASES))}."
    )


def split_config(config: dict[str, Any]) -> tuple[str, dict[str, Any], dict[str, Any]]:
    """Return variant, model kwargs, and train defaults from an RF-DETR wrapper YAML."""
    variant = VARIANT_ALIASES.get(str(config.get("variant", "nano")).lower().replace("_", "-"), config.get("variant"))
    if variant not in VARIANT_CLASSES:
        raise ValueError(f"Unsupported RF-DETR variant '{variant}'. Supported: {', '.join(VARIANT_CLASSES)}")

    model_kwargs = dict(config.get("model") or {})
    model_kwargs.update(config.get("backbone") or {})
    model_kwargs.update(config.get("head") or {})
    train_defaults = dict(config.get("train") or {})
    loss_cfg = dict(config.get("loss") or {})
    for key in ("cls_loss_coef", "ia_bce_loss", "segmentation_head", "num_select"):
        if key in loss_cfg:
            model_kwargs[key] = loss_cfg[key]
            train_defaults.setdefault(key, loss_cfg[key])
    if "nc" in config and "num_classes" not in model_kwargs:
        model_kwargs["num_classes"] = config["nc"]
    if "num_classes" in config and "num_classes" not in model_kwargs:
        model_kwargs["num_classes"] = config["num_classes"]
    return variant, model_kwargs, train_defaults


def normalize_train_kwargs(defaults: dict[str, Any], kwargs: dict[str, Any]) -> dict[str, Any]:
    """Map common Ultralytics train arguments to RF-DETR TrainConfig arguments."""
    args = {**defaults}
    for key, value in kwargs.items():
        args[TRAIN_FIELD_ALIASES.get(key, key)] = value

    project = args.pop("project", None)
    name = args.pop("name", None)
    if project or name:
        args["output_dir"] = str(Path(project or "runs/rf-detr") / (name or "train"))

    if "dataset_dir" in args:
        data_path = Path(str(args["dataset_dir"])).expanduser()
        if data_path.suffix in {".yaml", ".yml"}:
            args["dataset_dir"] = prepare_ultralytics_data_yaml(data_path, Path(str(args.get("output_dir", "runs/rf-detr/train"))))
            args.setdefault("dataset_file", "yolo")

    return {k: v for k, v in args.items() if k in TRAIN_FIELDS or k == "device" or k == "resolution"}


def _resolve_data_path(root: Path, value: Any, key: str) -> Path:
    """Resolve one image-directory entry from an Ultralytics data YAML."""
    if value is None:
        raise ValueError(f"RF-DETR data bridge requires '{key}' in the Ultralytics data YAML.")
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError(f"RF-DETR data bridge supports one directory for '{key}', got {value!r}.")
        value = value[0]
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def _labels_dir_from_images(images_dir: Path) -> Path:
    """Infer a YOLO labels directory from an images directory."""
    parts = list(images_dir.parts)
    if "images" in parts:
        idx = len(parts) - 1 - parts[::-1].index("images")
        parts[idx] = "labels"
        return Path(*parts)
    return images_dir.parent / "labels" / images_dir.name


def _link_dir(link: Path, target: Path) -> None:
    """Create a directory symlink if it does not exist."""
    if not target.exists():
        raise FileNotFoundError(f"RF-DETR data bridge expected directory does not exist: {target}")
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.exists() or link.is_symlink():
        return
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError:
        # Some filesystems disallow symlinks. Directory junction semantics are
        # not portable here, so fail with an actionable message instead of
        # silently copying a potentially large dataset.
        raise OSError(f"Could not create dataset symlink {link} -> {target}") from None


def prepare_ultralytics_data_yaml(data_yaml: Path, output_dir: Path) -> str:
    """Create an RF-DETR-compatible YOLO dataset view from an Ultralytics data YAML.

    RF-DETR's native YOLO loader expects:
        dataset/train/images, dataset/train/labels, dataset/valid/images, dataset/valid/labels

    Ultralytics YAMLs often use `val:` and arbitrary image directories. This
    function creates a small symlink view so RF-DETR can train without requiring
    users to duplicate or rearrange their dataset.
    """
    data_yaml = data_yaml.resolve()
    data = YAML.load(str(data_yaml))
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in data YAML: {data_yaml}")

    root = Path(str(data.get("path", data_yaml.parent))).expanduser()
    if not root.is_absolute():
        root = data_yaml.parent / root
    root = root.resolve()

    train_images = _resolve_data_path(root, data.get("train"), "train")
    val_images = _resolve_data_path(root, data.get("val", data.get("valid")), "val")
    train_labels = _labels_dir_from_images(train_images)
    val_labels = _labels_dir_from_images(val_images)

    view = output_dir / "_rfdetr_dataset"
    _link_dir(view / "train" / "images", train_images)
    _link_dir(view / "train" / "labels", train_labels)
    _link_dir(view / "valid" / "images", val_images)
    _link_dir(view / "valid" / "labels", val_labels)

    names = data.get("names", [])
    data_file = view / "data.yaml"
    data_file.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(names, dict):
        names_yaml = [names[k] for k in sorted(names, key=lambda x: int(x))]
    else:
        names_yaml = list(names)
    with data_file.open("w", encoding="utf-8", newline="\n") as f:
        f.write("names:\n")
        for name in names_yaml:
            f.write(f"  - {name}\n")
        f.write(f"nc: {len(names_yaml)}\n")
    return os.fspath(view)
