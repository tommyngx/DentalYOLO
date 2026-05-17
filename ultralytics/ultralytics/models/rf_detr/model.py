# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Ultralytics RF-DETR model-family wrapper.

This wrapper intentionally keeps RF-DETR as its own model family. It does not
route RF-DETR YAML through YOLO's `parse_model()` because RF-DETR has its own
backbone, transformer, Hungarian loss, postprocess, and training stack.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .predict import RFDETRPredictor, detections_to_results
from .train import RFDETRTrainer
from .utils import VARIANT_CLASSES, import_rfdetr, resolve_model_config, split_config
from .val import RFDETRValidator


class RFDETR:
    """RF-DETR model family exposed through the Ultralytics namespace.

    Examples:
        >>> from ultralytics import RFDETR
        >>> model = RFDETR("rfdetr-nano.yaml", pretrain_weights=None)
        >>> model.train(data="/path/to/yolo_dataset", epochs=100, batch=8)
    """

    def __init__(self, model: str | Path = "rfdetr-nano.yaml", task: str = "detect", verbose: bool = False, **kwargs):
        self.task = task
        self.predictor = RFDETRPredictor(self)
        self.validator = RFDETRValidator(self)
        self.trainer = None
        self.cfg, self.config = resolve_model_config(model)
        self.variant, model_kwargs, self.train_defaults = split_config(self.config)
        model_kwargs.update(kwargs)
        if "nc" in model_kwargs and "num_classes" not in model_kwargs:
            model_kwargs["num_classes"] = model_kwargs.pop("nc")

        rfdetr = import_rfdetr()
        model_path = Path(str(model)).expanduser()
        if model_path.exists() and model_path.suffix.lower() in {".pt", ".pth", ".ckpt"}:
            self._native = rfdetr.from_checkpoint(str(model_path), **model_kwargs)
        else:
            cls = getattr(rfdetr, VARIANT_CLASSES[self.variant])
            self._native = cls(**model_kwargs)

        self.model = self._native
        self.model_name = str(model)
        self.overrides = {"model": str(model), "task": self.task, **model_kwargs}
        self.verbose = verbose

    @classmethod
    def from_checkpoint(cls, path: str | Path, **kwargs):
        """Load an RF-DETR checkpoint through the Ultralytics wrapper."""
        return cls(path, **kwargs)

    @property
    def task_map(self) -> dict[str, dict[str, Any]]:
        """Return RF-DETR task adapters following the Ultralytics model-family pattern."""
        return {
            "detect": {
                "predictor": RFDETRPredictor,
                "validator": RFDETRValidator,
                "trainer": RFDETRTrainer,
                "model": type(self._native),
            }
        }

    @property
    def names(self) -> dict[int, str]:
        """Return class names as the dictionary format expected by Ultralytics Results."""
        class_names = getattr(self._native, "class_names", [])
        return {i: name for i, name in enumerate(class_names)}

    def __call__(self, source=None, stream: bool = False, **kwargs):
        """Alias for predict."""
        return self.predict(source=source, stream=stream, **kwargs)

    def predict(self, source=None, stream: bool = False, return_ultralytics: bool = True, **kwargs):
        """Run RF-DETR prediction.

        Args:
            source: Image path, PIL image, NumPy array, torch tensor, or list accepted by RF-DETR.
            stream: Present for Ultralytics API compatibility. RF-DETR prediction is currently eager.
            return_ultralytics: If True, convert supervision.Detections to Ultralytics Results.
            **kwargs: Forwarded to RF-DETR's native `predict()` method.
        """
        if stream:
            raise NotImplementedError("RF-DETR wrapper does not support stream=True yet.")
        kwargs.setdefault("include_source_image", True)
        threshold = kwargs.pop("conf", kwargs.pop("threshold", 0.5))
        detections = self._native.predict(source, threshold=threshold, **kwargs)
        return detections_to_results(detections, source, self.names) if return_ultralytics else detections

    def train(self, **kwargs):
        """Train RF-DETR using its native PyTorch Lightning trainer with Ultralytics-style argument aliases."""
        self.trainer = RFDETRTrainer(self, self.train_defaults)
        return self.trainer.train(**kwargs)

    def val(self, **kwargs):
        """Run validation. Full Ultralytics metrics bridging is not implemented yet."""
        return self.validator(**kwargs)

    def export(self, **kwargs):
        """Delegate export to RF-DETR's native exporter."""
        return self._native.export(**kwargs)

    def optimize_for_inference(self, **kwargs):
        """Delegate inference optimization to RF-DETR."""
        return self._native.optimize_for_inference(**kwargs)

    def info(self, verbose: bool = True):
        """Return a small parameter summary for the wrapped RF-DETR model."""
        context = getattr(self._native, "model", None)
        module = getattr(context, "model", None)
        if not isinstance(module, torch.nn.Module):
            return None
        params = sum(p.numel() for p in module.parameters())
        gradients = sum(p.numel() for p in module.parameters() if p.requires_grad)
        if verbose:
            print(f"RF-DETR {self.variant}: {params:,} parameters, {gradients:,} gradients")
        return params, gradients

    def __getattr__(self, name: str):
        """Forward unknown attributes to the native RF-DETR object."""
        native = self.__dict__.get("_native")
        if native is not None and hasattr(native, name):
            return getattr(native, name)
        raise AttributeError(f"{self.__class__.__name__!s} has no attribute {name!r}")
