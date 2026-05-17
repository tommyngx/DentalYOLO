# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Prediction helpers for the RF-DETR model-family wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from ultralytics.engine.results import Results


def _as_list(x: Any) -> list[Any]:
    return x if isinstance(x, list) else [x]


def detections_to_results(detections: Any, source: Any, names: dict[int, str]) -> list[Results]:
    """Convert supervision.Detections from RF-DETR into Ultralytics Results."""
    det_list = _as_list(detections)
    src_list = _as_list(source)
    if len(src_list) == 1 and len(det_list) > 1:
        src_list = src_list * len(det_list)

    results = []
    for det, src in zip(det_list, src_list):
        xyxy = getattr(det, "xyxy", np.zeros((0, 4), dtype=np.float32))
        conf = getattr(det, "confidence", None)
        cls = getattr(det, "class_id", None)
        n = len(xyxy)
        if conf is None:
            conf = np.zeros((n,), dtype=np.float32)
        if cls is None:
            cls = np.zeros((n,), dtype=np.float32)

        boxes = np.concatenate(
            [
                np.asarray(xyxy, dtype=np.float32),
                np.asarray(conf, dtype=np.float32).reshape(-1, 1),
                np.asarray(cls, dtype=np.float32).reshape(-1, 1),
            ],
            axis=1,
        ) if n else np.zeros((0, 6), dtype=np.float32)

        metadata = getattr(det, "metadata", {}) or {}
        orig_img = metadata.get("source_image")
        if orig_img is None:
            orig_img = np.zeros((1, 1, 3), dtype=np.uint8)
        elif orig_img.ndim == 2:
            orig_img = np.repeat(orig_img[..., None], 3, axis=2)

        path = str(src) if isinstance(src, (str, Path)) else ""
        results.append(Results(orig_img, path=path, names=names, boxes=torch.as_tensor(boxes)))
    return results


class RFDETRPredictor:
    """Small adapter around RF-DETR's native predict method."""

    def __init__(self, model):
        self.model = model

    def __call__(self, source=None, **kwargs):
        return self.model.predict(source=source, **kwargs)
