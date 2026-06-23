# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Validation adapter for the RF-DETR model-family wrapper."""

from __future__ import annotations


class RFDETRValidator:
    """Placeholder adapter for future native Ultralytics validation integration."""

    def __init__(self, model):
        self.model = model

    def __call__(self, **kwargs):
        raise NotImplementedError(
            "RF-DETR validation is not bridged to Ultralytics metrics yet. "
            "Use RF-DETR training/evaluation callbacks or run `model.train(..., run_test=True)` for now."
        )
