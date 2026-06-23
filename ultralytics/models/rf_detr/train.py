# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Training adapter for the RF-DETR model-family wrapper."""

from __future__ import annotations

from .utils import normalize_train_kwargs


class RFDETRTrainer:
    """Adapter that maps common Ultralytics train arguments to RF-DETR's native trainer."""

    def __init__(self, model, train_defaults=None):
        self.model = model
        self.train_defaults = train_defaults or {}

    def train(self, **kwargs):
        args = normalize_train_kwargs(self.train_defaults, kwargs)
        return self.model._native.train(**args)
