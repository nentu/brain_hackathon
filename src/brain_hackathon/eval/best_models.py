"""Combined inference with both best models."""

import numpy as np

from brain_hackathon.eval.aligned_model import (
    DEFAULT_ALIGNED_CHECKPOINT,
    AlignedBestModel,
)
from brain_hackathon.eval.base import ALIGNED_CHANNELS, DEFAULT_CLASSES
from brain_hackathon.eval.raw_model import DEFAULT_RAW_CHECKPOINT, RawBestModel

__all__ = [
    "ALIGNED_CHANNELS",
    "DEFAULT_ALIGNED_CHECKPOINT",
    "DEFAULT_CLASSES",
    "DEFAULT_RAW_CHECKPOINT",
    "AlignedBestModel",
    "BestModels",
    "RawBestModel",
]


class BestModels:
    """Wrapper around :class:`RawBestModel` and :class:`AlignedBestModel`."""

    def __init__(
        self,
        raw_checkpoint=DEFAULT_RAW_CHECKPOINT,
        aligned_checkpoint=DEFAULT_ALIGNED_CHECKPOINT,
        classes=DEFAULT_CLASSES,
        device=None,
    ):
        self.raw = RawBestModel(
            raw_checkpoint, classes=classes, device=device
        )
        self.aligned = AlignedBestModel(
            aligned_checkpoint, classes=classes, device=device
        )
        self.classes = self.raw.classes
        self.device = self.raw.device

    @property
    def num_classes(self):
        return self.raw.num_classes

    @staticmethod
    def detect_kind(image):
        if isinstance(image, dict):
            return "aligned"
        if isinstance(image, (list, tuple)):
            return "aligned"
        array = np.asarray(image)
        if array.ndim == 2:
            return "raw"
        if array.ndim == 3:
            channels = array.shape[2]
            if channels == ALIGNED_CHANNELS:
                return "aligned"
            if channels in (1, 3):
                return "raw"
        raise ValueError(f"cannot infer model kind from shape {array.shape}")

    def _model_for(self, kind):
        if kind == "aligned":
            return self.aligned
        if kind == "raw":
            return self.raw
        raise ValueError(f"unknown kind: {kind!r}")

    @staticmethod
    def _first_item(images):
        if isinstance(images, np.ndarray):
            if images.ndim == 4:
                return images[0]
            if images.ndim == 3 and images.shape[2] not in (1, 3, ALIGNED_CHANNELS):
                return images[0]
            return images
        items = list(images)
        return items[0] if items else None

    def predict(self, image, kind="auto"):
        """Return the class distribution for one image."""
        if kind == "auto":
            kind = self.detect_kind(image)
        return self._model_for(kind).predict(image)

    def predict_batch(self, images, kind="auto"):
        """Return ``(N, num_classes)`` distributions for a batch."""
        if kind == "auto":
            first = self._first_item(images)
            kind = "raw" if first is None else self.detect_kind(first)
        return self._model_for(kind).predict_batch(images)

    def predict_both(self, image, weights=(0.5, 0.5)):
        """Run both models on one image and return their distributions."""
        raw = self.raw.predict(image)
        aligned = self.aligned.predict(image)
        ensemble = weights[0] * raw + weights[1] * aligned
        return {"raw": raw, "aligned": aligned, "ensemble": ensemble}

    def predict_batch_both(self, images, weights=(0.5, 0.5)):
        """Run both models on a batch and return ``(N, num_classes)`` arrays."""
        raw = self.raw.predict_batch(images)
        aligned = self.aligned.predict_batch(images)
        ensemble = weights[0] * raw + weights[1] * aligned
        return {"raw": raw, "aligned": aligned, "ensemble": ensemble}

    def grad_cam(self, image, kind="auto", target_class=None, **kwargs):
        """Return a Grad-CAM++ heatmap dict, dispatching raw/aligned."""
        if kind == "auto":
            kind = self.detect_kind(image)
        return self._model_for(kind).grad_cam(
            image, target_class=target_class, **kwargs
        )

    def top_class(self, distribution):
        return self.raw.top_class(distribution)
