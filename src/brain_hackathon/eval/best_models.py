"""Inference with the best raw (1-channel) and aligned (6-channel) models."""

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from brain_hackathon.data.dataset import (
    foreground_stats,
    three_sigma_normalize,
)
from brain_hackathon.data.preprocessing import IMAGE_SIZE, resize_image
from brain_hackathon.train.core.engine import resolve_device
from brain_hackathon.train.evaluation.test_model import load_checkpoint

DEFAULT_RAW_CHECKPOINT = "wandb/run-20261017_202512-cyg0yhoz/files/best.pt"
DEFAULT_ALIGNED_CHECKPOINT = "wandb/run-20261017_215657-mb02wu62/files/best.pt"
DEFAULT_CLASSES = ("glioma", "meningioma", "notumor", "pituitary")
ALIGNED_CHANNELS = 6


def _to_gray(image):
    """Convert an HxW, HxWx1 or (BGR) HxWx3 image to HxW grayscale."""
    array = np.asarray(image)
    if array.ndim == 2:
        return array
    if array.ndim == 3 and array.shape[2] == 1:
        return array[:, :, 0]
    if array.ndim == 3 and array.shape[2] == 3:
        return cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
    raise ValueError(f"cannot convert image with shape {array.shape} to grayscale")


class BestModels:
    """Run the best raw and aligned checkpoints on numpy images.

    ``predict``/``predict_batch`` apply the matching preprocessing and return the
    softmax class distribution. ``kind="auto"`` selects the model from the input
    shape: 2D / 1-channel / 3-channel -> raw (1-channel model); 6 channels ->
    aligned (6-channel model).
    """

    def __init__(
        self,
        raw_checkpoint=DEFAULT_RAW_CHECKPOINT,
        aligned_checkpoint=DEFAULT_ALIGNED_CHECKPOINT,
        classes=DEFAULT_CLASSES,
        device=None,
    ):
        self.device = resolve_device(device)
        self.classes = tuple(classes)
        self.raw_model, self.raw_info = load_checkpoint(
            raw_checkpoint, self.device
        )
        self.aligned_model, self.aligned_info = load_checkpoint(
            aligned_checkpoint, self.device
        )
        self.raw_model.eval()
        self.aligned_model.eval()

    @property
    def num_classes(self):
        return len(self.classes)

    @staticmethod
    def preprocess_raw(image):
        """Grayscale -> 224x224 INTER_AREA -> [0, 1] -> ``(1, 224, 224)``."""
        gray = _to_gray(image)
        array = resize_image(gray, IMAGE_SIZE).astype(np.float32)
        if array.max() > 1.0:
            array /= 255.0
        return torch.from_numpy(np.ascontiguousarray(array[None, :, :]))

    @staticmethod
    def preprocess_aligned(image):
        """Build the 6-channel input and apply 3-sigma normalization.

        An HxWx6 input is used directly; any other shape is converted to grayscale
        and replicated across all 6 modality channels.
        """
        array = np.asarray(image)
        if array.ndim == 3 and array.shape[2] == ALIGNED_CHANNELS:
            stack = np.stack(
                [
                    resize_image(array[:, :, c], IMAGE_SIZE)
                    for c in range(ALIGNED_CHANNELS)
                ],
                axis=-1,
            ).astype(np.float32)
        else:
            gray = resize_image(_to_gray(array), IMAGE_SIZE).astype(np.float32)
            stack = np.repeat(gray[:, :, None], ALIGNED_CHANNELS, axis=2)

        mean, std = foreground_stats(stack)
        stack = three_sigma_normalize(stack, mean, std)
        return torch.from_numpy(np.ascontiguousarray(stack.transpose(2, 0, 1)))

    @staticmethod
    def detect_kind(image):
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

    def _preprocess(self, image, kind):
        if kind == "aligned":
            return self.preprocess_aligned(image)
        if kind == "raw":
            return self.preprocess_raw(image)
        raise ValueError(f"unknown kind: {kind!r}")

    def _predict_tensors(self, tensors, kind):
        model = self.aligned_model if kind == "aligned" else self.raw_model
        batch = torch.stack(tensors).to(self.device)
        with torch.no_grad():
            logits = model(batch)
            probabilities = F.softmax(logits, dim=1)
        return probabilities.cpu().numpy()

    def predict(self, image, kind="auto"):
        """Return the class distribution for one image."""
        if kind == "auto":
            kind = self.detect_kind(image)
        tensor = self._preprocess(image, kind)
        return self._predict_tensors([tensor], kind)[0]

    def predict_batch(self, images, kind="auto"):
        """Return ``(N, num_classes)`` distributions for a list of images."""
        items = self._as_items(images, kind)
        if not items:
            return np.empty((0, self.num_classes), dtype=np.float32)
        if kind == "auto":
            kind = self.detect_kind(items[0])
        tensors = [self._preprocess(image, kind) for image in items]
        return self._predict_tensors(tensors, kind)

    def predict_both(self, image, weights=(0.5, 0.5)):
        """Run both models on one image and return their distributions."""
        raw = self.predict(image, kind="raw")
        aligned = self.predict(image, kind="aligned")
        ensemble = weights[0] * raw + weights[1] * aligned
        return {"raw": raw, "aligned": aligned, "ensemble": ensemble}

    def predict_batch_both(self, images, weights=(0.5, 0.5)):
        """Run both models on a batch and return ``(N, num_classes)`` arrays."""
        raw = self.predict_batch(images, kind="raw")
        aligned = self.predict_batch(images, kind="aligned")
        ensemble = weights[0] * raw + weights[1] * aligned
        return {"raw": raw, "aligned": aligned, "ensemble": ensemble}

    def top_class(self, distribution):
        index = int(np.argmax(distribution))
        return self.classes[index], float(distribution[index])

    @staticmethod
    def _as_items(images, kind):
        if isinstance(images, np.ndarray):
            if images.ndim == 4:
                return list(images)
            if images.ndim == 3 and images.shape[2] not in (1, 3, ALIGNED_CHANNELS):
                return list(images)  # batch of grayscale HxW images
            return [images]
        return list(images)
