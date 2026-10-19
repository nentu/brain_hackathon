"""Shared inference base for the best models."""

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from brain_hackathon.train.core.engine import resolve_device
from brain_hackathon.train.evaluation.test_model import load_checkpoint

DEFAULT_CLASSES = ("glioma", "meningioma", "notumor", "pituitary")
ALIGNED_CHANNELS = 6


def to_gray(image):
    """Convert an HxW, HxWx1 or (BGR) HxWx3 image to HxW grayscale."""
    array = np.asarray(image)
    if array.ndim == 2:
        return array
    if array.ndim == 3 and array.shape[2] == 1:
        return array[:, :, 0]
    if array.ndim == 3 and array.shape[2] == 3:
        return cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
    raise ValueError(f"cannot convert image with shape {array.shape} to grayscale")


class BaseBestModel:
    """Shared single/batch inference for one checkpoint."""

    default_checkpoint = None

    def __init__(self, checkpoint=None, classes=DEFAULT_CLASSES, device=None):
        self.device = resolve_device(device)
        self.classes = tuple(classes)
        self.checkpoint = checkpoint or self.default_checkpoint
        self.model, self.info = load_checkpoint(self.checkpoint, self.device)
        self.model.eval()

    @property
    def num_classes(self):
        return len(self.classes)

    def preprocess(self, image):
        raise NotImplementedError

    @staticmethod
    def _as_items(images):
        raise NotImplementedError

    def _predict_tensors(self, tensors):
        batch = torch.stack(tensors).to(self.device)
        with torch.no_grad():
            probabilities = F.softmax(self.model(batch), dim=1)
        return probabilities.cpu().numpy()

    def predict(self, image):
        """Return the class distribution for one image."""
        return self._predict_tensors([self.preprocess(image)])[0]

    def predict_batch(self, images):
        """Return ``(N, num_classes)`` distributions."""
        items = self._as_items(images)
        if not items:
            return np.empty((0, self.num_classes), dtype=np.float32)
        return self._predict_tensors([self.preprocess(image) for image in items])

    def grad_cam(self, image, target_class=None, **kwargs):
        """Return a Grad-CAM++ heatmap dict for one image."""
        from brain_hackathon.eval.gradcam import grad_cam_plus_plus

        return grad_cam_plus_plus(self, image, target_class=target_class, **kwargs)

    def top_class(self, distribution):
        index = int(np.argmax(distribution))
        return self.classes[index], float(distribution[index])
