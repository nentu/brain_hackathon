"""Inference with the best 1-channel raw model."""

import numpy as np
import torch

from brain_hackathon.data.preprocessing import IMAGE_SIZE, resize_image
from brain_hackathon.eval.base import BaseBestModel, to_gray

DEFAULT_RAW_CHECKPOINT = "wandb/run-20261017_202512-cyg0yhoz/files/best.pt"


class RawBestModel(BaseBestModel):
    """1-channel model trained on the raw dataset."""

    default_checkpoint = DEFAULT_RAW_CHECKPOINT

    @staticmethod
    def preprocess(image):
        """Grayscale -> 224x224 INTER_AREA -> [0, 1] -> ``(1, 224, 224)``."""
        gray = to_gray(image)
        array = resize_image(gray, IMAGE_SIZE).astype(np.float32)
        if array.max() > 1.0:
            array /= 255.0
        return torch.from_numpy(np.ascontiguousarray(array[None, :, :]))

    @staticmethod
    def _as_items(images):
        if isinstance(images, np.ndarray):
            if images.ndim == 4:
                return list(images)
            if images.ndim == 3 and images.shape[2] not in (1, 3):
                return list(images)  # batch of grayscale HxW images
            return [images]
        return list(images)
