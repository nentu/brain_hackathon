"""Inference with the best 6-channel aligned model."""

import numpy as np
import torch

from brain_hackathon.data.dataset import (
    foreground_stats,
    three_sigma_normalize,
)
from brain_hackathon.data.preprocessing import IMAGE_SIZE, resize_image
from brain_hackathon.eval.base import ALIGNED_CHANNELS, BaseBestModel, to_gray

DEFAULT_ALIGNED_CHECKPOINT = "wandb/run-20261017_215657-mb02wu62/files/best.pt"


class AlignedBestModel(BaseBestModel):
    """6-channel model trained on the aligned dataset."""

    default_checkpoint = DEFAULT_ALIGNED_CHECKPOINT

    @staticmethod
    def preprocess(image):
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
            gray = resize_image(to_gray(array), IMAGE_SIZE).astype(np.float32)
            stack = np.repeat(gray[:, :, None], ALIGNED_CHANNELS, axis=2)

        mean, std = foreground_stats(stack)
        stack = three_sigma_normalize(stack, mean, std)
        return torch.from_numpy(np.ascontiguousarray(stack.transpose(2, 0, 1)))

    @staticmethod
    def _as_items(images):
        if isinstance(images, np.ndarray):
            if images.ndim == 4:
                return list(images)
            if images.ndim == 3 and images.shape[2] != ALIGNED_CHANNELS:
                return list(images)  # batch of grayscale HxW images
            return [images]
        return list(images)
