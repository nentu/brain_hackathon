"""Inference with the best 6-channel aligned model."""

import numpy as np
import torch

from brain_hackathon.data.dataset import (
    MODALITIES,
    foreground_stats,
    three_sigma_normalize,
)
from brain_hackathon.data.preprocessing import (
    IMAGE_SIZE,
    apply_augmentation,
    process_images,
    resize_image,
)
from brain_hackathon.eval.base import ALIGNED_CHANNELS, BaseBestModel, to_gray

DEFAULT_ALIGNED_CHECKPOINT = "wandb/run-20261017_215657-mb02wu62/files/best.pt"


def align_modalities(images):
    """Align raw modality images into one ``H x W x 6`` stack.

    ``images`` is a sequence ordered as ``MODALITIES`` or a mapping from modality
    name to image. Missing modalities stay as zero channels. Alignment uses
    :func:`process_images` and :func:`apply_augmentation` from the training
    preprocessing, so the result matches the aligned dataset layout.
    """
    if isinstance(images, dict):
        unknown = set(images) - set(MODALITIES)
        if unknown:
            raise ValueError(f"unknown modalities: {sorted(unknown)}")
        ordered = [images.get(modality) for modality in MODALITIES]
    else:
        ordered = list(images)
        if len(ordered) > len(MODALITIES):
            raise ValueError(
                f"expected at most {len(MODALITIES)} modality images, "
                f"got {len(ordered)}"
            )
        ordered.extend([None] * (len(MODALITIES) - len(ordered)))

    present = [i for i, image in enumerate(ordered) if image is not None]
    if not present:
        raise ValueError("at least one modality image is required")

    gray_images = [
        resize_image(to_gray(ordered[i]), IMAGE_SIZE) for i in present
    ]
    transforms = process_images(gray_images)
    aligned = [
        apply_augmentation(image, transform)
        for image, transform in zip(gray_images, transforms)
    ]

    height, width = aligned[0].shape[:2]
    stack = np.zeros((height, width, ALIGNED_CHANNELS), dtype=np.float32)
    for position, image in zip(present, aligned):
        stack[:, :, position] = image
    return stack


class AlignedBestModel(BaseBestModel):
    """6-channel model trained on the aligned dataset."""

    default_checkpoint = DEFAULT_ALIGNED_CHECKPOINT

    @staticmethod
    def preprocess(image):
        """Build the 6-channel input and apply 3-sigma normalization.

        A sequence or mapping of raw modality images is aligned on the fly with
        :func:`align_modalities`. An already aligned HxWx6 array is used directly.
        Any other shape is converted to grayscale and replicated across all 6
        modality channels.
        """
        if isinstance(image, (list, tuple, dict)):
            stack = align_modalities(image)
        else:
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
