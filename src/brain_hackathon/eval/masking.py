"""Foreground masking for attribution heatmaps."""

import cv2
import numpy as np

from brain_hackathon.eval.base import to_gray

__all__ = ["apply_heatmap_mask", "foreground_mask"]


def _to_uint8_gray(image):
    gray = to_gray(image)
    if gray.dtype == np.uint8:
        return gray

    values = gray.astype(np.float32)
    if values.size and float(values.max()) <= 1.0:
        values = values * 255.0
    return np.clip(values, 0.0, 255.0).astype(np.uint8)


def foreground_mask(image, radius=10.0):
    """Return a boolean mask of the largest non-background region.

    Pixels close to the modal image intensity are treated as background. The
    largest connected background component is removed, so the returned mask
    covers the brain/tissue region that surrounds it. ``radius`` is in the
    ``[0, 255]`` intensity scale and is rescaled to the image range.
    """
    radius = float(radius)
    if radius < 0.0:
        raise ValueError("radius must be non-negative")

    gray = _to_uint8_gray(image)
    values, edges = np.histogram(gray.flatten(), bins=100)
    center = float(edges[int(np.argmax(values))])
    intensity_range = float(int(gray.max()) - int(gray.min()))
    threshold = radius / 255.0 * intensity_range
    centered = gray.astype(np.float32) - center

    background = (-threshold < centered) & (centered < threshold)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        np.uint8(background)
    )
    if num_labels > 1:
        largest = int(np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1)
        background_mask = labels == largest
    else:
        background_mask = np.zeros_like(background, dtype=bool)

    foreground = (~background_mask).astype(np.uint8)
    foreground = cv2.morphologyEx(
        foreground, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)
    )
    return foreground.astype(bool)


def apply_heatmap_mask(heatmap, image, radius=10.0):
    """Multiply a heatmap by the foreground mask and renormalize to [0, 1]."""
    heatmap = np.asarray(heatmap, dtype=np.float32)
    mask = foreground_mask(image, radius)
    if not mask.any():
        return heatmap.copy()

    masked = heatmap * mask
    high = float(masked.max())
    if high <= 0.0:
        return heatmap.copy()
    return (masked / high).astype(np.float32)
