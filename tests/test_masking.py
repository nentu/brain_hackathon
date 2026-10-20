import cv2
import numpy as np
import pytest

from brain_hackathon.eval.masking import apply_heatmap_mask, foreground_mask


def _brain_like_image(size=64):
    image = np.zeros((size, size), dtype=np.uint8)
    image[size // 4 : 3 * size // 4, size // 4 : 3 * size // 4] = 128
    return image


def test_foreground_mask_selects_tissue_region():
    image = _brain_like_image()
    mask = foreground_mask(image, radius=10.0)

    assert mask.dtype == bool
    assert mask.shape == image.shape
    assert mask[32, 32]
    assert not mask[0, 0]


def test_foreground_mask_accepts_bgr():
    gray = _brain_like_image()
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    assert np.array_equal(
        foreground_mask(gray, radius=10.0),
        foreground_mask(bgr, radius=10.0),
    )


def test_apply_heatmap_mask_zeroes_background():
    image = _brain_like_image()
    heatmap = np.ones((64, 64), dtype=np.float32)

    masked = apply_heatmap_mask(heatmap, image, radius=10.0)

    assert masked[32, 32] == pytest.approx(1.0)
    assert masked[0, 0] == 0.0
    assert float(masked.max()) <= 1.0
    assert float(masked.min()) >= 0.0


def test_apply_heatmap_mask_renormalizes():
    image = _brain_like_image()
    heatmap = np.zeros((64, 64), dtype=np.float32)
    heatmap[0, 0] = 1.0
    heatmap[32, 32] = 0.25

    masked = apply_heatmap_mask(heatmap, image, radius=10.0)

    assert masked[32, 32] == pytest.approx(1.0)
    assert masked[0, 0] == 0.0


def test_apply_heatmap_mask_falls_back_when_foreground_is_empty():
    image = np.full((64, 64), 100, dtype=np.uint8)
    image[0:10, 0:10] = 50
    image[-10:, -10:] = 150

    assert not foreground_mask(image, radius=255.0).any()
    heatmap = np.full((64, 64), 0.5, dtype=np.float32)
    masked = apply_heatmap_mask(heatmap, image, radius=255.0)

    assert np.array_equal(masked, heatmap)


def test_foreground_mask_rejects_negative_radius():
    with pytest.raises(ValueError, match="non-negative"):
        foreground_mask(_brain_like_image(), radius=-1.0)
