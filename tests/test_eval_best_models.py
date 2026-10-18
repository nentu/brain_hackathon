import os

import numpy as np
import pytest

from brain_hackathon.eval import BestModels
from brain_hackathon.eval.best_models import (
    DEFAULT_ALIGNED_CHECKPOINT,
    DEFAULT_RAW_CHECKPOINT,
)

requires_models = pytest.mark.skipif(
    not (
        os.path.isfile(DEFAULT_RAW_CHECKPOINT)
        and os.path.isfile(DEFAULT_ALIGNED_CHECKPOINT)
    ),
    reason="best checkpoints are not available",
)


def test_preprocess_raw_shape_and_range():
    image = np.random.randint(0, 256, (64, 48), dtype=np.uint8)
    tensor = BestModels.preprocess_raw(image)

    assert tuple(tensor.shape) == (1, 224, 224)
    assert tensor.dtype.is_floating_point
    assert float(tensor.min()) >= 0.0
    assert float(tensor.max()) <= 1.0


def test_preprocess_raw_accepts_bgr():
    image = np.random.randint(0, 256, (64, 48, 3), dtype=np.uint8)
    assert tuple(BestModels.preprocess_raw(image).shape) == (1, 224, 224)


def test_preprocess_aligned_replicates_single_channel():
    image = np.random.randint(0, 256, (64, 48), dtype=np.uint8)
    tensor = BestModels.preprocess_aligned(image)

    assert tuple(tensor.shape) == (6, 224, 224)
    assert np.allclose(tensor[0].numpy(), tensor[5].numpy())
    assert float(tensor.min()) >= 0.0
    assert float(tensor.max()) <= 1.0


def test_preprocess_aligned_uses_six_channels():
    image = np.random.randint(0, 256, (64, 48, 6), dtype=np.uint8)
    assert tuple(BestModels.preprocess_aligned(image).shape) == (6, 224, 224)


def test_detect_kind():
    assert BestModels.detect_kind(np.zeros((10, 10), np.uint8)) == "raw"
    assert BestModels.detect_kind(np.zeros((10, 10, 3), np.uint8)) == "raw"
    assert BestModels.detect_kind(np.zeros((10, 10, 6), np.uint8)) == "aligned"


@pytest.fixture(scope="module")
def predictor():
    return BestModels(device="cpu")


@requires_models
def test_predict_single_raw(predictor):
    image = np.random.randint(0, 256, (224, 224), dtype=np.uint8)
    probabilities = predictor.predict(image)

    assert probabilities.shape == (len(predictor.classes),)
    assert probabilities.sum() == pytest.approx(1.0, abs=1e-5)


@requires_models
def test_predict_single_aligned(predictor):
    image = np.random.randint(0, 256, (224, 224, 6), dtype=np.uint8)
    probabilities = predictor.predict(image, kind="aligned")

    assert probabilities.shape == (len(predictor.classes),)
    assert probabilities.sum() == pytest.approx(1.0, abs=1e-5)


@requires_models
def test_predict_batch(predictor):
    images = [
        np.random.randint(0, 256, (224, 224), dtype=np.uint8) for _ in range(3)
    ]
    probabilities = predictor.predict_batch(images)

    assert probabilities.shape == (3, len(predictor.classes))
    assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5)


@requires_models
def test_predict_both(predictor):
    image = np.random.randint(0, 256, (224, 224), dtype=np.uint8)
    distributions = predictor.predict_both(image)

    assert set(distributions) == {"raw", "aligned", "ensemble"}
    for probabilities in distributions.values():
        assert probabilities.shape == (len(predictor.classes),)
        assert probabilities.sum() == pytest.approx(1.0, abs=1e-5)


@requires_models
def test_top_class(predictor):
    probabilities = np.array([0.1, 0.2, 0.6, 0.1], dtype=np.float32)
    name, confidence = predictor.top_class(probabilities)
    assert name == predictor.classes[2]
    assert confidence == pytest.approx(0.6)
