import os

import numpy as np
import pytest

from brain_hackathon.eval import (
    DEFAULT_ALIGNED_CHECKPOINT,
    DEFAULT_RAW_CHECKPOINT,
    AlignedBestModel,
    BestModels,
    RawBestModel,
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
    tensor = RawBestModel.preprocess(image)

    assert tuple(tensor.shape) == (1, 224, 224)
    assert tensor.dtype.is_floating_point
    assert float(tensor.min()) >= 0.0
    assert float(tensor.max()) <= 1.0


def test_preprocess_raw_accepts_bgr():
    image = np.random.randint(0, 256, (64, 48, 3), dtype=np.uint8)
    assert tuple(RawBestModel.preprocess(image).shape) == (1, 224, 224)


def test_preprocess_aligned_replicates_single_channel():
    image = np.random.randint(0, 256, (64, 48), dtype=np.uint8)
    tensor = AlignedBestModel.preprocess(image)

    assert tuple(tensor.shape) == (6, 224, 224)
    assert np.allclose(tensor[0].numpy(), tensor[5].numpy())
    assert float(tensor.min()) >= 0.0
    assert float(tensor.max()) <= 1.0


def test_preprocess_aligned_uses_six_channels():
    image = np.random.randint(0, 256, (64, 48, 6), dtype=np.uint8)
    assert tuple(AlignedBestModel.preprocess(image).shape) == (6, 224, 224)


def test_detect_kind():
    assert BestModels.detect_kind(np.zeros((10, 10), np.uint8)) == "raw"
    assert BestModels.detect_kind(np.zeros((10, 10, 3), np.uint8)) == "raw"
    assert BestModels.detect_kind(np.zeros((10, 10, 6), np.uint8)) == "aligned"


@pytest.fixture(scope="module")
def models():
    return BestModels(device="cpu")


@requires_models
def test_models_use_dedicated_classes(models):
    assert isinstance(models.raw, RawBestModel)
    assert isinstance(models.aligned, AlignedBestModel)


@requires_models
def test_predict_single_raw(models):
    image = np.random.randint(0, 256, (224, 224), dtype=np.uint8)
    probabilities = models.raw.predict(image)

    assert probabilities.shape == (models.num_classes,)
    assert probabilities.sum() == pytest.approx(1.0, abs=1e-5)


@requires_models
def test_predict_single_aligned(models):
    image = np.random.randint(0, 256, (224, 224, 6), dtype=np.uint8)
    probabilities = models.aligned.predict(image)

    assert probabilities.shape == (models.num_classes,)
    assert probabilities.sum() == pytest.approx(1.0, abs=1e-5)


@requires_models
def test_dispatch_by_kind(models):
    raw_image = np.random.randint(0, 256, (224, 224), dtype=np.uint8)
    aligned_image = np.random.randint(0, 256, (224, 224, 6), dtype=np.uint8)

    assert models.predict(raw_image).shape == (models.num_classes,)
    assert models.predict(aligned_image).shape == (models.num_classes,)


@requires_models
def test_predict_batch(models):
    images = [
        np.random.randint(0, 256, (224, 224), dtype=np.uint8) for _ in range(3)
    ]
    probabilities = models.raw.predict_batch(images)

    assert probabilities.shape == (3, models.num_classes)
    assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5)


@requires_models
def test_predict_both(models):
    image = np.random.randint(0, 256, (224, 224), dtype=np.uint8)
    distributions = models.predict_both(image)

    assert set(distributions) == {"raw", "aligned", "ensemble"}
    for probabilities in distributions.values():
        assert probabilities.shape == (models.num_classes,)
        assert probabilities.sum() == pytest.approx(1.0, abs=1e-5)


@requires_models
def test_top_class(models):
    probabilities = np.array([0.1, 0.2, 0.6, 0.1], dtype=np.float32)
    name, confidence = models.top_class(probabilities)
    assert name == models.classes[2]
    assert confidence == pytest.approx(0.6)
