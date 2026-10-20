import itertools
import math

import cv2
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from brain_hackathon.eval import (
    AlignedBestModel,
    BestModels,
    RawBestModel,
    shap,
    sharp,
)
from brain_hackathon.eval.base import DEFAULT_CLASSES, BaseBestModel
from brain_hackathon.eval.shapley import _solve_shapley
from brain_hackathon.model.resnet_rad import (
    RadImageNetResNet50,
    RadImageNetResNet50Single,
)


class _Adapter:
    """Minimal model wrapper for SHAP tests (no checkpoint needed)."""

    def __init__(self, model, preprocess):
        self.model = model
        self.device = torch.device("cpu")
        self.classes = DEFAULT_CLASSES
        self._preprocess = preprocess

    def preprocess(self, image):
        return self._preprocess(image)

    def shap(self, image, target_class=None, **kwargs):
        return BaseBestModel.shap(self, image, target_class=target_class, **kwargs)

    sharp = shap


def _raw_image(seed=0):
    return np.random.default_rng(seed).integers(0, 256, (224, 224), dtype=np.uint8)


def _aligned_image(seed=1):
    return np.random.default_rng(seed).integers(0, 256, (224, 224, 6), dtype=np.uint8)


@pytest.fixture(scope="module")
def raw_adapter():
    torch.manual_seed(0)
    model = RadImageNetResNet50Single(dim=[4], pretrained=False, verbose=False)
    return _Adapter(model, RawBestModel.preprocess)


@pytest.fixture(scope="module")
def aligned_adapter():
    torch.manual_seed(0)
    model = RadImageNetResNet50(dim=[4], in_channels=6, pretrained=False, verbose=False)
    return _Adapter(model, AlignedBestModel.preprocess)


def _exact_shapley(masks, values):
    n_features = masks.shape[1]
    lookup = {tuple(mask): value for mask, value in zip(masks, values)}
    phi = np.zeros(n_features, dtype=np.float64)
    for feature in range(n_features):
        for mask, value in zip(masks, values):
            if mask[feature]:
                continue
            size = int(mask.sum())
            with_feature = mask.copy()
            with_feature[feature] = True
            weight = (
                math.factorial(size)
                * math.factorial(n_features - size - 1)
                / math.factorial(n_features)
            )
            phi[feature] += weight * (lookup[tuple(with_feature)] - value)
    return phi


def test_solver_matches_exact_shapley():
    n_features = 4
    masks = np.array(
        list(itertools.product([False, True], repeat=n_features)), dtype=bool
    )
    feature_weights = np.array([0.4, -0.2, 0.7, 0.1])
    values = np.sin(masks.astype(float) @ feature_weights)
    values += 0.3 * masks[:, 0] * masks[:, 2]

    phi, value_empty, value_full = _solve_shapley(masks, values, n_features)
    expected = _exact_shapley(masks, values)

    assert np.allclose(phi, expected, atol=1e-7)
    assert float(phi.sum()) == pytest.approx(value_full - value_empty, abs=1e-7)


def test_shap_raw(raw_adapter):
    result = shap(
        raw_adapter,
        _raw_image(),
        patch_grid=3,
        n_samples=32,
        batch_size=16,
    )

    assert set(result) == {
        "heatmap",
        "overlay",
        "class_index",
        "class_name",
        "probability",
    }
    assert result["heatmap"].shape == (224, 224)
    assert result["heatmap"].dtype == np.float32
    assert np.isfinite(result["heatmap"]).all()
    assert float(result["heatmap"].min()) >= 0.0
    assert float(result["heatmap"].max()) <= 1.0
    assert float(result["heatmap"].std()) > 0.0
    assert result["overlay"].shape == (224, 224, 3)
    assert result["overlay"].dtype == np.uint8

    index = result["class_index"]
    assert 0 <= index < len(DEFAULT_CLASSES)
    assert result["class_name"] == DEFAULT_CLASSES[index]
    assert 0.0 <= result["probability"] <= 1.0


def test_shap_with_mask(raw_adapter):
    result = shap(
        raw_adapter,
        _raw_image(seed=5),
        patch_grid=3,
        n_samples=32,
        batch_size=16,
        mask=True,
        mask_radius=10.0,
    )

    assert result["heatmap"].shape == (224, 224)
    assert result["overlay"].shape == (224, 224, 3)
    assert float(result["heatmap"].min()) >= 0.0
    assert float(result["heatmap"].max()) <= 1.0


def test_shap_aligned(aligned_adapter):
    result = shap(
        aligned_adapter,
        _aligned_image(),
        patch_grid=3,
        n_samples=32,
        batch_size=16,
        base_modality="BR",
    )

    assert result["heatmap"].shape == (224, 224)
    assert result["overlay"].shape == (224, 224, 3)
    assert result["overlay"].dtype == np.uint8


def test_shap_target_class(raw_adapter):
    image = _raw_image(seed=2)
    default = shap(raw_adapter, image, patch_grid=3, n_samples=32, batch_size=16)
    expected = (default["class_index"] + 1) % len(DEFAULT_CLASSES)

    by_name = shap(
        raw_adapter,
        image,
        target_class=DEFAULT_CLASSES[expected],
        patch_grid=3,
        n_samples=32,
        batch_size=16,
    )
    by_index = shap(
        raw_adapter,
        image,
        target_class=expected,
        patch_grid=3,
        n_samples=32,
        batch_size=16,
    )

    assert by_name["class_index"] == expected
    assert by_index["class_index"] == expected
    assert np.array_equal(by_name["heatmap"], by_index["heatmap"])


def test_shap_invalid_arguments(raw_adapter):
    image = _raw_image()

    with pytest.raises(ValueError, match="too small"):
        shap(raw_adapter, image, patch_grid=3, n_samples=5)
    with pytest.raises(ValueError, match="dimensions must be positive"):
        shap(raw_adapter, image, patch_grid=0, n_samples=8)
    with pytest.raises(ValueError, match="alpha"):
        shap(raw_adapter, image, alpha=1.5)
    with pytest.raises(ValueError, match="baseline shape"):
        shap(
            raw_adapter,
            image,
            patch_grid=2,
            n_samples=8,
            baseline=np.zeros((2, 2, 2, 2), dtype=np.float32),
        )


def test_shap_deterministic(raw_adapter):
    kwargs = {"patch_grid": 3, "n_samples": 32, "batch_size": 16, "seed": 11}
    first = shap(raw_adapter, _raw_image(), **kwargs)
    second = shap(raw_adapter, _raw_image(), **kwargs)

    assert np.array_equal(first["heatmap"], second["heatmap"])


def test_shap_saves_overlay(raw_adapter, tmp_path):
    path = tmp_path / "nested" / "shap.png"
    result = shap(
        raw_adapter,
        _raw_image(),
        patch_grid=3,
        n_samples=32,
        batch_size=16,
        output_path=path,
    )

    assert path.is_file()
    written = cv2.imread(str(path))
    assert written is not None
    assert written.shape == result["overlay"].shape


def test_shap_methods(raw_adapter):
    image = _raw_image(seed=3)
    kwargs = {"patch_grid": 3, "n_samples": 32, "batch_size": 16}

    by_shap = BaseBestModel.shap(raw_adapter, image, **kwargs)
    by_sharp = BaseBestModel.sharp(raw_adapter, image, **kwargs)
    by_adapter = raw_adapter.shap(image, **kwargs)

    assert np.array_equal(by_shap["heatmap"], by_sharp["heatmap"])
    assert np.array_equal(by_shap["heatmap"], by_adapter["heatmap"])


def test_sharp_alias_is_shap():
    assert sharp is shap


def test_shap_does_not_accumulate_gradients(raw_adapter):
    network = raw_adapter.model
    network.eval()
    network.zero_grad(set_to_none=True)

    shap(
        raw_adapter,
        _raw_image(),
        patch_grid=3,
        n_samples=32,
        batch_size=16,
    )

    assert all(parameter.grad is None for parameter in network.parameters())


def test_shap_restores_training_mode(raw_adapter):
    network = raw_adapter.model
    network.train()
    try:
        shap(
            raw_adapter,
            _raw_image(),
            patch_grid=3,
            n_samples=32,
            batch_size=16,
        )
        assert network.training
    finally:
        network.eval()


def test_shap_no_grad_context(raw_adapter):
    image = _raw_image(seed=4)
    kwargs = {"patch_grid": 3, "n_samples": 32, "batch_size": 16}
    normal = shap(raw_adapter, image, **kwargs)
    with torch.no_grad():
        inside = shap(raw_adapter, image, **kwargs)

    assert np.array_equal(normal["heatmap"], inside["heatmap"])


def test_best_models_shap_dispatch(raw_adapter, aligned_adapter):
    models = object.__new__(BestModels)
    models.raw = raw_adapter
    models.aligned = aligned_adapter
    models.classes = DEFAULT_CLASSES

    kwargs = {"patch_grid": 3, "n_samples": 32, "batch_size": 16}
    raw_result = models.shap(_raw_image(), **kwargs)
    aligned_result = models.sharp(_aligned_image(), **kwargs)

    assert raw_result["overlay"].shape == (224, 224, 3)
    assert aligned_result["overlay"].shape == (224, 224, 3)
