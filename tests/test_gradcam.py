import cv2
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from brain_hackathon.eval import (
    AlignedBestModel,
    BestModels,
    RawBestModel,
    grad_cam_plus_plus,
)
from brain_hackathon.eval.base import DEFAULT_CLASSES, BaseBestModel
from brain_hackathon.model.resnet_rad import (
    RadImageNetResNet50,
    RadImageNetResNet50Single,
)


class _Adapter:
    """Minimal model wrapper for Grad-CAM tests (no checkpoint needed)."""

    def __init__(self, model, preprocess):
        self.model = model
        self.device = torch.device("cpu")
        self.classes = DEFAULT_CLASSES
        self._preprocess = preprocess

    def preprocess(self, image):
        return self._preprocess(image)

    def grad_cam(self, image, target_class=None, **kwargs):
        return BaseBestModel.grad_cam(self, image, target_class=target_class, **kwargs)


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


def test_grad_cam_raw(raw_adapter):
    result = grad_cam_plus_plus(raw_adapter, _raw_image())

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

    assert result["overlay"].shape == (224, 224, 3)
    assert result["overlay"].dtype == np.uint8

    index = result["class_index"]
    assert 0 <= index < len(DEFAULT_CLASSES)
    assert result["class_name"] == DEFAULT_CLASSES[index]
    assert 0.0 <= result["probability"] <= 1.0


def test_grad_cam_target_class(raw_adapter):
    image = _raw_image(seed=2)
    default = grad_cam_plus_plus(raw_adapter, image)
    expected = (default["class_index"] + 1) % len(DEFAULT_CLASSES)

    by_name = grad_cam_plus_plus(
        raw_adapter, image, target_class=DEFAULT_CLASSES[expected]
    )
    by_index = grad_cam_plus_plus(raw_adapter, image, target_class=expected)

    assert by_name["class_index"] == expected
    assert by_index["class_index"] == expected
    assert by_name["class_name"] == DEFAULT_CLASSES[expected]


def test_grad_cam_invalid_arguments(raw_adapter):
    image = _raw_image()

    with pytest.raises(ValueError, match="unknown class"):
        grad_cam_plus_plus(raw_adapter, image, target_class="missing")
    with pytest.raises(ValueError, match="out of range"):
        grad_cam_plus_plus(raw_adapter, image, target_class=99)
    with pytest.raises(ValueError, match="alpha"):
        grad_cam_plus_plus(raw_adapter, image, alpha=1.5)
    with pytest.raises(ValueError, match="unknown layer"):
        grad_cam_plus_plus(raw_adapter, image, layer="no.such.layer")


def test_grad_cam_layer_override(raw_adapter):
    result = grad_cam_plus_plus(raw_adapter, _raw_image(seed=3), layer="backbone.7")
    assert result["heatmap"].shape == (224, 224)


def test_grad_cam_aligned(aligned_adapter):
    image = _aligned_image()
    result = grad_cam_plus_plus(aligned_adapter, image)

    assert result["heatmap"].shape == (224, 224)
    assert result["overlay"].shape == (224, 224, 3)
    assert result["overlay"].dtype == np.uint8

    modality_result = grad_cam_plus_plus(aligned_adapter, image, base_modality="BR")
    assert modality_result["overlay"].shape == (224, 224, 3)


def test_grad_cam_aligned_base_modality_errors(aligned_adapter):
    image = np.zeros((224, 224, 6), dtype=np.uint8)
    image[:, :, 0] = 128

    with pytest.raises(ValueError, match="missing or empty"):
        grad_cam_plus_plus(aligned_adapter, image, base_modality="DA")
    with pytest.raises(ValueError, match="unknown modality"):
        grad_cam_plus_plus(aligned_adapter, image, base_modality="XX")
    with pytest.raises(ValueError, match="all aligned channels are empty"):
        grad_cam_plus_plus(aligned_adapter, np.zeros((224, 224, 6), np.uint8))


def test_grad_cam_raw_rejects_base_modality(raw_adapter):
    with pytest.raises(ValueError, match="only supported for aligned"):
        grad_cam_plus_plus(raw_adapter, _raw_image(), base_modality="BR")


def test_grad_cam_saves_overlay(raw_adapter, tmp_path):
    path = tmp_path / "nested" / "heatmap.png"
    result = grad_cam_plus_plus(raw_adapter, _raw_image(), output_path=path)

    assert path.is_file()
    written = cv2.imread(str(path))
    assert written is not None
    assert written.shape == result["overlay"].shape


def test_base_best_model_grad_cam_method(raw_adapter):
    result = BaseBestModel.grad_cam(raw_adapter, _raw_image())
    assert result["overlay"].shape == (224, 224, 3)


def test_grad_cam_restores_training_mode(raw_adapter):
    network = raw_adapter.model
    network.train()
    try:
        grad_cam_plus_plus(raw_adapter, _raw_image())
        assert network.training
    finally:
        network.eval()


def test_grad_cam_preserves_parameter_gradients(raw_adapter):
    network = raw_adapter.model
    network.eval()
    network.zero_grad(set_to_none=True)
    parameter = next(network.parameters())
    parameter.grad = torch.ones_like(parameter)

    try:
        grad_cam_plus_plus(raw_adapter, _raw_image())
        assert torch.equal(parameter.grad, torch.ones_like(parameter))
        assert all(p.grad is None for p in network.parameters() if p is not parameter)
    finally:
        network.zero_grad(set_to_none=True)


def test_grad_cam_does_not_accumulate_gradients(raw_adapter):
    network = raw_adapter.model
    network.eval()
    network.zero_grad(set_to_none=True)

    grad_cam_plus_plus(raw_adapter, _raw_image())
    assert all(p.grad is None for p in network.parameters())


def test_grad_cam_no_grad_context(raw_adapter):
    image = _raw_image(seed=4)
    normal = grad_cam_plus_plus(raw_adapter, image)
    with torch.no_grad():
        inside = grad_cam_plus_plus(raw_adapter, image)

    assert np.array_equal(normal["heatmap"], inside["heatmap"])


def test_best_models_grad_cam_dispatch(raw_adapter, aligned_adapter):
    models = object.__new__(BestModels)
    models.raw = raw_adapter
    models.aligned = aligned_adapter
    models.classes = DEFAULT_CLASSES

    raw_result = models.grad_cam(_raw_image())
    aligned_result = models.grad_cam(_aligned_image())

    assert raw_result["overlay"].shape == (224, 224, 3)
    assert aligned_result["overlay"].shape == (224, 224, 3)
