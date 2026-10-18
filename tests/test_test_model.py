import os

import cv2
import numpy as np
import pytest
import torch

from brain_hackathon.data.dataset import MODALITIES, AlignedMRIClassificationDataset
from brain_hackathon.model.resnet_rad import (
    RadImageNetResNet50,
    RadImageNetResNet50Single,
)
from brain_hackathon.train.test_model import (
    cross_validation_test_indices,
    infer_model_config,
    load_checkpoint,
    train_holdout_indices,
)
from brain_hackathon.train.utils import ALL_SPLITS


def _write_series(class_dir, series_id):
    os.makedirs(class_dir, exist_ok=True)
    image = np.full((8, 8), (int(series_id) % 200) + 1, dtype=np.uint8)
    for modality in MODALITIES:
        cv2.imwrite(os.path.join(class_dir, f"{series_id}_{modality}.jpg"), image)


def test_cross_validation_test_indices(tmp_path):
    root = tmp_path / "aligned"
    for i in range(1, 21):
        _write_series(root / "Train" / "glioma", str(i))
    for i in range(1, 21):
        _write_series(root / "Train" / "meningioma", str(i))

    indices, classes = cross_validation_test_indices(str(root))
    base = AlignedMRIClassificationDataset(
        root=str(root), split=ALL_SPLITS, augment=None
    )

    assert classes == base.classes
    assert len(indices) == 8  # 20% of 40 series
    assert all(0 <= index < len(base) for index in indices)
    assert len(set(indices)) == len(indices)


def test_train_holdout_indices(tmp_path):
    root = tmp_path / "aligned"
    for i in range(1, 21):
        _write_series(root / "Train" / "glioma", str(i))
    for i in range(1, 21):
        _write_series(root / "Train" / "meningioma", str(i))

    indices, classes = train_holdout_indices(str(root), "aligned")
    base = AlignedMRIClassificationDataset(
        root=str(root), split="Train", augment=None
    )

    assert classes == base.classes
    assert len(indices) == 8  # 20% of 40 series
    assert all(0 <= index < len(base) for index in indices)
    assert len(set(indices)) == len(indices)


@pytest.mark.parametrize("channels", [1, 6])
def test_infer_model_config_sequential_head(channels):
    state = {
        "backbone.0.weight": torch.zeros(64, channels, 7, 7),
        "head.0.weight": torch.zeros(16, 2048),
        "head.2.weight": torch.zeros(4, 16),
    }
    assert infer_model_config(state) == (channels, [16, 4])


def test_infer_model_config_single_linear_head():
    state = {
        "backbone.0.weight": torch.zeros(64, 6, 7, 7),
        "head.weight": torch.zeros(4, 2048),
    }
    assert infer_model_config(state) == (6, [4])


def test_infer_model_config_rejects_missing_parts():
    with pytest.raises(ValueError):
        infer_model_config({"head.weight": torch.zeros(4, 2048)})
    with pytest.raises(ValueError):
        infer_model_config({"backbone.0.weight": torch.zeros(64, 6, 7, 7)})


def _assert_roundtrip(tmp_path, model, channels):
    path = tmp_path / "best.pt"
    torch.save({"model": model.state_dict(), "epoch": 3, "val_loss": 0.5}, path)

    loaded, info = load_checkpoint(str(path), device="cpu")

    assert info["in_channels"] == channels
    assert info["head_dims"] == [8, 4]
    assert info["epoch"] == 3
    assert info["val_loss"] == 0.5
    with torch.no_grad():
        assert loaded(torch.zeros(1, channels, 224, 224)).shape == (1, 4)


def test_load_checkpoint_roundtrip_raw(tmp_path):
    model = RadImageNetResNet50Single(
        dim=[8, 4], pretrained=False, verbose=False
    )
    _assert_roundtrip(tmp_path, model, 1)


def test_load_checkpoint_roundtrip_aligned(tmp_path):
    model = RadImageNetResNet50(
        dim=[8, 4], in_channels=6, pretrained=False, verbose=False
    )
    _assert_roundtrip(tmp_path, model, 6)
