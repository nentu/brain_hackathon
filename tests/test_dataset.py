import os

import cv2
import numpy as np
import pytest
import torch

from brain_hackathon.data.dataset import (
    MODALITIES,
    AlignedMRIClassificationDataset,
    AlignedMRIUnlabeledDataset,
    build_series_index,
)

SHAPE = (8, 8)


def write_series(class_dir, series_id, modalities, shape=SHAPE):
    os.makedirs(class_dir, exist_ok=True)
    for i, modality in enumerate(modalities):
        value = 10 * (MODALITIES.index(modality) + 1)
        image = np.full(shape, value, dtype=np.uint8)
        cv2.imwrite(os.path.join(class_dir, f"{series_id}_{modality}.jpg"), image)


@pytest.fixture
def aligned_root(tmp_path):
    root = tmp_path / "aligned"
    glioma = root / "Train" / "glioma"
    meningioma = root / "Train" / "meningioma"

    write_series(glioma, "1", MODALITIES)
    write_series(glioma, "2", [m for m in MODALITIES if m != "HF"])
    write_series(meningioma, "3", MODALITIES)

    return str(root)


def test_index_and_classes(aligned_root):
    classes, index = build_series_index(aligned_root, "Train")
    assert classes == ["glioma", "meningioma"]
    assert [entry[0] for entry in index] == ["1", "2", "3"]
    assert set(index[0][2]) == set(MODALITIES)
    assert "HF" not in index[1][2]


def test_length_and_channel_names(aligned_root):
    dataset = AlignedMRIClassificationDataset(root=aligned_root, split="Train")
    assert len(dataset) == 3
    assert dataset.channel_names == MODALITIES
    assert dataset.classes == ["glioma", "meningioma"]


def test_tensor_shape_and_dtype(aligned_root):
    dataset = AlignedMRIClassificationDataset(root=aligned_root, split="Train")
    tensor, label = dataset[0]
    assert isinstance(tensor, torch.Tensor)
    assert tensor.dtype == torch.float32
    assert tensor.shape == (len(MODALITIES), *SHAPE)
    assert label == dataset.class_to_idx["glioma"]


def test_missing_modality_is_filled(aligned_root):
    dataset = AlignedMRIClassificationDataset(root=aligned_root, split="Train")
    tensor, _ = dataset[1]

    hf = MODALITIES.index("HF")
    assert torch.all(tensor[hf] == 0.0)

    br = MODALITIES.index("BR")
    assert torch.all(tensor[br] > 0.0)


def test_custom_fill_value(aligned_root):
    dataset = AlignedMRIClassificationDataset(
        root=aligned_root, split="Train", fill_value=1.0
    )
    tensor, _ = dataset[1]
    hf = MODALITIES.index("HF")
    assert torch.allclose(tensor[hf], torch.ones(*SHAPE))


def test_normalize_toggle(aligned_root):
    raw = AlignedMRIClassificationDataset(
        root=aligned_root, split="Train", normalize=False
    )
    tensor, _ = raw[0]
    assert tensor[MODALITIES.index("BR")].unique().tolist() == [10.0]
    assert tensor[MODALITIES.index("VF")].unique().tolist() == [60.0]

    normalized = AlignedMRIClassificationDataset(
        root=aligned_root, split="Train", normalize=True
    )
    tensor, _ = normalized[0]
    assert tensor.max().item() == pytest.approx(60.0 / 255.0)


def test_unlabeled_dataset(aligned_root):
    dataset = AlignedMRIUnlabeledDataset(root=aligned_root, split="Train")
    tensor = dataset[0]
    assert torch.is_tensor(tensor)
    assert tensor.shape == (len(MODALITIES), *SHAPE)


def test_class_filter(aligned_root):
    dataset = AlignedMRIClassificationDataset(
        root=aligned_root, split="Train", classes=["meningioma"]
    )
    assert len(dataset) == 1
    tensor, label = dataset[0]
    assert dataset.classes == ["meningioma"]
    assert label == 0
