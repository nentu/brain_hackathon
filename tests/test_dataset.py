import os

import albumentations as A
import cv2
import numpy as np
import pytest
import torch

from brain_hackathon.data.dataset import (
    MODALITIES,
    AlignedMRIClassificationDataset,
    AlignedMRIUnlabeledDataset,
    build_series_index,
    compute_foreground_stats,
    foreground_stats,
    three_sigma_normalize,
)

SHAPE = (8, 8)


def _value(modality):
    return 10 * (MODALITIES.index(modality) + 1)


def write_series(class_dir, series_id, modalities, shape=SHAPE):
    os.makedirs(class_dir, exist_ok=True)
    for modality in modalities:
        image = np.full(shape, _value(modality), dtype=np.uint8)
        cv2.imwrite(os.path.join(class_dir, f"{series_id}_{modality}.jpg"), image)


def write_shared_series(class_dir, series_id, image):
    os.makedirs(class_dir, exist_ok=True)
    for modality in MODALITIES:
        cv2.imwrite(os.path.join(class_dir, f"{series_id}_{modality}.jpg"), image)


@pytest.fixture
def aligned_root(tmp_path):
    root = tmp_path / "aligned"
    glioma = root / "Train" / "glioma"
    meningioma = root / "Train" / "meningioma"

    write_series(glioma, "1", MODALITIES)
    write_series(glioma, "2", [m for m in MODALITIES if m != "HF"])
    write_series(meningioma, "3", MODALITIES)

    gradient = np.arange(np.prod(SHAPE), dtype=np.uint8).reshape(SHAPE) * 3
    write_shared_series(glioma, "4", gradient)

    return str(root)


def test_index_and_classes(aligned_root):
    classes, index = build_series_index(aligned_root, "Train")
    assert classes == ["glioma", "meningioma"]
    assert [entry[0] for entry in index] == ["1", "2", "4", "3"]
    assert set(index[0][2]) == set(MODALITIES)
    assert "HF" not in index[1][2]


def test_length_and_channel_names(aligned_root):
    dataset = AlignedMRIClassificationDataset(root=aligned_root, split="Train")
    assert len(dataset) == 4
    assert dataset.channel_names == MODALITIES
    assert dataset.classes == ["glioma", "meningioma"]


def test_tensor_shape_and_dtype(aligned_root):
    dataset = AlignedMRIClassificationDataset(
        root=aligned_root, split="Train", normalization=None
    )
    tensor, label = dataset[0]
    assert isinstance(tensor, torch.Tensor)
    assert tensor.dtype == torch.float32
    assert tensor.shape == (len(MODALITIES), *SHAPE)
    assert label == dataset.class_to_idx["glioma"]


def test_missing_modality_is_filled(aligned_root):
    dataset = AlignedMRIClassificationDataset(
        root=aligned_root, split="Train", normalization=None
    )
    tensor, _ = dataset[1]
    assert torch.all(tensor[MODALITIES.index("HF")] == 0.0)
    assert torch.all(tensor[MODALITIES.index("BR")] == 10.0)


def test_custom_fill_value(aligned_root):
    dataset = AlignedMRIClassificationDataset(
        root=aligned_root, split="Train", normalization=None, fill_value=5
    )
    tensor, _ = dataset[1]
    assert torch.all(tensor[MODALITIES.index("HF")] == 5.0)


def test_no_normalization_keeps_raw_values(aligned_root):
    dataset = AlignedMRIClassificationDataset(
        root=aligned_root, split="Train", normalization=None
    )
    tensor, _ = dataset[0]
    assert tensor[MODALITIES.index("BR")].unique().tolist() == [10.0]
    assert tensor[MODALITIES.index("VF")].unique().tolist() == [60.0]


def test_three_sigma_normalization_with_fixed_stats(aligned_root):
    dataset = AlignedMRIClassificationDataset(
        root=aligned_root, split="Train", norm_stats=(30.0, 10.0)
    )
    tensor, _ = dataset[0]
    assert tensor[MODALITIES.index("BR")].max().item() == pytest.approx(10.0 / 60.0)
    assert tensor[MODALITIES.index("VF")].max().item() == pytest.approx(1.0)


def test_three_sigma_normalization_range(aligned_root):
    dataset = AlignedMRIClassificationDataset(root=aligned_root, split="Train")
    tensor, _ = dataset[0]
    assert float(tensor.min()) >= 0.0
    assert float(tensor.max()) <= 1.0
    assert tensor[MODALITIES.index("VF")].mean() > tensor[MODALITIES.index("BR")].mean()


def test_foreground_stats_ignores_zero():
    mean, std = foreground_stats(np.array([[0, 0, 2], [4, 0, 0]], dtype=np.float32))
    assert mean == pytest.approx(3.0)
    assert std == pytest.approx(1.0)


def test_foreground_stats_ignores_zero_channel():
    stack = np.stack(
        [np.full((4, 4), 10, np.float32), np.zeros((4, 4), np.float32)], axis=-1
    )
    mean, std = foreground_stats(stack)
    assert mean == pytest.approx(10.0)
    assert std == pytest.approx(0.0)


def test_three_sigma_normalize_formula():
    array = np.array([-100, -20, 10, 40, 100], dtype=np.float32)
    out = three_sigma_normalize(array, mean=10.0, std=10.0)
    assert out[1] == pytest.approx(0.0)  # mean - 3 sigma
    assert out[2] == pytest.approx(0.5)  # mean
    assert out[3] == pytest.approx(1.0)  # mean + 3 sigma
    assert out[0] == pytest.approx(0.0)  # clipped low
    assert out[4] == pytest.approx(1.0)  # clipped high


def test_compute_foreground_stats():
    mean, std = compute_foreground_stats([np.array([[0, 2, 4]], dtype=np.float32)])
    assert mean == pytest.approx(3.0)
    assert std == pytest.approx(1.0)


def test_augmentation_is_shared_across_channels(aligned_root):
    dataset = AlignedMRIClassificationDataset(
        root=aligned_root,
        split="Train",
        augment=A.Compose([A.HorizontalFlip(p=1.0)]),
        normalization=None,
    )
    index = dataset.series_ids().index("4")
    tensor, _ = dataset[index]

    for c in range(1, len(MODALITIES)):
        assert torch.equal(tensor[0], tensor[c])

    stored = cv2.imread(
        os.path.join(aligned_root, "Train", "glioma", "4_BR.jpg"),
        cv2.IMREAD_GRAYSCALE,
    )
    assert np.array_equal(tensor[0].numpy(), cv2.flip(stored, 1).astype(np.float32))


def test_augmentation_applies_and_keeps_shape(aligned_root):
    dataset = AlignedMRIClassificationDataset(
        root=aligned_root,
        split="Train",
        augment=A.Compose([A.HorizontalFlip(p=1.0), A.Affine(rotate=(-15, 15), p=1.0)]),
        normalization=None,
    )
    tensor, _ = dataset[0]
    assert tensor.shape == (len(MODALITIES), *SHAPE)


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
    _tensor, label = dataset[0]
    assert dataset.classes == ["meningioma"]
    assert label == 0


def test_single_split_is_normalized_to_tuple(aligned_root):
    dataset = AlignedMRIClassificationDataset(root=aligned_root, split="Train")
    assert dataset.splits == ("Train",)


def test_multiple_splits_combined(tmp_path):
    root = tmp_path / "aligned"
    write_series(root / "Train" / "glioma", "1", MODALITIES)
    write_series(root / "Train" / "meningioma", "3", MODALITIES)
    write_series(root / "Test" / "glioma", "2", MODALITIES)

    dataset = AlignedMRIClassificationDataset(root=str(root), split=("Train", "Test"))

    assert dataset.splits == ("Train", "Test")
    assert dataset.classes == ["glioma", "meningioma"]
    assert len(dataset) == 3
    assert [(sid, cls) for sid, cls, _ in dataset.index] == [
        ("1", "glioma"),
        ("3", "meningioma"),
        ("2", "glioma"),
    ]

    tensor, label = dataset[2]
    assert tensor.shape == (len(MODALITIES), *SHAPE)
    assert dataset.classes[label] == "glioma"
