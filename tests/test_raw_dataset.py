import os

import cv2
import numpy as np
import pytest

from brain_hackathon.data.raw_dataset import (
    RawImageDataset,
    base_image_path,
    build_raw_index,
)


def write_image(path, value=50, shape=(8, 8)):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, np.full(shape, value, dtype=np.uint8))


@pytest.fixture
def raw_root(tmp_path):
    root = tmp_path / "raw"
    glioma = root / "Train" / "glioma"
    meningioma = root / "Train" / "meningioma"
    test_glioma = root / "Test" / "glioma"

    write_image(glioma / "G_1.jpg", 10)
    write_image(glioma / "G_1_VF_.jpg", 11)
    write_image(glioma / "G_1_RO_.jpg", 12)
    write_image(glioma / "100.jpg", 13)
    write_image(glioma / "G_3_BR_.jpg", 14)  # no base anywhere -> skipped

    write_image(meningioma / "M_1.jpg", 20)
    write_image(meningioma / "M_1_HF_.jpg", 21)

    write_image(test_glioma / "G_2.jpg", 30)
    write_image(test_glioma / "G_2_DA_.jpg", 31)
    write_image(test_glioma / "G_4.jpg", 40)
    # base lives in Test while the group member is in Train
    write_image(glioma / "G_4_BR_.jpg", 41)

    return str(root)


def test_base_image_path(raw_root):
    glioma = os.path.join(raw_root, "Train", "glioma")

    base = base_image_path(os.path.join(glioma, "G_1_VF_.jpg"))
    assert os.path.basename(base) == "G_1.jpg"

    plain = os.path.join(glioma, "G_1.jpg")
    assert base_image_path(plain) == plain

    assert base_image_path(os.path.join(glioma, "G_3_BR_.jpg")) is None


def test_index_collapses_groups(raw_root):
    classes, index, missing = build_raw_index(raw_root, "Train")
    names = {os.path.basename(path) for path, _ in index}

    assert classes == ["glioma", "meningioma"]
    assert "G_1.jpg" in names
    assert "G_1_VF_.jpg" not in names
    assert "G_1_RO_.jpg" not in names
    assert "M_1.jpg" in names
    assert "M_1_HF_.jpg" not in names
    assert "100.jpg" in names
    # base recovered from the Test split
    assert "G_4.jpg" in names
    assert not any(name.endswith("_BR_.jpg") for name in names)
    # group without a base is skipped and reported
    assert any(os.path.basename(path) == "G_3_BR_.jpg" for path in missing)

    labels = {os.path.basename(path): label for path, label in index}
    assert labels["G_1.jpg"] == classes.index("glioma")
    assert labels["M_1.jpg"] == classes.index("meningioma")


def test_dataset_item_shape(raw_root):
    dataset = RawImageDataset(raw_root, split="Train", image_size=16)
    image, label = dataset[0]

    assert tuple(image.shape) == (1, 16, 16)
    assert image.dtype.is_floating_point
    assert float(image.min()) >= 0.0
    assert float(image.max()) <= 1.0
    assert label in range(len(dataset.classes))


def test_combined_splits_are_deduplicated(raw_root):
    dataset = RawImageDataset(raw_root, split=("Train", "Test"))
    paths = [os.path.abspath(path) for path in dataset.paths()]

    assert len(paths) == len(set(paths))
    assert dataset.classes == ["glioma", "meningioma"]
    # G_4 base is referenced by both the Test plain file and the Train group
    assert sum(p.endswith("G_4.jpg") for p in paths) == 1
