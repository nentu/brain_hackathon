import os
from collections import defaultdict

import cv2
import numpy as np
import pytest

from brain_hackathon.data.dataset import MODALITIES, AlignedMRIClassificationDataset
from brain_hackathon.train.utils import build_datasets, stratified_indices


def write_series(class_dir, series_id, shape=(8, 8)):
    os.makedirs(class_dir, exist_ok=True)
    image = np.arange(np.prod(shape), dtype=np.uint8).reshape(shape)
    for modality in MODALITIES:
        cv2.imwrite(os.path.join(class_dir, f"{series_id}_{modality}.jpg"), image)


@pytest.fixture
def aligned_root(tmp_path):
    root = tmp_path / "aligned"
    for i in range(1, 21):
        write_series(root / "Train" / "glioma", str(i))
    for i in range(21, 31):
        write_series(root / "Test" / "glioma", str(i))
    for i in range(1, 21):
        write_series(root / "Train" / "meningioma", str(i))
    # same series id 5 present in both splits (leak case)
    write_series(root / "Test" / "glioma", "5", shape=(4, 4))
    return str(root)


def test_build_datasets_partitions_all_samples(aligned_root):
    base = AlignedMRIClassificationDataset(
        root=aligned_root, split=("Train", "Test")
    )
    train, val, test = build_datasets(
        root=aligned_root,
        splits=("Train", "Test"),
        train_transform=None,
        eval_transform=None,
    )
    assert len(train) + len(val) + len(test) == len(base)
    assert len(train) > len(val) > 0
    assert len(test) > 0


def test_build_datasets_stratified(aligned_root):
    base = AlignedMRIClassificationDataset(
        root=aligned_root, split=("Train", "Test")
    )
    train, val, test = build_datasets(
        root=aligned_root,
        splits=("Train", "Test"),
        train_transform=None,
        eval_transform=None,
    )

    def class_counts(indices):
        counts = defaultdict(int)
        for index in indices:
            counts[base.index[index][1]] += 1
        return counts

    total = class_counts(range(len(base)))
    train_counts = class_counts(train.indices)
    for class_name, n in total.items():
        assert train_counts[class_name] / n == pytest.approx(0.7, abs=0.15)


def test_no_index_overlap(aligned_root):
    train, val, test = build_datasets(
        root=aligned_root,
        splits=("Train", "Test"),
        train_transform=None,
        eval_transform=None,
    )
    train_idx, val_idx, test_idx = set(train.indices), set(val.indices), set(test.indices)
    assert not train_idx & val_idx
    assert not train_idx & test_idx
    assert not val_idx & test_idx


def test_series_never_spans_folds(aligned_root):
    base = AlignedMRIClassificationDataset(
        root=aligned_root, split=("Train", "Test")
    )
    train, val, test = build_datasets(
        root=aligned_root,
        splits=("Train", "Test"),
        train_transform=None,
        eval_transform=None,
    )
    fold_of = {}
    for name, subset in (("train", train), ("val", val), ("test", test)):
        for index in subset.indices:
            series_id, class_name, _ = base.index[index]
            fold_of.setdefault((class_name, series_id), set()).add(name)

    assert all(len(folds) == 1 for folds in fold_of.values())


def test_stratified_indices_rejects_bad_fractions():
    with pytest.raises(ValueError):
        stratified_indices([0, 0, 1, 1], fractions=(0.7, 0.4))
    with pytest.raises(ValueError):
        stratified_indices([0, 0, 1, 1], fractions=(0.5, 0.5, 0.5))
