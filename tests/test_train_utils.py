import os
from collections import defaultdict

import cv2
import numpy as np
import pytest

from brain_hackathon.data.dataset import MODALITIES, AlignedMRIClassificationDataset
from brain_hackathon.train.utils import (
    build_datasets,
    build_split_indices,
    content_groups,
    stratified_indices,
)


def write_series(class_dir, series_id, shape=(8, 8), value=None):
    os.makedirs(class_dir, exist_ok=True)
    if value is None:
        value = (int(series_id) % 250) + 1
    image = np.full(shape, value, dtype=np.uint8)
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
    # same series id in both splits (series-level leak)
    write_series(root / "Test" / "glioma", "5", shape=(4, 4), value=200)
    return str(root)


def _base(aligned_root):
    return AlignedMRIClassificationDataset(root=aligned_root, split=("Train", "Test"))


def test_build_datasets_partitions_all_samples(aligned_root):
    base = _base(aligned_root)
    train, val, test = build_datasets(
        root=aligned_root,
        splits=("Train", "Test"),
        train_transform=None,
        eval_transform=None,
        dedup=False,
        group_by_content=False,
    )
    assert len(train) + len(val) + len(test) == len(base)
    assert len(train) > len(val) > 0
    assert len(test) > 0


def test_build_datasets_dedups_series_level_leak(aligned_root):
    base = _base(aligned_root)
    train, val, test = build_datasets(
        root=aligned_root, splits=("Train", "Test"), train_transform=None,
        eval_transform=None,
    )
    # glioma id 5 exists in both splits -> one duplicate removed
    assert len(train) + len(val) + len(test) == len(base) - 1


def test_content_dedup_identical_images(aligned_root):
    # add a second id with exactly the same content as glioma id 1
    write_series(os.path.join(aligned_root, "Train", "glioma"), "999", value=2)
    base = _base(aligned_root)

    groups = content_groups(base.index)
    e1 = next(i for i, (s, c, _) in enumerate(base.index) if s == "1" and c == "glioma")
    e9 = next(i for i, (s, c, _) in enumerate(base.index) if s == "999")
    assert groups[e1] == groups[e9]


def test_content_duplicates_stay_in_same_fold(tmp_path):
    root = tmp_path / "aligned"
    for cls in ("glioma", "meningioma"):
        for i in range(1, 16):
            write_series(root / "Train" / cls, str(i))
    # id 100 duplicates id 1's exact content (same class)
    write_series(root / "Train" / "glioma", "100", value=(1 % 250) + 1)

    train, val, test = build_datasets(
        root=str(root), splits=("Train",), train_transform=None, eval_transform=None
    )
    base = AlignedMRIClassificationDataset(root=str(root), split=("Train",))
    e1 = next(i for i, (s, c, _) in enumerate(base.index) if s == "1" and c == "glioma")
    e100 = next(i for i, (s, c, _) in enumerate(base.index) if s == "100")
    fold = {}
    for name, subset in (("train", train), ("val", val), ("test", test)):
        for i in subset.indices:
            fold[i] = name
    # the identical duplicate is dropped, the original stays
    assert e100 not in fold
    assert e1 in fold


def test_build_datasets_stratified(aligned_root):
    base = _base(aligned_root)
    train, val, test = build_datasets(
        root=aligned_root, splits=("Train", "Test"), train_transform=None,
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
        root=aligned_root, splits=("Train", "Test"), train_transform=None,
        eval_transform=None,
    )
    ti, vi, ei = set(train.indices), set(val.indices), set(test.indices)
    assert not ti & vi and not ti & ei and not vi & ei


def test_series_never_spans_folds(aligned_root):
    base = _base(aligned_root)
    train, val, test = build_datasets(
        root=aligned_root, splits=("Train", "Test"), train_transform=None,
        eval_transform=None,
    )
    fold_of = {}
    for name, subset in (("train", train), ("val", val), ("test", test)):
        for index in subset.indices:
            series_id, class_name, _ = base.index[index]
            fold_of.setdefault((class_name, series_id), set()).add(name)
    assert all(len(folds) == 1 for folds in fold_of.values())


def test_build_split_indices_reports_removed(aligned_root):
    base = _base(aligned_root)
    labels = [base.class_to_idx[c] for _, c, _ in base.index]
    train, val, test, removed = build_split_indices(base.index, labels)
    assert removed == 1
    kept = set(train) | set(val) | set(test)
    assert len(kept) == len(base) - 1


def test_stratified_indices_rejects_bad_fractions():
    with pytest.raises(ValueError):
        stratified_indices([0, 0, 1, 1], fractions=(0.7, 0.4))
    with pytest.raises(ValueError):
        stratified_indices([0, 0, 1, 1], fractions=(0.5, 0.5, 0.5))
