import os

import cv2
import numpy as np
import pytest

from brain_hackathon.data.dataset import MODALITIES
from brain_hackathon.train.cross_validate import METRIC_KEYS, run_cross_validation
from brain_hackathon.train.utils import stratified_kfold


def write_series(class_dir, series_id, shape=(8, 8)):
    os.makedirs(class_dir, exist_ok=True)
    image = np.full(shape, (int(series_id) % 200) + 1, dtype=np.uint8)
    for modality in MODALITIES:
        cv2.imwrite(os.path.join(class_dir, f"{series_id}_{modality}.jpg"), image)


@pytest.fixture
def aligned_root(tmp_path):
    root = tmp_path / "aligned"
    for i in range(1, 21):
        write_series(root / "Train" / "glioma", str(i))
    for i in range(1, 21):
        write_series(root / "Train" / "meningioma", str(i))
    return str(root)


def test_stratified_kfold_partitions():
    labels = [0] * 20 + [1] * 20
    folds = stratified_kfold(labels, n_splits=4, seed=1)

    assert len(folds) == 4
    val_sets = []
    for train, val in folds:
        assert val and train
        assert not set(train) & set(val)
        assert sorted(set(train) | set(val)) == list(range(40))
        val_sets.append(set(val))

    for i in range(len(val_sets)):
        for j in range(i + 1, len(val_sets)):
            assert not val_sets[i] & val_sets[j]
    assert set().union(*val_sets) == set(range(40))


def test_stratified_kfold_is_label_stratified():
    labels = [0] * 20 + [1] * 20
    for _, val in stratified_kfold(labels, n_splits=4, seed=1):
        assert any(labels[i] == 0 for i in val)
        assert any(labels[i] == 1 for i in val)


def test_stratified_kfold_keeps_groups():
    labels = [0] * 10
    groups = [i // 2 for i in range(10)]  # pairs share a group
    for _, val in stratified_kfold(labels, groups, n_splits=2, seed=3):
        for group in set(groups):
            members = {i for i, g in enumerate(groups) if g == group}
            assert members <= set(val) or not members & set(val)


def fake_train(model, train_ds, val_ds, test_dataset=None, **kwargs):
    fake_train.calls.append(
        (len(train_ds), len(val_ds), None if test_dataset is None else len(test_dataset))
    )
    record = {"loss/train": 1.0, "loss/val": 0.5, "loss/test": 0.6}
    for key in METRIC_KEYS:
        record[f"{key}/val"] = 0.8
        record[f"{key}/test"] = 0.7
    return {"history": [record]}


def test_run_cross_validation(aligned_root, tmp_path):
    fake_train.calls = []
    result = run_cross_validation(
        root=aligned_root,
        n_splits=4,
        train_frac=0.8,
        epochs=1,
        seed=42,
        checkpoint_dir=str(tmp_path / "cv"),
        verbose=False,
        train_fn=fake_train,
        model_factory=lambda: None,
    )

    assert len(result["folds"]) == 4
    assert set(result["mean"]) == {"loss", *METRIC_KEYS}
    assert result["mean"]["acc"][0] == pytest.approx(0.8)
    assert result["final"]["acc"] == pytest.approx(0.7)

    sizes = result["sizes"]
    assert sizes["train"] + sizes["test"] == 40
    assert 0.75 <= sizes["train"] / 40 <= 0.85
    assert sum(sizes["fold_val"]) == sizes["train"]

    # four folds + one final run
    assert len(fake_train.calls) == 5
    assert fake_train.calls[-1][2] == sizes["test"]
