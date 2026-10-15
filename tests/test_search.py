import json
import os

import cv2
import numpy as np
import pytest

from brain_hackathon.data.dataset import MODALITIES
from brain_hackathon.train.search import (
    config_name,
    grid_search,
    iter_configs,
    select_best,
)
from brain_hackathon.train.utils import build_datasets


def write_series(class_dir, series_id, shape=(8, 8)):
    os.makedirs(class_dir, exist_ok=True)
    image = np.full(shape, (int(series_id) % 250) + 1, dtype=np.uint8)
    for modality in MODALITIES:
        cv2.imwrite(os.path.join(class_dir, f"{series_id}_{modality}.jpg"), image)


@pytest.fixture
def aligned_root(tmp_path):
    root = tmp_path / "aligned"
    for i in range(1, 15):
        write_series(root / "Train" / "glioma", str(i))
    for i in range(1, 15):
        write_series(root / "Train" / "meningioma", str(i))
    return str(root)


def fake_runner(config, context):
    loss = 1.0
    loss -= 0.5 if config["optimizer"] == "adamw" else 0.0
    loss -= 0.4 if config["lr"] == 1e-3 else 0.0
    return {**config, "val_loss": loss, "val_f1": 1.0 - loss, "val_acc": 1.0 - loss}


def test_iter_configs_product():
    configs = iter_configs(("adamw", "sgd"), (1e-3,), (0.0, 1e-4), (None, 0))
    assert len(configs) == 8
    assert {
        "optimizer": "adamw",
        "lr": 1e-3,
        "weight_decay": 0.0,
        "freeze_first_n": None,
    } in configs


def test_select_best():
    results = [
        {"val_loss": 0.5, "val_f1": 0.7},
        {"val_loss": 0.3, "val_f1": 0.6},
        {"val_loss": 0.4, "val_f1": 0.9},
    ]
    assert select_best(results, "val_loss")["val_loss"] == 0.3
    assert select_best(results, "val_f1", maximize=True)["val_f1"] == 0.9
    with pytest.raises(ValueError):
        select_best([])


def test_grid_search_with_runner(aligned_root, tmp_path):
    datasets = build_datasets(
        root=aligned_root, train_transform=None, eval_transform=None
    )
    results_path = tmp_path / "results.json"

    results = grid_search(
        datasets=datasets,
        optimizers=("adamw", "sgd"),
        lrs=(1e-3,),
        weight_decays=(0.0,),
        freezes=(None, 0),
        runner=fake_runner,
        results_path=str(results_path),
    )

    assert len(results) == 4
    assert results_path.is_file()
    assert len(json.loads(results_path.read_text())) == 4
    assert select_best(results, "val_loss")["optimizer"] == "adamw"


def test_config_name():
    assert (
        config_name(
            {"optimizer": "adamw", "lr": 1e-3, "weight_decay": 0.0, "freeze_first_n": None}
        )
        == "adamw-lr0.001-wd0-freezeall"
    )
    assert config_name(
        {"optimizer": "sgd", "lr": 1e-2, "weight_decay": 1e-4, "freeze_first_n": 2}
    ) == "sgd-lr0.01-wd0.0001-freeze2"


def test_grid_search_passes_logger_factory(aligned_root):
    datasets = build_datasets(
        root=aligned_root, train_transform=None, eval_transform=None
    )
    seen = {}
    sentinel = object()

    def runner(config, context):
        seen["factory"] = context.get("logger_factory")
        return {**config, "val_loss": 0.0, "val_f1": 0.0, "val_acc": 0.0}

    grid_search(
        datasets=datasets,
        optimizers=("adamw",),
        lrs=(1e-3,),
        weight_decays=(0.0,),
        freezes=(None,),
        runner=runner,
        logger_factory=sentinel,
    )
    assert seen["factory"] is sentinel


def test_grid_search_max_runs(aligned_root):
    datasets = build_datasets(
        root=aligned_root, train_transform=None, eval_transform=None
    )
    results = grid_search(
        datasets=datasets,
        optimizers=("adamw", "sgd"),
        lrs=(1e-2, 1e-3),
        weight_decays=(0.0,),
        freezes=(None,),
        runner=fake_runner,
        max_runs=2,
    )
    assert len(results) == 2
