import json
import os

import cv2
import numpy as np
import pytest
import torch
from torch import nn

from brain_hackathon.data.dataset import MODALITIES
from brain_hackathon.train.early_stopping import EarlyStopping
from brain_hackathon.train.logger import Logger
from brain_hackathon.train.main import build_optimizer, train
from brain_hackathon.train.metrics import Metrics
from brain_hackathon.train.utils import build_datasets


class TinyModel(nn.Module):
    def __init__(self, num_classes, channels=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, 4, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(4, num_classes),
        )

    def forward(self, x):
        return self.net(x)


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


def test_metrics_values():
    metrics = Metrics(3)
    logits = torch.tensor(
        [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0], [10.0, 0.0, 0.0]]
    )
    targets = torch.tensor([0, 1, 2, 1])
    metrics.update(logits, targets)
    result = metrics.compute()

    assert result["acc"] == pytest.approx(0.75)
    assert set(["acc", "f1", "precision", "recall", "sensitivity", "specificity"]) <= set(
        result
    )
    assert result["sensitivity"] == result["recall"]
    for value in result.values():
        assert 0.0 <= value <= 1.0


def test_early_stopping_patience():
    stopping = EarlyStopping(patience=2)
    assert stopping.step(1.0) is False
    assert stopping.step(1.0) is False
    assert stopping.step(1.0) is True

    improving = EarlyStopping(patience=2)
    assert improving.step(1.0) is False
    assert improving.step(0.5) is False
    assert improving.step(0.4) is False


def test_build_optimizer_default_adamw():
    model = TinyModel(2)
    opt = build_optimizer(model, "adamw", lr=1e-3, weight_decay=0.0)
    assert isinstance(opt, torch.optim.AdamW)
    assert build_optimizer(model, torch.optim.SGD, lr=1e-2, weight_decay=0.0).__class__ is (
        torch.optim.SGD
    )
    with pytest.raises(ValueError):
        build_optimizer(model, "nope", lr=1e-3, weight_decay=0.0)


def test_build_optimizer_skips_frozen_params():
    model = TinyModel(2)
    for param in model.net[0].parameters():
        param.requires_grad = False

    opt = build_optimizer(model, "adamw", lr=1e-3, weight_decay=0.0)
    optimized = {id(p) for group in opt.param_groups for p in group["params"]}
    assert id(model.net[0].weight) not in optimized
    assert id(model.net[-1].weight) in optimized


def test_build_optimizer_all_frozen_raises():
    model = TinyModel(2)
    for param in model.parameters():
        param.requires_grad = False
    with pytest.raises(ValueError):
        build_optimizer(model, "adamw", lr=1e-3, weight_decay=0.0)


def test_logger_sets_wandb_env_defaults(monkeypatch, tmp_path):
    pytest.importorskip("wandb")
    from brain_hackathon.train.logger import (
        DEFAULT_WANDB_API_KEY,
        DEFAULT_WANDB_BASE_URL,
    )

    monkeypatch.delenv("WANDB_BASE_URL", raising=False)
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    Logger(mode="disabled", checkpoint_dir=str(tmp_path), enabled=True)

    assert os.environ["WANDB_BASE_URL"] == DEFAULT_WANDB_BASE_URL
    assert os.environ["WANDB_API_KEY"] == DEFAULT_WANDB_API_KEY


def test_logger_keeps_existing_wandb_env(monkeypatch, tmp_path):
    pytest.importorskip("wandb")
    monkeypatch.setenv("WANDB_BASE_URL", "http://example:1234")
    monkeypatch.setenv("WANDB_API_KEY", "existing-key")
    Logger(mode="disabled", checkpoint_dir=str(tmp_path), enabled=True)

    assert os.environ["WANDB_BASE_URL"] == "http://example:1234"
    assert os.environ["WANDB_API_KEY"] == "existing-key"


def test_logger_dumps_results(tmp_path):
    logger = Logger(checkpoint_dir=str(tmp_path / "ckpt"), enabled=False)
    logger.log_epoch(1, {"val/loss": 0.5}, step=1)
    logger.log_checkpoint(str(tmp_path / "ckpt" / "best.pt"), "best")
    path = logger.finish()

    assert os.path.isfile(path)
    payload = json.loads(open(path).read())
    assert payload["history"][0]["epoch"] == 1
    assert payload["best_model_path"].endswith("best.pt")


def test_train_early_stopping_and_checkpoints(aligned_root, tmp_path):
    train_ds, val_ds, test_ds = build_datasets(
        root=aligned_root, train_transform=None, eval_transform=None
    )
    model = TinyModel(len(train_ds.dataset.classes))
    checkpoint_dir = str(tmp_path / "ckpt")
    logger = Logger(checkpoint_dir=checkpoint_dir, enabled=False)

    result = train(
        model,
        train_ds,
        val_ds,
        test_ds,
        optimizer="adamw",
        lr=0.0,
        batch_size=4,
        epochs=10,
        patience=2,
        loss_period=2,
        logger=logger,
        checkpoint_dir=checkpoint_dir,
        device="cpu",
    )

    assert result["stopped_early"] is True
    assert result["epochs_run"] < 10
    assert os.path.isfile(result["best_model_path"])
    assert os.path.isfile(result["last_model_path"])
    assert os.path.isfile(os.path.join(checkpoint_dir, "results.json"))
    assert result["step_test_losses"]

    record = result["history"][0]
    assert "train/loss" in record and "val/loss" in record
    assert "val/acc" in record and "test/loss" in record


def test_train_saves_best_and_last_paths(aligned_root, tmp_path):
    train_ds, val_ds, test_ds = build_datasets(
        root=aligned_root, train_transform=None, eval_transform=None
    )
    model = TinyModel(len(train_ds.dataset.classes))
    checkpoint_dir = str(tmp_path / "ckpt")

    result = train(
        model,
        train_ds,
        val_ds,
        test_ds,
        lr=1e-3,
        batch_size=4,
        epochs=2,
        patience=10,
        loss_period=1000,
        checkpoint_dir=checkpoint_dir,
        device="cpu",
    )

    assert result["best_model_path"] is not None
    assert result["last_model_path"] is not None
    assert result["epochs_run"] == 2
    assert result["step_test_losses"] == []
