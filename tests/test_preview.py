import os

import cv2
import numpy as np
import pytest
import torch

from brain_hackathon.data import preview
from brain_hackathon.data.dataset import MODALITIES
from brain_hackathon.data.preview import build_montage, main, supports_gui


@pytest.fixture
def aligned_root(tmp_path):
    root = tmp_path / "aligned"
    class_dir = root / "Train" / "glioma"
    os.makedirs(class_dir, exist_ok=True)
    gradient = np.arange(16, dtype=np.uint8).reshape(4, 4) * 3
    for series_id in ("1", "2"):
        for modality in MODALITIES:
            cv2.imwrite(
                os.path.join(class_dir, f"{series_id}_{modality}.jpg"), gradient
            )
    return str(root)


def test_build_montage_shape():
    tensor = torch.zeros(len(MODALITIES), 4, 4)
    montage = build_montage(tensor, list(MODALITIES), cols=3, padding=6)
    assert montage.shape == (2 * 4 + 3 * 6, 3 * 4 + 4 * 6)
    assert montage.dtype == np.uint8


def test_supports_gui_is_bool():
    assert isinstance(supports_gui(), bool)


def test_empty_dataset_returns_error(tmp_path):
    empty = tmp_path / "empty"
    os.makedirs(empty / "Train", exist_ok=True)
    code = main(["--root", str(empty)])
    assert code == 1


def test_no_gui_returns_error(monkeypatch, aligned_root):
    monkeypatch.setattr(preview, "supports_gui", lambda: False)
    assert main(["--root", aligned_root, "--split", "Train", "--no-augment"]) == 1


def test_preview_shows_window(monkeypatch, aligned_root):
    monkeypatch.setattr(preview, "supports_gui", lambda: True)
    shown = []
    monkeypatch.setattr(preview.cv2, "imshow", lambda *a, **k: shown.append(a))
    monkeypatch.setattr(preview.cv2, "waitKey", lambda *a, **k: ord("q"))
    monkeypatch.setattr(preview.cv2, "destroyAllWindows", lambda: None)

    code = main(
        ["--root", aligned_root, "--split", "Train", "--index", "0", "--no-augment"]
    )
    assert code == 0
    assert len(shown) == 1
    _, montage = shown[0]
    assert montage.dtype == np.uint8
