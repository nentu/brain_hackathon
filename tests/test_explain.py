import cv2
import numpy as np
import pytest

from brain_hackathon.eval.explain import (
    _build_montage,
    _load_aligned_items,
    _load_raw_items,
    _safe_name,
    _select_indices,
    main,
    parse_args,
)


def test_parse_args_defaults():
    args = parse_args([])

    assert args.dataset == "both"
    assert args.split == "Test"
    assert args.limit == 3
    assert args.methods == "both"
    assert args.shap_patch_grid == 7
    assert args.shap_samples == 128


def test_parse_args_overrides():
    args = parse_args(
        [
            "--dataset",
            "raw",
            "--limit",
            "1",
            "--methods",
            "gradcam",
            "--classes",
            "glioma",
        ]
    )

    assert args.dataset == "raw"
    assert args.limit == 1
    assert args.methods == "gradcam"
    assert args.classes == ["glioma"]


def test_safe_name():
    assert _safe_name("aligned/glioma 123") == "aligned_glioma_123"
    assert _safe_name("...") == "item"


def test_select_indices_is_deterministic_and_bounded():
    first = _select_indices(10, 3, seed=42)
    second = _select_indices(10, 3, seed=42)

    assert first == second
    assert first == sorted(first)
    assert len(first) == 3
    assert all(0 <= index < 10 for index in first)


def test_select_indices_handles_limits():
    assert _select_indices(2, 10, seed=0) == [0, 1]
    assert _select_indices(10, 0, seed=0) == []
    assert _select_indices(0, 3, seed=0) == []


def test_build_montage_shape():
    images = [
        np.zeros((8, 8, 3), dtype=np.uint8),
        np.full((8, 8, 3), 255, dtype=np.uint8),
    ]
    montage = _build_montage(images, ["original", "gradcam"], "title")

    assert montage.ndim == 3 and montage.shape[2] == 3
    assert montage.shape[0] > 8
    assert montage.shape[1] > 16
    assert montage.dtype == np.uint8


def test_load_raw_items(tmp_path):
    path = tmp_path / "image.png"
    cv2.imwrite(str(path), np.full((8, 8), 100, dtype=np.uint8))

    class Dataset:
        def __init__(self):
            self.index = [(str(path), 0)]
            self.classes = ["glioma"]

    items = _load_raw_items(Dataset(), [0])

    assert len(items) == 1
    assert items[0]["kind"] == "raw"
    assert items[0]["true_class"] == "glioma"
    assert items[0]["image"].shape == (8, 8)


def test_load_aligned_items(tmp_path):
    paths = {}
    for modality in ("BR", "DA"):
        path = tmp_path / f"7_{modality}.png"
        cv2.imwrite(str(path), np.full((8, 8), 100, dtype=np.uint8))
        paths[modality] = str(path)

    class Dataset:
        def __init__(self):
            self.index = [("7", "glioma", paths)]
            self.classes = ["glioma"]

    items = _load_aligned_items(Dataset(), [0])

    assert len(items) == 1
    assert items[0]["kind"] == "aligned"
    assert items[0]["true_class"] == "glioma"
    assert set(items[0]["image"]) == {"BR", "DA"}


def test_main_rejects_raw_base_modality():
    with pytest.raises(SystemExit, match="only supported for aligned"):
        main(["--dataset", "raw", "--base-modality", "BR"])


def test_main_rejects_unknown_target_class():
    with pytest.raises(SystemExit, match="unknown class"):
        main(["--dataset", "raw", "--target-class", "missing"])


def test_main_rejects_invalid_shap_grid():
    with pytest.raises(SystemExit, match="patch-grid must be positive"):
        main(["--dataset", "raw", "--shap-patch-grid", "0"])
