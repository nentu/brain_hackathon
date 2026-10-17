import pytest
import torch

from brain_hackathon.model.resnet_rad import RadImageNetResNet50Single
from brain_hackathon.train.test_model import infer_model_config, load_checkpoint


def test_infer_model_config_sequential_head():
    state = {
        "backbone.0.weight": torch.zeros(64, 1, 7, 7),
        "head.0.weight": torch.zeros(16, 2048),
        "head.2.weight": torch.zeros(4, 16),
    }
    assert infer_model_config(state) == (1, [16, 4])


def test_infer_model_config_single_linear_head():
    state = {
        "backbone.0.weight": torch.zeros(64, 6, 7, 7),
        "head.weight": torch.zeros(4, 2048),
    }
    assert infer_model_config(state) == (6, [4])


def test_infer_model_config_rejects_missing_parts():
    with pytest.raises(ValueError):
        infer_model_config({"head.weight": torch.zeros(4, 2048)})
    with pytest.raises(ValueError):
        infer_model_config({"backbone.0.weight": torch.zeros(64, 1, 7, 7)})


def test_load_checkpoint_roundtrip(tmp_path):
    model = RadImageNetResNet50Single(
        dim=[8, 4], pretrained=False, verbose=False
    )
    path = tmp_path / "best.pt"
    torch.save(
        {"model": model.state_dict(), "epoch": 3, "val_loss": 0.5}, path
    )

    loaded, info = load_checkpoint(str(path), device="cpu")

    assert info["in_channels"] == 1
    assert info["head_dims"] == [8, 4]
    assert info["epoch"] == 3
    assert info["val_loss"] == 0.5
    with torch.no_grad():
        assert loaded(torch.zeros(1, 1, 224, 224)).shape == (1, 4)
