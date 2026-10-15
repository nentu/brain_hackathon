import math

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from huggingface_hub import hf_hub_download

from brain_hackathon.model.resnet_rad import RadImageNetResNet50


@pytest.fixture(scope="module")
def model():
    try:
        return RadImageNetResNet50(verbose=False)
    except Exception as exc:  # network / weights unavailable
        pytest.skip(f"RadImageNet weights unavailable: {exc}")


def test_input_size(model):
    assert model.input_size == (6, 224, 224)
    assert model.backbone[0].in_channels == 6


def test_forward_output_shape(model):
    x = torch.zeros(1, *model.input_size, device=model.device)
    with torch.no_grad():
        output = model(x)
    assert output.shape == (1, model.dim)


def test_custom_dim():
    try:
        model = RadImageNetResNet50(dim=7, verbose=False)
    except Exception as exc:
        pytest.skip(f"RadImageNet weights unavailable: {exc}")
    x = torch.zeros(1, *model.input_size, device=model.device)
    with torch.no_grad():
        assert model(x).shape == (1, 7)


def test_first_conv_weights_duplicated(model):
    path = hf_hub_download(repo_id=model.repo_id, filename=model.filename)
    state_dict = torch.load(path, map_location="cpu", weights_only=True)
    pretrained = state_dict["backbone.0.weight"]
    conv = model.backbone[0].weight.detach().cpu()

    assert conv.shape == (64, 6, 7, 7)
    assert torch.equal(conv[:, :3], pretrained)
    assert torch.equal(conv[:, 3:], pretrained)


def _new_model(**kwargs):
    try:
        return RadImageNetResNet50(verbose=False, **kwargs)
    except Exception as exc:  # network / weights unavailable
        pytest.skip(f"RadImageNet weights unavailable: {exc}")


def test_freeze_all_except_keeps_first_n_and_head():
    model = _new_model()
    model.freeze_all_except(2)

    assert all(p.requires_grad for p in model.head.parameters())
    assert all(p.requires_grad for p in model.backbone[0].parameters())
    assert all(p.requires_grad for p in model.backbone[1].parameters())
    assert not any(p.requires_grad for p in model.backbone[4].parameters())
    assert not any(p.requires_grad for p in model.backbone[7].parameters())


def test_freeze_all_except_zero_only_head():
    model = _new_model()
    model.freeze_all_except(0)
    assert all(p.requires_grad for p in model.head.parameters())
    assert not any(
        p.requires_grad for name, p in model.named_parameters() if name.startswith("backbone.")
    )


def test_unfreeze_first_n_flag():
    model = _new_model(unfreeze_first_n=3)
    trainable = {name for name, p in model.named_parameters() if p.requires_grad}

    assert any(name.startswith("head.") for name in trainable)
    assert "backbone.0.weight" in trainable
    assert "backbone.1.weight" in trainable
    assert "backbone.4.0.conv1.weight" not in trainable


def test_head_is_xavier_initialised(model):
    weight = model.head.weight.detach().cpu()
    bound = math.sqrt(6.0 / (weight.shape[0] + weight.shape[1]))
    assert weight.abs().max().item() <= bound * 1.001
    assert weight.abs().sum().item() > 0.0
    assert torch.count_nonzero(model.head.bias) == 0
