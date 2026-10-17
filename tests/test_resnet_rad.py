import math

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from huggingface_hub import hf_hub_download

from brain_hackathon.model.resnet_rad import (
    RadImageNetResNet50,
    RadImageNetResNet50Single,
)


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
    assert output.shape == (1, model.dim[-1])


def test_custom_dim_list_builds_mlp_head():
    try:
        model = RadImageNetResNet50(dim=[16, 7], verbose=False)
    except Exception as exc:
        pytest.skip(f"RadImageNet weights unavailable: {exc}")

    linears = [m for m in model.head if isinstance(m, torch.nn.Linear)]
    assert [layer.out_features for layer in linears] == [16, 7]
    assert isinstance(model.head[1], torch.nn.ReLU)

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


def test_freeze_backbone_flag():
    frozen = _new_model(freeze_backbone=True)
    assert all(p.requires_grad for p in frozen.head.parameters())
    assert not any(p.requires_grad for p in frozen.backbone.parameters())

    trainable = _new_model(freeze_backbone=False)
    assert all(p.requires_grad for p in trainable.backbone.parameters())
    assert all(p.requires_grad for p in trainable.head.parameters())


def test_single_channel_model():
    try:
        model = RadImageNetResNet50Single(dim=[4], verbose=False)
    except Exception as exc:
        pytest.skip(f"RadImageNet weights unavailable: {exc}")

    assert model.input_size == (1, 224, 224)
    conv = model.backbone[0]
    assert conv.in_channels == 1
    assert tuple(conv.weight.shape) == (64, 1, 7, 7)

    path = hf_hub_download(repo_id=model.repo_id, filename=model.filename)
    state_dict = torch.load(path, map_location="cpu", weights_only=True)
    expected = state_dict["backbone.0.weight"].mean(dim=1, keepdim=True)
    assert torch.allclose(conv.weight.detach().cpu(), expected)

    x = torch.zeros(1, 1, 224, 224, device=model.device)
    with torch.no_grad():
        assert model(x).shape == (1, 4)


def test_head_is_xavier_initialised(model):
    linears = [m for m in model.head if isinstance(m, torch.nn.Linear)]
    assert linears
    for linear in linears:
        weight = linear.weight.detach().cpu()
        bound = math.sqrt(6.0 / (weight.shape[0] + weight.shape[1]))
        assert weight.abs().max().item() <= bound * 1.001
        assert weight.abs().sum().item() > 0.0
        assert torch.count_nonzero(linear.bias) == 0
