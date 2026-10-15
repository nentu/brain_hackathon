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
    assert model.input_size == (3, 224, 224)


def test_forward_output_shape(model):
    x = torch.zeros(1, *model.input_size, device=model.device)
    with torch.no_grad():
        output = model(x)
    assert output.shape == (1, model.feature_dim)


def test_weights_match_checkpoint(model):
    path = hf_hub_download(repo_id=model.repo_id, filename=model.filename)
    state_dict = torch.load(path, map_location="cpu", weights_only=True)
    loaded = model.backbone[0].weight.detach().cpu()
    assert torch.equal(loaded, state_dict["backbone.0.weight"])
