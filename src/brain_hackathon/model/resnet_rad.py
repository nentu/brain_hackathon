import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from torchvision.models import resnet50


class RadImageNetResNet50(nn.Module):
    """ResNet50 feature extractor pretrained on RadImageNet.

    Weights: Hugging Face ``Lab-Rasool/RadImageNet`` / ``ResNet50.pt``.

    The published checkpoint is a state dict of the ResNet50 convolution trunk with a
    ``backbone.`` prefix and numeric child indices (``backbone.0``, ``backbone.4.0``...),
    i.e. ``nn.Sequential(*resnet50().children()[:-2])`` without the average pool and the
    classifier. Input is RGB ``(3, 224, 224)``; ``forward`` returns globally pooled
    2048-d features.
    """

    repo_id = "Lab-Rasool/RadImageNet"
    filename = "ResNet50.pt"
    input_size = (3, 224, 224)
    feature_dim = 2048

    def __init__(self, device=None, verbose=True):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        trunk = list(resnet50(weights=None).children())[:-2]
        self.backbone = nn.Sequential(*trunk)
        self.pool = nn.AdaptiveAvgPool2d(1)

        model_path = hf_hub_download(repo_id=self.repo_id, filename=self.filename)
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        backbone_state = {
            key[len("backbone.") :]: value
            for key, value in state_dict.items()
            if key.startswith("backbone.")
        }
        missing, unexpected = self.backbone.load_state_dict(backbone_state, strict=False)
        if unexpected:
            raise RuntimeError(f"unexpected checkpoint keys: {unexpected[:5]}")
        if missing:
            raise RuntimeError(f"missing checkpoint keys: {missing[:5]}")

        self.to(self.device).eval()
        if verbose:
            self.print_input_shape()

    def forward(self, x):
        return torch.flatten(self.pool(self.backbone(x)), 1)

    @torch.no_grad()
    def print_input_shape(self):
        dummy = torch.zeros(1, *self.input_size, device=self.device)
        output = self(dummy)
        print(
            f"{type(self).__name__}: input (N, {self.input_size[0]}, "
            f"{self.input_size[1]}, {self.input_size[2]}) -> output {tuple(output.shape)}"
        )
        return self.input_size


if __name__ == "__main__":
    RadImageNetResNet50()
