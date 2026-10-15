import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from torchvision.models import resnet50


class RadImageNetResNet50(nn.Module):
    """ResNet50 classifier pretrained on RadImageNet for multi-modal MRI input.

    Weights: Hugging Face ``Lab-Rasool/RadImageNet`` / ``ResNet50.pt``.

    The published checkpoint is a state dict of the ResNet50 convolution trunk with a
    ``backbone.`` prefix and numeric child indices (``backbone.0``, ``backbone.4.0``...),
    i.e. ``nn.Sequential(*resnet50().children()[:-2])`` without the average pool and the
    classifier.

    The pretrained first convolution (3 input channels) is expanded to ``in_channels``
    by duplicating its weights, and a Xavier-initialised ``nn.Linear`` head with ``dim``
    outputs is appended.
    """

    repo_id = "Lab-Rasool/RadImageNet"
    filename = "ResNet50.pt"
    feature_dim = 2048

    def __init__(self, dim=4, in_channels=6, device=None, verbose=True):
        super().__init__()
        self.dim = dim
        self.in_channels = in_channels
        self.input_size = (in_channels, 224, 224)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.backbone = nn.Sequential(*list(resnet50(weights=None).children())[:-2])
        self._load_pretrained_backbone()
        self._expand_first_conv(in_channels)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(self.feature_dim, dim)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

        self.to(self.device).eval()
        if verbose:
            self.print_input_shape()

    def _load_pretrained_backbone(self):
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

    def _expand_first_conv(self, in_channels):
        conv = self.backbone[0]
        if in_channels == conv.in_channels:
            return
        if in_channels % conv.in_channels != 0:
            raise ValueError(
                f"in_channels={in_channels} is not a multiple of the pretrained "
                f"{conv.in_channels} channels"
            )

        repeats = in_channels // conv.in_channels
        expanded = nn.Conv2d(
            in_channels,
            conv.out_channels,
            conv.kernel_size,
            conv.stride,
            conv.padding,
            bias=conv.bias is not None,
            device=conv.weight.device,
            dtype=conv.weight.dtype,
        )
        with torch.no_grad():
            expanded.weight.copy_(torch.cat([conv.weight.detach()] * repeats, dim=1))
            if conv.bias is not None:
                expanded.bias.copy_(conv.bias.detach())
        self.backbone[0] = expanded

    def forward(self, x):
        return self.head(torch.flatten(self.pool(self.backbone(x)), 1))

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
