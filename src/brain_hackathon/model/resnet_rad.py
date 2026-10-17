import os

import torch
from huggingface_hub import hf_hub_download
from torch import nn
from torchvision.models import resnet50

# Optional environment override for the checkpoint path (e.g. weights mounted from a
# read-only dataset instead of downloading from the Hub).
WEIGHTS_ENV = "BRAIN_HACKATHON_WEIGHTS"


class RadImageNetResNet50(nn.Module):
    """ResNet50 classifier pretrained on RadImageNet for multi-modal MRI input.

    Weights: Hugging Face ``Lab-Rasool/RadImageNet`` / ``ResNet50.pt``.

    The published checkpoint is a state dict of the ResNet50 convolution trunk with a
    ``backbone.`` prefix and numeric child indices (``backbone.0``, ``backbone.4.0``...),
    i.e. ``nn.Sequential(*resnet50().children()[:-2])`` without the average pool and the
    classifier.

    The pretrained first convolution (3 input channels) is expanded to ``in_channels``
    by duplicating its weights. A head is appended after the pooled features: ``dim`` is
    a list of layer widths, producing ``Linear -> ReLU`` blocks and a final
    ``Linear(feature_dim -> dim[-1])`` with no trailing ReLU. All head linears are
    Xavier-initialised.
    """

    repo_id = "Lab-Rasool/RadImageNet"
    filename = "ResNet50.pt"
    feature_dim = 2048

    def __init__(
        self,
        dim=(4,),
        in_channels=6,
        device=None,
        verbose=True,
        freeze_backbone=False,
        weights_path=None,
        cache_dir=None,
        local_files_only=False,
    ):
        super().__init__()
        if isinstance(dim, int):
            dim = [dim]
        self.dim = [int(width) for width in dim]
        if not self.dim or any(width <= 0 for width in self.dim):
            raise ValueError("dim must be a non-empty list of positive layer sizes")
        self.in_channels = in_channels
        self.input_size = (in_channels, 224, 224)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.weights_path = weights_path
        self.cache_dir = cache_dir
        self.local_files_only = local_files_only

        self.backbone = nn.Sequential(*list(resnet50(weights=None).children())[:-2])
        self._load_pretrained_backbone()
        self._expand_first_conv(in_channels)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = self._build_head(self.dim)

        if freeze_backbone:
            self.freeze_all_except(0)

        self.to(self.device).eval()
        if verbose:
            self.print_input_shape()

    def _build_head(self, widths):
        layers = []
        in_features = self.feature_dim
        for i, out_features in enumerate(widths):
            layers.append(nn.Linear(in_features, out_features))
            if i < len(widths) - 1:
                layers.append(nn.ReLU())
            in_features = out_features

        for module in layers:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        return nn.Sequential(*layers)

    def freeze_all_except(self, first_n=0):
        """Freeze all parameters except the first ``first_n`` backbone layers and the head.

        With ``first_n=0`` only the classifier head stays trainable.
        """
        for param in self.parameters():
            param.requires_grad = False
        for layer in list(self.backbone.children())[: max(int(first_n), 0)]:
            for param in layer.parameters():
                param.requires_grad = True
        for param in self.head.parameters():
            param.requires_grad = True
        return self

    def _resolve_weights_path(self):
        path = self.weights_path or os.environ.get(WEIGHTS_ENV)
        if path:
            print(f"[model] loading pretrained weights from {path}", flush=True)
            return path
        print(
            f"[model] loading pretrained weights from HF: "
            f"{self.repo_id}/{self.filename}",
            flush=True,
        )
        return hf_hub_download(
            repo_id=self.repo_id,
            filename=self.filename,
            cache_dir=self.cache_dir,
            local_files_only=self.local_files_only,
        )

    def _load_pretrained_backbone(self):
        model_path = self._resolve_weights_path()
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
        print(f"[model] backbone loaded: {len(backbone_state)} tensors", flush=True)

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

    def forward_features(self, x):
        """Pooled ``feature_dim`` features from the pretrained backbone (no head)."""
        return torch.flatten(self.pool(self.backbone(x)), 1)

    def forward(self, x):
        return self.head(self.forward_features(x))

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
