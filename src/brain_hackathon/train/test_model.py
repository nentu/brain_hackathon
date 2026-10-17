"""Evaluate a saved checkpoint on the Test split.

    uv run test_model --checkpoint checkpoints/best.pt --dataset raw
"""

import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader

from brain_hackathon.data.dataset import (
    DEFAULT_ROOT,
    AlignedMRIClassificationDataset,
)
from brain_hackathon.data.raw_dataset import RAW_ROOT, RawImageDataset
from brain_hackathon.model.resnet_rad import (
    RadImageNetResNet50,
    RadImageNetResNet50Single,
)
from brain_hackathon.train.engine import evaluate, resolve_device


def infer_model_config(state_dict):
    """Infer ``(in_channels, head_dims)`` from a checkpoint state dict."""
    conv = state_dict.get("backbone.0.weight")
    if conv is None:
        raise ValueError("checkpoint has no 'backbone.0.weight'")
    in_channels = int(conv.shape[1])

    linears = []
    for key, value in state_dict.items():
        if key.startswith("head.") and key.endswith(".weight") and value.dim() == 2:
            parts = key.split(".")
            if len(parts) == 3:  # head.<index>.weight (nn.Sequential)
                linears.append((int(parts[1]), int(value.shape[0])))
    if linears:
        dims = [width for _, width in sorted(linears)]
    elif "head.weight" in state_dict:  # old single nn.Linear head
        dims = [int(state_dict["head.weight"].shape[0])]
    else:
        raise ValueError("checkpoint has no head weights")
    return in_channels, dims


def build_model(in_channels, dims, device=None):
    if in_channels == 1:
        model = RadImageNetResNet50Single(
            dim=dims, verbose=False, pretrained=False
        )
    else:
        model = RadImageNetResNet50(
            dim=dims, in_channels=in_channels, verbose=False, pretrained=False
        )
    return model.to(device) if device is not None else model


def load_checkpoint(path, device="cpu"):
    """Load a checkpoint and return ``(model_in_eval_mode, info)``."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    state_dict = checkpoint.get("model", checkpoint) if isinstance(
        checkpoint, dict
    ) else checkpoint

    in_channels, dims = infer_model_config(state_dict)
    model = build_model(in_channels, dims, device)
    model.load_state_dict(state_dict)
    model.eval()

    info = {
        "in_channels": in_channels,
        "head_dims": dims,
        "epoch": checkpoint.get("epoch") if isinstance(checkpoint, dict) else None,
        "val_loss": checkpoint.get("val_loss")
        if isinstance(checkpoint, dict)
        else None,
    }
    return model, info


def make_test_dataset(kind, root):
    if kind == "raw":
        return RawImageDataset(root=root, split="Test", normalize=True)
    return AlignedMRIClassificationDataset(root=root, split="Test", augment=None)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Evaluate a checkpoint on the Test split."
    )
    parser.add_argument(
        "--checkpoint", required=True, help="path to best.pt / last.pt"
    )
    parser.add_argument("--dataset", choices=("raw", "aligned"), default="raw")
    parser.add_argument(
        "--root",
        default=None,
        help="dataset root (default: raw dataset root for 'raw', data/aligned "
        "for 'aligned')",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default=None)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    device = resolve_device(args.device)
    model, info = load_checkpoint(args.checkpoint, device)

    root = args.root or (RAW_ROOT if args.dataset == "raw" else DEFAULT_ROOT)
    dataset = make_test_dataset(args.dataset, root)
    if len(dataset) == 0:
        raise SystemExit(
            f"empty Test split for dataset={args.dataset} under root={root!r}"
        )

    sample, _ = dataset[0]
    channels = int(sample.shape[0])
    if channels != info["in_channels"]:
        raise SystemExit(
            f"checkpoint expects {info['in_channels']} channels but "
            f"--dataset {args.dataset} yields {channels}; use the matching --dataset"
        )
    if info["head_dims"][-1] != len(dataset.classes):
        raise SystemExit(
            f"checkpoint outputs {info['head_dims'][-1]} classes but Test has "
            f"{len(dataset.classes)} ({dataset.classes})"
        )

    print(
        f"[test] checkpoint={args.checkpoint} epoch={info['epoch']} "
        f"val_loss={info['val_loss']} in_channels={info['in_channels']} "
        f"head_dims={info['head_dims']}",
        flush=True,
    )
    print(
        f"[test] dataset={args.dataset} root={root!r} samples={len(dataset)} "
        f"classes={dataset.classes} device={device}",
        flush=True,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    loss, metrics = evaluate(
        model, loader, nn.CrossEntropyLoss(), device, len(dataset.classes)
    )
    print(
        f"[test] loss={loss:.4f} "
        + " ".join(f"{key}={value:.4f}" for key, value in metrics.items()),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    main()
