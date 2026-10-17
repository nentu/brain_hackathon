"""Evaluate a saved checkpoint on the aligned Test split.

    uv run test_model --checkpoint checkpoints/best.pt
"""

import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from brain_hackathon.data.augmentation import val_aug
from brain_hackathon.data.dataset import (
    DEFAULT_ROOT,
    AlignedMRIClassificationDataset,
)
from brain_hackathon.model.resnet_rad import RadImageNetResNet50
from brain_hackathon.train.cross_validate import _dedup_indices
from brain_hackathon.train.engine import evaluate, resolve_device
from brain_hackathon.train.utils import ALL_SPLITS, stratified_indices


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
    model = RadImageNetResNet50(
        dim=dims,
        in_channels=in_channels,
        verbose=False,
        pretrained=False,
    )
    return model.to(device) if device is not None else model


def load_checkpoint(path, device="cpu"):
    """Load a checkpoint and return ``(model_in_eval_mode, info)``."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    state_dict = (
        checkpoint.get("model", checkpoint)
        if isinstance(checkpoint, dict)
        else checkpoint
    )

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


def cross_validation_test_indices(root, train_frac=0.8, seed=42):
    """Return the indices of the 20% holdout used by cross-validation.

    Uses the same dedup + stratified 80/20 split as
    :func:`brain_hackathon.train.cross_validate.run_cross_validation`.
    """
    base = AlignedMRIClassificationDataset(
        root=root, split=ALL_SPLITS, augment=None
    )
    keep, groups = _dedup_indices(base.index)
    labels = [base.class_to_idx[base.index[i][1]] for i in keep]
    kept_groups = [groups[i] for i in keep]
    _, test_pos, _ = stratified_indices(
        labels,
        kept_groups,
        (train_frac, 1.0 - train_frac, 0.0),
        seed,
    )
    return [keep[p] for p in test_pos], base.classes


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Evaluate a checkpoint on a test split."
    )
    parser.add_argument(
        "--checkpoint", required=True, help="path to best.pt / last.pt"
    )
    parser.add_argument("--root", default=DEFAULT_ROOT, help="aligned dataset root")
    parser.add_argument(
        "--split",
        choices=("cv", "test"),
        default="cv",
        help="cv: same 80/20 holdout as cross-validation (default); "
        "test: the original data/aligned/Test directory",
    )
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default=None)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    device = resolve_device(args.device)
    model, info = load_checkpoint(args.checkpoint, device)

    if args.split == "cv":
        indices, classes = cross_validation_test_indices(
            args.root, args.train_frac, args.seed
        )
        dataset = Subset(
            AlignedMRIClassificationDataset(
                root=args.root, split=ALL_SPLITS, augment=val_aug
            ),
            indices,
        )
        split_label = (
            f"cross-validation holdout "
            f"({round(args.train_frac * 100)}/"
            f"{round((1 - args.train_frac) * 100)}, seed={args.seed})"
        )
    else:
        dataset = AlignedMRIClassificationDataset(
            root=args.root, split="Test", augment=None
        )
        classes = dataset.classes
        split_label = "original Test directory"

    if len(dataset) == 0:
        raise SystemExit(f"empty test split under root={args.root!r}")

    sample, _ = dataset[0]
    channels = int(sample.shape[0])
    if channels != info["in_channels"]:
        raise SystemExit(
            f"checkpoint expects {info['in_channels']} channels but "
            f"the split yields {channels}"
        )
    if info["head_dims"][-1] != len(classes):
        raise SystemExit(
            f"checkpoint outputs {info['head_dims'][-1]} classes but the split "
            f"has {len(classes)} ({classes})"
        )

    print(
        f"[test] checkpoint={args.checkpoint} epoch={info['epoch']} "
        f"val_loss={info['val_loss']} in_channels={info['in_channels']} "
        f"head_dims={info['head_dims']}",
        flush=True,
    )
    print(
        f"[test] split={split_label} root={args.root!r} "
        f"samples={len(dataset)} classes={classes} device={device}",
        flush=True,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    loss, metrics = evaluate(
        model, loader, nn.CrossEntropyLoss(), device, len(classes)
    )
    print(
        f"[test] loss={loss:.4f} "
        + " ".join(f"{key}={value:.4f}" for key, value in metrics.items()),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    main()
