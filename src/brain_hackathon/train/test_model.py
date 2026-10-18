"""Evaluate a saved checkpoint on a test split.

    # 20% holdout of the Train dir (80/20, same seed as training)
    uv run test_model --checkpoint checkpoints/best.pt --dataset aligned --split test
    uv run test_model --checkpoint checkpoints/raw_best.pt --dataset raw --split test

    # aligned: same holdout used by cross-validation
    uv run test_model --checkpoint checkpoints/best.pt --dataset aligned --split cv

    # original Test directory
    uv run test_model --checkpoint checkpoints/best.pt --dataset aligned --split test-dir
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
from brain_hackathon.data.raw_dataset import RAW_ROOT, RawImageDataset
from brain_hackathon.model.resnet_rad import (
    RadImageNetResNet50,
    RadImageNetResNet50Single,
)
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
    if in_channels == 1:
        model = RadImageNetResNet50Single(
            dim=dims, verbose=False, pretrained=False
        )
    else:
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
    """Return the indices of the 20% aligned holdout used by cross-validation."""
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


def train_holdout_indices(root, dataset_kind, train_frac=0.8, seed=42):
    """Indices of the held-out part of the Train dir (80/20, stratified)."""
    if dataset_kind == "raw":
        dataset = RawImageDataset(root=root, split="Train", normalize=True)
        labels = [int(entry[1]) for entry in dataset.index]
        groups = list(range(len(labels)))
    else:
        dataset = AlignedMRIClassificationDataset(
            root=root, split="Train", augment=None
        )
        labels = [dataset.class_to_idx[entry[1]] for entry in dataset.index]
        groups = [(entry[1], entry[0]) for entry in dataset.index]

    _, test_pos, _ = stratified_indices(
        labels,
        groups,
        (train_frac, 1.0 - train_frac, 0.0),
        seed,
    )
    return test_pos, dataset.classes


def make_test_dataset(dataset_kind, split_mode, root, train_frac=0.8, seed=42):
    """Return ``(dataset, classes, split_label)``."""
    if split_mode == "test":
        indices, classes = train_holdout_indices(
            root, dataset_kind, train_frac, seed
        )
        if dataset_kind == "raw":
            dataset = Subset(
                RawImageDataset(root=root, split="Train", normalize=True),
                indices,
            )
        else:
            dataset = Subset(
                AlignedMRIClassificationDataset(
                    root=root, split="Train", augment=val_aug
                ),
                indices,
            )
        label = (
            f"{dataset_kind} Train holdout "
            f"({round(train_frac * 100)}/"
            f"{round((1 - train_frac) * 100)}, seed={seed})"
        )
        return dataset, classes, label

    if split_mode == "cv":
        if dataset_kind != "aligned":
            raise ValueError("--split cv is only supported for --dataset aligned")
        indices, classes = cross_validation_test_indices(root, train_frac, seed)
        dataset = Subset(
            AlignedMRIClassificationDataset(
                root=root, split=ALL_SPLITS, augment=val_aug
            ),
            indices,
        )
        label = (
            f"aligned cross-validation holdout "
            f"({round(train_frac * 100)}/"
            f"{round((1 - train_frac) * 100)}, seed={seed})"
        )
        return dataset, classes, label

    # split_mode == "test-dir": the original Test directory
    if dataset_kind == "raw":
        dataset = RawImageDataset(root=root, split="Test", normalize=True)
    else:
        dataset = AlignedMRIClassificationDataset(
            root=root, split="Test", augment=None
        )
    return dataset, dataset.classes, f"{dataset_kind} original Test directory"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Evaluate a checkpoint on a test split."
    )
    parser.add_argument(
        "--checkpoint", required=True, help="path to best.pt / last.pt"
    )
    parser.add_argument(
        "--dataset",
        choices=("raw", "aligned"),
        default="aligned",
        help="raw: single-channel Test split + 1-channel model; "
        "aligned: 6-channel dataset + 6-channel model",
    )
    parser.add_argument(
        "--split",
        choices=("cv", "test", "test-dir"),
        default="cv",
        help="test: 20%% holdout of the Train dir (80/20, same seed); "
        "cv: aligned cross-validation holdout (aligned only); "
        "test-dir: the original Test directory",
    )
    parser.add_argument(
        "--root",
        default=None,
        help="dataset root (default: raw dataset root for 'raw', "
        "data/aligned for 'aligned')",
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

    root = args.root or (RAW_ROOT if args.dataset == "raw" else DEFAULT_ROOT)
    try:
        dataset, classes, split_label = make_test_dataset(
            args.dataset, args.split, root, args.train_frac, args.seed
        )
    except ValueError as exc:
        raise SystemExit(str(exc))

    if len(dataset) == 0:
        raise SystemExit(
            f"empty test split for dataset={args.dataset} under root={root!r}"
        )

    sample, _ = dataset[0]
    channels = int(sample.shape[0])
    if channels != info["in_channels"]:
        raise SystemExit(
            f"checkpoint expects {info['in_channels']} channels but "
            f"--dataset {args.dataset} yields {channels}; use the matching --dataset"
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
        f"[test] dataset={args.dataset} split={split_label} root={root!r} "
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
