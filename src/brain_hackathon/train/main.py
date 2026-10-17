"""Training CLI."""

import argparse

from brain_hackathon.train.early_stopping import EarlyStopping
from brain_hackathon.train.engine import (
    build_optimizer,
    evaluate,
    evaluate_loss,
    num_classes_of,
    resolve_device,
    save_checkpoint,
)
from brain_hackathon.train.logger import Logger
from brain_hackathon.train.metrics import METRIC_NAMES, Metrics
from brain_hackathon.train.trainer import (
    BEST_LR,
    BEST_OPTIMIZER,
    BEST_WEIGHT_DECAY,
    train,
)

__all__ = [
    "BEST_LR",
    "BEST_OPTIMIZER",
    "BEST_WEIGHT_DECAY",
    "METRIC_NAMES",
    "EarlyStopping",
    "Logger",
    "Metrics",
    "build_optimizer",
    "evaluate",
    "evaluate_loss",
    "main",
    "num_classes_of",
    "resolve_device",
    "save_checkpoint",
    "train",
]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Train on the dataset's Train split and evaluate on Test."
    )
    parser.add_argument(
        "--dataset",
        choices=("raw", "aligned"),
        default="raw",
        help="raw: single-channel images + 1-channel model; "
        "aligned: 6-channel aligned dataset + 6-channel model",
    )
    parser.add_argument(
        "--root",
        default=None,
        help="dataset root (default: raw dataset root for 'raw', "
        "data/aligned for 'aligned')",
    )
    parser.add_argument(
        "--weights",
        default=None,
        help="path to ResNet50.pt (default: download from the Hugging Face Hub)",
    )
    parser.add_argument("--optimizer", default=BEST_OPTIMIZER)
    parser.add_argument("--lr", type=float, default=BEST_LR)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--weight-decay", type=float, default=BEST_WEIGHT_DECAY)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--loss-period", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument(
        "--freeze",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="freeze the whole pretrained backbone (train only the head)",
    )
    parser.add_argument("--wandb-project", default="brain-hackathon-raw")
    parser.add_argument("--wandb-mode", default=None)
    parser.add_argument("--no-wandb", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    from brain_hackathon.data.augmentation import train_aug, val_aug
    from brain_hackathon.data.dataset import (
        DEFAULT_ROOT,
        AlignedMRIClassificationDataset,
    )
    from brain_hackathon.data.raw_dataset import RAW_ROOT, RawImageDataset
    from brain_hackathon.model.resnet_rad import (
        RadImageNetResNet50,
        RadImageNetResNet50Single,
    )

    args = parse_args(argv)
    root = args.root or (RAW_ROOT if args.dataset == "raw" else DEFAULT_ROOT)

    if args.dataset == "raw":
        train_ds = RawImageDataset(root=root, split="Train", verbose=True)
        test_ds = RawImageDataset(root=root, split="Test", verbose=True)
        model_class = RadImageNetResNet50Single
    else:
        train_ds = AlignedMRIClassificationDataset(
            root=root, split="Train", augment=train_aug
        )
        test_ds = AlignedMRIClassificationDataset(
            root=root, split="Test", augment=val_aug
        )
        model_class = RadImageNetResNet50

    if len(train_ds) == 0 or len(test_ds) == 0:
        raise SystemExit(
            f"empty split under root={root!r} "
            f"(train={len(train_ds)}, test={len(test_ds)})"
        )
    if train_ds.classes != test_ds.classes:
        raise SystemExit(
            f"class mismatch: train={train_ds.classes} test={test_ds.classes}"
        )

    print(
        f"[data] dataset={args.dataset} root={root!r} train={len(train_ds)} "
        f"test={len(test_ds)} classes={train_ds.classes}",
        flush=True,
    )
    print("[data] no code split: Test is also used as validation", flush=True)

    model = model_class(
        dim=[len(train_ds.classes)],
        freeze_backbone=args.freeze,
        weights_path=args.weights,
    )
    print(f"[model] head dims={model.dim} freeze_backbone={args.freeze}", flush=True)
    logger = Logger(
        project=args.wandb_project,
        mode=args.wandb_mode,
        checkpoint_dir=args.checkpoint_dir,
        enabled=not args.no_wandb,
        config=vars(args),
    )
    result = train(
        model,
        train_ds,
        test_ds,  # validation (no code split)
        test_ds,  # test
        optimizer=args.optimizer,
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        weight_decay=args.weight_decay,
        patience=args.patience,
        loss_period=args.loss_period,
        logger=logger,
        checkpoint_dir=args.checkpoint_dir,
        seed=args.seed,
    )
    print(
        f"[done] epochs={result['epochs_run']} "
        f"best_val_loss={result['best_val_loss']:.4f} "
        f"best={result['best_model_path']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    main()
