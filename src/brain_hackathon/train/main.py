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
        description="Train RadImageNet on the aligned MRI dataset."
    )
    parser.add_argument("--root", default="data/aligned")
    parser.add_argument(
        "--weights",
        default=None,
        help="path to ResNet50.pt (default: download from the Hugging Face Hub)",
    )
    parser.add_argument("--optimizer", default=BEST_OPTIMIZER)
    parser.add_argument("--lr", type=float, default=BEST_LR)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
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
    parser.add_argument("--wandb-project", default="brain-hackathon")
    parser.add_argument("--wandb-mode", default=None)
    parser.add_argument("--no-wandb", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    from brain_hackathon.model.resnet_rad import RadImageNetResNet50
    from brain_hackathon.train.utils import build_datasets

    args = parse_args(argv)

    train_ds, val_ds, test_ds = build_datasets(root=args.root)
    model = RadImageNetResNet50(
        dim=[len(train_ds.dataset.classes)],
        freeze_backbone=args.freeze,
        weights_path=args.weights,
    )
    logger = Logger(
        project=args.wandb_project,
        mode=args.wandb_mode,
        checkpoint_dir=args.checkpoint_dir,
        enabled=not args.no_wandb,
        config=vars(args),
    )
    return train(
        model,
        train_ds,
        val_ds,
        test_ds,
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


if __name__ == "__main__":
    main()
