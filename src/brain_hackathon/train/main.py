"""Training loop with metrics, Weights & Biases logging and early stopping."""

import argparse
import os

import torch
from torch import nn
from torch.utils.data import DataLoader

from brain_hackathon.train.early_stopping import EarlyStopping
from brain_hackathon.train.logger import Logger
from brain_hackathon.train.metrics import METRIC_NAMES, Metrics

# Defaults from the hyperparameter search (best val_loss/F1):
# adamw, lr=1e-3, weight_decay=0, no freezing (all layers trainable).
BEST_OPTIMIZER = "adamw"
BEST_LR = 1e-3
BEST_WEIGHT_DECAY = 0.0
BEST_FREEZE_FIRST_N = None

__all__ = [
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


def resolve_device(device=None):
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_optimizer(model, optimizer, lr, weight_decay):
    """Resolve ``optimizer`` (name, class or instance) with AdamW as the default."""
    if isinstance(optimizer, torch.optim.Optimizer):
        return optimizer
    if isinstance(optimizer, str):
        name = optimizer.lower()
        if name in ("adamw", "adam_w"):
            cls = torch.optim.AdamW
        elif name == "adam":
            cls = torch.optim.Adam
        elif name == "sgd":
            cls = torch.optim.SGD
        else:
            raise ValueError(f"unknown optimizer: {optimizer!r}")
    elif isinstance(optimizer, type) and issubclass(optimizer, torch.optim.Optimizer):
        cls = optimizer
    else:
        raise TypeError(f"unsupported optimizer: {optimizer!r}")
    params = [param for param in model.parameters() if param.requires_grad]
    if not params:
        raise ValueError("no trainable parameters (all layers are frozen)")
    return cls(params, lr=lr, weight_decay=weight_decay)


def num_classes_of(dataset):
    base = getattr(dataset, "dataset", dataset)
    classes = getattr(base, "classes", None)
    if classes:
        return len(classes)
    raise ValueError("cannot infer number of classes from dataset")


@torch.no_grad()
def evaluate(model, loader, criterion, device, num_classes):
    model.eval()
    metrics = Metrics(num_classes)
    total = 0.0
    count = 0
    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)
        logits = model(images)
        loss = criterion(logits, targets)
        total += loss.item() * images.size(0)
        count += images.size(0)
        metrics.update(logits, targets)
    return total / max(count, 1), metrics.compute()


@torch.no_grad()
def evaluate_loss(model, loader, criterion, device):
    model.eval()
    total = 0.0
    count = 0
    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)
        loss = criterion(model(images), targets)
        total += loss.item() * images.size(0)
        count += images.size(0)
    return total / max(count, 1)


def save_checkpoint(model, optimizer, epoch, val_loss, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "epoch": epoch,
            "val_loss": val_loss,
        },
        path,
    )
    return path


def train(
    model,
    train_dataset,
    val_dataset,
    test_dataset=None,
    *,
    optimizer=BEST_OPTIMIZER,
    lr=BEST_LR,
    batch_size=32,
    epochs=50,
    weight_decay=BEST_WEIGHT_DECAY,
    patience=10,
    min_delta=0.0,
    loss_period=50,
    device=None,
    logger=None,
    num_workers=0,
    checkpoint_dir="checkpoints",
    seed=42,
):
    """Train ``model`` with softmax cross-entropy.

    Test loss is measured every ``loss_period`` optimizer steps; val metrics are
    computed every epoch. Training stops when val loss has not decreased for
    ``patience`` epochs. Best and last checkpoints are saved and logged.
    """
    torch.manual_seed(seed)
    device = resolve_device(device)
    model.to(device)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = (
        DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        if test_dataset is not None
        else None
    )

    criterion = nn.CrossEntropyLoss()
    opt = build_optimizer(model, optimizer, lr, weight_decay)
    num_classes = num_classes_of(train_dataset)

    if logger is not None and checkpoint_dir is not None:
        logger.checkpoint_dir = checkpoint_dir
    os.makedirs(checkpoint_dir, exist_ok=True)

    early = EarlyStopping(patience=patience, min_delta=min_delta)

    global_step = 0
    history = []
    step_test_losses = []
    best_val = float("inf")
    best_path = None
    last_path = None
    stopped_early = False

    for epoch in range(1, epochs + 1):
        model.train()
        metrics = Metrics(num_classes)
        running = 0.0
        count = 0

        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)

            opt.zero_grad()
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            opt.step()

            global_step += 1
            running += loss.item() * images.size(0)
            count += images.size(0)
            metrics.update(logits, targets)

            if (
                test_loader is not None
                and loss_period
                and global_step % loss_period == 0
            ):
                test_step_loss = evaluate_loss(model, test_loader, criterion, device)
                step_test_losses.append((global_step, test_step_loss))
                if logger is not None:
                    logger.log({"test/loss_step": test_step_loss}, step=global_step)

        train_loss = running / max(count, 1)
        val_loss, val_metrics = evaluate(
            model, val_loader, criterion, device, num_classes
        )

        record = {
            "global_step": global_step,
            "train/loss": train_loss,
            **{f"train/{k}": v for k, v in metrics.compute().items()},
            "val/loss": val_loss,
            **{f"val/{k}": v for k, v in val_metrics.items()},
            "lr": opt.param_groups[0]["lr"],
        }
        if test_loader is not None:
            test_loss, test_metrics = evaluate(
                model, test_loader, criterion, device, num_classes
            )
            record["test/loss"] = test_loss
            record.update({f"test/{k}": v for k, v in test_metrics.items()})

        history.append(record)
        if logger is not None:
            logger.log_epoch(epoch, record, step=global_step)

        if val_loss < best_val - min_delta:
            best_val = val_loss
            best_path = save_checkpoint(
                model, opt, epoch, val_loss, os.path.join(checkpoint_dir, "best.pt")
            )
            if logger is not None:
                logger.log_checkpoint(best_path, "best")

        last_path = save_checkpoint(
            model, opt, epoch, val_loss, os.path.join(checkpoint_dir, "last.pt")
        )
        if logger is not None:
            logger.log_checkpoint(last_path, "last")

        if early.step(val_loss):
            stopped_early = True
            break

    result = {
        "history": history,
        "best_val_loss": best_val,
        "best_model_path": best_path,
        "last_model_path": last_path,
        "stopped_early": stopped_early,
        "step_test_losses": step_test_losses,
        "epochs_run": len(history),
    }
    if logger is not None:
        logger.finish()
    return result


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
        "--freeze-all",
        action="store_true",
        default=True,
        help="freeze every layer except the first --first-n backbone layers and the head",
    )
    parser.add_argument(
        "--first-n",
        type=int,
        default=2,
        dest="first_n",
        help="number of first backbone layers to keep trainable with --freeze-all",
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
        dim=len(train_ds.dataset.classes),
        unfreeze_first_n=args.first_n if args.freeze_all else BEST_FREEZE_FIRST_N,
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
