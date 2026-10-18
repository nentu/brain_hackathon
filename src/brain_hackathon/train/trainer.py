"""Training loop."""

import os

import torch
from torch import nn
from torch.utils.data import DataLoader

from brain_hackathon.train.early_stopping import EarlyStopping
from brain_hackathon.train.engine import (
    build_optimizer,
    evaluate,
    evaluate_loss,
    num_classes_of,
    resolve_device,
    save_checkpoint,
)
from brain_hackathon.train.metrics import Metrics

# Defaults from the hyperparameter search (best val_loss/F1).
BEST_OPTIMIZER = "adamw"
BEST_LR = 1e-3
BEST_WEIGHT_DECAY = 1e-4  # 1e-4


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
    log_every=5,
    initial_eval=True,
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
    optimizer_name = (
        optimizer if isinstance(optimizer, str) else type(optimizer).__name__
    )
    print(
        f"[train] device={device} epochs={epochs} batch_size={batch_size} "
        f"optimizer={optimizer_name} lr={lr}",
        flush=True,
    )
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

    if logger is not None:
        logger.log_config(
            {
                "optimizer": optimizer
                if isinstance(optimizer, str)
                else type(opt).__name__,
                "lr": lr,
                "weight_decay": weight_decay,
                "batch_size": batch_size,
                "epochs": epochs,
                "patience": patience,
                "min_delta": min_delta,
                "loss_period": loss_period,
                "log_every": log_every,
                "initial_eval": initial_eval,
                "seed": seed,
                "device": str(device),
                "num_classes": num_classes,
                "head_dims": list(getattr(model, "dim", [])),
                "n_train": len(train_dataset),
                "n_val": len(val_dataset),
                "n_test": len(test_dataset) if test_dataset is not None else 0,
                "n_params": sum(param.numel() for param in model.parameters()),
                "n_trainable": sum(
                    param.numel() for param in model.parameters() if param.requires_grad
                ),
            }
        )

    if initial_eval:
        train_loss0, train_metrics0 = evaluate(
            model, train_loader, criterion, device, num_classes
        )
        val_loss0, val_metrics0 = evaluate(
            model, val_loader, criterion, device, num_classes
        )
        message = (
            f"[epoch   0] loss train={train_loss0:.4f} val={val_loss0:.4f}"
        )
        message += (
            f" | train acc={train_metrics0['acc']:.4f} "
            f"f1={train_metrics0['f1']:.4f}"
        )
        message += (
            f" | val acc={val_metrics0['acc']:.4f} "
            f"f1={val_metrics0['f1']:.4f}"
        )

        record0 = {
            "loss/train": train_loss0,
            **{f"{k}/train": v for k, v in train_metrics0.items()},
            "loss/val": val_loss0,
            **{f"{k}/val": v for k, v in val_metrics0.items()},
        }
        if test_loader is not None:
            test_loss0, test_metrics0 = evaluate(
                model, test_loader, criterion, device, num_classes
            )
            message += f" | test loss={test_loss0:.4f}"
            record0["loss/test"] = test_loss0
            record0.update(
                {f"{k}/test": v for k, v in test_metrics0.items()}
            )
        print(message, flush=True)

        if logger is not None:
            logger.log(record0, step=0)

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
                    logger.log({"test_step/loss": test_step_loss}, step=global_step)

        train_loss = running / max(count, 1)
        val_loss, val_metrics = evaluate(
            model, val_loader, criterion, device, num_classes
        )

        record = {
            "global_step": global_step,
            "loss/train": train_loss,
            **{f"{k}/train": v for k, v in metrics.compute().items()},
            "loss/val": val_loss,
            **{f"{k}/val": v for k, v in val_metrics.items()},
            "lr": opt.param_groups[0]["lr"],
        }
        if test_loader is not None:
            test_loss, test_metrics = evaluate(
                model, test_loader, criterion, device, num_classes
            )
            record["loss/test"] = test_loss
            record.update({f"{k}/test": v for k, v in test_metrics.items()})

        if log_every and (epoch == 1 or epoch % log_every == 0 or epoch == epochs):
            message = (
                f"[epoch {epoch:3d}] loss train={record['loss/train']:.4f} "
                f"val={record['loss/val']:.4f}"
            )
            if "loss/test" in record:
                message += f" test={record['loss/test']:.4f}"
            message += f" | val acc={record['acc/val']:.4f} f1={record['f1/val']:.4f}"
            if "acc/test" in record:
                message += (
                    f" | test acc={record['acc/test']:.4f} f1={record['f1/test']:.4f}"
                )
            print(message, flush=True)

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
