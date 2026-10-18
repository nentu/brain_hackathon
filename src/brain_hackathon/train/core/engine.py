"""Training helpers: device, optimizer, evaluation and checkpoints."""

import os

import torch

from brain_hackathon.train.core.metrics import Metrics


def resolve_device(device=None):
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_optimizer(model, optimizer, lr, weight_decay):
    """Resolve ``optimizer`` (name, class or instance) with AdamW as the default."""
    if isinstance(optimizer, torch.optim.Optimizer):
        return optimizer
    name = optimizer.lower() if isinstance(optimizer, str) else None
    if name is not None:
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
    if name == "sgd":
        return cls(
            params, lr=lr, weight_decay=weight_decay, nesterov=True, momentum=0.9
        )
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
