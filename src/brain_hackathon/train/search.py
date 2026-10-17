"""Small hyperparameter grid search for the RadImageNet classifier."""

import argparse
import itertools
import json
import os

from brain_hackathon.data.dataset import DEFAULT_ROOT
from brain_hackathon.model.resnet_rad import RadImageNetResNet50
from brain_hackathon.train.logger import Logger
from brain_hackathon.train.main import train
from brain_hackathon.train.utils import build_datasets

DEFAULT_OPTIMIZERS = ("adamw", "sgd")
DEFAULT_LRS = (1e-2, 1e-3, 1e-4)
DEFAULT_WEIGHT_DECAYS = (0.0, 1e-4, 1e-2)
DEFAULT_FREEZES = (None, 0)


def iter_configs(optimizers, lrs, weight_decays, freezes):
    """Cartesian product of the search space as a list of config dicts."""
    return [
        {
            "optimizer": optimizer,
            "lr": lr,
            "weight_decay": weight_decay,
            "freeze_first_n": freeze,
        }
        for optimizer, lr, weight_decay, freeze in itertools.product(
            optimizers, lrs, weight_decays, freezes
        )
    ]


def config_name(config):
    freeze = config["freeze_first_n"]
    freeze_label = "all" if freeze is None else freeze
    return (
        f"{config['optimizer']}-lr{config['lr']:g}-"
        f"wd{config['weight_decay']:g}-freeze{freeze_label}"
    )


def evaluate_config(config, context):
    """Train one configuration and return its final-epoch metrics."""
    model = RadImageNetResNet50(
        dim=[context["num_classes"]],
        verbose=False,
        freeze_backbone=config["freeze_first_n"] is not None,
    )
    logger_factory = context.get("logger_factory")
    logger = logger_factory(config) if logger_factory is not None else None
    result = train(
        model,
        context["train_ds"],
        context["val_ds"],
        context["test_ds"],
        optimizer=config["optimizer"],
        lr=config["lr"],
        weight_decay=config["weight_decay"],
        batch_size=context["batch_size"],
        epochs=context["epochs"],
        loss_period=10**9,
        patience=10**9,
        device=context["device"],
        logger=logger,
        num_workers=context["num_workers"],
        checkpoint_dir=context["checkpoint_dir"],
        seed=context["seed"],
    )
    last = result["history"][-1]
    return {
        **config,
        "epochs_run": result["epochs_run"],
        "train_loss": last["loss/train"],
        "val_loss": last["loss/val"],
        "val_acc": last["val/acc"],
        "val_f1": last["val/f1"],
        "test_loss": last.get("loss/test"),
        "test_acc": last.get("test/acc"),
        "test_f1": last.get("test/f1"),
    }


def grid_search(
    datasets=None,
    root=DEFAULT_ROOT,
    optimizers=DEFAULT_OPTIMIZERS,
    lrs=DEFAULT_LRS,
    weight_decays=DEFAULT_WEIGHT_DECAYS,
    freezes=DEFAULT_FREEZES,
    epochs=2,
    batch_size=32,
    device=None,
    seed=42,
    num_workers=4,
    max_runs=None,
    results_path=None,
    checkpoint_dir=os.path.join("checkpoints", "search"),
    runner=None,
    logger_factory=None,
):
    """Run the grid and return the per-config records.

    ``runner`` can be replaced (``runner(config, context) -> record``) to stub training.
    ``logger_factory(config) -> Logger`` creates a per-config logger for W&B logging.
    """
    if datasets is None:
        datasets = build_datasets(root=root)
    train_ds, val_ds, test_ds = datasets

    context = {
        "train_ds": train_ds,
        "val_ds": val_ds,
        "test_ds": test_ds,
        "num_classes": len(train_ds.dataset.classes),
        "epochs": epochs,
        "batch_size": batch_size,
        "device": device,
        "seed": seed,
        "num_workers": num_workers,
        "checkpoint_dir": checkpoint_dir,
        "logger_factory": logger_factory,
    }

    runner = runner or evaluate_config
    configs = iter_configs(optimizers, lrs, weight_decays, freezes)
    if max_runs is not None:
        configs = configs[:max_runs]

    results = []
    for index, config in enumerate(configs, 1):
        record = runner(config, context)
        results.append(record)
        print(
            f"[{index}/{len(configs)}] {config['optimizer']} lr={config['lr']} "
            f"wd={config['weight_decay']} freeze={config['freeze_first_n']} -> "
            f"val_loss={record['val_loss']:.4f} val_f1={record['val_f1']:.3f} "
            f"val_acc={record['val_acc']:.3f}",
            flush=True,
        )
        if results_path:
            os.makedirs(os.path.dirname(os.path.abspath(results_path)), exist_ok=True)
            with open(results_path, "w", encoding="utf-8") as handle:
                json.dump(results, handle, indent=2)

    return results


def select_best(results, key="val_loss", maximize=False):
    if not results:
        raise ValueError("no results")
    chooser = max if maximize else min
    return chooser(results, key=lambda record: record[key])


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Hyperparameter grid search.")
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--optimizers", nargs="+", default=list(DEFAULT_OPTIMIZERS))
    parser.add_argument("--lrs", nargs="+", type=float, default=list(DEFAULT_LRS))
    parser.add_argument("--weight-decays", nargs="+", type=float, default=list(DEFAULT_WEIGHT_DECAYS))
    parser.add_argument(
        "--freezes",
        nargs="+",
        type=int,
        default=[-1 if f is None else f for f in DEFAULT_FREEZES],
        help="-1 means all layers trainable",
    )
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--results-path", default=os.path.join("checkpoints", "search", "results.json"))
    parser.add_argument("--checkpoint-dir", default=os.path.join("checkpoints", "search"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--wandb-project", default="brain-hackathon")
    parser.add_argument("--wandb-mode", default=None)
    parser.add_argument("--no-wandb", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    freezes = [None if value < 0 else value for value in args.freezes]

    logger_factory = None
    if not args.no_wandb:

        def logger_factory(config):
            return Logger(
                project=args.wandb_project,
                run_name=config_name(config),
                config=config,
                mode=args.wandb_mode,
                checkpoint_dir=args.checkpoint_dir,
                log_artifacts=False,
            )

    results = grid_search(
        root=args.root,
        optimizers=args.optimizers,
        lrs=args.lrs,
        weight_decays=args.weight_decays,
        freezes=freezes,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        num_workers=args.num_workers,
        max_runs=args.max_runs,
        results_path=args.results_path,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
        logger_factory=logger_factory,
    )

    best_loss = select_best(results, key="val_loss")
    best_f1 = select_best(results, key="val_f1", maximize=True)
    print("\nBEST by val_loss:", json.dumps(best_loss))
    print("BEST by val_f1:  ", json.dumps(best_f1))
    print("results saved to", args.results_path)
    return results


if __name__ == "__main__":
    main()
