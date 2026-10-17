"""K-fold cross-validation on an 80% train split + final full-train/test run.

    uv run cross_validate --root data/aligned --epochs 30 --n-splits 4
"""

import argparse
import os
from statistics import mean, pstdev

from torch.utils.data import Subset

from brain_hackathon.data.augmentation import train_aug, val_aug
from brain_hackathon.data.dataset import DEFAULT_ROOT, AlignedMRIClassificationDataset
from brain_hackathon.model.resnet_rad import RadImageNetResNet50
from brain_hackathon.train.trainer import train
from brain_hackathon.train.utils import (
    ALL_SPLITS,
    content_groups,
    stratified_indices,
    stratified_kfold,
)

METRIC_KEYS = ("acc", "f1", "precision", "recall", "sensitivity", "specificity")


def _dedup_indices(index):
    groups = content_groups(index)
    keep = []
    seen = set()
    for i, group in enumerate(groups):
        if group not in seen:
            seen.add(group)
            keep.append(i)
    return keep, groups


def _metrics_from_record(record, split):
    metrics = {"loss": record[f"loss/{split}"]}
    metrics.update({key: record[f"{key}/{split}"] for key in METRIC_KEYS})
    return metrics


def run_cross_validation(
    root=DEFAULT_ROOT,
    n_splits=4,
    train_frac=0.8,
    epochs=30,
    batch_size=32,
    seed=42,
    weights=None,
    freeze_backbone=True,
    device=None,
    checkpoint_dir=os.path.join("checkpoints", "cv"),
    verbose=True,
    train_fn=None,
    model_factory=None,
):
    """Run n-fold CV on the train portion, then train on all of it and test.

    Returns ``{"folds": [...], "mean": {...}, "final": {...}, "sizes": {...}}``.
    """
    base = AlignedMRIClassificationDataset(
        root=root, split=ALL_SPLITS, augment=None
    )
    if len(base) < 3:
        raise ValueError(f"not enough series under root={root!r}")

    keep, groups = _dedup_indices(base.index)
    labels = [base.class_to_idx[base.index[i][1]] for i in keep]
    kept_groups = [groups[i] for i in keep]

    train_pos, test_pos, _ = stratified_indices(
        labels, kept_groups, (train_frac, 1.0 - train_frac, 0.0), seed
    )
    train_orig = [keep[p] for p in train_pos]
    test_orig = [keep[p] for p in test_pos]

    train_labels = [labels[p] for p in train_pos]
    train_groups = [kept_groups[p] for p in train_pos]
    folds = stratified_kfold(
        train_labels, train_groups, n_splits=n_splits, seed=seed
    )

    if verbose:
        print(
            f"[cv] root={root!r} classes={base.classes} series={len(base.index)}",
            flush=True,
        )
        print(f"[cv] dedup kept {len(keep)}/{len(base.index)}", flush=True)
        print(
            f"[cv] split {round(train_frac * 100)}/"
            f"{round((1 - train_frac) * 100)}: "
            f"train={len(train_orig)} test={len(test_orig)}",
            flush=True,
        )
        print(
            f"[cv] {n_splits}-fold val sizes="
            f"{[len(val) for _, val in folds]}",
            flush=True,
        )

    train_ds = AlignedMRIClassificationDataset(
        root=root, split=ALL_SPLITS, augment=train_aug
    )
    eval_ds = AlignedMRIClassificationDataset(
        root=root, split=ALL_SPLITS, augment=val_aug
    )
    num_classes = len(base.classes)

    train_fn = train_fn or train
    if model_factory is None:

        def model_factory():
            return RadImageNetResNet50(
                dim=[num_classes],
                freeze_backbone=freeze_backbone,
                verbose=False,
                weights_path=weights,
            )

    common = dict(
        epochs=epochs,
        batch_size=batch_size,
        patience=epochs,
        loss_period=10**9,
        device=device,
        logger=None,
        seed=seed,
    )

    fold_metrics = []
    fold_sizes = []
    for fold_index, (fit_pos, val_pos) in enumerate(folds, 1):
        fit_orig = [keep[train_pos[p]] for p in fit_pos]
        val_orig = [keep[train_pos[p]] for p in val_pos]
        fold_sizes.append(len(val_orig))

        result = train_fn(
            model_factory(),
            Subset(train_ds, fit_orig),
            Subset(eval_ds, val_orig),
            checkpoint_dir=os.path.join(checkpoint_dir, f"fold{fold_index}"),
            **common,
        )
        metrics = _metrics_from_record(result["history"][-1], "val")
        fold_metrics.append(metrics)
        if verbose:
            print(
                f"fold {fold_index}/{n_splits} (train={len(fit_orig)}, "
                f"val={len(val_orig)}): "
                + " ".join(f"{k}={v:.4f}" for k, v in metrics.items())
            )

    summary = {}
    if fold_metrics:
        for key in fold_metrics[0]:
            values = [metrics[key] for metrics in fold_metrics]
            summary[key] = (mean(values), pstdev(values))
    if verbose:
        print(
            "CV mean±std: "
            + " ".join(
                f"{key}={mu:.4f}±{sd:.4f}" for key, (mu, sd) in summary.items()
            )
        )

    test_subset = Subset(eval_ds, test_orig)
    final = train_fn(
        model_factory(),
        Subset(train_ds, train_orig),
        test_subset,
        test_subset,
        checkpoint_dir=os.path.join(checkpoint_dir, "final"),
        **common,
    )
    final_metrics = _metrics_from_record(final["history"][-1], "test")
    if verbose:
        print(
            f"final (train={len(train_orig)}, test={len(test_orig)}): "
            + " ".join(f"{k}={v:.4f}" for k, v in final_metrics.items())
        )

    return {
        "folds": fold_metrics,
        "mean": summary,
        "final": final_metrics,
        "sizes": {
            "train": len(train_orig),
            "test": len(test_orig),
            "fold_val": fold_sizes,
        },
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Cross-validate, then train on the full train split."
    )
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--n-splits", type=int, default=4)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weights", default=None)
    parser.add_argument(
        "--freeze",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="freeze the pretrained backbone (train only the head)",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--checkpoint-dir", default=os.path.join("checkpoints", "cv")
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    return run_cross_validation(
        root=args.root,
        n_splits=args.n_splits,
        train_frac=args.train_frac,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        weights=args.weights,
        freeze_backbone=args.freeze,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
    )


if __name__ == "__main__":
    main()
