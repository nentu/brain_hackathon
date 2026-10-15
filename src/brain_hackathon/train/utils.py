"""Dataset helpers for training.

Build a dataset over all splits and split it into stratified train/val/test subsets.
"""

import random
from collections import defaultdict

from torch.utils.data import Subset

from brain_hackathon.data.augmentation import train_aug, val_aug
from brain_hackathon.data.dataset import (
    DEFAULT_ROOT,
    AlignedMRIClassificationDataset,
)

ALL_SPLITS = ("Train", "Test")
DEFAULT_FRACTIONS = (0.7, 0.2, 0.1)


def _allocate(total, fractions):
    """Split ``total`` items into ``len(fractions)`` counts (largest remainder)."""
    raw = [total * fraction for fraction in fractions]
    counts = [int(value) for value in raw]
    remainder = total - sum(counts)
    order = sorted(
        range(len(fractions)), key=lambda i: raw[i] - counts[i], reverse=True
    )
    for i in range(remainder):
        counts[order[i % len(order)]] += 1
    return counts


def stratified_indices(labels, groups=None, fractions=DEFAULT_FRACTIONS, seed=42):
    """Partition sample indices into ``(train, val, test)`` stratified by label.

    ``groups`` optionally assigns each sample to a group (e.g. ``(class, series_id)``);
    all samples of a group are placed in the same split, so a series never spans folds.
    Returns three lists of sample indices.
    """
    if len(fractions) != 3:
        raise ValueError("fractions must be (train, val, test)")
    if not 0.0 < sum(fractions) <= 1.000001:
        raise ValueError("fractions must sum to 1")

    if groups is None:
        groups = list(range(len(labels)))

    group_to_samples = defaultdict(list)
    group_label = {}
    for sample, (label, group) in enumerate(zip(labels, groups)):
        group_to_samples[group].append(sample)
        group_label[group] = label

    class_to_groups = defaultdict(list)
    for group, label in group_label.items():
        class_to_groups[label].append(group)

    rng = random.Random(seed)
    train, val, test = [], [], []
    for label in sorted(class_to_groups):
        class_groups = class_to_groups[label]
        rng.shuffle(class_groups)
        n_train, n_val, _ = _allocate(len(class_groups), fractions)
        for group in class_groups[:n_train]:
            train.extend(group_to_samples[group])
        for group in class_groups[n_train : n_train + n_val]:
            val.extend(group_to_samples[group])
        for group in class_groups[n_train + n_val :]:
            test.extend(group_to_samples[group])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def build_datasets(
    root=DEFAULT_ROOT,
    splits=ALL_SPLITS,
    fractions=DEFAULT_FRACTIONS,
    seed=42,
    train_transform=train_aug,
    eval_transform=val_aug,
    **dataset_kwargs,
):
    """Build stratified ``(train, val, test)`` subsets over all ``splits``.

    The dataset is created once for indexing/labels, then train/val/test are subsets of
    three dataset instances sharing the same index but using ``train_transform`` and
    ``eval_transform`` respectively. Extra keyword arguments are forwarded to
    :class:`AlignedMRIClassificationDataset`.
    """
    dataset_kwargs.pop("augment", None)
    base = AlignedMRIClassificationDataset(root=root, split=splits, **dataset_kwargs)
    labels = [base.class_to_idx[class_name] for _, class_name, _ in base.index]
    groups = [
        (class_name, series_id) for series_id, class_name, _ in base.index
    ]

    train_idx, val_idx, test_idx = stratified_indices(
        labels, groups, fractions, seed
    )

    shared = dict(root=root, split=splits, **dataset_kwargs)
    train_ds = AlignedMRIClassificationDataset(augment=train_transform, **shared)
    val_ds = AlignedMRIClassificationDataset(augment=eval_transform, **shared)
    test_ds = AlignedMRIClassificationDataset(augment=eval_transform, **shared)

    return (
        Subset(train_ds, train_idx),
        Subset(val_ds, val_idx),
        Subset(test_ds, test_idx),
    )
