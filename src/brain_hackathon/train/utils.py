"""Dataset helpers for training.

Build a dataset over all splits and split it into stratified train/val/test subsets,
deduplicating by content so identical images never span folds.
"""

import hashlib
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


def _md5(path, block=1 << 16):
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(block), b""):
            digest.update(chunk)
    return digest.hexdigest()


def content_groups(index, max_shared_files=4, use_series_identity=True):
    """Assign each index entry a group id via union-find.

    Entries are merged when they share the same ``(class, series_id)`` or when they
    share a byte-identical modality file (within the same class). Files shared by more
    than ``max_shared_files`` series are ignored so a common/blank image cannot chain
    many unrelated series together.
    """
    count = len(index)
    parent = list(range(count))

    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(a, b):
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[max(root_a, root_b)] = min(root_a, root_b)

    if use_series_identity:
        first_of_series = {}
        for i, (series_id, class_name, _) in enumerate(index):
            key = (class_name, series_id)
            if key in first_of_series:
                union(i, first_of_series[key])
            else:
                first_of_series[key] = i

    by_hash = defaultdict(list)
    for i, (_, _, modalities) in enumerate(index):
        for path in modalities.values():
            by_hash[_md5(path)].append(i)

    for members in by_hash.values():
        members = sorted(set(members))
        if len(members) < 2 or len(members) > max_shared_files:
            continue
        by_class = defaultdict(list)
        for i in members:
            by_class[index[i][1]].append(i)
        for same_class in by_class.values():
            if len(same_class) > 1:
                for other in same_class[1:]:
                    union(same_class[0], other)

    roots = [find(i) for i in range(count)]
    remap = {}
    return [remap.setdefault(root, len(remap)) for root in roots]


def stratified_indices(labels, groups=None, fractions=DEFAULT_FRACTIONS, seed=42):
    """Partition sample indices into ``(train, val, test)`` stratified by label.

    ``groups`` optionally assigns each sample to a group; all samples of a group are
    placed in the same split. Returns three lists of positions into the inputs.
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


def stratified_kfold(labels, groups=None, n_splits=4, seed=42):
    """Return ``n_splits`` ``(train_positions, val_positions)`` stratified by label.

    Groups are kept intact: samples of one group always land in the same fold.
    """
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")
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
    folds = [[] for _ in range(n_splits)]
    for label in sorted(class_to_groups):
        class_groups = class_to_groups[label]
        rng.shuffle(class_groups)
        for index, group in enumerate(class_groups):
            folds[index % n_splits].extend(group_to_samples[group])

    all_positions = set(range(len(labels)))
    result = []
    for fold in folds:
        val = sorted(fold)
        train = sorted(all_positions - set(val))
        result.append((train, val))
    return result


def build_split_indices(
    index,
    labels,
    fractions=DEFAULT_FRACTIONS,
    seed=42,
    group_by_content=True,
    dedup=True,
    max_shared_files=4,
):
    """Return ``(train, val, test, removed)`` original dataset indices.

    Duplicate series (same ``(class, id)`` or identical images) are collapsed to one
    representative and grouped so a group never spans folds.
    """
    if group_by_content:
        groups = content_groups(index, max_shared_files=max_shared_files)
    else:
        groups = []
        seen = {}
        for series_id, class_name, _ in index:
            key = (class_name, series_id)
            groups.append(seen.setdefault(key, len(seen)))

    keep = list(range(len(index)))
    if dedup:
        keep = []
        kept_groups = set()
        for i, group in enumerate(groups):
            if group not in kept_groups:
                kept_groups.add(group)
                keep.append(i)

    kept_labels = [labels[i] for i in keep]
    kept_groups = [groups[i] for i in keep]
    train_pos, val_pos, test_pos = stratified_indices(
        kept_labels, kept_groups, fractions, seed
    )
    return (
        [keep[p] for p in train_pos],
        [keep[p] for p in val_pos],
        [keep[p] for p in test_pos],
        len(index) - len(keep),
    )


def build_datasets(
    root=DEFAULT_ROOT,
    splits=ALL_SPLITS,
    fractions=DEFAULT_FRACTIONS,
    seed=42,
    train_transform=train_aug,
    eval_transform=val_aug,
    group_by_content=True,
    dedup=True,
    max_shared_files=4,
    verbose=False,
    **dataset_kwargs,
):
    """Build stratified ``(train, val, test)`` subsets over all ``splits``.

    The dataset is created once for indexing/labels, then train/val/test are subsets of
    three dataset instances sharing the same index but using ``train_transform`` and
    ``eval_transform`` respectively. Duplicate series are removed and grouped by content
    before splitting. Extra keyword arguments go to :class:`AlignedMRIClassificationDataset`.
    """
    dataset_kwargs.pop("augment", None)
    base = AlignedMRIClassificationDataset(root=root, split=splits, **dataset_kwargs)
    if len(base.index) == 0:
        raise ValueError(
            f"no series found for root={root!r} and splits={tuple(splits)!r}. "
            "Expected <root>/<split>/<class>/<series_id>_<MODALITY>.jpg; check the path "
            "and that it contains the split folders (e.g. 'Train'/'Test')."
        )
    labels = [base.class_to_idx[class_name] for _, class_name, _ in base.index]

    train_idx, val_idx, test_idx, removed = build_split_indices(
        base.index,
        labels,
        fractions,
        seed,
        group_by_content=group_by_content,
        dedup=dedup,
        max_shared_files=max_shared_files,
    )
    if not train_idx or not val_idx:
        raise ValueError(
            f"empty split after dedup/stratify (train={len(train_idx)}, "
            f"val={len(val_idx)}, test={len(test_idx)}); need more series per class."
        )
    if verbose:
        print(
            f"[data] root={root!r} splits={tuple(splits)!r} "
            f"classes={base.classes}",
            flush=True,
        )
        print(
            f"[data] dedup: {removed} removed "
            f"({len(base.index)} -> {len(base.index) - removed})",
            flush=True,
        )
        print(
            f"[data] split sizes: train={len(train_idx)} val={len(val_idx)} "
            f"test={len(test_idx)}",
            flush=True,
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
