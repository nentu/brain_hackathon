import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

MODALITIES = ("BR", "DA", "HF", "RO", "SP", "VF")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
DEFAULT_ROOT = os.path.join("data", "aligned")


def as_splits(split):
    """Normalize a split argument (one name or an iterable of names) to a tuple."""
    if isinstance(split, str):
        return (split,)
    return tuple(split)


def discover_classes(root, splits):
    """Sorted union of class directories across one or more splits."""
    classes = set()
    for split in as_splits(splits):
        split_dir = os.path.join(root, split)
        if not os.path.isdir(split_dir):
            continue
        classes.update(
            name
            for name in os.listdir(split_dir)
            if os.path.isdir(os.path.join(split_dir, name))
        )
    return sorted(classes)


def _series_sort_key(series_id):
    return (0, int(series_id)) if series_id.isdigit() else (1, series_id)


def build_series_index(root, splits, classes=None):
    """Index the aligned dataset across one or more splits.

    ``splits`` is a split name or an iterable of split names; entries from every split
    are concatenated (split-major order). Returns ``(classes, index)`` where ``classes``
    is the sorted class list and ``index`` is a list of
    ``(series_id, class_name, {modality: path})``. Modalities that are absent for a
    series are simply missing from the dict.
    """
    splits = as_splits(splits)
    if classes is None:
        classes = discover_classes(root, splits)
    classes = list(classes)

    index = []
    for split in splits:
        for class_name in classes:
            class_dir = os.path.join(root, split, class_name)
            if not os.path.isdir(class_dir):
                continue

            series = {}
            for fname in sorted(os.listdir(class_dir)):
                stem, ext = os.path.splitext(fname)
                if ext.lower() not in IMAGE_EXTENSIONS or "_" not in stem:
                    continue
                series_id, modality = stem.rsplit("_", 1)
                modality = modality.upper()
                if modality not in MODALITIES:
                    continue
                series.setdefault(series_id, {})[modality] = os.path.join(
                    class_dir, fname
                )

            for series_id in sorted(series, key=_series_sort_key):
                index.append((series_id, class_name, series[series_id]))

    return classes, index


def foreground_stats(image):
    """Mean and std over non-black (non-zero) pixels of ``image``."""
    values = np.asarray(image)[np.asarray(image) > 0].astype(np.float64)
    if values.size == 0:
        return 0.0, 0.0
    return float(values.mean()), float(values.std())


def three_sigma_normalize(image, mean, std):
    """Map ``[mean - 3*std, mean + 3*std]`` to ``[0, 1]`` and clip."""
    low = mean - 3.0 * std
    span = 6.0 * std
    image = np.asarray(image, dtype=np.float32)
    if span <= 0.0:
        return np.zeros_like(image)
    return np.clip((image - low) / span, 0.0, 1.0)


def compute_foreground_stats(images):
    """Global foreground mean/std over an iterable of arrays (e.g. a whole split)."""
    count = 0
    total = 0.0
    total_sq = 0.0
    for image in images:
        array = np.asarray(image)
        values = array[array > 0].astype(np.float64)
        count += values.size
        total += values.sum()
        total_sq += float((values * values).sum())
    if count == 0:
        return 0.0, 0.0
    mean = total / count
    variance = max(total_sq / count - mean * mean, 0.0)
    return float(mean), float(variance**0.5)


class AlignedMRIDataset(Dataset):
    """Multimodal MRI dataset over the aligned dataset layout.

    Layout: ``<root>/<split>/<class>/<series_id>_<MODALITY>.<ext>``.

    ``split`` may be a single split name or an iterable of splits (e.g.
    ``("Train", "Test")``); series from every split are combined.

    Every item is a float tensor of shape ``(len(MODALITIES), H, W)`` with channels
    ordered as ``MODALITIES`` (BR, DA, HF, RO, SP, VF). Modalities missing for a
    series are filled with ``fill_value``.

    ``augment`` is an Albumentations ``Compose`` (see ``augmentation.py``). It is
    applied to the stacked ``(H, W, 6)`` image so every geometric transform is shared
    by all channels. Normalization (``"three_sigma"``) uses statistics over non-black
    pixels only; pass ``norm_stats=(mean, std)`` for fixed (e.g. dataset-wide) stats.
    """

    channel_names = MODALITIES

    def __init__(
        self,
        root=DEFAULT_ROOT,
        split="Train",
        classes=None,
        augment=None,
        normalization="three_sigma",
        norm_stats=None,
        fill_value=0,
        transform=None,
    ):
        self.root = root
        self.split = split
        self.splits = as_splits(split)
        self.augment = augment
        self.normalization = normalization
        self.norm_stats = norm_stats
        self.fill_value = int(fill_value)
        self.transform = transform
        self.classes, self.index = build_series_index(root, self.splits, classes)
        self.class_to_idx = {name: i for i, name in enumerate(self.classes)}

    def __len__(self):
        return len(self.index)

    def series_ids(self):
        return [series_id for series_id, _, _ in self.index]

    def _load_hwc(self, modality_paths):
        images = {}
        for modality in MODALITIES:
            path = modality_paths.get(modality)
            if path is None:
                continue
            image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if image is not None:
                images[modality] = image

        if images:
            height, width = next(iter(images.values())).shape[:2]
        else:
            height = width = 0

        stack = np.full(
            (height, width, len(MODALITIES)), self.fill_value, dtype=np.uint8
        )
        for i, modality in enumerate(MODALITIES):
            if modality in images:
                stack[:, :, i] = images[modality]
        return stack

    def load_series(self, index):
        _, _, modality_paths = self.index[index]
        stack = self._load_hwc(modality_paths)

        if self.augment is not None:
            stack = self.augment(image=stack)["image"]

        array = np.asarray(stack, dtype=np.float32)
        if self.normalization == "three_sigma":
            mean, std = self.norm_stats if self.norm_stats else foreground_stats(array)
            array = three_sigma_normalize(array, mean, std)
        elif self.normalization not in (None, "none"):
            raise ValueError(f"unknown normalization: {self.normalization!r}")

        tensor = torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1)))
        if self.transform is not None:
            tensor = self.transform(tensor)
        return tensor

    def __getitem__(self, index):
        raise NotImplementedError


class AlignedMRIClassificationDataset(AlignedMRIDataset):
    """Returns ``(tensor[C, H, W], label)`` where ``label`` indexes ``classes``."""

    def __getitem__(self, index):
        _, class_name, _ = self.index[index]
        return self.load_series(index), self.class_to_idx[class_name]


class AlignedMRIUnlabeledDataset(AlignedMRIDataset):
    """Returns only the ``(C, H, W)`` tensor, for inference."""

    def __getitem__(self, index):
        return self.load_series(index)
