import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

MODALITIES = ("BR", "DA", "HF", "RO", "SP", "VF")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
DEFAULT_ROOT = os.path.join("data", "aligned")


def discover_classes(root, split):
    split_dir = os.path.join(root, split)
    if not os.path.isdir(split_dir):
        return []
    return sorted(
        name
        for name in os.listdir(split_dir)
        if os.path.isdir(os.path.join(split_dir, name))
    )


def _series_sort_key(series_id):
    return (0, int(series_id)) if series_id.isdigit() else (1, series_id)


def build_series_index(root, split, classes=None):
    """Index the aligned dataset.

    Returns ``(classes, index)`` where ``classes`` is the sorted class list and
    ``index`` is a list of ``(series_id, class_name, {modality: path})``. Modalities
    that are absent for a series are simply missing from the dict.
    """
    if classes is None:
        classes = discover_classes(root, split)
    classes = list(classes)

    index = []
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
            series.setdefault(series_id, {})[modality] = os.path.join(class_dir, fname)

        for series_id in sorted(series, key=_series_sort_key):
            index.append((series_id, class_name, series[series_id]))

    return classes, index


class AlignedMRIDataset(Dataset):
    """Multimodal MRI dataset over the aligned dataset layout.

    Layout: ``<root>/<split>/<class>/<series_id>_<MODALITY>.<ext>``.

    Every item is a float tensor of shape ``(len(MODALITIES), H, W)`` with channels
    ordered as ``MODALITIES`` (BR, DA, HF, RO, SP, VF). Modalities missing for a
    series are filled with ``fill_value``.
    """

    channel_names = MODALITIES

    def __init__(
        self,
        root=DEFAULT_ROOT,
        split="Train",
        classes=None,
        normalize=True,
        fill_value=0.0,
        transform=None,
    ):
        self.root = root
        self.split = split
        self.normalize = normalize
        self.fill_value = float(fill_value)
        self.transform = transform
        self.classes, self.index = build_series_index(root, split, classes)
        self.class_to_idx = {name: i for i, name in enumerate(self.classes)}

    def __len__(self):
        return len(self.index)

    def series_ids(self):
        return [series_id for series_id, _, _ in self.index]

    def _load_stack(self, modality_paths):
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

        scale = 1.0 / 255.0 if self.normalize else 1.0
        stack = np.full(
            (len(MODALITIES), height, width), self.fill_value, dtype=np.float32
        )
        for i, modality in enumerate(MODALITIES):
            if modality in images:
                stack[i] = images[modality].astype(np.float32) * scale
        return stack

    def load_series(self, index):
        _, _, modality_paths = self.index[index]
        tensor = torch.from_numpy(self._load_stack(modality_paths))
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
