"""Dataset over single images from the raw dataset.

Modality groups (e.g. ``G_86_VF_.jpg``, ``G_86_RO_.jpg``) are collapsed to their base
image ``G_86.jpg``, so every item is one unaugmented scan. Plain single images are kept
as-is.
"""

import os
import re

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from brain_hackathon.data.preprocessing import resize_image

RAW_ROOT = os.path.join("data", "Epic and CSCR hospital Dataset")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
MODALITY_STEM = re.compile(r"^(?P<stem>.+)_(?P<id>\d+)_(?P<modality>[A-Za-z]{2})_$")


def as_splits(splits):
    return (splits,) if isinstance(splits, str) else tuple(splits)


def base_image_path(path, search_dirs=()):
    """Return the base image for a group member, or ``None`` if it does not exist.

    ``G_86_VF_.jpg`` -> ``G_86.jpg`` (looked up in the same directory first, then in
    ``search_dirs``). Files that are not group members are returned unchanged.
    """
    directory, filename = os.path.split(path)
    stem, extension = os.path.splitext(filename)
    match = MODALITY_STEM.match(stem)
    if match is None:
        return path

    base_stem = f"{match.group('stem')}_{match.group('id')}"
    for candidate_dir in (directory, *search_dirs):
        for candidate_ext in (extension, ".jpg", ".jpeg", ".png"):
            candidate = os.path.join(candidate_dir, base_stem + candidate_ext)
            if os.path.isfile(candidate):
                return candidate
    return None


def discover_classes(root, splits):
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


def build_raw_index(root=RAW_ROOT, splits="Train", classes=None):
    """Index one base image per group plus plain single images.

    Returns ``(classes, index, missing)`` where ``index`` is
    ``[(image_path, class_index), ...]`` and ``missing`` lists group files whose base
    image does not exist anywhere in the dataset.
    """
    splits = as_splits(splits)
    if classes is None:
        classes = discover_classes(root, splits)
    classes = list(classes)
    class_to_idx = {name: i for i, name in enumerate(classes)}

    all_splits = [
        name
        for name in os.listdir(root)
        if os.path.isdir(os.path.join(root, name))
    ]
    class_dirs = {
        class_name: [
            os.path.join(root, split, class_name)
            for split in all_splits
            if os.path.isdir(os.path.join(root, split, class_name))
        ]
        for class_name in classes
    }

    index = []
    seen = set()
    missing = []
    for split in splits:
        for class_name in classes:
            directory = os.path.join(root, split, class_name)
            if not os.path.isdir(directory):
                continue
            search_dirs = [
                path for path in class_dirs[class_name] if path != directory
            ]

            for filename in sorted(os.listdir(directory)):
                stem, extension = os.path.splitext(filename)
                if extension.lower() not in IMAGE_EXTENSIONS:
                    continue
                path = os.path.join(directory, filename)
                if MODALITY_STEM.match(stem):
                    base = base_image_path(path, search_dirs)
                    if base is None:
                        missing.append(path)
                        continue
                    path = base
                absolute = os.path.abspath(path)
                if absolute in seen:
                    continue
                seen.add(absolute)
                index.append((path, class_to_idx[class_name]))

    return classes, index, missing


class RawImageDataset(Dataset):
    """One grayscale image per item, collapsing augmented groups to their base image.

    Returns ``(tensor[1, H, W] float32, label)``. Images are resized to
    ``image_size`` (set ``None`` to keep the original size) and optionally scaled to
    ``[0, 1]`` with ``normalize=True``.
    """

    def __init__(
        self,
        root=RAW_ROOT,
        split="Train",
        classes=None,
        image_size=224,
        normalize=True,
        transform=None,
        verbose=False,
    ):
        self.root = root
        self.splits = as_splits(split)
        self.classes, self.index, self.missing_bases = build_raw_index(
            root, split, classes
        )
        self.class_to_idx = {
            name: i for i, name in enumerate(self.classes)
        }
        self.image_size = image_size
        self.normalize = normalize
        self.transform = transform

        if verbose:
            print(
                f"[raw] root={root!r} splits={self.splits} classes={self.classes} "
                f"images={len(self.index)} skipped_groups={len(self.missing_bases)}",
                flush=True,
            )

    def __len__(self):
        return len(self.index)

    def paths(self):
        return [path for path, _ in self.index]

    def __getitem__(self, index):
        path, label = self.index[index]
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise RuntimeError(f"cannot read image: {path}")

        if self.image_size is not None:
            image = resize_image(image, (self.image_size, self.image_size))

        array = image.astype(np.float32)
        if self.normalize:
            array /= 255.0
        tensor = torch.from_numpy(array[None, :, :])

        if self.transform is not None:
            tensor = self.transform(tensor)
        return tensor, label
