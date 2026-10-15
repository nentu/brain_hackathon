import argparse
import os
import random
import re

import cv2
import numpy as np

from brain_hackathon.data.augmentation import train_aug
from brain_hackathon.data.dataset import (
    DEFAULT_ROOT,
    MODALITIES,
    AlignedMRIClassificationDataset,
)

WINDOW = "preview"


def supports_gui():
    """True only when this OpenCV build has a GUI backend (not ``GUI: NONE``)."""
    return re.search(r"GUI:\s+NONE", cv2.getBuildInformation()) is None


def to_display(channel):
    channel = channel.astype(np.float32)
    low, high = float(channel.min()), float(channel.max())
    if high > low:
        channel = (channel - low) / (high - low)
    else:
        channel = np.zeros_like(channel)
    return (channel * 255.0).astype(np.uint8)


def build_montage(tensor, labels, cols=3, padding=6, label_scale=0.6):
    array = tensor.detach().cpu().numpy()
    channels, height, width = array.shape
    rows = (channels + cols - 1) // cols

    montage = np.zeros(
        (rows * height + (rows + 1) * padding, cols * width + (cols + 1) * padding),
        dtype=np.uint8,
    )

    for i in range(channels):
        row, col = divmod(i, cols)
        y = padding + row * (height + padding)
        x = padding + col * (width + padding)
        montage[y : y + height, x : x + width] = to_display(array[i])

        label = labels[i] if labels else str(i)
        cv2.putText(
            montage,
            label,
            (x + 4, y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            label_scale,
            255,
            1,
            cv2.LINE_AA,
        )

    return montage


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Preview the six aligned modalities of a series with OpenCV."
    )
    parser.add_argument("--root", default=DEFAULT_ROOT, help="aligned dataset root")
    parser.add_argument("--split", default="Test", help="split directory")
    parser.add_argument(
        "--classes", nargs="+", default=None, help="classes to use (default: all)"
    )
    parser.add_argument(
        "--index",
        type=int,
        default=None,
        help="series index to start from (default: random)",
    )
    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="preview aligned images without the training augmentation",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if not supports_gui():
        print(
            "OpenCV GUI is unavailable in this build (GUI: NONE). "
            "Reinstall a GUI build, e.g. `uv pip install --reinstall opencv-python`."
        )
        return 1

    augment = None if args.no_augment else train_aug
    dataset = AlignedMRIClassificationDataset(
        root=args.root, split=args.split, classes=args.classes, augment=augment
    )
    if len(dataset) == 0:
        print(f"No series found under {os.path.join(args.root, args.split)}")
        return 1

    if args.index is not None:
        if not 0 <= args.index < len(dataset):
            print(f"index {args.index} out of range (dataset has {len(dataset)} series)")
            return 1
        start = args.index
    else:
        start = random.randrange(len(dataset))

    order = list(range(start, len(dataset))) + list(range(start))

    for index in order:
        tensor, label = dataset[index]
        series_id, class_name, _ = dataset.index[index]
        print(
            f"{args.split}/{class_name}/{series_id} "
            f"shape={tuple(tensor.shape)} label={dataset.classes[label]}"
        )

        cv2.imshow(WINDOW, build_montage(tensor, list(MODALITIES)))
        if cv2.waitKey(0) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
