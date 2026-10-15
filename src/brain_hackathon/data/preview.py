import argparse
import os
import random

import cv2
import numpy as np

from brain_hackathon.data.dataset import (
    DEFAULT_ROOT,
    MODALITIES,
    AlignedMRIClassificationDataset,
)


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
        help="dataset index to preview (default: random)",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    dataset = AlignedMRIClassificationDataset(
        root=args.root, split=args.split, classes=args.classes
    )
    if len(dataset) == 0:
        print(f"No series found under {os.path.join(args.root, args.split)}")
        return 1

    index = args.index if args.index is not None else random.randrange(len(dataset))
    if not 0 <= index < len(dataset):
        print(f"index {index} out of range (dataset has {len(dataset)} series)")
        return 1
    for index in range(len(dataset)):
        tensor, label = dataset[index]
        series_id, class_name, _ = dataset.index[index]
        title = (
            f"{args.split}/{class_name}/{series_id} "
            f"shape={tuple(tensor.shape)} label={dataset.classes[label]}"
        )
        print(title)

        montage = build_montage(tensor, list(MODALITIES))
        cv2.imshow("preview", montage)
        key = cv2.waitKey(0)
        if key == ord("q"):
            break
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
