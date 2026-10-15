import argparse
import os
import re
import sys

import cv2

from brain_hackathon.data.load import get_modality_dict
from brain_hackathon.data.preprocessing import apply_augmentation, process_images

MOD_RE = re.compile(r"_([A-Za-z0-9]{2})_\.jpg$", re.IGNORECASE)

DEFAULT_ROOT = os.path.join("data", "Epic and CSCR hospital Dataset")
DEFAULT_OUT = os.path.join("data", "aligned")
DEFAULT_SPLITS = ("Train", "Test")


def discover_classes(src_root, split):
    split_dir = os.path.join(src_root, split)
    if not os.path.isdir(split_dir):
        return []
    return sorted(
        name
        for name in os.listdir(split_dir)
        if os.path.isdir(os.path.join(split_dir, name))
    )


def modality_code(fname):
    match = MOD_RE.search(fname)
    return match.group(1).upper() if match else None


def build_aligned_dataset(src_root, out_root, splits, classes=None):
    stats = {"series": 0, "written": 0, "skipped": 0}

    for split in splits:
        split_classes = classes if classes else discover_classes(src_root, split)

        for class_name in split_classes:
            class_dir = os.path.join(src_root, split, class_name)
            if not os.path.isdir(class_dir):
                continue

            group_dict = get_modality_dict(src_root, split, class_name)
            out_dir = os.path.join(out_root, split, class_name)
            os.makedirs(out_dir, exist_ok=True)

            written = skipped = 0

            for series_id, filenames in sorted(group_dict.items()):
                stats["series"] += 1
                filenames = sorted(filenames)

                grays = []
                readable = True
                for fname in filenames:
                    gray = cv2.imread(
                        os.path.join(class_dir, fname), cv2.IMREAD_GRAYSCALE
                    )
                    if gray is None:
                        print(
                            f"WARN unreadable image, skipping series "
                            f"{split}/{class_name}/{series_id}: {fname}",
                            file=sys.stderr,
                        )
                        readable = False
                        break
                    grays.append(gray)

                if not readable:
                    skipped += len(filenames)
                    stats["skipped"] += len(filenames)
                    continue

                transforms = process_images(grays)

                seen_modalities = set()
                for fname, gray, transform in zip(filenames, grays, transforms):
                    if transform.get("angle_deg") is None:
                        print(
                            f"WARN no alignment, skipping "
                            f"{split}/{class_name}/{fname}",
                            file=sys.stderr,
                        )
                        skipped += 1
                        stats["skipped"] += 1
                        continue

                    code = modality_code(fname)
                    if code is None:
                        print(
                            f"WARN cannot parse modality, skipping "
                            f"{split}/{class_name}/{fname}",
                            file=sys.stderr,
                        )
                        skipped += 1
                        stats["skipped"] += 1
                        continue

                    if code in seen_modalities:
                        print(
                            f"WARN duplicate modality {code}, overwriting "
                            f"{split}/{class_name}/{series_id}",
                            file=sys.stderr,
                        )
                    seen_modalities.add(code)

                    aligned = apply_augmentation(gray, transform)
                    out_path = os.path.join(out_dir, f"{series_id}_{code}.jpg")
                    if not cv2.imwrite(out_path, aligned):
                        print(f"WARN write failed: {out_path}", file=sys.stderr)
                        skipped += 1
                        stats["skipped"] += 1
                        continue

                    written += 1
                    stats["written"] += 1

            print(
                f"{split}/{class_name}: series={len(group_dict)} "
                f"written={written} skipped={skipped}"
            )

    return stats


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Build a geometrically aligned copy of the dataset."
    )
    parser.add_argument(
        "--root", default=DEFAULT_ROOT, help="source dataset root"
    )
    parser.add_argument(
        "--out", default=DEFAULT_OUT, help="output root for the aligned dataset"
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(DEFAULT_SPLITS),
        help="splits to process (default: Train Test)",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=None,
        help="classes to process (default: auto-discover per split)",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    stats = build_aligned_dataset(args.root, args.out, args.splits, args.classes)
    print(
        f"Total: series={stats['series']} "
        f"written={stats['written']} skipped={stats['skipped']}"
    )
    return stats


if __name__ == "__main__":
    main()
