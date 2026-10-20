"""Draw Grad-CAM++ and SHAP explanations for test-split items.

uv run explain_data --dataset both --limit 3 --out outputs/attributions
uv run explain_data --dataset raw --classes glioma --limit 5 --methods gradcam
uv run explain_data --dataset aligned --limit 2 --base-modality BR
"""

import argparse
import csv
import os
import re

import cv2
import numpy as np

from brain_hackathon.data.dataset import (
    DEFAULT_ROOT,
    MODALITIES,
    AlignedMRIClassificationDataset,
)
from brain_hackathon.data.raw_dataset import RAW_ROOT, RawImageDataset
from brain_hackathon.eval.aligned_model import (
    DEFAULT_ALIGNED_CHECKPOINT,
    AlignedBestModel,
)
from brain_hackathon.eval.base import DEFAULT_CLASSES
from brain_hackathon.eval.gradcam import _display_base
from brain_hackathon.eval.raw_model import DEFAULT_RAW_CHECKPOINT, RawBestModel


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Draw Grad-CAM++ and Kernel SHAP for test-split items."
    )
    parser.add_argument(
        "--dataset",
        choices=("raw", "aligned", "both"),
        default="both",
        help="dataset kind to explain",
    )
    parser.add_argument("--raw-root", default=RAW_ROOT, help="raw dataset root")
    parser.add_argument(
        "--aligned-root", default=DEFAULT_ROOT, help="aligned dataset root"
    )
    parser.add_argument("--split", default="Test", help="dataset split")
    parser.add_argument("--classes", nargs="+", default=None, help="classes to include")
    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="number of items per dataset (default: 3)",
    )
    parser.add_argument("--seed", type=int, default=42, help="sampling seed")
    parser.add_argument(
        "--out",
        default=os.path.join("outputs", "attributions"),
        help="output directory",
    )
    parser.add_argument(
        "--methods",
        choices=("both", "gradcam", "shap"),
        default="both",
        help="attributions to draw",
    )
    parser.add_argument(
        "--target-class",
        default=None,
        help="class name to explain (default: predicted class)",
    )
    parser.add_argument(
        "--base-modality",
        default=None,
        help="aligned modality for the overlay background (default: mean)",
    )
    parser.add_argument("--raw-checkpoint", default=DEFAULT_RAW_CHECKPOINT)
    parser.add_argument("--aligned-checkpoint", default=DEFAULT_ALIGNED_CHECKPOINT)
    parser.add_argument("--device", default=None)
    parser.add_argument("--gradcam-alpha", type=float, default=0.45)
    parser.add_argument("--shap-alpha", type=float, default=0.45)
    parser.add_argument(
        "--mask",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="mask heatmaps to the largest foreground region (default: enabled)",
    )
    parser.add_argument(
        "--mask-radius",
        type=float,
        default=10.0,
        help="intensity tolerance for the foreground mask (default: 10)",
    )
    parser.add_argument("--shap-patch-grid", type=int, default=7)
    parser.add_argument("--shap-samples", type=int, default=128)
    parser.add_argument("--shap-batch-size", type=int, default=64)
    parser.add_argument("--shap-baseline", type=float, default=0.0)
    parser.add_argument("--shap-seed", type=int, default=0)
    return parser.parse_args(argv)


def _safe_name(value):
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    return name or "item"


def _write_image(path, image):
    if not cv2.imwrite(path, image):
        raise OSError(f"failed to write image to {path!r}")


def _select_indices(length, limit, seed):
    if length <= 0 or limit <= 0:
        return []
    count = min(int(limit), length)
    rng = np.random.default_rng(seed)
    return sorted(int(index) for index in rng.choice(length, count, replace=False))


def _build_montage(images, labels, title, padding=8, header=40):
    prepared = [
        image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        for image in images
    ]
    height = max(image.shape[0] for image in prepared)
    width = max(image.shape[1] for image in prepared)
    prepared = [cv2.resize(image, (width, height)) for image in prepared]

    canvas = np.full(
        (header + height + 2 * padding, len(prepared) * (width + padding) + padding, 3),
        32,
        dtype=np.uint8,
    )
    cv2.putText(
        canvas,
        title,
        (padding, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    for position, (image, label) in enumerate(zip(prepared, labels)):
        x = padding + position * (width + padding)
        y = header + padding
        canvas[y : y + height, x : x + width] = image
        cv2.putText(
            canvas,
            label,
            (x + 4, y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return canvas


def _load_raw_items(dataset, indices):
    items = []
    for index in indices:
        path, label = dataset.index[index]
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"[skip] cannot read {path}", flush=True)
            continue
        item_id = os.path.splitext(os.path.basename(path))[0]
        class_name = dataset.classes[label]
        items.append(
            {
                "kind": "raw",
                "name": _safe_name(f"raw_{class_name}_{item_id}"),
                "true_class": class_name,
                "image": image,
            }
        )
    return items


def _load_aligned_items(dataset, indices):
    items = []
    for index in indices:
        series_id, class_name, paths = dataset.index[index]
        modalities = {}
        for modality, path in paths.items():
            image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if image is not None:
                modalities[modality] = image
        if not modalities:
            print(
                f"[skip] no readable modalities for {series_id}/{class_name}",
                flush=True,
            )
            continue
        items.append(
            {
                "kind": "aligned",
                "name": _safe_name(f"aligned_{class_name}_{series_id}"),
                "true_class": class_name,
                "image": modalities,
            }
        )
    return items


def _load_items(args):
    items = []

    if args.dataset in ("raw", "both"):
        dataset = RawImageDataset(
            root=args.raw_root, split=args.split, classes=args.classes
        )
        indices = _select_indices(len(dataset), args.limit, args.seed)
        print(
            f"[raw] split={args.split} items={len(dataset)} selected={len(indices)}",
            flush=True,
        )
        items.extend(_load_raw_items(dataset, indices))

    if args.dataset in ("aligned", "both"):
        dataset = AlignedMRIClassificationDataset(
            root=args.aligned_root,
            split=args.split,
            classes=args.classes,
            augment=None,
        )
        indices = _select_indices(len(dataset), args.limit, args.seed)
        print(
            f"[aligned] split={args.split} items={len(dataset)} "
            f"selected={len(indices)}",
            flush=True,
        )
        items.extend(_load_aligned_items(dataset, indices))

    return items


def _load_model(kind, args):
    if kind == "raw":
        return RawBestModel(checkpoint=args.raw_checkpoint, device=args.device)
    return AlignedBestModel(checkpoint=args.aligned_checkpoint, device=args.device)


def _run_attributions(model, item, args):
    kwargs = {}
    if args.target_class is not None:
        kwargs["target_class"] = args.target_class
    if item["kind"] == "aligned" and args.base_modality is not None:
        kwargs["base_modality"] = args.base_modality.upper()

    results = {}
    mask_kwargs = {"mask": args.mask, "mask_radius": args.mask_radius}
    if args.methods in ("both", "gradcam"):
        results["gradcam"] = model.grad_cam(
            item["image"], alpha=args.gradcam_alpha, **mask_kwargs, **kwargs
        )
    if args.methods in ("both", "shap"):
        results["shap"] = model.shap(
            item["image"],
            patch_grid=args.shap_patch_grid,
            n_samples=args.shap_samples,
            batch_size=args.shap_batch_size,
            baseline=args.shap_baseline,
            seed=args.shap_seed,
            alpha=args.shap_alpha,
            **mask_kwargs,
            **kwargs,
        )
    return results


def _original_image(item, args):
    channels = 1 if item["kind"] == "raw" else len(MODALITIES)
    base_modality = (
        args.base_modality.upper()
        if item["kind"] == "aligned" and args.base_modality is not None
        else None
    )
    return _display_base(item["image"], channels, None, base_modality)


def _save_item(item, results, args):
    original = _original_image(item, args)
    saved = {"original": f"{item['name']}_original.png"}
    _write_image(os.path.join(args.out, saved["original"]), original)

    images = [original]
    labels = ["original"]
    for method in ("gradcam", "shap"):
        if method not in results:
            continue
        result = results[method]
        filename = f"{item['name']}_{method}.png"
        _write_image(os.path.join(args.out, filename), result["overlay"])
        saved[method] = filename
        images.append(result["overlay"])
        labels.append(f"{method} {result['class_name']} {result['probability']:.2f}")

    title = f"{item['kind']} | {item['name']} | true={item['true_class']}"
    montage = _build_montage(images, labels, title)
    saved["montage"] = f"{item['name']}_montage.png"
    _write_image(os.path.join(args.out, saved["montage"]), montage)
    return saved


def _write_summary(rows, out_dir):
    path = os.path.join(out_dir, "summary.csv")
    fieldnames = [
        "dataset",
        "item",
        "true_class",
        "predicted_class",
        "probability",
        "gradcam",
        "shap",
        "montage",
    ]
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def main(argv=None):
    args = parse_args(argv)
    if args.limit < 0:
        raise SystemExit("--limit must be non-negative")
    if args.base_modality is not None:
        if args.dataset == "raw":
            raise SystemExit("--base-modality is only supported for aligned items")
        if args.base_modality.upper() not in MODALITIES:
            raise SystemExit(
                f"unknown modality {args.base_modality!r}; expected one of {MODALITIES}"
            )
    if args.target_class is not None and args.target_class not in DEFAULT_CLASSES:
        raise SystemExit(
            f"unknown class {args.target_class!r}; expected one of {DEFAULT_CLASSES}"
        )
    if args.mask and args.mask_radius < 0.0:
        raise SystemExit("--mask-radius must be non-negative")
    if args.methods in ("both", "shap"):
        if args.shap_patch_grid < 1:
            raise SystemExit("--shap-patch-grid must be positive")
        if args.shap_batch_size < 1:
            raise SystemExit("--shap-batch-size must be positive")
        minimum = args.shap_patch_grid * args.shap_patch_grid + 1
        if args.shap_samples < minimum:
            raise SystemExit(
                f"--shap-samples must be at least {minimum} for "
                f"a {args.shap_patch_grid}x{args.shap_patch_grid} patch grid"
            )

    items = _load_items(args)
    if not items:
        print("No items found.")
        return 1

    models = {}
    for kind in sorted({item["kind"] for item in items}):
        print(f"[model] loading {kind} checkpoint", flush=True)
        models[kind] = _load_model(kind, args)

    os.makedirs(args.out, exist_ok=True)
    rows = []
    for item in items:
        print(
            f"[explain] {item['kind']} {item['name']} true={item['true_class']}",
            flush=True,
        )
        try:
            results = _run_attributions(models[item["kind"]], item, args)
            saved = _save_item(item, results, args)
        except Exception as exc:  # noqa: BLE001
            print(f"[error] {item['name']}: {exc}", flush=True)
            continue

        reference = results.get("gradcam") or results.get("shap")
        rows.append(
            {
                "dataset": item["kind"],
                "item": item["name"],
                "true_class": item["true_class"],
                "predicted_class": reference["class_name"],
                "probability": f"{reference['probability']:.6f}",
                "gradcam": saved.get("gradcam", ""),
                "shap": saved.get("shap", ""),
                "montage": saved["montage"],
            }
        )

    if not rows:
        print("No explanations were produced.")
        return 1

    summary = _write_summary(rows, args.out)
    print(f"[done] wrote {len(rows)} items to {args.out}", flush=True)
    print(f"[done] summary: {summary}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
