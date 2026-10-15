"""Combine hyperparameter search results into a markdown/CSV report."""

import argparse
import csv
import json
import os

FIELDS = [
    "optimizer",
    "lr",
    "weight_decay",
    "freeze_first_n",
    "train_loss",
    "val_loss",
    "val_acc",
    "val_f1",
    "test_loss",
    "test_acc",
    "test_f1",
    "epochs_run",
]

PARAM_FIELDS = ("optimizer", "lr", "weight_decay", "freeze_first_n")


def load_results(paths):
    """Load and concatenate one or more JSON result lists."""
    records = []
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            records.extend(json.load(handle))
    return records


def sort_records(records, key="val_loss"):
    return sorted(records, key=lambda record: record[key])


def _format(record, field):
    value = record.get(field)
    if field == "freeze_first_n":
        return "all" if value is None else str(value)
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def to_markdown(records):
    header = ["rank"] + FIELDS
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * len(header)) + "|",
    ]
    for rank, record in enumerate(records, 1):
        row = [str(rank)] + [_format(record, field) for field in FIELDS]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def write_report(paths, out_md, out_csv=None, key="val_loss"):
    """Sort all results by ``key`` and write markdown (and optional CSV)."""
    records = sort_records(load_results(paths), key)

    os.makedirs(os.path.dirname(os.path.abspath(out_md)), exist_ok=True)
    with open(out_md, "w", encoding="utf-8") as handle:
        handle.write(to_markdown(records) + "\n")

    if out_csv:
        os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
        with open(out_csv, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            for record in records:
                writer.writerow({field: record.get(field) for field in FIELDS})

    return records


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Build a hyperparameter report.")
    parser.add_argument("--inputs", nargs="+", required=True, help="result JSON files")
    parser.add_argument("--out-md", default=os.path.join("checkpoints", "search", "report.md"))
    parser.add_argument("--out-csv", default=os.path.join("checkpoints", "search", "report.csv"))
    parser.add_argument("--sort-by", default="val_loss")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    records = write_report(args.inputs, args.out_md, args.out_csv, args.sort_by)
    print(to_markdown(records))
    print(f"\n{len(records)} configs -> {args.out_md}, {args.out_csv}")
    return records


if __name__ == "__main__":
    main()
