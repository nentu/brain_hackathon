import csv
import json

from brain_hackathon.train.report import (
    load_results,
    sort_records,
    to_markdown,
    write_report,
)

RECORDS = [
    {
        "optimizer": "sgd",
        "lr": 0.01,
        "weight_decay": 0.0,
        "freeze_first_n": 2,
        "train_loss": 1.5,
        "val_loss": 1.2,
        "val_acc": 0.5,
        "val_f1": 0.4,
        "test_loss": 1.3,
        "test_acc": 0.5,
        "test_f1": 0.4,
        "epochs_run": 2,
    },
    {
        "optimizer": "adamw",
        "lr": 0.001,
        "weight_decay": 0.0,
        "freeze_first_n": None,
        "train_loss": 0.9,
        "val_loss": 0.5,
        "val_acc": 0.8,
        "val_f1": 0.79,
        "test_loss": 0.54,
        "test_acc": 0.8,
        "test_f1": 0.77,
        "epochs_run": 2,
    },
]


def test_sort_records():
    ordered = sort_records(RECORDS, "val_loss")
    assert ordered[0]["optimizer"] == "adamw"


def test_to_markdown_contents():
    markdown = to_markdown(sort_records(RECORDS, "val_loss"))
    lines = markdown.splitlines()
    assert lines[0].startswith("| rank | optimizer")
    assert "all" in lines[2]  # freeze None rendered as "all"
    assert "adamw" in lines[2]


def test_write_report(tmp_path):
    src = tmp_path / "a.json"
    src.write_text(json.dumps(RECORDS))
    out_md = tmp_path / "report.md"
    out_csv = tmp_path / "report.csv"

    records = write_report([str(src)], str(out_md), str(out_csv))

    assert len(records) == 2
    assert out_md.is_file() and out_csv.is_file()
    with open(out_csv, newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert rows[0]["optimizer"] == "adamw"
    assert len(load_results([str(src)])) == 2
