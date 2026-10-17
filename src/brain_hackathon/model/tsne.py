"""t-SNE of RadImageNet backbone features, colored by class.

Uses the pretrained backbone only: the classifier head added for fine-tuning is
bypassed (``forward_features``), so no newly-initialised/trained head weights are used.

Run directly, e.g.:
    uv run python src/brain_hackathon/model/tsne.py --root data/aligned --dash
"""

import argparse
import base64
import os

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import torch
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Subset

from brain_hackathon.data.dataset import (
    DEFAULT_ROOT,
    MODALITIES,
    AlignedMRIClassificationDataset,
)
from brain_hackathon.model.resnet_rad import RadImageNetResNet50


def select_indices(dataset, limit=None, seed=42):
    indices = list(range(len(dataset)))
    if limit is not None and 0 < limit < len(indices):
        rng = np.random.default_rng(seed)
        indices = sorted(rng.choice(indices, size=limit, replace=False).tolist())
    return indices


def extract_features(model, dataset, indices, batch_size=32, device=None):
    device = (
        torch.device(device)
        if device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = model.to(device).eval()
    loader = DataLoader(
        Subset(dataset, indices), batch_size=batch_size, shuffle=False, num_workers=0
    )

    features, labels = [], []
    with torch.no_grad():
        for images, targets in loader:
            features.append(model.forward_features(images.to(device)).cpu().numpy())
            labels.append(targets.numpy())
    return np.concatenate(features), np.concatenate(labels)


def series_files(dataset, index):
    """File names of every modality in a series."""
    modalities = dataset.index[index][2]
    return sorted(os.path.basename(path) for path in modalities.values())


def build_figure(dataset, indices, embedding, labels):
    files = [series_files(dataset, i) for i in indices]
    frame = pd.DataFrame(
        {
            "x": embedding[:, 0],
            "y": embedding[:, 1],
            "label": [dataset.classes[int(label)] for label in labels],
            "series_id": [dataset.index[i][0] for i in indices],
            "file": [names[0] if names else "" for names in files],
            "files": ["<br>".join(names) for names in files],
            "index": list(indices),
        }
    )
    return px.scatter(
        frame,
        x="x",
        y="y",
        color="label",
        hover_data=["file", "files", "series_id"],
        custom_data=["index"],
        title=f"t-SNE of RadImageNet features ({len(frame)} series)",
        labels={"x": "t-SNE 1", "y": "t-SNE 2", "label": "class"},
    )


def montage_data_uri(dataset, index, tile=72, pad=2):
    """Base64 JPEG data URI of a 2x3 montage of the series' modalities."""
    paths = dataset.index[index][2]
    rows, cols = 2, 3
    canvas = np.zeros(
        (rows * tile + (rows + 1) * pad, cols * tile + (cols + 1) * pad, 3),
        dtype=np.uint8,
    )
    for i, modality in enumerate(MODALITIES):
        row, col = divmod(i, cols)
        y = pad + row * (tile + pad)
        x = pad + col * (tile + pad)
        path = paths.get(modality)
        if path:
            image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if image is not None:
                image = cv2.resize(image, (tile, tile))
                canvas[y : y + tile, x : x + tile] = cv2.cvtColor(
                    image, cv2.COLOR_GRAY2BGR
                )
        cv2.putText(
            canvas,
            modality,
            (x + 2, y + 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    ok, buffer = cv2.imencode(".jpg", canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(buffer.tobytes()).decode()


def run_dash(dataset, indices, embedding, labels, port=8050, jupyter_mode=None):
    """Interactive Dash viewer: hover a point to see the series' modality images."""
    from dash import Dash, Input, Output, dcc, html, no_update

    figure = build_figure(dataset, indices, embedding, labels)
    lookup = {int(index): int(label) for index, label in zip(indices, labels)}

    app = Dash(__name__)
    app.layout = html.Div(
        [
            dcc.Graph(id="tsne-graph", figure=figure, clear_on_unhover=True),
            dcc.Tooltip(id="graph-tooltip"),
        ]
    )

    @app.callback(
        Output("graph-tooltip", "show"),
        Output("graph-tooltip", "bbox"),
        Output("graph-tooltip", "children"),
        Input("tsne-graph", "hoverData"),
    )
    def display_hover(hover_data):
        if not hover_data or not hover_data.get("points"):
            return False, no_update, no_update

        point = hover_data["points"][0]
        custom = point.get("customdata")
        index = custom[0] if isinstance(custom, (list, tuple)) else custom
        if index is None:
            return False, no_update, no_update
        index = int(index)

        label = dataset.classes[lookup[index]]
        files = series_files(dataset, index)
        children = [
            html.Div(
                [
                    html.B(label),
                    html.Br(),
                    html.Small(", ".join(files)),
                ]
            ),
            html.Img(
                src=montage_data_uri(dataset, index),
                style={"width": "240px", "border": "1px solid #ccc"},
            ),
        ]
        return True, point["bbox"], children

    run_kwargs = {"debug": False, "port": port}
    if jupyter_mode:
        run_kwargs["jupyter_mode"] = jupyter_mode
    app.run(**run_kwargs)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="t-SNE of RadImageNet features colored by class."
    )
    parser.add_argument("--root", default=DEFAULT_ROOT, help="aligned dataset root")
    parser.add_argument(
        "--splits", nargs="+", default=["Train", "Test"], help="splits to include"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="max number of series to embed"
    )
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weights", default=None, help="optional path to ResNet50.pt")
    parser.add_argument(
        "--output", default="./previews/tsne.html", help="output HTML file ('' to skip)"
    )
    parser.add_argument("--no-show", action="store_true", help="do not open the plot")
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--dash",
        action="store_true",
        help="interactive Dash viewer with image-on-hover tooltips",
    )
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument(
        "--jupyter", action="store_true", help="run the Dash app inline in a notebook"
    )
    return parser.parse_args(argv)

def compute_embedding(
    root=DEFAULT_ROOT,
    splits=("Train", "Test"),
    limit=None,
    perplexity=30.0,
    batch_size=32,
    seed=42,
    weights=None,
    device=None,
):
    """Embed the unaugmented dataset with t-SNE.

    Returns ``(dataset, indices, embedding, labels)``.
    """
    dataset = AlignedMRIClassificationDataset(
        root=root, split=tuple(splits), augment=None
    )
    if len(dataset) == 0:
        raise ValueError(
            f"no series found under root={root!r}; expected "
            "<root>/<split>/<class>/<id>_<MODALITY>.jpg"
        )
    if len(dataset) < 2:
        raise ValueError("need at least 2 series for t-SNE")

    model = RadImageNetResNet50(
        dim=[len(dataset.classes)], verbose=False, weights_path=weights
    )
    indices = select_indices(dataset, limit, seed)
    features, labels = extract_features(model, dataset, indices, batch_size, device)

    features = StandardScaler().fit_transform(features)
    effective_perplexity = min(perplexity, max(2.0, (len(features) - 1) / 3.0))
    embedding = TSNE(
        n_components=2,
        perplexity=effective_perplexity,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    ).fit_transform(features)
    return dataset, indices, embedding, labels


def draw_embeddding(
    root=DEFAULT_ROOT,
    splits=("Train", "Test"),
    limit=None,
    perplexity=30.0,
    batch_size=32,
    seed=42,
    weights=None,
    device=None,
):
    """Return a ready-to-display Plotly figure of the t-SNE embedding.

    In a notebook, ``draw_embeddding()`` as the last expression of a cell displays it.
    """
    dataset, indices, embedding, labels = compute_embedding(
        root=root,
        splits=splits,
        limit=limit,
        perplexity=perplexity,
        batch_size=batch_size,
        seed=seed,
        weights=weights,
        device=device,
    )
    return build_figure(dataset, indices, embedding, labels)


# Correctly-spelled alias.
draw_embedding = draw_embeddding

def main(argv=None):
    args = parse_args(argv)

    try:
        dataset, indices, embedding, labels = compute_embedding(
            root=args.root,
            splits=args.splits,
            limit=args.limit,
            perplexity=args.perplexity,
            batch_size=args.batch_size,
            seed=args.seed,
            weights=args.weights,
            device=args.device,
        )
    except ValueError as exc:
        raise SystemExit(str(exc))

    print(f"embedded {len(indices)} series")

    if args.dash:
        run_dash(
            dataset,
            indices,
            embedding,
            labels,
            port=args.port,
            jupyter_mode="inline" if args.jupyter else None,
        )
        return embedding

    figure = build_figure(dataset, indices, embedding, labels)
    if args.output:
        figure.write_html(args.output)
        print(f"saved {args.output}")
    if not args.no_show:
        figure.show()

    return embedding


if __name__ == "__main__":
    main()
