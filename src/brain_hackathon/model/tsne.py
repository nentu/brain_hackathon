"""t-SNE of RadImageNet backbone features, colored by class.

Uses the pretrained backbone only: the classifier head added for fine-tuning is
bypassed (``forward_features``), so no newly-initialised/trained head weights are used.

Run directly, e.g.:
    uv run python src/brain_hackathon/model/t-sne.py --root data/aligned
"""

import argparse

import numpy as np
import pandas as pd
import plotly.express as px
import torch
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Subset

from brain_hackathon.data.dataset import DEFAULT_ROOT, AlignedMRIClassificationDataset
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


def build_figure(dataset, indices, embedding, labels):
    frame = pd.DataFrame(
        {
            "x": embedding[:, 0],
            "y": embedding[:, 1],
            "label": [dataset.classes[int(label)] for label in labels],
            "series_id": [dataset.index[i][0] for i in indices],
        }
    )
    return px.scatter(
        frame,
        x="x",
        y="y",
        color="label",
        hover_data=["series_id"],
        title=f"t-SNE of RadImageNet features ({len(frame)} series)",
        labels={"x": "t-SNE 1", "y": "t-SNE 2", "label": "class"},
    )


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
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    dataset = AlignedMRIClassificationDataset(
        root=args.root, split=tuple(args.splits), augment=None
    )
    if len(dataset) == 0:
        raise SystemExit(
            f"no series found under root={args.root!r}; expected "
            "<root>/<split>/<class>/<id>_<MODALITY>.jpg"
        )
    if len(dataset) < 2:
        raise SystemExit("need at least 2 series for t-SNE")

    model = RadImageNetResNet50(
        dim=len(dataset.classes), verbose=False, weights_path=args.weights
    )

    indices = select_indices(dataset, args.limit, args.seed)
    features, labels = extract_features(
        model, dataset, indices, args.batch_size, args.device
    )
    print(f"features: {features.shape} for {len(indices)} series")

    features = StandardScaler().fit_transform(features)
    perplexity = min(args.perplexity, max(2.0, (len(features) - 1) / 3.0))
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=args.seed,
    )
    embedding = tsne.fit_transform(features)

    figure = build_figure(dataset, indices, embedding, labels)
    if args.output:
        figure.write_html(args.output)
        print(f"saved {args.output}")
    if not args.no_show:
        figure.show()

    return embedding


if __name__ == "__main__":
    main()
