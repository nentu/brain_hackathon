"""Download the best (or last) checkpoint of a W&B run from the configured server.

The server is taken from ``WANDB_BASE_URL`` / ``WANDB_API_KEY`` (same as training).

    uv run fetch_model --run-id latxhxbe --output checkpoints
"""

import argparse
import glob
import os
import shutil


def local_checkpoint(run_id, output="checkpoints", alias="best", wandb_dir="wandb"):
    """Copy ``<wandb_dir>/run-*<run_id>/files/<alias>.pt`` to ``output`` if present."""
    pattern = os.path.join(wandb_dir, f"run-*-{run_id}", "files", f"{alias}.pt")
    matches = glob.glob(pattern)
    if not matches:
        return None
    os.makedirs(output, exist_ok=True)
    source = os.path.realpath(matches[0])
    destination = os.path.join(output, f"{alias}.pt")
    shutil.copyfile(source, destination)
    print(f"copied {source} -> {destination}")
    return destination


def fetch_checkpoint(
    run_path,
    output="checkpoints",
    alias="best",
    entity=None,
    project=None,
    wandb_dir="wandb",
):
    """Return the local path of the requested checkpoint for ``entity/project/run``.

    Looks in the local W&B run directory first (works with openwandb/offline runs),
    then falls back to the W&B public API.
    """
    run_id = run_path.rstrip("/").split("/")[-1]

    local = local_checkpoint(run_id, output, alias, wandb_dir)
    if local is not None:
        return local

    import wandb  # lazy: importing this module should not require wandb

    overrides = {}
    if entity:
        overrides["entity"] = entity
    if project:
        overrides["project"] = project
    api = wandb.Api(overrides=overrides or None)
    run = api.run(run_path)

    for artifact in run.logged_artifacts():
        if artifact.type != "model":
            continue
        if alias in artifact.aliases or alias in artifact.name.lower():
            path = artifact.download(root=output)
            print(f"downloaded artifact {artifact.name} ({artifact.version}) -> {path}")
            return path

    # Fallback: files uploaded with wandb.save (e.g. log_artifacts=False runs).
    for file in run.files():
        name = os.path.basename(file.name)
        if name.startswith(alias) and name.endswith(".pt"):
            file.download(root=output, replace=True)
            path = os.path.join(output, file.name)
            print(f"downloaded run file {file.name} -> {path}")
            return path

    summary_path = run.summary.get(f"{alias}_model_path")
    raise SystemExit(
        f"no '{alias}' checkpoint artifact/file for {run_path}. "
        f"summary {alias}_model_path={summary_path!r}"
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Download a W&B run's best/last checkpoint."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--entity", default="default")
    parser.add_argument("--project", default="brain-hackathon")
    parser.add_argument("--alias", default="best", choices=["best", "last"])
    parser.add_argument("--output", default="checkpoints")
    parser.add_argument(
        "--wandb-dir", default="wandb", help="local wandb directory for offline runs"
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    run_path = f"{args.entity}/{args.project}/{args.run_id}"
    return fetch_checkpoint(
        run_path,
        args.output,
        args.alias,
        entity=args.entity,
        project=args.project,
        wandb_dir=args.wandb_dir,
    )


if __name__ == "__main__":
    main()
