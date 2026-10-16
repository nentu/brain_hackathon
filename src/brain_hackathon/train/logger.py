"""Weights & Biases logging with a local results dump."""

import json
import os
import subprocess


def git_hash():
    """Short current git commit hash, or None when unavailable."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None


class Logger:
    """Log losses/metrics to Weights & Biases and dump all results to disk.

    Tracks the last and best checkpoints. When ``enabled`` is False (or wandb is not
    installed) it only writes the local results file.
    """

    def __init__(
        self,
        project="brain-hackathon",
        run_name=None,
        config=None,
        mode=None,
        dir=None,
        checkpoint_dir="checkpoints",
        enabled=True,
        log_artifacts=True,
        base_url=None,
        api_key=None,
    ):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.log_artifacts = log_artifacts
        self.history = []
        self.best_path = None
        self.last_path = None
        self.run = None
        self._wandb = None
        self.config = dict(config) if config else {}
        commit = git_hash()
        if commit is not None:
            self.config.setdefault("git_hash", commit)

        self.enabled = enabled
        if self.enabled:
            # W&B reads WANDB_BASE_URL / WANDB_API_KEY from the environment. Only set
            # them here when explicitly provided; otherwise the env vars are used as-is.
            if base_url:
                os.environ.setdefault("WANDB_BASE_URL", base_url)
            if api_key:
                os.environ.setdefault("WANDB_API_KEY", api_key)
            try:
                import wandb

                self._wandb = wandb
                self.run = wandb.init(
                    project=project,
                    name=run_name,
                    config=self.config,
                    mode=mode,
                    dir=dir,
                )
            except Exception as exc:  # pragma: no cover - depends on environment
                print(f"W&B disabled: {exc}")
                self.enabled = False

    def log_config(self, data):
        """Merge the training configuration into the run config."""
        self.config.update(data)
        if self.run is not None:
            try:
                self.run.config.update(data, allow_val_change=True)
            except Exception:
                pass
        return self.config

    def log(self, data, step=None):
        if self.run is not None:
            self._wandb.log(data, step=step)

    def log_epoch(self, epoch, data, step=None):
        record = {"epoch": epoch, **data}
        self.history.append(record)
        self.log(data, step=step)
        return record

    def log_checkpoint(self, path, alias):
        if alias == "best":
            self.best_path = path
        elif alias == "last":
            self.last_path = path
        if (self.run is not None) and (alias == "best"):
            try:
                self.run.summary[f"{alias}_model_path"] = path
                if self.log_artifacts:
                    self._wandb.save(
                        path, base_path=os.path.dirname(os.path.abspath(path))
                    )
            except Exception:
                pass

    def log_model_artifact(self, path, name=None, aliases=("best",)):
        """Upload a checkpoint to W&B as a versioned model artifact."""
        if self.run is None or not self.log_artifacts or not path:
            return None
        try:
            artifact = self._wandb.Artifact(
                name=name or f"model-{self.run.id}", type="model"
            )
            artifact.add_file(path)
            self.run.log_artifact(artifact, aliases=list(aliases))
            return artifact
        except Exception:
            return None

    def finish(self, results_path=None):
        if results_path is None:
            results_path = os.path.join(self.checkpoint_dir, "results.json")
        os.makedirs(os.path.dirname(os.path.abspath(results_path)), exist_ok=True)
        payload = {
            "history": self.history,
            "best_model_path": self.best_path,
            "last_model_path": self.last_path,
        }
        with open(results_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

        if self.run is not None:
            try:
                self.log({"results_path": results_path})
                self._wandb.save(
                    results_path,
                    base_path=os.path.dirname(os.path.abspath(results_path)),
                )
                if self.log_artifacts:
                    run_id = self.run.id
                    self.log_model_artifact(
                        self.best_path, name=f"model-{run_id}-best", aliases=("best",)
                    )
                    self.log_model_artifact(
                        self.last_path, name=f"model-{run_id}-last", aliases=("last",)
                    )
                self.run.finish()
            except Exception:
                pass
        return results_path
