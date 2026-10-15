"""Weights & Biases logging with a local results dump."""

import json
import os


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
    ):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.history = []
        self.best_path = None
        self.last_path = None
        self.run = None
        self._wandb = None

        self.enabled = enabled
        if self.enabled:
            try:
                import wandb

                self._wandb = wandb
                self.run = wandb.init(
                    project=project,
                    name=run_name,
                    config=config,
                    mode=mode,
                    dir=dir,
                )
            except Exception as exc:  # pragma: no cover - depends on environment
                print(f"W&B disabled: {exc}")
                self.enabled = False

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
        if self.run is not None:
            try:
                self.run.summary[f"{alias}_model_path"] = path
                self._wandb.save(
                    path, base_path=os.path.dirname(os.path.abspath(path))
                )
            except Exception:
                pass

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
                self.run.finish()
            except Exception:
                pass
        return results_path
