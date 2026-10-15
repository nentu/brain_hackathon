"""Classification metrics."""

import numpy as np
import torch

METRIC_NAMES = ("acc", "f1", "precision", "recall", "sensitivity", "specificity")


def _safe_div(numerator, denominator):
    denominator = np.where(denominator > 0, denominator, 1.0)
    return np.where(numerator > 0, numerator / denominator, 0.0)


class Metrics:
    """Accumulate multiclass predictions and compute macro one-vs-rest metrics.

    ``acc`` is plain accuracy; ``f1``, ``precision``, ``recall``, ``sensitivity`` and
    ``specificity`` are macro averages over classes.
    """

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.tp = np.zeros(self.num_classes, dtype=np.float64)
        self.fp = np.zeros(self.num_classes, dtype=np.float64)
        self.fn = np.zeros(self.num_classes, dtype=np.float64)
        self.correct = 0
        self.total = 0

    @torch.no_grad()
    def update(self, logits, targets):
        preds = logits.argmax(dim=1).detach().cpu().numpy()
        targets = targets.detach().cpu().numpy()
        for c in range(self.num_classes):
            pred_c = preds == c
            true_c = targets == c
            self.tp[c] += np.logical_and(pred_c, true_c).sum()
            self.fp[c] += np.logical_and(pred_c, ~true_c).sum()
            self.fn[c] += np.logical_and(~pred_c, true_c).sum()
        self.correct += int((preds == targets).sum())
        self.total += int(targets.size)

    def compute(self):
        tn = self.total - self.tp - self.fp - self.fn
        precision = _safe_div(self.tp, self.tp + self.fp)
        recall = _safe_div(self.tp, self.tp + self.fn)
        specificity = _safe_div(tn, tn + self.fp)
        f1 = _safe_div(2.0 * precision * recall, precision + recall)
        accuracy = self.correct / self.total if self.total else 0.0
        return {
            "acc": float(accuracy),
            "f1": float(f1.mean()),
            "precision": float(precision.mean()),
            "recall": float(recall.mean()),
            "sensitivity": float(recall.mean()),
            "specificity": float(specificity.mean()),
        }
