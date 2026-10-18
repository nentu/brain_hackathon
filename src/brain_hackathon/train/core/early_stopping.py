"""Early stopping helper."""


class EarlyStopping:
    """Signals stop after ``patience`` epochs without a decrease in the monitored value."""

    def __init__(self, patience=10, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.counter = 0

    def step(self, value):
        if value < self.best - self.min_delta:
            self.best = value
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience
