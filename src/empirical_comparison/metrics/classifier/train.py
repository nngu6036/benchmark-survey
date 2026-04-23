from __future__ import annotations
import numpy as np

class LogisticRegressionPlaceholder:
    def fit(self, x: np.ndarray, y: np.ndarray):
        self.class_means_ = {0: x[y == 0].mean(axis=0), 1: x[y == 1].mean(axis=0)}
        return self

    def predict_score(self, x: np.ndarray) -> np.ndarray:
        d0 = ((x - self.class_means_[0]) ** 2).sum(axis=1)
        d1 = ((x - self.class_means_[1]) ** 2).sum(axis=1)
        logits = d1 - d0
        return 1.0 / (1.0 + np.exp(-logits))
