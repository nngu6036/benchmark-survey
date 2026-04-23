from __future__ import annotations
import numpy as np
from scipy.stats import wasserstein_distance

def feature_wasserstein(xs: list[np.ndarray], ys: list[np.ndarray]) -> float:
    x = np.asarray(xs, dtype=float).reshape(len(xs), -1).mean(axis=1)
    y = np.asarray(ys, dtype=float).reshape(len(ys), -1).mean(axis=1)
    return float(wasserstein_distance(x, y))
