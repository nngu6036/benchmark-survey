from __future__ import annotations
import numpy as np

def rbf_kernel(x: np.ndarray, y: np.ndarray, sigma: float = 1.0) -> float:
    return float(np.exp(-np.sum((x - y) ** 2) / (2 * sigma ** 2)))

def mmd_unbiased(xs: list[np.ndarray], ys: list[np.ndarray], sigma: float = 1.0) -> float:
    m, n = len(xs), len(ys)
    if m < 2 or n < 2:
        return 0.0
    k_xx = sum(rbf_kernel(xs[i], xs[j], sigma) for i in range(m) for j in range(m) if i != j) / (m * (m - 1))
    k_yy = sum(rbf_kernel(ys[i], ys[j], sigma) for i in range(n) for j in range(n) if i != j) / (n * (n - 1))
    k_xy = sum(rbf_kernel(x, y, sigma) for x in xs for y in ys) / (m * n)
    return float(k_xx + k_yy - 2 * k_xy)
