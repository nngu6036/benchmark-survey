from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np


def _as_2d_array(features: Sequence[np.ndarray]) -> np.ndarray:
    arr = np.asarray(features, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D feature array, got shape {arr.shape}")
    return arr


def _pairwise_sq_dists(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Compute pairwise squared Euclidean distances between rows of x and y.
    """
    x_norm = np.sum(x * x, axis=1, keepdims=True)
    y_norm = np.sum(y * y, axis=1, keepdims=True).T
    d2 = x_norm + y_norm - 2.0 * (x @ y.T)
    return np.maximum(d2, 0.0)


def _rbf_kernel(x: np.ndarray, y: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    d2 = _pairwise_sq_dists(x, y)
    return np.exp(-d2 / (2.0 * sigma * sigma))


def _median_heuristic_sigma(x: np.ndarray, y: np.ndarray) -> float:
    """
    Estimate a sensible RBF bandwidth using the median pairwise distance.
    """
    z = np.vstack([x, y])
    d2 = _pairwise_sq_dists(z, z)

    # Remove diagonal zeros
    mask = ~np.eye(d2.shape[0], dtype=bool)
    vals = d2[mask]

    # Keep only positive distances
    vals = vals[vals > 1e-12]

    if vals.size == 0:
        return 1.0

    median_d2 = float(np.median(vals))
    sigma = np.sqrt(max(median_d2, 1e-12))
    return sigma


def mmd_unbiased(
    x: np.ndarray,
    y: np.ndarray,
    sigma: Optional[float] = None,
) -> float:
    """
    Unbiased MMD^2 estimate with an RBF kernel.

    Returns:
        A nonnegative scalar (clipped at zero for numerical stability).
    """
    x = _as_2d_array(x)
    y = _as_2d_array(y)

    n = x.shape[0]
    m = y.shape[0]

    if n < 2 or m < 2:
        raise ValueError("Need at least 2 samples in each set for unbiased MMD.")

    if sigma is None:
        sigma = _median_heuristic_sigma(x, y)

    k_xx = _rbf_kernel(x, x, sigma)
    k_yy = _rbf_kernel(y, y, sigma)
    k_xy = _rbf_kernel(x, y, sigma)

    # Remove diagonal terms for unbiased estimate
    sum_xx = (np.sum(k_xx) - np.trace(k_xx)) / (n * (n - 1))
    sum_yy = (np.sum(k_yy) - np.trace(k_yy)) / (m * (m - 1))
    sum_xy = np.sum(k_xy) / (n * m)

    mmd2 = float(sum_xx + sum_yy - 2.0 * sum_xy)

    # Numerical guard: small negatives can appear due to finite precision
    return max(mmd2, 0.0)


def feature_mmd(
    ref_feats: Sequence[np.ndarray],
    gen_feats: Sequence[np.ndarray],
    sigma: Optional[float] = None,
) -> float:
    """
    Compute MMD^2 between reference and generated feature distributions.

    If sigma is None, uses the median heuristic on the pooled feature set.
    """
    x = _as_2d_array(ref_feats)
    y = _as_2d_array(gen_feats)
    return mmd_unbiased(x, y, sigma=sigma)