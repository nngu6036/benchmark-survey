from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from scipy.stats import wasserstein_distance


def _as_2d_array(xs: Sequence[np.ndarray] | np.ndarray) -> np.ndarray:
    arr = np.asarray(xs, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D feature/descriptor array, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("MMD input contains NaN or Inf values.")
    return arr


def pairwise_sq_dists(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x_norm = np.sum(x * x, axis=1, keepdims=True)
    y_norm = np.sum(y * y, axis=1, keepdims=True).T
    d2 = x_norm + y_norm - 2.0 * (x @ y.T)
    return np.maximum(d2, 0.0)


def median_heuristic_sigma(x: np.ndarray, y: np.ndarray) -> float:
    z = np.vstack([x, y])
    d2 = pairwise_sq_dists(z, z)
    mask = ~np.eye(d2.shape[0], dtype=bool)
    vals = d2[mask]
    vals = vals[vals > 1e-12]
    if vals.size == 0:
        return 1.0
    return float(np.sqrt(max(np.median(vals), 1e-12)))


def rbf_kernel_matrix(x: np.ndarray, y: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    d2 = pairwise_sq_dists(x, y)
    return np.exp(-d2 / (2.0 * sigma * sigma))


def mmd_unbiased(
    xs: Sequence[np.ndarray] | np.ndarray,
    ys: Sequence[np.ndarray] | np.ndarray,
    sigma: Optional[float] = None,
) -> float:
    """Unbiased MMD^2 estimate using an RBF kernel.

    This is used for fixed-length Euclidean descriptors such as spectral
    histograms or learned/random feature vectors.
    """
    x = _as_2d_array(xs)
    y = _as_2d_array(ys)
    n, m = x.shape[0], y.shape[0]
    if n < 2 or m < 2:
        raise ValueError("Need at least 2 samples in each set for unbiased MMD.")

    if sigma is None:
        sigma = median_heuristic_sigma(x, y)

    k_xx = rbf_kernel_matrix(x, x, sigma)
    k_yy = rbf_kernel_matrix(y, y, sigma)
    k_xy = rbf_kernel_matrix(x, y, sigma)

    xx = (np.sum(k_xx) - np.trace(k_xx)) / (n * (n - 1))
    yy = (np.sum(k_yy) - np.trace(k_yy)) / (m * (m - 1))
    xy = np.sum(k_xy) / (n * m)
    return max(float(xx + yy - 2.0 * xy), 0.0)


def gaussian_emd_kernel_matrix(x: np.ndarray, y: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Gaussian kernel whose distance is 1D Wasserstein/EMD over bins.

    This follows the common graph-generation MMD implementation used for
    histogram-like descriptors such as degree, clustering, and graphlet/orbit
    count histograms.
    """
    x = _as_2d_array(x)
    y = _as_2d_array(y)
    if x.shape[1] != y.shape[1]:
        raise ValueError(f"Descriptor dimensions differ: {x.shape[1]} vs {y.shape[1]}")
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")

    bins = np.arange(x.shape[1], dtype=np.float64)
    k = np.zeros((x.shape[0], y.shape[0]), dtype=np.float64)
    for i, xi in enumerate(x):
        for j, yj in enumerate(y):
            # scipy.stats.wasserstein_distance accepts weights that do not sum to
            # one, but normalized histograms are preferred.
            emd = wasserstein_distance(bins, bins, xi, yj)
            k[i, j] = np.exp(-(emd * emd) / (2.0 * sigma * sigma))
    return k


def mmd_gaussian_emd(
    xs: Sequence[np.ndarray] | np.ndarray,
    ys: Sequence[np.ndarray] | np.ndarray,
    sigma: float = 1.0,
    unbiased: bool = False,
) -> float:
    """MMD using the Gaussian-EMD kernel.

    The reference implementation in many graph-generation repositories uses the
    biased V-statistic, i.e. mean(Kxx) + mean(Kyy) - 2 mean(Kxy).  Set
    ``unbiased=True`` to remove diagonal terms.
    """
    x = _as_2d_array(xs)
    y = _as_2d_array(ys)
    n, m = x.shape[0], y.shape[0]
    if n < 1 or m < 1:
        raise ValueError("Need non-empty samples for MMD.")

    k_xx = gaussian_emd_kernel_matrix(x, x, sigma=sigma)
    k_yy = gaussian_emd_kernel_matrix(y, y, sigma=sigma)
    k_xy = gaussian_emd_kernel_matrix(x, y, sigma=sigma)

    if unbiased:
        if n < 2 or m < 2:
            raise ValueError("Need at least 2 samples in each set for unbiased MMD.")
        xx = (np.sum(k_xx) - np.trace(k_xx)) / (n * (n - 1))
        yy = (np.sum(k_yy) - np.trace(k_yy)) / (m * (m - 1))
    else:
        xx = np.mean(k_xx)
        yy = np.mean(k_yy)
    xy = np.mean(k_xy)
    return max(float(xx + yy - 2.0 * xy), 0.0)
