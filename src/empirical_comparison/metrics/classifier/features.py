from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import networkx as nx
import numpy as np


def _hist(values: Sequence[float], bins: int, value_range: tuple[float, float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return np.zeros(bins, dtype=np.float64)

    h, _ = np.histogram(arr, bins=bins, range=value_range, density=False)
    h = h.astype(np.float64)
    s = h.sum()
    return h / s if s > 0 else h


def _safe_avg_clustering(g: nx.Graph) -> float:
    try:
        return float(nx.average_clustering(g))
    except Exception:
        return 0.0


def _safe_triangle_count(g: nx.Graph) -> float:
    try:
        return float(sum(nx.triangles(g).values()) / 3.0)
    except Exception:
        return 0.0


def _safe_spectrum(g: nx.Graph, k: int = 20) -> np.ndarray:
    n = g.number_of_nodes()
    if n == 0:
        return np.zeros(k, dtype=np.float64)

    try:
        lap = nx.normalized_laplacian_matrix(g).astype(float).toarray()
        eigvals = np.linalg.eigvalsh(lap)
        eigvals = np.sort(np.real(eigvals))
    except Exception:
        return np.zeros(k, dtype=np.float64)

    if eigvals.size >= k:
        return eigvals[:k].astype(np.float64)

    out = np.zeros(k, dtype=np.float64)
    out[: eigvals.size] = eigvals
    return out


@dataclass
class GraphDescriptorFeaturizer:
    """
    Descriptor-based graph featurizer for classifier-based evaluation.

    This is a practical PolyGraphScore-style feature representation:
    each graph is mapped to a fixed vector of structural descriptors.
    """

    degree_bins: int = 20
    clustering_bins: int = 20
    spectral_k: int = 20
    max_degree: int = 100

    def transform_one(self, g: nx.Graph) -> np.ndarray:
        n = g.number_of_nodes()
        m = g.number_of_edges()

        degrees = [d for _, d in g.degree()]
        degree_hist = _hist(
            degrees,
            bins=self.degree_bins,
            value_range=(0.0, float(self.max_degree)),
        )

        clustering_values = list(nx.clustering(g).values()) if n > 0 else []
        clustering_hist = _hist(
            clustering_values,
            bins=self.clustering_bins,
            value_range=(0.0, 1.0),
        )

        spectrum = _safe_spectrum(g, k=self.spectral_k)

        density = nx.density(g) if n > 1 else 0.0
        avg_degree = (2.0 * m / n) if n > 0 else 0.0
        avg_clustering = _safe_avg_clustering(g)
        triangles = _safe_triangle_count(g)

        try:
            num_components = nx.number_connected_components(g)
            largest_cc = max((len(c) for c in nx.connected_components(g)), default=0)
            largest_cc_ratio = largest_cc / n if n > 0 else 0.0
        except Exception:
            num_components = 0
            largest_cc_ratio = 0.0

        scalar_features = np.array(
            [
                np.log1p(n),
                np.log1p(m),
                density,
                avg_degree,
                avg_clustering,
                np.log1p(triangles),
                float(num_components),
                largest_cc_ratio,
            ],
            dtype=np.float64,
        )

        return np.concatenate(
            [
                scalar_features,
                degree_hist,
                clustering_hist,
                spectrum,
            ],
            axis=0,
        )

    def transform(self, graphs: Sequence[nx.Graph]) -> np.ndarray:
        x = np.asarray([self.transform_one(g) for g in graphs], dtype=np.float64)

        if x.ndim != 2:
            raise ValueError(f"Expected 2D feature matrix, got shape {x.shape}")

        if not np.all(np.isfinite(x)):
            raise ValueError("Descriptor features contain NaN or Inf.")

        return x