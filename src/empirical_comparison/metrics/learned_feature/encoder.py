from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np


def _safe_triangle_count(graph: nx.Graph) -> float:
    try:
        return float(sum(nx.triangles(graph).values()) / 3.0)
    except Exception:
        return 0.0


def _safe_avg_clustering(graph: nx.Graph) -> float:
    try:
        return float(nx.average_clustering(graph))
    except Exception:
        return 0.0


def _hist(values, bins: int, value_range: tuple[float, float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return np.zeros(bins, dtype=np.float64)
    h, _ = np.histogram(arr, bins=bins, range=value_range, density=False)
    h = h.astype(np.float64)
    s = h.sum()
    return h / s if s > 0 else h


@dataclass
class StructuralFeatureEncoder:
    """Deterministic structural encoder for learned-feature-style MMD.

    This is stronger and more stable than the previous random placeholder: it
    maps each graph to a fixed descriptor vector with scalar, degree,
    clustering, and spectrum features.  It is still not a trained neural GNN;
    the JSON output labels it accordingly.  The class is dependency-light and
    suitable as the default benchmark fallback when no trained encoder is
    provided.
    """

    degree_bins: int = 32
    clustering_bins: int = 32
    spectral_k: int = 32
    max_degree: int = 128
    normalize_output: bool = True

    name: str = "StructuralFeatureEncoder"

    def _spectrum(self, graph: nx.Graph) -> np.ndarray:
        n = graph.number_of_nodes()
        if n == 0:
            return np.zeros(self.spectral_k, dtype=np.float64)
        try:
            lap = nx.normalized_laplacian_matrix(graph).astype(float).toarray()
            eigvals = np.sort(np.real(np.linalg.eigvalsh(lap)))
        except Exception:
            return np.zeros(self.spectral_k, dtype=np.float64)
        out = np.zeros(self.spectral_k, dtype=np.float64)
        out[: min(self.spectral_k, eigvals.size)] = eigvals[: self.spectral_k]
        return out

    def encode(self, graph: nx.Graph) -> np.ndarray:
        n = float(graph.number_of_nodes())
        e = float(graph.number_of_edges())
        density = float(nx.density(graph)) if n > 1 else 0.0
        avg_degree = (2.0 * e / n) if n > 0 else 0.0
        triangles = _safe_triangle_count(graph)
        avg_clust = _safe_avg_clustering(graph)
        try:
            components = float(nx.number_connected_components(graph)) if n > 0 else 0.0
            largest_cc = max((len(c) for c in nx.connected_components(graph)), default=0)
            largest_cc_ratio = float(largest_cc / n) if n > 0 else 0.0
        except Exception:
            components = 0.0
            largest_cc_ratio = 0.0

        scalar = np.asarray(
            [
                np.log1p(n),
                np.log1p(e),
                density,
                avg_degree,
                avg_clust,
                np.log1p(triangles),
                components,
                largest_cc_ratio,
            ],
            dtype=np.float64,
        )
        degree_hist = _hist([d for _, d in graph.degree()], self.degree_bins, (0.0, float(self.max_degree)))
        clustering_hist = _hist(nx.clustering(graph).values() if n > 0 else [], self.clustering_bins, (0.0, 1.0))
        z = np.concatenate([scalar, degree_hist, clustering_hist, self._spectrum(graph)]).astype(np.float64)
        if self.normalize_output:
            norm = np.linalg.norm(z)
            if norm > 1e-12:
                z = z / norm
        return z


@dataclass
class RandomGINPlaceholder:
    """Backward-compatible fixed random-feature placeholder.

    Important: this is not a trained GIN. It is kept only for reproducing older
    benchmark runs and should not be reported as a trained learned-feature
    metric.
    """

    feature_dim: int = 128
    seed: int = 42
    normalize_output: bool = True
    name: str = "RandomGINPlaceholder"

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self.base_dim = 6
        self.proj = self.rng.normal(0.0, 1.0 / np.sqrt(self.base_dim), size=(self.base_dim, self.feature_dim))
        self.bias = self.rng.normal(0.0, 0.1, size=(self.feature_dim,))

    def _base_features(self, graph: nx.Graph) -> np.ndarray:
        n = float(graph.number_of_nodes())
        e = float(graph.number_of_edges())
        return np.array(
            [
                np.log1p(n),
                np.log1p(e),
                float(nx.density(graph)) if n > 1 else 0.0,
                np.log1p(_safe_triangle_count(graph)),
                _safe_avg_clustering(graph),
                (2.0 * e / n) if n > 0 else 0.0,
            ],
            dtype=np.float64,
        )

    def encode(self, graph: nx.Graph) -> np.ndarray:
        z = np.tanh(self._base_features(graph) @ self.proj + self.bias)
        if self.normalize_output:
            norm = np.linalg.norm(z)
            if norm > 1e-12:
                z = z / norm
        return z.astype(np.float64)
