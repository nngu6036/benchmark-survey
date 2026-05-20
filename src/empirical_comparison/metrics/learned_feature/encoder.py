from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

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
class WLSubtreeFeatureEncoder:
    """Fitted Weisfeiler-Lehman subtree feature encoder.

    The encoder learns a subtree-label vocabulary and optional IDF weights from
    reference graphs, then maps every graph to a fixed vector of normalized WL
    subtree counts.  If ``feature_dim`` is smaller than the vocabulary size and
    scikit-learn is available, a TruncatedSVD projection is fitted on reference
    features.  This is a dependency-light fitted graph representation for
    feature-space MMD; it is not a supervised classifier and does not use
    generated labels.
    """

    num_iterations: int = 3
    feature_dim: int | None = 128
    node_label_attr: str = "node_label"
    use_node_labels: bool = True
    use_idf: bool = True
    normalize_output: bool = True
    seed: int = 42
    name: str = "WLSubtreeFeatureEncoder"

    def __post_init__(self) -> None:
        self.vocabulary_: dict[str, int] = {}
        self.idf_: np.ndarray | None = None
        self.reducer_ = None

    @staticmethod
    def _stable_token(parts: tuple[str, ...]) -> str:
        import hashlib

        payload = "\x1f".join(parts).encode("utf-8", errors="replace")
        return hashlib.blake2b(payload, digest_size=12).hexdigest()

    def _initial_labels(self, graph: nx.Graph) -> dict:
        labels = {}
        for node, data in graph.nodes(data=True):
            if self.use_node_labels and self.node_label_attr in data:
                labels[node] = f"atom={data.get(self.node_label_attr)}"
            else:
                labels[node] = f"deg={graph.degree(node)}"
        return labels

    def _tokens(self, graph: nx.Graph) -> list[str]:
        if graph.number_of_nodes() == 0:
            return ["empty_graph"]
        labels = self._initial_labels(graph)
        tokens: list[str] = [f"wl0:{v}" for v in labels.values()]
        for depth in range(1, max(0, int(self.num_iterations)) + 1):
            new_labels = {}
            for node in graph.nodes():
                neigh = sorted(labels.get(nb, "") for nb in graph.neighbors(node))
                token = self._stable_token((labels.get(node, ""), *neigh))
                new_labels[node] = token
            labels = new_labels
            tokens.extend(f"wl{depth}:{v}" for v in labels.values())
        return tokens

    def fit(self, graphs: Sequence[nx.Graph]) -> "WLSubtreeFeatureEncoder":
        from collections import Counter

        df: Counter[str] = Counter()
        for graph in graphs:
            df.update(set(self._tokens(graph)))
        self.vocabulary_ = {token: idx for idx, token in enumerate(sorted(df))}
        n_docs = max(1, len(graphs))
        if self.vocabulary_ and self.use_idf:
            self.idf_ = np.ones(len(self.vocabulary_), dtype=np.float64)
            for token, idx in self.vocabulary_.items():
                self.idf_[idx] = np.log((1.0 + n_docs) / (1.0 + df[token])) + 1.0
        else:
            self.idf_ = None
        if self.feature_dim is not None and self.vocabulary_:
            target_dim = int(self.feature_dim)
            if 0 < target_dim < len(self.vocabulary_):
                x = self._raw_transform(graphs)
                try:
                    from sklearn.decomposition import TruncatedSVD

                    n_components = max(1, min(target_dim, x.shape[0] - 1, x.shape[1] - 1))
                    if n_components >= 1:
                        self.reducer_ = TruncatedSVD(n_components=n_components, random_state=self.seed)
                        self.reducer_.fit(x)
                except Exception:
                    self.reducer_ = None
        return self

    def _raw_transform(self, graphs: Sequence[nx.Graph]) -> np.ndarray:
        width = max(len(self.vocabulary_), 1)
        x = np.zeros((len(graphs), width), dtype=np.float64)
        if not self.vocabulary_:
            return x
        for row, graph in enumerate(graphs):
            for token in self._tokens(graph):
                idx = self.vocabulary_.get(token)
                if idx is not None:
                    x[row, idx] += 1.0
            total = x[row].sum()
            if total > 0:
                x[row] /= total
        if self.idf_ is not None and self.idf_.size == x.shape[1]:
            x *= self.idf_[None, :]
        return x

    def transform(self, graphs: Sequence[nx.Graph]) -> np.ndarray:
        x = self._raw_transform(graphs)
        if self.reducer_ is not None:
            x = self.reducer_.transform(x)
        if self.normalize_output and x.size:
            norms = np.linalg.norm(x, axis=1, keepdims=True)
            x = x / np.maximum(norms, 1e-12)
        return np.nan_to_num(x.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)

    def encode(self, graph: nx.Graph) -> np.ndarray:
        return self.transform([graph])[0]


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
