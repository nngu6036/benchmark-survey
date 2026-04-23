from __future__ import annotations
import networkx as nx
import numpy as np

def degree_histogram(graph: nx.Graph, bins: int = 16) -> np.ndarray:
    deg = np.array([d for _, d in graph.degree()], dtype=float)
    hi = max(1, int(deg.max()) if len(deg) else 1)
    hist, _ = np.histogram(deg, bins=bins, range=(0, hi))
    return hist.astype(float)

def clustering_histogram(graph: nx.Graph, bins: int = 16) -> np.ndarray:
    vals = np.array(list(nx.clustering(graph).values()), dtype=float)
    hist, _ = np.histogram(vals, bins=bins, range=(0.0, 1.0))
    return hist.astype(float)

def spectral_histogram(graph: nx.Graph, bins: int = 16) -> np.ndarray:
    lap = nx.normalized_laplacian_matrix(graph).toarray()
    eig = np.linalg.eigvalsh(lap)
    hist, _ = np.histogram(eig, bins=bins, range=(0.0, 2.0))
    return hist.astype(float)

def orbit_placeholder(graph: nx.Graph) -> np.ndarray:
    tri = sum(nx.triangles(graph).values()) / 3.0
    return np.array([graph.number_of_nodes(), graph.number_of_edges(), tri], dtype=float)
