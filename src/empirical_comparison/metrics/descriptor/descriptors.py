from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable, Sequence

import networkx as nx
import numpy as np

# ORCA executable can be supplied either through --orca-exec in the script or
# through this environment variable.
ORCA_EXEC = os.environ.get("ORCA_EXEC")

# Multiplicities for the 15 node orbits of connected graphlets up to size 4.
# Same convention as the implementation supplied in eval.py.
ORCA_4_NODE_ORBIT_MULTIPLICITY = np.asarray(
    [2, 2, 1, 6, 2, 1, 2, 4, 2, 1, 1, 2, 2, 4, 1], dtype=np.float64
)


def _normalized_hist(values, bins: int, value_range: tuple[float, float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return np.zeros(bins, dtype=np.float64)
    lo, hi = value_range
    arr = np.clip(arr, lo, hi)
    hist, _ = np.histogram(arr, bins=bins, range=value_range, density=False)
    hist = hist.astype(np.float64)
    total = hist.sum()
    return hist / total if total > 0 else hist


def degree_histogram(graph: nx.Graph, bins: int = 20, max_degree: int = 100) -> np.ndarray:
    """Normalized degree histogram with a fixed global range."""
    deg = [d for _, d in graph.degree()]
    return _normalized_hist(deg, bins=bins, value_range=(0.0, float(max_degree)))


def clustering_histogram(graph: nx.Graph, bins: int = 20) -> np.ndarray:
    """Normalized histogram of local clustering coefficients."""
    try:
        vals = list(nx.clustering(graph).values())
    except Exception:
        vals = []
    return _normalized_hist(vals, bins=bins, value_range=(0.0, 1.0))


def spectral_histogram(graph: nx.Graph, bins: int = 20) -> np.ndarray:
    """Normalized histogram of normalized Laplacian eigenvalues in [0, 2]."""
    n = graph.number_of_nodes()
    if n == 0:
        return np.zeros(bins, dtype=np.float64)
    try:
        lap = nx.normalized_laplacian_matrix(graph).astype(float).toarray()
        eig = np.linalg.eigvalsh(lap)
        eig = np.real(eig)
    except Exception:
        eig = np.zeros(n, dtype=np.float64)
    return _normalized_hist(eig, bins=bins, value_range=(0.0, 2.0))


def structural_summary(graph: nx.Graph) -> np.ndarray:
    """Small fallback structural summary used when true ORCA counts are disabled."""
    n = float(graph.number_of_nodes())
    m = float(graph.number_of_edges())
    try:
        triangles = float(sum(nx.triangles(graph).values()) / 3.0)
    except Exception:
        triangles = 0.0
    try:
        avg_clustering = float(nx.average_clustering(graph))
    except Exception:
        avg_clustering = 0.0
    try:
        num_components = float(nx.number_connected_components(graph))
        largest_cc = max((len(c) for c in nx.connected_components(graph)), default=0)
        largest_cc_ratio = float(largest_cc / n) if n > 0 else 0.0
    except Exception:
        num_components = 0.0
        largest_cc_ratio = 0.0
    density = float(nx.density(graph)) if n > 1 else 0.0
    avg_degree = float(2.0 * m / n) if n > 0 else 0.0
    return np.array(
        [
            np.log1p(n),
            np.log1p(m),
            density,
            avg_degree,
            avg_clustering,
            np.log1p(triangles),
            num_components,
            largest_cc_ratio,
        ],
        dtype=np.float64,
    )


def _resolve_orca_exec(orca_exec: str | None = None) -> str:
    exe = orca_exec or ORCA_EXEC or os.environ.get("ORCA_EXEC")
    if not exe:
        raise FileNotFoundError(
            "ORCA executable is not configured. Set ORCA_EXEC=/path/to/orca "
            "or pass --orca-exec to evaluate_descriptor_metrics.py."
        )
    exe_path = Path(exe)
    if not exe_path.exists():
        raise FileNotFoundError(f"ORCA executable not found: {exe}")
    return str(exe_path)


def _simple_undirected_graph(graph: nx.Graph) -> nx.Graph:
    """Convert to a simple undirected graph with no self-loops for ORCA."""
    g = nx.Graph()
    g.add_nodes_from(graph.nodes())
    g.add_edges_from((u, v) for u, v in graph.edges() if u != v)
    return g


def count_orca_4node_orbits(graph: nx.Graph, orca_exec: str | None = None) -> np.ndarray:
    """Count 4-node graphlet orbits using ORCA.

    ORCA returns node-level orbit counts. We sum over nodes and divide by the
    standard orbit multiplicity vector to obtain graph-level occurrence counts,
    matching the compute_mmd_orbit implementation supplied by the user.

    Returns:
        A length-15 float vector of graphlet/orbit occurrence counts.
    """
    exe = _resolve_orca_exec(orca_exec)
    g = _simple_undirected_graph(graph)
    nodes = sorted(g.nodes())
    node_map = {node: idx for idx, node in enumerate(nodes)}

    if len(nodes) == 0:
        return np.zeros(15, dtype=np.float64)

    input_path = None
    output_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f_in:
            input_path = f_in.name
            f_in.write(f"{g.number_of_nodes()} {g.number_of_edges()}\n")
            for u, v in g.edges():
                f_in.write(f"{node_map[u]} {node_map[v]}\n")

        with tempfile.NamedTemporaryFile(mode="r", delete=False) as f_out:
            output_path = f_out.name

        # ORCA command format: orca node 4 input output
        proc = subprocess.run(
            [exe, "node", "4", input_path, output_path],
            check=True,
            capture_output=True,
            text=True,
        )

        with open(output_path, "r", encoding="utf-8") as f:
            rows = [line.strip().split() for line in f if line.strip()]

        if not rows:
            return np.zeros(15, dtype=np.float64)

        node_orbit_counts = np.asarray([[float(v) for v in row] for row in rows], dtype=np.float64)
        total_counts = node_orbit_counts.sum(axis=0)

        # Defensive handling in case a different ORCA binary returns a different
        # number of columns.
        if total_counts.size < ORCA_4_NODE_ORBIT_MULTIPLICITY.size:
            padded = np.zeros(ORCA_4_NODE_ORBIT_MULTIPLICITY.size, dtype=np.float64)
            padded[: total_counts.size] = total_counts
            total_counts = padded
        elif total_counts.size > ORCA_4_NODE_ORBIT_MULTIPLICITY.size:
            total_counts = total_counts[: ORCA_4_NODE_ORBIT_MULTIPLICITY.size]

        return total_counts / ORCA_4_NODE_ORBIT_MULTIPLICITY

    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr or ""
        stdout = exc.stdout or ""
        raise RuntimeError(f"ORCA execution failed. stdout={stdout!r} stderr={stderr!r}") from exc
    finally:
        for path in (input_path, output_path):
            if path and os.path.exists(path):
                os.remove(path)


def orbit_count_vector(
    graph: nx.Graph,
    *,
    orca_exec: str | None = None,
    normalize: bool = True,
    log_transform: bool = False,
) -> np.ndarray:
    """Return a graph-level 4-node orbit-count vector.

    Args:
        graph: input NetworkX graph.
        orca_exec: optional explicit path to ORCA executable.
        normalize: if True, convert counts into a histogram-like vector.
        log_transform: if True, apply log1p before optional normalization.
    """
    counts = count_orca_4node_orbits(graph, orca_exec=orca_exec).astype(np.float64)
    if log_transform:
        counts = np.log1p(counts)
    if normalize:
        total = counts.sum()
        if total > 0:
            counts = counts / total
    return counts


# Backward-compatible alias for older scripts. Now this is a true ORCA-based
# orbit vector if ORCA is configured; otherwise it raises FileNotFoundError.
def orbit_placeholder(graph: nx.Graph) -> np.ndarray:
    return orbit_count_vector(graph)
