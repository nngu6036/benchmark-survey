#!/usr/bin/env python3
"""Compute PolyGraphScore with the official polygraph-benchmark implementation.

Drop this file into the survey repo as:

    scripts/evaluate_polygraphscore_official.py

The only command-line selectors are the same benchmark selectors used by the
other scripts: datasets, models, and run ids.  All metric parameters are read
from configs/experiment.yaml, preferably under metrics.polygraphscore_official.

Example:

    export POLYGRAPH_REPO=/path/to/polygraph-benchmark
    export ORCA_EXEC=/path/to/orca

    python scripts/evaluate_polygraphscore_official.py \
      --datasets planar sbm \
      --models digress construct \
      --run-ids 0 1 2

The script calls polygraph-benchmark's PolyGraphDiscrepancy/
PolyGraphDiscrepancyInterval classes.  It also works from an unpacked source
checkout via polygraph_root, so the polygraph package itself does not have to be
installed with pip.  In normal benchmark runs, define POLYGRAPH_REPO as the
path to this unpacked repository instead of setting polygraph_root in the YAML.

If pip fails on the optional binary dependency `orbit-count`, keep the official
orbit descriptors enabled and define ORCA_EXEC as the path to a compiled ORCA
executable.  The script then injects an ORCA-backed compatibility module named
`orbit_count` before importing the official implementation.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import importlib.metadata as importlib_metadata
import importlib.util
import json
import os
import pickle
import random
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import types
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import yaml

DEFAULT_DESCRIPTORS = ["orbit4", "orbit5", "clustering", "degree", "spectral", "gin"]
DEFAULT_METRIC_FILENAME = "polygraphscore_official.json"
ORBIT_DIMS = {4: 15, 5: 73}
DEPENDENCY_SHIMS: dict[str, str] = {}


def log(msg: str, *args: Any) -> None:
    print("[official-pgs] " + (msg % args if args else msg), file=sys.stderr)


def json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if hasattr(obj, "mean") and hasattr(obj, "std"):
        out = {"mean": float(obj.mean), "std": float(obj.std)}
        for attr in ("low", "high", "coverage"):
            value = getattr(obj, attr, None)
            if value is not None:
                out[attr] = float(value)
        return out
    return str(obj)




def patch_scipy_compat() -> None:
    """Patch SciPy/NetworkX compatibility for older NetworkX spectral code.

    Some NetworkX versions call scipy.errstate inside normalized_laplacian_matrix.
    Newer SciPy builds do not expose scipy.errstate; numpy.errstate is the intended
    context manager.  PolyGraph's EigenvalueHistogram can trigger this path.
    """
    try:
        import scipy as _scipy  # type: ignore
    except Exception as exc:
        DEPENDENCY_SHIMS.setdefault("scipy", f"SciPy import failed before spectral descriptor use: {exc}")
        return
    if not hasattr(_scipy, "errstate"):
        try:
            setattr(_scipy, "errstate", np.errstate)
            DEPENDENCY_SHIMS["scipy.errstate"] = "patched scipy.errstate = numpy.errstate for NetworkX spectral descriptor compatibility"
        except Exception as exc:
            DEPENDENCY_SHIMS["scipy.errstate"] = f"failed to patch scipy.errstate: {exc}"

def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise TypeError(f"YAML root must be a mapping: {path}")
    return data


def save_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")


def as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def int_or_none(value: Any) -> int | None:
    if value in (None, "", "null", "None", 0, "0"):
        return None
    return int(value)


def infer_survey_root(explicit: str | None) -> Path:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser().resolve())
    here = Path(__file__).resolve()
    candidates.extend([Path.cwd().resolve(), here.parent.resolve(), here.parent.parent.resolve()])
    for root in candidates:
        if (root / "configs" / "experiment.yaml").exists():
            return root
    return Path.cwd().resolve()


def resolve_path(value: str | Path | None, *, base: Path) -> Path | None:
    if value in (None, ""):
        return None
    # Accept config values such as "${POLYGRAPH_REPO}" or "$ORCA_EXEC".
    path = Path(os.path.expandvars(str(value))).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def metric_cfg(experiment_cfg: Mapping[str, Any]) -> dict[str, Any]:
    metrics = experiment_cfg.get("metrics", {}) or {}
    for key in ("polygraphscore_official", "official_polygraphscore", "polygraph_benchmark"):
        block = metrics.get(key)
        if isinstance(block, Mapping):
            out = dict(block)
            out["_source_config_key"] = f"metrics.{key}"
            return out
    # Fallback lets users try the script before editing experiment.yaml.
    block = metrics.get("classifier", {}) or {}
    out = dict(block) if isinstance(block, Mapping) else {}
    out["_source_config_key"] = "metrics.classifier/fallback"
    return out


def dataset_num_runs(dataset: str, experiment_cfg: Mapping[str, Any]) -> int:
    single = {str(x).lower() for x in experiment_cfg.get("single_run_datasets", []) or []}
    single |= {str(x).lower() for x in experiment_cfg.get("real_datasets", []) or []}
    single |= {"qm9", "zinc"}
    if dataset.lower() in single:
        return int(experiment_cfg.get("real_dataset_num_runs", 1) or 1)
    return int(experiment_cfg.get("num_runs", 1) or 1)


def run_seed(base_seed: int, run_id: int | None, stride: int = 1000) -> int:
    return int(base_seed) if run_id is None else int(base_seed) + int(run_id) * stride


def run_dir_name(run_id: int) -> str:
    return f"run_{run_id:03d}"


def dataset_split_path(root: Path, dataset_root: str, dataset: str, split: str) -> Path:
    base = resolve_path(dataset_root, base=root) or (root / "outputs" / "datasets")
    return base / dataset / f"{split}.pkl"


def sample_path(root: Path, samples_root: str, dataset: str, model: str, run_id: int | None) -> Path:
    base = resolve_path(samples_root, base=root) or (root / "outputs" / "samples")
    if run_id is None:
        return base / dataset / f"{model}.pkl"
    path = base / dataset / model / f"{run_dir_name(run_id)}.pkl"
    if path.exists():
        return path
    if run_id == 0:
        legacy = base / dataset / f"{model}.pkl"
        if legacy.exists():
            return legacy
    return path


def metric_path(root: Path, metrics_root: str, dataset: str, model: str, filename: str, run_id: int | None) -> Path:
    base = resolve_path(metrics_root, base=root) or (root / "outputs" / "metrics")
    if run_id is None:
        return base / dataset / model / filename
    return base / dataset / model / run_dir_name(run_id) / filename


def aggregate_path(root: Path, metrics_root: str, dataset: str, model: str, filename: str) -> Path:
    base = resolve_path(metrics_root, base=root) or (root / "outputs" / "metrics")
    return base / dataset / model / f"{Path(filename).stem}.aggregate.json"


def batch_path(root: Path, metrics_root: str, filename: str) -> Path:
    base = resolve_path(metrics_root, base=root) or (root / "outputs" / "metrics")
    return base / f"{Path(filename).stem}.batch.json"


def open_maybe_gzip(path: Path, mode: str):
    return gzip.open(path, mode) if path.suffix == ".gz" else open(path, mode)


def load_pickle(path: Path) -> Any:
    with open_maybe_gzip(path, "rb") as f:
        return pickle.load(f)


def load_graph_list(path: Path, *, split: str | None = None) -> list[Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = load_pickle(path)
    if isinstance(payload, Mapping):
        if split and split in payload:
            payload = payload[split]
        else:
            for key in ("graphs", "samples", "data", "generated_graphs", "reference_graphs"):
                if key in payload:
                    payload = payload[key]
                    break
    if hasattr(payload, "to_nx"):
        payload = payload.to_nx()
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        graphs = list(payload)
    elif isinstance(payload, Iterable) and not isinstance(payload, (str, bytes, bytearray)):
        graphs = list(payload)
    else:
        graphs = [payload]
    if not graphs:
        raise ValueError(f"No graphs loaded from {path}")
    return graphs


def normalize_graphs(graphs: Sequence[Any], *, simple: bool = True, relabel: bool = True, drop_empty: bool = False) -> list[Any]:
    import networkx as nx

    out = []
    dropped = 0
    for graph in graphs:
        if not isinstance(graph, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)):
            graph = nx.Graph(graph)
        g = nx.Graph(graph) if simple else copy.deepcopy(graph)
        if relabel:
            g = nx.convert_node_labels_to_integers(g, ordering="sorted")
        g.remove_edges_from(nx.selfloop_edges(g))
        if g.number_of_nodes() == 0:
            if drop_empty:
                dropped += 1
                continue
            raise ValueError("Empty graph encountered. Set drop_empty_graphs: true to drop empty samples.")
        out.append(g)
    if dropped:
        log("dropped %d empty graph(s)", dropped)
    if not out:
        raise ValueError("No graphs remain after normalization")
    return out


def subsample(graphs: Sequence[Any], n: int | None, rng: np.random.Generator) -> list[Any]:
    graphs = list(graphs)
    if n is None or n <= 0 or len(graphs) <= n:
        return graphs
    idx = rng.choice(len(graphs), size=int(n), replace=False)
    return [graphs[int(i)] for i in idx]


def balance_graphs(ref: Sequence[Any], gen: Sequence[Any], *, max_ref: int | None, max_gen: int | None, seed: int) -> tuple[list[Any], list[Any]]:
    rng = np.random.default_rng(seed)
    ref2 = subsample(ref, max_ref, rng)
    gen2 = subsample(gen, max_gen, rng)
    n = min(len(ref2), len(gen2))
    if n < 8:
        raise ValueError(f"Official PGS needs enough graphs for fit/test plus 4-fold CV; got reference={len(ref2)}, generated={len(gen2)}")
    return subsample(ref2, n, rng), subsample(gen2, n, rng)


def shuffle_graphs(graphs: Sequence[Any], seed: int) -> list[Any]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(graphs))
    return [graphs[int(i)] for i in idx]


# ---------------------------------------------------------------------------
# Import shims for polygraph-benchmark source trees and optional deps
# ---------------------------------------------------------------------------


def add_polygraph_root(polygraph_root: str | None, *, survey_root: Path) -> str:
    candidates: list[Path] = []
    explicit = resolve_path(polygraph_root, base=survey_root)
    if explicit is not None:
        candidates.append(explicit)
    # POLYGRAPH_REPO is the project-level environment variable used by the
    # survey benchmark.  Keep the older aliases for backwards compatibility.
    for env in ("POLYGRAPH_REPO", "POLYGRAPH_BENCHMARK_ROOT", "POLYGRAPH_ROOT"):
        if os.environ.get(env):
            candidates.append(Path(os.path.expandvars(os.environ[env])).expanduser().resolve())
    candidates.extend([
        survey_root / "polygraph-benchmark",
        survey_root.parent / "polygraph-benchmark",
        Path.cwd() / "polygraph-benchmark",
        Path.cwd().parent / "polygraph-benchmark",
        Path("/mnt/data/polygraph-benchmark"),
    ])
    for root in candidates:
        try:
            root = root.resolve()
        except Exception:
            pass
        if (root / "polygraph" / "__init__.py").exists():
            if str(root) not in sys.path:
                sys.path.insert(0, str(root))
            patch_distribution_version()
            return str(root)
    patch_distribution_version()
    return "installed-or-pythonpath"


def patch_distribution_version() -> None:
    try:
        importlib_metadata.version("polygraph-benchmark")
        return
    except importlib_metadata.PackageNotFoundError:
        pass
    original = importlib_metadata.version

    def version(name: str) -> str:
        if name == "polygraph-benchmark":
            return "1.1.0+source"
        return original(name)

    importlib_metadata.version = version  # type: ignore[assignment]


def ensure_executable(path: Path) -> Path:
    if os.access(path, os.X_OK):
        return path
    try:
        path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        if os.access(path, os.X_OK):
            return path
    except Exception:
        pass
    copied = Path(tempfile.gettempdir()) / f"orca_{os.getpid()}"
    shutil.copy2(path, copied)
    copied.chmod(copied.stat().st_mode | stat.S_IXUSR)
    return copied


def compile_orca(survey_root: Path) -> Path | None:
    if shutil.which("g++") is None:
        return None
    for cpp in [
        survey_root / "external" / "DisCo" / "orca" / "orca.cpp",
        survey_root / "external" / "ConStruct" / "ConStruct" / "analysis" / "orca" / "orca.cpp",
        survey_root / "external" / "EDP-GNN" / "evaluation" / "orca.cpp",
    ]:
        if not cpp.exists():
            continue
        out = survey_root / "outputs" / "tools" / "orca"
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            subprocess.run(["g++", "-O2", "-std=c++11", str(cpp.name), "-o", str(out)], cwd=str(cpp.parent), check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            out.chmod(0o755)
            return out
        except Exception as exc:
            log("could not compile ORCA from %s: %s", cpp, exc)
    return None


def find_orca(value: str | None, *, survey_root: Path, auto_compile: bool) -> Path | None:
    candidates: list[Path] = []
    # ORCA_EXEC is the project-level environment variable used by the survey
    # benchmark.  Prefer it over YAML to avoid stale relative config paths.
    if os.environ.get("ORCA_EXEC"):
        candidates.append(Path(os.path.expandvars(os.environ["ORCA_EXEC"])).expanduser().resolve())
    explicit = resolve_path(value, base=survey_root)
    if explicit is not None:
        candidates.append(explicit)
    candidates.extend([
        survey_root / "external" / "DisCo" / "orca" / "orca",
        survey_root / "outputs" / "tools" / "orca",
        Path("/mnt/data/benchmark-survey/external/DisCo/orca/orca"),
    ])
    for path in candidates:
        if path.exists() and path.is_file():
            return ensure_executable(path)
    return compile_orca(survey_root) if auto_compile else None


def install_orbit_count_shim(orca_exec: Path | None) -> None:
    mod = types.ModuleType("orbit_count")

    def count_one(graph: Any, graphlet_size: int) -> np.ndarray:
        if orca_exec is None:
            raise ImportError("orbit-count is unavailable and no ORCA executable was found. Set orca_exec or skip_orbits: true.")
        import networkx as nx

        g = nx.convert_node_labels_to_integers(nx.Graph(graph), ordering="sorted")
        g.remove_edges_from(nx.selfloop_edges(g))
        if g.number_of_nodes() == 0:
            raise ValueError("ORCA orbit descriptor cannot evaluate an empty graph")
        with tempfile.TemporaryDirectory(prefix="pgs_orca_") as tmp:
            inp = Path(tmp) / "graph.txt"
            out = Path(tmp) / "orbits.txt"
            with inp.open("w", encoding="utf-8") as f:
                f.write(f"{g.number_of_nodes()} {g.number_of_edges()}\n")
                for u, v in g.edges():
                    f.write(f"{int(u)} {int(v)}\n")
            proc = subprocess.run([str(orca_exec), "node", str(int(graphlet_size)), str(inp), str(out)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
            if proc.returncode != 0 or not out.exists():
                raise RuntimeError(f"ORCA failed: returncode={proc.returncode}; stdout={proc.stdout[:500]!r}; stderr={proc.stderr[:500]!r}")
            rows = [[float(x) for x in line.split()] for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]
        arr = np.asarray(rows, dtype=np.float64)
        expected = ORBIT_DIMS[int(graphlet_size)]
        if arr.shape != (g.number_of_nodes(), expected):
            raise RuntimeError(f"Unexpected ORCA output shape {arr.shape}, expected {(g.number_of_nodes(), expected)}")
        return arr

    def batched_node_orbit_counts(graphs: Iterable[Any], graphlet_size: int = 4) -> list[np.ndarray]:
        return [count_one(g, int(graphlet_size)) for g in graphs]

    def batched_edge_orbit_counts(graphs: Iterable[Any], graphlet_size: int = 4) -> list[np.ndarray]:
        raise NotImplementedError("The ORCA shim implements node orbit counts only; StandardPGD uses node orbit counts.")

    mod.batched_node_orbit_counts = batched_node_orbit_counts  # type: ignore[attr-defined]
    mod.batched_edge_orbit_counts = batched_edge_orbit_counts  # type: ignore[attr-defined]
    sys.modules["orbit_count"] = mod
    DEPENDENCY_SHIMS["orbit_count"] = f"ORCA shim ({orca_exec})"


def install_tabpfn_stub_if_logistic(classifier_name: str) -> None:
    normalized = classifier_name.lower().replace("_", "-")
    if normalized not in {"logistic", "logistic-regression", "lr"}:
        return
    if importlib.util.find_spec("tabpfn") is not None:
        return
    tabpfn_mod = types.ModuleType("tabpfn")
    classifier_mod = types.ModuleType("tabpfn.classifier")

    class MissingTabPFNClassifier:  # pragma: no cover
        @classmethod
        def create_default_for_version(cls, *args: Any, **kwargs: Any):
            raise ImportError("TabPFN is not installed")

    class ModelVersion:
        V2_5 = "V2_5"

    tabpfn_mod.TabPFNClassifier = MissingTabPFNClassifier  # type: ignore[attr-defined]
    classifier_mod.ModelVersion = ModelVersion  # type: ignore[attr-defined]
    sys.modules["tabpfn"] = tabpfn_mod
    sys.modules["tabpfn.classifier"] = classifier_mod
    DEPENDENCY_SHIMS["tabpfn"] = "import stub for logistic_regression mode"


def torch_geometric_is_usable() -> tuple[bool, str | None]:
    """Return whether the installed torch_geometric stack can be imported.

    A common failure mode on shared servers is that torch_geometric itself is
    installed, but one of its binary extension packages, e.g. torch_cluster,
    torch_scatter, or torch_sparse, was compiled for a different PyTorch ABI.
    In that case importlib.find_spec succeeds but importing torch_geometric
    raises OSError with an undefined symbol.  The official PolyGraph package
    imports torch_geometric at module import time because of the RandomGIN
    descriptor, so we need to detect that failure before importing PolyGraph.
    """
    try:
        import torch_geometric  # noqa: F401
        from torch_geometric.data import Batch  # noqa: F401
        from torch_geometric.utils import degree, from_networkx  # noqa: F401
        from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool  # noqa: F401
        from torch_geometric.nn.conv import MessagePassing  # noqa: F401
        return True, None
    except Exception as exc:  # noqa: BLE001 - includes OSError from broken binary wheels
        return False, f"{type(exc).__name__}: {exc}"


def install_torch_geometric_shim(reason: str) -> None:
    """Install a small pure-PyTorch subset of torch_geometric used by RandomGIN.

    The official PolyGraph RandomGIN descriptor only needs:
      * torch_geometric.utils.from_networkx
      * torch_geometric.utils.degree
      * torch_geometric.data.Batch.from_data_list
      * torch_geometric.nn.global_{add,mean,max}_pool
      * torch_geometric.nn.conv.MessagePassing with add/mean/max aggregation

    This shim avoids importing broken compiled PyG extensions such as
    torch_cluster while preserving the official PolyGraph descriptor code path.
    It is intended for evaluation only, not for training arbitrary PyG models.
    """
    import torch
    import torch.nn as nn

    tg = types.ModuleType("torch_geometric")
    data = types.ModuleType("torch_geometric.data")
    utils = types.ModuleType("torch_geometric.utils")
    nn_mod = types.ModuleType("torch_geometric.nn")
    conv = types.ModuleType("torch_geometric.nn.conv")

    class DataObj:
        def __init__(self, **kwargs: Any) -> None:
            self.__dict__.update(kwargs)

        def to(self, device: str | torch.device):
            for key, value in list(self.__dict__.items()):
                if torch.is_tensor(value):
                    setattr(self, key, value.to(device))
            return self

    def from_networkx(g: Any, group_node_attrs: Any = None, group_edge_attrs: Any = None):
        nodes = list(g.nodes())
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        edges: list[tuple[int, int]] = []
        edge_attrs: list[list[float]] = []

        # PyG's from_networkx converts undirected NetworkX graphs to directed
        # edge lists.  Mirroring that behavior keeps the RandomGIN descriptor as
        # close as possible to the official PyG-backed path.
        for u, v, attrs in g.edges(data=True):
            pairs = [(u, v)] if getattr(g, "is_directed", lambda: False)() else [(u, v), (v, u)]
            for a, b in pairs:
                edges.append((node_to_idx[a], node_to_idx[b]))
                if group_edge_attrs:
                    edge_attrs.append([float(attrs.get(k, 0.0)) for k in group_edge_attrs])

        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        x = None
        if group_node_attrs:
            x = torch.tensor(
                [[float(g.nodes[node].get(k, 0.0)) for k in group_node_attrs] for node in nodes],
                dtype=torch.float32,
            )

        edge_attr = None
        if group_edge_attrs:
            edge_attr = torch.tensor(edge_attrs, dtype=torch.float32) if edge_attrs else torch.empty((0, len(group_edge_attrs)), dtype=torch.float32)

        return DataObj(edge_index=edge_index, num_nodes=len(nodes), x=x, edge_attr=edge_attr)

    def degree(index: torch.Tensor, num_nodes: int | None = None, dtype: Any = None):
        if num_nodes is None:
            num_nodes = int(index.max().item()) + 1 if index.numel() else 0
        out = torch.bincount(index.to(torch.long), minlength=num_nodes)
        if dtype is not None:
            return out.to(dtype)
        return out.to(torch.float32)

    class Batch(DataObj):
        @classmethod
        def from_data_list(cls, data_list: Sequence[Any]):
            edge_indices = []
            xs = []
            edge_attrs = []
            batch_parts = []
            offset = 0
            for graph_idx, d in enumerate(data_list):
                n = int(d.num_nodes)
                if getattr(d, "edge_index", None) is not None:
                    edge_indices.append(d.edge_index + offset)
                if getattr(d, "x", None) is not None:
                    xs.append(d.x)
                if getattr(d, "edge_attr", None) is not None:
                    edge_attrs.append(d.edge_attr)
                batch_parts.append(torch.full((n,), graph_idx, dtype=torch.long))
                offset += n
            edge_index = torch.cat(edge_indices, dim=1) if edge_indices else torch.empty((2, 0), dtype=torch.long)
            batch = torch.cat(batch_parts, dim=0) if batch_parts else torch.empty((0,), dtype=torch.long)
            x = torch.cat(xs, dim=0) if xs else None
            edge_attr = torch.cat(edge_attrs, dim=0) if edge_attrs else None
            return cls(edge_index=edge_index, batch=batch, num_nodes=offset, x=x, edge_attr=edge_attr)

    def _num_graphs(batch: torch.Tensor) -> int:
        return int(batch.max().item()) + 1 if batch.numel() else 0

    def global_add_pool(x: torch.Tensor, batch: torch.Tensor):
        out = torch.zeros((_num_graphs(batch), x.size(-1)), dtype=x.dtype, device=x.device)
        if batch.numel():
            out.index_add_(0, batch, x)
        return out

    def global_mean_pool(x: torch.Tensor, batch: torch.Tensor):
        out = global_add_pool(x, batch)
        counts = torch.bincount(batch, minlength=out.size(0)).clamp(min=1).to(x.device).to(x.dtype).unsqueeze(-1)
        return out / counts

    def global_max_pool(x: torch.Tensor, batch: torch.Tensor):
        n = _num_graphs(batch)
        if n == 0:
            return torch.empty((0, x.size(-1)), dtype=x.dtype, device=x.device)
        outs = []
        for i in range(n):
            mask = batch == i
            if mask.any():
                outs.append(x[mask].max(dim=0).values)
            else:
                outs.append(torch.zeros((x.size(-1),), dtype=x.dtype, device=x.device))
        return torch.stack(outs, dim=0)

    class MessagePassing(nn.Module):
        def __init__(self, aggr: str = "add", *args: Any, **kwargs: Any) -> None:
            super().__init__()
            self.aggr = aggr

        def propagate(self, edge_index: torch.Tensor, x: torch.Tensor, edge_weight: Any = None, edge_attr: Any = None):
            src = edge_index[0]
            dst = edge_index[1]
            messages = self.message(x[src], edge_weight=edge_weight, edge_attr=edge_attr)
            out = torch.zeros((x.size(0), messages.size(-1)), dtype=messages.dtype, device=messages.device)
            if self.aggr in {"add", "sum"}:
                if dst.numel():
                    out.index_add_(0, dst, messages)
            elif self.aggr == "mean":
                if dst.numel():
                    out.index_add_(0, dst, messages)
                counts = torch.bincount(dst, minlength=x.size(0)).clamp(min=1).to(messages.device).to(messages.dtype).unsqueeze(-1)
                out = out / counts
            elif self.aggr == "max":
                # Simple, robust implementation for evaluation-sized graphs.
                out[:] = -torch.inf
                for e in range(dst.numel()):
                    out[dst[e]] = torch.maximum(out[dst[e]], messages[e])
                out[out == -torch.inf] = 0
            else:
                raise NotImplementedError(f"torch_geometric shim does not support aggr={self.aggr!r}")
            return out

        def message(self, x_j: torch.Tensor, edge_weight: Any = None, edge_attr: Any = None):
            return x_j

    utils.degree = degree  # type: ignore[attr-defined]
    utils.from_networkx = from_networkx  # type: ignore[attr-defined]
    data.Batch = Batch  # type: ignore[attr-defined]
    data.Data = DataObj  # type: ignore[attr-defined]
    nn_mod.global_add_pool = global_add_pool  # type: ignore[attr-defined]
    nn_mod.global_mean_pool = global_mean_pool  # type: ignore[attr-defined]
    nn_mod.global_max_pool = global_max_pool  # type: ignore[attr-defined]
    conv.MessagePassing = MessagePassing  # type: ignore[attr-defined]

    tg.data = data  # type: ignore[attr-defined]
    tg.utils = utils  # type: ignore[attr-defined]
    tg.nn = nn_mod  # type: ignore[attr-defined]
    nn_mod.conv = conv  # type: ignore[attr-defined]

    sys.modules.update({
        "torch_geometric": tg,
        "torch_geometric.data": data,
        "torch_geometric.utils": utils,
        "torch_geometric.nn": nn_mod,
        "torch_geometric.nn.conv": conv,
    })
    DEPENDENCY_SHIMS["torch_geometric"] = reason


def install_torch_geometric_stub_if_needed(descriptors: Sequence[str], cfg: Mapping[str, Any]) -> None:
    uses_gin = "gin" in descriptors
    usable, reason = torch_geometric_is_usable()
    if usable:
        return
    allow_shim = as_bool(cfg.get("allow_torch_geometric_shim"), True)
    if not allow_shim:
        raise ImportError(
            "torch_geometric cannot be imported. This is usually caused by PyG binary wheels "
            "compiled for a different PyTorch/CUDA version. Install matching torch-scatter, "
            "torch-sparse, and torch-cluster wheels, or set allow_torch_geometric_shim: true. "
            f"Import error was: {reason}"
        )
    if uses_gin:
        install_torch_geometric_shim("pure-PyTorch shim because installed torch_geometric is unavailable or ABI-incompatible: " + str(reason))
    else:
        install_torch_geometric_shim("import shim because gin descriptor is disabled and installed torch_geometric is unavailable or ABI-incompatible: " + str(reason))

def canonical_descriptors(cfg: Mapping[str, Any]) -> list[str]:
    raw = cfg.get("descriptors") or DEFAULT_DESCRIPTORS
    raw_names = [raw] if isinstance(raw, str) else list(raw)
    out: list[str] = []
    for name in map(lambda x: str(x).lower(), raw_names):
        if name in {"orbit", "orbits"}:
            out.extend(["orbit4", "orbit5"])
        elif name in {"orb4"}:
            out.append("orbit4")
        elif name in {"orb5"}:
            out.append("orbit5")
        elif name in {"deg"}:
            out.append("degree")
        elif name in {"clust"}:
            out.append("clustering")
        elif name in {"eig", "spectrum", "eigenvalues"}:
            out.append("spectral")
        else:
            out.append(name)
    if as_bool(cfg.get("skip_orbits"), False) or as_bool(cfg.get("skip_orbit"), False):
        out = [d for d in out if not d.startswith("orbit")]
    if as_bool(cfg.get("skip_gin"), False):
        out = [d for d in out if d != "gin"]
    return list(dict.fromkeys(out))


def prepare_imports(cfg: Mapping[str, Any], descriptors: list[str], classifier_name: str, *, survey_root: Path) -> list[str]:
    install_tabpfn_stub_if_logistic(classifier_name)
    install_torch_geometric_stub_if_needed(descriptors, cfg)
    needs_orbits = any(d.startswith("orbit") for d in descriptors)
    if importlib.util.find_spec("orbit_count") is None:
        skip_if_unavailable = (
            as_bool(cfg.get("skip_orbits_if_unavailable"), False)
            or as_bool(cfg.get("skip_orbits_when_unavailable"), False)
        )
        if needs_orbits and skip_if_unavailable:
            descriptors = [d for d in descriptors if not d.startswith("orbit")]
            install_orbit_count_shim(None)
            DEPENDENCY_SHIMS["orbit_count"] = "import shim installed; orbit descriptors skipped because orbit_count is unavailable"
        elif needs_orbits:
            allow_fallback = as_bool(cfg.get("allow_orbit_count_fallback"), True)
            if not allow_fallback:
                raise ImportError(
                    "orbit-count is unavailable and allow_orbit_count_fallback is false. "
                    "Install orbit-count, set skip_orbits: true, or export ORCA_EXEC."
                )
            orca = find_orca(
                str(cfg.get("orca_exec")) if cfg.get("orca_exec") else None,
                survey_root=survey_root,
                auto_compile=as_bool(cfg.get("auto_compile_orca"), True),
            )
            if orca is None:
                raise ImportError(
                    "orbit-count is unavailable and no ORCA executable was found. "
                    "Export ORCA_EXEC, set metrics.polygraphscore_official.orca_exec, "
                    "or set skip_orbits: true for a non-paper fallback."
                )
            install_orbit_count_shim(orca)
        else:
            # polygraph-benchmark imports orbit_count at module import time even
            # when orbit descriptors are not used. This shim lets non-orbit
            # descriptor runs avoid the optional orbit-count dependency.
            install_orbit_count_shim(None)
            DEPENDENCY_SHIMS["orbit_count"] = "import shim because orbit descriptors are disabled"
    return descriptors


def import_polygraph_objects(polygraph_root: str | None, *, survey_root: Path) -> dict[str, Any]:
    resolved = add_polygraph_root(polygraph_root, survey_root=survey_root)
    patch_scipy_compat()
    try:
        from polygraph.metrics.base import PolyGraphDiscrepancy, PolyGraphDiscrepancyInterval, default_classifier
        from polygraph.utils.descriptors import ClusteringHistogram, EigenvalueHistogram, OrbitCounts, RandomGIN, SparseDegreeHistogram
    except ModuleNotFoundError as exc:
        hint = "Set POLYGRAPH_REPO=/path/to/polygraph-benchmark or install the package."
        raise RuntimeError(f"Could not import polygraph-benchmark or one of its dependencies; missing {exc.name!r}. {hint}") from exc
    return {
        "PolyGraphDiscrepancy": PolyGraphDiscrepancy,
        "PolyGraphDiscrepancyInterval": PolyGraphDiscrepancyInterval,
        "ClusteringHistogram": ClusteringHistogram,
        "EigenvalueHistogram": EigenvalueHistogram,
        "OrbitCounts": OrbitCounts,
        "RandomGIN": RandomGIN,
        "SparseDegreeHistogram": SparseDegreeHistogram,
        "default_classifier": default_classifier,
        "polygraph_root_resolved": resolved,
    }


def make_descriptors(poly: Mapping[str, Any], names: Sequence[str], cfg: Mapping[str, Any], seed: int) -> dict[str, Any]:
    descriptors: dict[str, Any] = {}
    for name in names:
        if name == "orbit4":
            descriptors[name] = poly["OrbitCounts"](graphlet_size=4)
        elif name == "orbit5":
            descriptors[name] = poly["OrbitCounts"](graphlet_size=5)
        elif name == "clustering":
            descriptors[name] = poly["ClusteringHistogram"](bins=int(cfg.get("clustering_bins", 100)))
        elif name == "degree":
            descriptors[name] = poly["SparseDegreeHistogram"]()
        elif name == "spectral":
            descriptors[name] = poly["EigenvalueHistogram"](n_bins=int(cfg.get("spectral_bins", 200)))
        elif name == "gin":
            descriptors[name] = poly["RandomGIN"](node_feat_loc=None, input_dim=1, edge_feat_loc=None, edge_feat_dim=0, seed=int(cfg.get("gin_seed", seed)), device=str(cfg.get("gin_device", "cpu")))
        else:
            raise ValueError(f"Unknown descriptor {name!r}")
    if not descriptors:
        raise ValueError("Descriptor set is empty")
    return descriptors


class PositiveClassProbabilityAdapter:
    """Adapter that makes predict_proba(X)[:, 1] mean P(y == 1).

    The official PolyGraph implementation indexes column 1 directly.  That is
    safe for scikit-learn classifiers whose classes_ are usually [0, 1], but
    some TabPFN versions may preserve first-seen label order.  Because
    PolyGraph trains with reference labels first (1) and generated labels second
    (0), such a classifier can expose classes_ == [1, 0].  Without this adapter,
    column 1 is then P(y == 0), which makes a strong discriminator look
    anti-correlated and the JS lower bound is clipped to 0.
    """

    def __init__(self, base: Any):
        self.base = base
        self.classes_ = None

    def fit(self, X: Any, y: Any) -> "PositiveClassProbabilityAdapter":
        self.base.fit(X, y)
        self.classes_ = getattr(self.base, "classes_", None)
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        proba = np.asarray(self.base.predict_proba(X))
        classes = getattr(self.base, "classes_", self.classes_)
        if classes is None:
            return proba
        classes_list = list(classes)
        if 1 in classes_list and proba.ndim == 2:
            pos_idx = classes_list.index(1)
            p1 = proba[:, pos_idx]
            # Return columns in the convention expected by PolyGraph internals:
            # column 0 = P(y == 0), column 1 = P(y == 1).
            return np.column_stack([1.0 - p1, p1])
        return proba

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base, name)


def make_classifier(name: str, cfg: Mapping[str, Any], seed: int, poly: Mapping[str, Any]) -> tuple[Any | None, str]:
    normalized = name.lower().replace("_", "-")
    if normalized in {"official-default", "default", "tabpfn", "tabpfn-v25"}:
        return PositiveClassProbabilityAdapter(poly["default_classifier"]()), "polygraph-benchmark default TabPFN + P(y=1) probability adapter"
    if normalized in {"logistic", "logistic-regression", "lr"}:
        from sklearn.linear_model import LogisticRegression

        base = LogisticRegression(max_iter=int(cfg.get("logistic_max_iter", 5000)), random_state=seed)
        return PositiveClassProbabilityAdapter(base), "sklearn LogisticRegression + P(y=1) probability adapter"
    if normalized == "auto":
        if importlib.util.find_spec("tabpfn") is not None:
            return PositiveClassProbabilityAdapter(poly["default_classifier"]()), "polygraph-benchmark default TabPFN(auto) + P(y=1) probability adapter"
        from sklearn.linear_model import LogisticRegression

        base = LogisticRegression(max_iter=int(cfg.get("logistic_max_iter", 5000)), random_state=seed)
        return PositiveClassProbabilityAdapter(base), "sklearn LogisticRegression(auto fallback) + P(y=1) probability adapter"
    raise ValueError(f"Unknown classifier {name!r}")


def interval_to_dict(x: Any) -> dict[str, float]:
    return {"mean": float(x.mean), "std": float(x.std)}


def format_point_result(raw: Mapping[str, Any], variant: str) -> dict[str, Any]:
    score = float(raw["pgd"])
    subscores = {str(k): float(v) for k, v in dict(raw.get("subscores", {})).items()}
    out: dict[str, Any] = {
        "pgs": score,
        "polygraphscore": score,
        "pgd": score,
        "pgs_descriptor": str(raw["pgd_descriptor"]),
        "pgd_descriptor": str(raw["pgd_descriptor"]),
        "subscores": subscores,
    }
    out["pgs_js_distance" if variant == "jsd" else "pgs_tv_informedness"] = score
    for k, v in subscores.items():
        out[f"pgs_subscore_{k}"] = v
    return out


def format_interval_result(raw: Mapping[str, Any], variant: str) -> dict[str, Any]:
    pgd = raw["pgd"]
    score = float(pgd.mean)
    sub = {str(k): interval_to_dict(v) for k, v in dict(raw.get("subscores", {})).items()}
    freq = {str(k): float(v) for k, v in dict(raw.get("pgd_descriptor", {})).items()}
    out: dict[str, Any] = {
        "pgs": score,
        "polygraphscore": score,
        "pgd": score,
        "pgs_mean": score,
        "pgs_std": float(pgd.std),
        "pgs_descriptor_frequency": freq,
        "subscores": sub,
    }
    out["pgs_js_distance" if variant == "jsd" else "pgs_tv_informedness"] = score
    out[("pgs_js_distance" if variant == "jsd" else "pgs_tv_informedness") + "_std"] = float(pgd.std)
    for k, v in sub.items():
        out[f"pgs_subscore_{k}"] = v["mean"]
        out[f"pgs_subscore_{k}_std"] = v["std"]
    for k, v in freq.items():
        out[f"pgs_descriptor_frequency_{k}"] = v
    return out


def compute_pgs(poly: Mapping[str, Any], ref: Sequence[Any], gen: Sequence[Any], descriptors: Mapping[str, Any], classifier: Any | None, cfg: Mapping[str, Any], seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
    variant = str(cfg.get("variant", "jsd")).lower()
    estimate = str(cfg.get("mode") or cfg.get("estimate", "point")).lower()
    if estimate == "subsampling":
        estimate = "interval"
    if variant not in {"jsd", "informedness"}:
        raise ValueError("variant must be jsd or informedness")
    if estimate == "point":
        num_splits = max(1, int(cfg.get("num_splits", 1) or 1))
        split_results = []
        for split_id in range(num_splits):
            split_seed = seed + split_id * 9973
            metric = poly["PolyGraphDiscrepancy"](reference_graphs=shuffle_graphs(ref, split_seed), descriptors=dict(descriptors), variant=variant, classifier=classifier)
            split_results.append(format_point_result(metric.compute(shuffle_graphs(gen, split_seed + 1)), variant))
        if num_splits == 1:
            results = split_results[0]
        else:
            values: dict[str, list[float]] = defaultdict(list)
            for item in split_results:
                for k, v in item.items():
                    if isinstance(v, (int, float, np.number)) and np.isfinite(float(v)):
                        values[k].append(float(v))
            results = dict(split_results[0])
            for k, vals in values.items():
                arr = np.asarray(vals)
                results[k] = float(arr.mean())
                results[f"{k}_split_std"] = float(arr.std(ddof=0))
        return results, {"estimate": "point", "official_class": "PolyGraphDiscrepancy", "num_splits": num_splits}

    subsample_size = int_or_none(cfg.get("subsample_size"))
    if subsample_size is None:
        subsample_size = min(2048, min(len(ref), len(gen)) // 2)
    num_samples = int(cfg.get("num_samples", cfg.get("num_subsamples", 10)) or 10)
    metric = poly["PolyGraphDiscrepancyInterval"](reference_graphs=ref, descriptors=dict(descriptors), variant=variant, classifier=classifier, subsample_size=subsample_size, num_samples=num_samples)
    return format_interval_result(metric.compute(gen), variant), {"estimate": "interval", "official_class": "PolyGraphDiscrepancyInterval", "subsample_size": subsample_size, "num_samples": num_samples}


def package_versions() -> dict[str, str | None]:
    out = {}
    for name in ["polygraph-benchmark", "orbit-count", "tabpfn", "torch-geometric", "networkx", "numpy", "scikit-learn"]:
        try:
            out[name] = importlib_metadata.version(name)
        except Exception:
            out[name] = None
    return out


def evaluate_one(dataset: str, model: str, run_id: int | None, cfg: Mapping[str, Any], root: Path, poly: Mapping[str, Any], descriptors_names: Sequence[str], classifier: Any | None, classifier_resolved: str, seed: int, output_path: Path | None) -> dict[str, Any]:
    started = time.perf_counter()
    dataset_root = str(cfg.get("dataset_root", "outputs/datasets"))
    samples_root = str(cfg.get("samples_root", "outputs/samples"))
    reference_split = str(cfg.get("reference_split", "test"))
    ref_path = dataset_split_path(root, dataset_root, dataset, reference_split)
    gen_path = sample_path(root, samples_root, dataset, model, run_id)
    ref_raw = load_graph_list(ref_path)
    gen_raw = load_graph_list(gen_path)
    ref_graphs = normalize_graphs(ref_raw, simple=not as_bool(cfg.get("keep_multigraph"), False), relabel=not as_bool(cfg.get("no_relabel"), False), drop_empty=as_bool(cfg.get("drop_empty_graphs"), False))
    gen_graphs = normalize_graphs(gen_raw, simple=not as_bool(cfg.get("keep_multigraph"), False), relabel=not as_bool(cfg.get("no_relabel"), False), drop_empty=as_bool(cfg.get("drop_empty_graphs"), False))
    max_ref = int_or_none(cfg.get("max_reference_graphs"))
    max_gen = int_or_none(cfg.get("max_generated_graphs"))
    max_graphs = int_or_none(cfg.get("max_graphs"))
    if max_graphs is not None:
        max_ref = min(max_ref, max_graphs) if max_ref is not None else max_graphs
        max_gen = min(max_gen, max_graphs) if max_gen is not None else max_graphs
    ref, gen = balance_graphs(ref_graphs, gen_graphs, max_ref=max_ref, max_gen=max_gen, seed=seed)
    descriptors = make_descriptors(poly, descriptors_names, cfg, seed)
    results, protocol_extra = compute_pgs(poly, ref, gen, descriptors, classifier, cfg, seed)
    numeric_scores = [float(v) for k, v in results.items() if k.startswith("pgs_subscore_") and isinstance(v, (int, float, np.number))]
    result_diagnostics = {}
    if numeric_scores and all(abs(v) < 1e-12 for v in numeric_scores):
        result_diagnostics["all_descriptor_scores_zero"] = True
        result_diagnostics["hint"] = "All descriptor scores are exactly zero. This can be genuine, but it often indicates classifier failures swallowed by official PolyGraph or a predict_proba class-column ordering issue. This wrapper uses a P(y=1) adapter to guard against the class-column issue."
    payload = {
        "dataset": dataset,
        "model": model,
        "run_id": run_id,
        "metric_family": "polygraphscore_official",
        "metric_name": "PolyGraphScore-JS" if str(cfg.get("variant", "jsd")).lower() == "jsd" else "PolyGraphScore-TV",
        "implementation_name_in_package": "PolyGraphDiscrepancy",
        "runtime_seconds": time.perf_counter() - started,
        "reference_path": str(ref_path),
        "generated_path": str(gen_path),
        "results": results,
        "protocol": {
            "source_config_key": cfg.get("_source_config_key"),
            "reference_split": reference_split,
            "variant": str(cfg.get("variant", "jsd")),
            "descriptors": list(descriptors.keys()),
            "classifier_requested": str(cfg.get("classifier", "official_default")),
            "classifier_resolved": classifier_resolved,
            "classifier_object": "polygraph-benchmark default" if classifier is None else classifier.__class__.__name__,
            "num_reference_graphs_loaded": len(ref_raw),
            "num_generated_graphs_loaded": len(gen_raw),
            "num_reference_graphs_used": len(ref),
            "num_generated_graphs_used": len(gen),
            "official_internal_fit_per_class": len(ref) // 2,
            "official_internal_test_per_class": len(ref) - len(ref) // 2,
            "official_cv_folds": 4,
            "seed": seed,
            "fit_test_and_cv": "Delegated to official polygraph-benchmark: fit/test split, 4-fold stratified CV descriptor selection on fit, and held-out test score.",
            "orbit_count_fallback": "orbit_count" in DEPENDENCY_SHIMS,
            "orca_exec": DEPENDENCY_SHIMS.get("orbit_count"),
            **protocol_extra,
        },
        "result_diagnostics": result_diagnostics,
        "dependency_shims": dict(DEPENDENCY_SHIMS),
        "polygraph_root_resolved": poly.get("polygraph_root_resolved"),
        "versions": package_versions(),
    }
    if output_path is not None:
        save_json(payload, output_path)
        log("saved dataset=%s model=%s run_id=%s pgs=%.6f -> %s", dataset, model, run_id, float(results.get("pgs", float("nan"))), output_path)
    return payload


def aggregate_payloads(payloads: Sequence[dict[str, Any]]) -> dict[str, Any]:
    values: dict[str, list[float]] = defaultdict(list)
    for payload in payloads:
        for k, v in dict(payload.get("results", {})).items():
            if isinstance(v, (int, float, np.number)) and np.isfinite(float(v)):
                values[k].append(float(v))
    flat: dict[str, Any] = {}
    nested: dict[str, Any] = {}
    for k, vals in values.items():
        arr = np.asarray(vals, dtype=float)
        flat[k] = float(arr.mean())
        flat[f"{k}_mean"] = float(arr.mean())
        flat[f"{k}_std"] = float(arr.std(ddof=0))
        nested[k] = {"mean": float(arr.mean()), "std": float(arr.std(ddof=0)), "num_runs": int(arr.size)}
    return {"flat": flat, "nested": nested}


def resolve_datasets_models(args: argparse.Namespace, experiment_cfg: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    if args.dataset and args.datasets:
        raise ValueError("Use either --dataset or --datasets, not both")
    if args.model and args.models:
        raise ValueError("Use either --model or --models, not both")
    datasets = [args.dataset] if args.dataset else (list(args.datasets) if args.datasets else list(experiment_cfg.get("datasets", [])))
    models = [args.model] if args.model else (list(args.models) if args.models else list(experiment_cfg.get("models", [])))
    if not datasets or not models:
        raise ValueError("No datasets/models selected; use CLI selectors or define datasets/models in config")
    return [str(x) for x in datasets], [str(x) for x in models]


def resolve_run_ids(args: argparse.Namespace, dataset: str, experiment_cfg: Mapping[str, Any]) -> list[int | None]:
    if args.run_id is not None and args.run_ids:
        raise ValueError("Use either --run-id or --run-ids, not both")
    if args.run_id is not None:
        return [int(args.run_id)]
    if args.run_ids:
        return [int(x) for x in args.run_ids]
    n = max(1, int(args.num_runs if args.num_runs is not None else dataset_num_runs(dataset, experiment_cfg)))
    return [None] if n == 1 else list(range(n))


def merge_cfg(args: argparse.Namespace, experiment_cfg: Mapping[str, Any]) -> dict[str, Any]:
    cfg = metric_cfg(experiment_cfg)
    cfg.setdefault("reference_split", "test")
    cfg.setdefault("classifier", "official_default")
    cfg.setdefault("variant", "jsd")
    # Accept both names. Existing survey config often uses `mode`; older
    # versions of this wrapper used `estimate`. Keep them synchronized.
    cfg.setdefault("mode", cfg.get("estimate", "point"))
    cfg.setdefault("estimate", cfg.get("mode", "point"))
    cfg.setdefault("descriptors", DEFAULT_DESCRIPTORS)
    cfg.setdefault("dataset_root", "outputs/datasets")
    cfg.setdefault("samples_root", "outputs/samples")
    cfg.setdefault("metrics_root", "outputs/metrics")
    if experiment_cfg.get("num_reference_graphs") is not None:
        cfg.setdefault("max_reference_graphs", int(experiment_cfg["num_reference_graphs"]))
    if experiment_cfg.get("num_generated_graphs") is not None:
        cfg.setdefault("max_generated_graphs", int(experiment_cfg["num_generated_graphs"]))
    for key in [
        "polygraph_root",
        "dataset_root",
        "samples_root",
        "metrics_root",
        "reference_split",
        "classifier",
        "variant",
        "mode",
        "estimate",
        "subsample_size",
        "num_samples",
        "num_splits",
        "max_graphs",
        "max_reference_graphs",
        "max_generated_graphs",
        "degree_bins",
        "clustering_bins",
        "spectral_bins",
        "max_degree",
        "gin_dim",
        "gin_device",
        "logistic_max_iter",
        "orca_exec",
        "cv_folds",
        "attribute_schema_enabled",
        "node_label_attr",
        "node_feature_attr",
        "edge_label_attr",
        "edge_feature_attr",
        "graph_label_attr",
    ]:
        value = getattr(args, key, None)
        if value is not None:
            cfg[key] = value
    if getattr(args, "device", None) is not None and cfg.get("gin_device") is None:
        cfg["gin_device"] = args.device
    if args.descriptors:
        cfg["descriptors"] = args.descriptors
    if args.skip_orbits or getattr(args, "skip_orbit", False):
        cfg["skip_orbits"] = True
    if getattr(args, "no_attribute_descriptor", False):
        cfg["no_attribute_descriptor"] = True
    if args.skip_gin:
        cfg["skip_gin"] = True
    cfg["seed"] = int(args.seed if args.seed is not None else experiment_cfg.get("seed", 42))
    return cfg


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute official PolyGraphScore for survey datasets/models/run ids")
    p.add_argument("--survey-root", default=None)
    p.add_argument("--experiment-config", default="configs/experiment.yaml")
    p.add_argument("--dataset", default=None)
    p.add_argument("--datasets", nargs="+", default=None)
    p.add_argument("--model", default=None)
    p.add_argument("--models", nargs="+", default=None)
    p.add_argument("--run-id", type=int, default=None)
    p.add_argument("--run-ids", type=int, nargs="+", default=None)
    p.add_argument("--num-runs", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    # Optional overrides; normally place these in metrics.polygraphscore_official.
    p.add_argument("--polygraph-root", default=None, help="Optional override; normally use POLYGRAPH_REPO instead")
    p.add_argument("--dataset-root", default=None)
    p.add_argument("--samples-root", default=None)
    p.add_argument("--metrics-root", default=None)
    p.add_argument("--reference-split", choices=["train", "val", "test"], default=None)
    p.add_argument("--classifier", choices=["official_default", "default", "auto", "tabpfn", "tabpfn-v25", "logistic_regression", "logistic", "lr"], default=None)
    p.add_argument("--variant", choices=["jsd", "informedness"], default=None)
    p.add_argument("--estimate", choices=["point", "interval", "subsampling"], default=None)
    p.add_argument("--mode", dest="estimate", choices=["point", "interval", "subsampling"], default=None, help="Alias for --estimate")
    p.add_argument("--descriptors", nargs="+", default=None)
    p.add_argument("--skip-orbit", action="store_true", help="Backward-compatible alias for --skip-orbits")
    p.add_argument("--skip-orbits", action="store_true")
    p.add_argument("--skip-gin", action="store_true")
    p.add_argument("--max-graphs", type=int, default=None)
    p.add_argument("--max-reference-graphs", type=int, default=None)
    p.add_argument("--max-generated-graphs", type=int, default=None)
    p.add_argument("--num-splits", type=int, default=None, help="Repeated official point estimates for this sampled graph set.")
    p.add_argument("--cv-folds", type=int, default=None, help="Accepted for compatibility; official polygraph-benchmark currently controls CV internally.")
    p.add_argument("--degree-bins", type=int, default=None, help="Accepted for compatibility; official sparse-degree descriptor controls its own bins.")
    p.add_argument("--clustering-bins", type=int, default=None)
    p.add_argument("--spectral-bins", type=int, default=None)
    p.add_argument("--max-degree", type=int, default=None, help="Accepted for compatibility; official sparse-degree descriptor controls its own support.")
    p.add_argument("--gin-dim", type=int, default=None, help="Accepted for compatibility; official RandomGIN controls its output dimension.")
    p.add_argument("--no-attribute-descriptor", action="store_true", help="Accepted for compatibility; official descriptor set has no benchmark attribute descriptor.")
    p.add_argument("--attribute-schema-enabled", choices=["auto", "true", "false"], default=None, help="Accepted for compatibility; official PGS ignores benchmark attribute schema.")
    p.add_argument("--node-label-attr", default=None, help="Accepted for compatibility; official PGS ignores benchmark attribute schema.")
    p.add_argument("--node-feature-attr", default=None, help="Accepted for compatibility; official PGS ignores benchmark attribute schema.")
    p.add_argument("--edge-label-attr", default=None, help="Accepted for compatibility; official PGS ignores benchmark attribute schema.")
    p.add_argument("--edge-feature-attr", default=None, help="Accepted for compatibility; official PGS ignores benchmark attribute schema.")
    p.add_argument("--graph-label-attr", default=None, help="Accepted for compatibility; official PGS ignores benchmark attribute schema.")
    p.add_argument("--device", default=None, help="Compatibility alias for --gin-device when --gin-device is not set.")
    p.add_argument("--subsample-size", type=int, default=None)
    p.add_argument("--num-samples", type=int, default=None)
    p.add_argument("--gin-device", default=None)
    p.add_argument("--logistic-max-iter", type=int, default=None)
    p.add_argument("--orca-exec", default=None, help="Optional override; normally use ORCA_EXEC instead")
    p.add_argument("--output", default=None, help="Only for exactly one dataset/model/run")
    p.add_argument("--force", action="store_true")
    p.add_argument("--continue-on-error", action="store_true", default=None)
    p.add_argument("--fail-fast", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = infer_survey_root(args.survey_root)
    cfg_path = resolve_path(args.experiment_config, base=root)
    assert cfg_path is not None
    experiment_cfg = load_yaml(cfg_path)
    cfg = merge_cfg(args, experiment_cfg)
    datasets, models = resolve_datasets_models(args, experiment_cfg)
    seed = int(cfg.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    descriptors = canonical_descriptors(cfg)
    filename = str(cfg.get("metric_filename", DEFAULT_METRIC_FILENAME))
    metrics_root = str(cfg.get("metrics_root", "outputs/metrics"))
    force = bool(args.force or experiment_cfg.get("force", False) or cfg.get("force", False))
    continue_on_error = bool(experiment_cfg.get("continue_on_error", True)) if args.continue_on_error is None else bool(args.continue_on_error)
    if args.fail_fast:
        continue_on_error = False

    # A dry run should only expand the dataset/model/run matrix and output paths;
    # it should not require optional polygraph dependencies such as TabPFN,
    # torch-geometric, or orbit-count to be installed.
    poly: Mapping[str, Any] | None = None
    classifier: Any | None = None
    classifier_resolved = "not constructed in --dry-run"
    if not args.dry_run:
        descriptors = prepare_imports(cfg, descriptors, str(cfg.get("classifier", "official_default")), survey_root=root)
        poly = import_polygraph_objects(str(cfg.get("polygraph_root")) if cfg.get("polygraph_root") else None, survey_root=root)
        classifier, classifier_resolved = make_classifier(str(cfg.get("classifier", "official_default")), cfg, seed, poly)

    batch_records: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for dataset in datasets:
        run_ids = resolve_run_ids(args, dataset, experiment_cfg)
        for model in models:
            payloads: list[dict[str, Any]] = []
            for run_id in run_ids:
                out_path = Path(args.output) if args.output and len(datasets) == len(models) == len(run_ids) == 1 else metric_path(root, metrics_root, dataset, model, filename, run_id)
                if out_path.exists() and not force:
                    log("skip existing %s", out_path)
                    try:
                        payloads.append(json.loads(out_path.read_text(encoding="utf-8")))
                    except Exception:
                        pass
                    continue
                if args.dry_run:
                    log("would evaluate dataset=%s model=%s run_id=%s -> %s", dataset, model, run_id, out_path)
                    continue
                try:
                    assert poly is not None
                    payload = evaluate_one(dataset, model, run_id, cfg, root, poly, descriptors, classifier, classifier_resolved, run_seed(seed, run_id), out_path)
                    payloads.append(payload)
                    batch_records.append({"dataset": dataset, "model": model, "run_id": run_id, "pgs": payload["results"].get("pgs"), "metric_path": str(out_path)})
                except Exception as exc:
                    errors.append({"dataset": dataset, "model": model, "run_id": run_id, "error": repr(exc)})
                    log("ERROR dataset=%s model=%s run_id=%s: %s", dataset, model, run_id, exc)
                    if not continue_on_error:
                        raise
            if len(payloads) > 1 and not args.dry_run:
                agg = aggregate_payloads(payloads)
                agg_payload = {"dataset": dataset, "model": model, "metric_family": "polygraphscore_official", "is_aggregate": True, "run_ids": [p.get("run_id") for p in payloads], "num_runs": len(payloads), "results": agg["flat"], "run_result_summary": agg["nested"]}
                agg_path = aggregate_path(root, metrics_root, dataset, model, filename)
                save_json(agg_payload, agg_path)
                log("saved aggregate -> %s", agg_path)
    if not args.dry_run:
        save_json({"metric_family": "polygraphscore_official", "datasets": datasets, "models": models, "descriptors": descriptors, "classifier_resolved": classifier_resolved, "dependency_shims": DEPENDENCY_SHIMS, "records": batch_records, "errors": errors}, batch_path(root, metrics_root, filename))
    return 0 if not errors or continue_on_error else 1


if __name__ == "__main__":
    raise SystemExit(main())
