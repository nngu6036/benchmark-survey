from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence

import networkx as nx
import numpy as np

from empirical_comparison.datasets.base import BaseDatasetBuilder


def _optional_int(value: Any, default: int | None = None) -> int | None:
    if value is None or value == "":
        return default
    return int(value)


def _as_numpy(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    try:
        arr = np.asarray(value)
    except Exception:
        return None
    if arr.size == 0:
        return None
    return arr


class _MolecularDatasetBuilder(BaseDatasetBuilder):
    """Common PyG-to-NetworkX conversion for molecular graph datasets.

    The benchmark uses a canonical NetworkX schema:
    - node_label: categorical atom label index or atomic number before canonicalization;
    - feats: numeric atom feature vector;
    - edge_type: categorical bond label, with 0 reserved for dense no-edge states;
    - edge_attr: numeric bond feature vector;
    - graph.graph["molecular_target"]: optional regression target vector.
    """

    pyg_dataset_name = "molecular"

    def __init__(self, config: dict[str, Any], root: str | Path = "outputs/datasets") -> None:
        super().__init__(config, root=root)
        default_pyg_root = Path("outputs/raw_datasets") / self.pyg_dataset_name
        self.pyg_root = Path(config.get("pyg_root") or config.get("download_root") or default_pyg_root)
        self.max_graphs = _optional_int(config.get("max_graphs", config.get("num_graphs")), None)
        self.max_train_graphs = _optional_int(config.get("max_train_graphs"), None)
        self.max_val_graphs = _optional_int(config.get("max_val_graphs"), None)
        self.max_test_graphs = _optional_int(config.get("max_test_graphs"), None)
        self.include_node_features = bool(config.get("include_node_features", True))
        self.include_edge_features = bool(config.get("include_edge_features", True))
        self.include_positions = bool(config.get("include_positions", False))
        self.include_targets = bool(config.get("include_targets", True))
        self.target_index = _optional_int(config.get("target_index"), None)
        filter_cfg = (config.get("preprocessing") or {}).get("graph_size_filter", {})
        self.min_nodes = _optional_int(config.get("min_nodes", filter_cfg.get("min_nodes")), None)
        self.max_nodes = _optional_int(config.get("max_nodes", filter_cfg.get("max_nodes")), None)
        self.shuffle = bool(config.get("shuffle", True))

    def _import_pyg_dataset(self, class_name: str):
        try:
            from torch_geometric import datasets as pyg_datasets  # type: ignore
        except ModuleNotFoundError as exc:  # pragma: no cover - depends on optional install.
            raise ModuleNotFoundError(
                "QM9/ZINC dataset preparation requires PyTorch Geometric. Install `torch-geometric` "
                "or use `pip install -r requirements.txt`."
            ) from exc
        try:
            return getattr(pyg_datasets, class_name)
        except AttributeError as exc:  # pragma: no cover - depends on PyG version.
            raise AttributeError(f"torch_geometric.datasets does not provide {class_name}.") from exc

    def _take_indices(self, n: int, limit: int | None, *, seed_offset: int = 0) -> list[int]:
        indices = list(range(n))
        if self.shuffle:
            rng = np.random.default_rng(self.seed + int(seed_offset))
            rng.shuffle(indices)
        if limit is not None and limit > 0:
            indices = indices[: min(int(limit), len(indices))]
        return indices

    def _filter_graph(self, graph: nx.Graph) -> bool:
        n = graph.number_of_nodes()
        if self.min_nodes is not None and n < self.min_nodes:
            return False
        if self.max_nodes is not None and n > self.max_nodes:
            return False
        return True

    def _convert_many(self, dataset: Sequence[Any], *, limit: int | None, seed_offset: int = 0) -> list[nx.Graph]:
        graphs: list[nx.Graph] = []
        for idx in self._take_indices(len(dataset), limit, seed_offset=seed_offset):
            graph = self._data_to_networkx(dataset[idx])
            graph = self.normalize_graph(graph)
            if self._filter_graph(graph):
                graphs.append(graph)
        return graphs

    def _node_label(self, i: int, *, x: np.ndarray | None, z: np.ndarray | None) -> int:
        if z is not None and i < int(z.reshape(-1).shape[0]):
            return int(z.reshape(-1)[i])
        if x is not None and i < x.shape[0]:
            xi = np.asarray(x[i]).reshape(-1)
            if xi.size == 1:
                return int(round(float(xi[0])))
            if xi.size > 1:
                return int(np.argmax(xi))
        return 0

    def _node_features(self, i: int, *, x: np.ndarray | None, pos: np.ndarray | None, label: int) -> list[float]:
        values: list[float] = []
        if self.include_node_features and x is not None and i < x.shape[0]:
            values.extend(float(v) for v in np.asarray(x[i], dtype=np.float64).reshape(-1))
        if self.include_positions and pos is not None and i < pos.shape[0]:
            values.extend(float(v) for v in np.asarray(pos[i], dtype=np.float64).reshape(-1))
        if not values and self.include_node_features:
            values = [float(label)]
        return values

    @staticmethod
    def _edge_label(edge_attr_row: np.ndarray | None) -> int:
        if edge_attr_row is None:
            return 1
        arr = np.asarray(edge_attr_row).reshape(-1)
        if arr.size == 0:
            return 1
        if arr.size == 1:
            value = int(round(float(arr[0])))
            return value if value > 0 else value + 1
        return int(np.argmax(arr)) + 1

    def _data_to_networkx(self, data: Any) -> nx.Graph:
        x = _as_numpy(getattr(data, "x", None))
        z = _as_numpy(getattr(data, "z", None))
        pos = _as_numpy(getattr(data, "pos", None))
        edge_index = _as_numpy(getattr(data, "edge_index", None))
        edge_attr = _as_numpy(getattr(data, "edge_attr", None))
        y = _as_numpy(getattr(data, "y", None))

        num_nodes = int(getattr(data, "num_nodes", 0) or 0)
        if num_nodes <= 0:
            if x is not None:
                num_nodes = int(x.shape[0])
            elif z is not None:
                num_nodes = int(z.reshape(-1).shape[0])
        graph = nx.Graph()
        for i in range(num_nodes):
            label = self._node_label(i, x=x, z=z)
            node_payload: dict[str, Any] = {
                "node_label": label,
                "feats": self._node_features(i, x=x, pos=pos, label=label),
            }
            if z is not None and i < int(z.reshape(-1).shape[0]):
                node_payload["atomic_number"] = int(z.reshape(-1)[i])
            graph.add_node(i, **node_payload)

        edge_payloads: dict[tuple[int, int], dict[str, Any]] = {}
        if edge_index is not None and edge_index.size:
            edge_index = np.asarray(edge_index, dtype=np.int64)
            if edge_index.shape[0] != 2 and edge_index.shape[-1] == 2:
                edge_index = edge_index.T
            num_edges = int(edge_index.shape[1]) if edge_index.ndim == 2 else 0
            for k in range(num_edges):
                u, v = int(edge_index[0, k]), int(edge_index[1, k])
                if u == v or u < 0 or v < 0 or u >= num_nodes or v >= num_nodes:
                    continue
                a, b = (u, v) if u < v else (v, u)
                if (a, b) in edge_payloads:
                    continue
                attr_row = None
                if edge_attr is not None and k < edge_attr.shape[0]:
                    attr_row = np.asarray(edge_attr[k])
                payload: dict[str, Any] = {"edge_type": self._edge_label(attr_row)}
                if self.include_edge_features and attr_row is not None:
                    payload["edge_attr"] = [float(vv) for vv in attr_row.reshape(-1)]
                edge_payloads[(a, b)] = payload

        for (u, v), attrs in edge_payloads.items():
            graph.add_edge(u, v, **attrs)

        if self.include_targets and y is not None:
            y_flat = np.asarray(y, dtype=np.float64).reshape(-1)
            graph.graph["molecular_target"] = [float(v) for v in y_flat]
            if self.target_index is not None and 0 <= self.target_index < y_flat.size:
                graph.graph["molecular_target_value"] = float(y_flat[self.target_index])
        graph.graph["source_dataset"] = self.pyg_dataset_name
        return graph


class QM9DatasetBuilder(_MolecularDatasetBuilder):
    pyg_dataset_name = "qm9"

    def build(self) -> dict[str, list[nx.Graph]]:
        QM9 = self._import_pyg_dataset("QM9")
        dataset = QM9(root=str(self.pyg_root))
        graphs = self._convert_many(dataset, limit=self.max_graphs, seed_offset=0)
        return self.split_graphs(graphs)


class ZINCDatasetBuilder(_MolecularDatasetBuilder):
    pyg_dataset_name = "zinc"

    def build(self) -> dict[str, list[nx.Graph]]:
        ZINC = self._import_pyg_dataset("ZINC")
        subset = bool(self.config.get("subset", True))
        use_official_splits = bool(self.config.get("use_official_splits", True))
        if use_official_splits:
            limits = {"train": self.max_train_graphs, "val": self.max_val_graphs, "test": self.max_test_graphs}
            if self.max_graphs is not None and not any(v is not None for v in limits.values()):
                split_cfg = self.config.get("split", {"train": 0.8, "val": 0.1, "test": 0.1})
                total = int(self.max_graphs)
                limits["train"] = int(total * float(split_cfg.get("train", 0.8)))
                limits["val"] = int(total * float(split_cfg.get("val", 0.1)))
                limits["test"] = max(0, total - int(limits["train"] or 0) - int(limits["val"] or 0))
            return {
                split: self._convert_many(
                    ZINC(root=str(self.pyg_root), subset=subset, split=split),
                    limit=limits[split],
                    seed_offset={"train": 0, "val": 10_000, "test": 20_000}[split],
                )
                for split in ("train", "val", "test")
            }

        all_graphs: list[nx.Graph] = []
        for split, offset in (("train", 0), ("val", 10_000), ("test", 20_000)):
            all_graphs.extend(
                self._convert_many(
                    ZINC(root=str(self.pyg_root), subset=subset, split=split),
                    limit=None,
                    seed_offset=offset,
                )
            )
        if self.shuffle:
            rng = np.random.default_rng(self.seed)
            rng.shuffle(all_graphs)
        if self.max_graphs is not None:
            all_graphs = all_graphs[: int(self.max_graphs)]
        return self.split_graphs(all_graphs)
