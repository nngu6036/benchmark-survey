from __future__ import annotations

import contextlib
import importlib
import json
import os
import random
import shutil
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import torch
from omegaconf import OmegaConf
from pytorch_lightning import Trainer

from empirical_comparison.models.base import BaseGenerator
from empirical_comparison.utils.logging import get_logger
from empirical_comparison.utils.progress import update_progress


LOGGER = get_logger(__name__)


class _NoOpSamplingMetrics(torch.nn.Module):
    """Fallback for DiGress sampling metrics when graph_tool/ORCA is unavailable.

    DiGress imports graph-tool through its SPECTRE sampling metrics. The benchmark
    wrapper can still train and sample generic graphs without those metrics, because
    benchmark-level metrics are computed separately by this repository.
    """

    def __init__(self, reason: str) -> None:
        super().__init__()
        self.reason = reason

    def reset(self) -> None:
        return None

    def forward(self, *args, **kwargs) -> dict[str, Any]:
        return {"disabled": True, "reason": self.reason}


class _BenchmarkMolecularInfos:
    """Dataset info object for benchmark-owned molecular splits.

    This mirrors the attributes DiGress needs from upstream molecular infos
    without importing dataset modules that may require RDKit even when benchmark
    molecular data has already been prepared from PyG archives.
    """

    need_to_strip = False

    def __init__(self, datamodule: Any, cfg: Any, distribution_nodes: Any, utils_mod: Any) -> None:
        self.name = str(cfg.dataset.name)
        self.remove_h = bool(cfg.dataset.remove_h)
        self._distribution_nodes = distribution_nodes
        self.n_nodes = datamodule.node_counts(max_nodes_possible=256)
        self.node_types = datamodule.node_types()
        self.edge_types = datamodule.edge_counts()
        self.num_atom_types = int(self.node_types.shape[0])
        if self.name == "qm9" and self.remove_h:
            self.atom_decoder = ["C", "N", "O", "F"]
            self.valencies = [4, 3, 2, 1][: self.num_atom_types]
            qm9_weights = [12, 14, 16, 19]
            self.atom_weights = {i: qm9_weights[i] for i in range(self.num_atom_types)}
        elif self.name == "qm9":
            self.atom_decoder = ["H", "C", "N", "O", "F"]
            self.valencies = [1, 4, 3, 2, 1][: self.num_atom_types]
            qm9_weights = [1, 12, 14, 16, 19]
            self.atom_weights = {i: qm9_weights[i] for i in range(self.num_atom_types)}
        else:
            self.atom_decoder = [f"Atom{i}" for i in range(self.num_atom_types)]
            self.valencies = [4 for _ in range(self.num_atom_types)]
            self.atom_weights = {i: 1 for i in range(self.num_atom_types)}
        self.atom_encoder = {atom: i for i, atom in enumerate(self.atom_decoder)}
        self.max_n_nodes = len(self.n_nodes) - 1
        self.max_weight = max(1, self.max_n_nodes * max(self.atom_weights.values()))
        self.valency_distribution = datamodule.valency_count(self.max_n_nodes)
        self.complete_infos(n_nodes=self.n_nodes, node_types=self.node_types)

    def complete_infos(self, n_nodes, node_types) -> None:
        self.input_dims = None
        self.output_dims = None
        self.num_classes = len(node_types)
        self.max_n_nodes = len(n_nodes) - 1
        self.nodes_dist = self._distribution_nodes(n_nodes)

    def compute_input_output_dims(self, datamodule, extra_features, domain_features) -> None:
        example_batch = next(iter(datamodule.train_dataloader()))
        utils_mod = importlib.import_module("src.utils")
        ex_dense, node_mask = utils_mod.to_dense(
            example_batch.x,
            example_batch.edge_index,
            example_batch.edge_attr,
            example_batch.batch,
        )
        example_data = {"X_t": ex_dense.X, "E_t": ex_dense.E, "y_t": example_batch["y"], "node_mask": node_mask}

        self.input_dims = {
            "X": example_batch["x"].size(1),
            "E": example_batch["edge_attr"].size(1),
            "y": example_batch["y"].size(1) + 1,
        }
        ex_extra_feat = extra_features(example_data)
        self.input_dims["X"] += ex_extra_feat.X.size(-1)
        self.input_dims["E"] += ex_extra_feat.E.size(-1)
        self.input_dims["y"] += ex_extra_feat.y.size(-1)

        ex_extra_molecular_feat = domain_features(example_data)
        self.input_dims["X"] += ex_extra_molecular_feat.X.size(-1)
        self.input_dims["E"] += ex_extra_molecular_feat.E.size(-1)
        self.input_dims["y"] += ex_extra_molecular_feat.y.size(-1)

        self.output_dims = {
            "X": example_batch["x"].size(1),
            "E": example_batch["edge_attr"].size(1),
            "y": 0,
        }


class DiGressWrapper(BaseGenerator):
    """Benchmark adapter for the upstream DiGress repository.

    The upstream DiGress code is a Hydra/PyTorch-Lightning project. Its ``main.py``
    imports ``graph_tool`` at module load time, so this wrapper intentionally does
    not import or call ``main.py``. Instead, it mirrors the synthetic-graph branch
    of ``main.py`` and directly constructs:

    * ``SpectreGraphDataModule``
    * ``SpectreDatasetInfos``
    * ``TrainAbstractMetricsDiscrete``
    * ``ExtraFeatures`` / ``DummyExtraFeatures``
    * ``DiscreteDenoisingDiffusion``

    For featureless synthetic datasets (``sbm``, ``planar``, ``comm20``), it
    materializes benchmark NetworkX splits into DiGress's SPECTRE raw format:
    ``raw/train.pt``, ``raw/val.pt`` and ``raw/test.pt``, each storing a list of
    dense binary adjacency tensors. For ``qm9``, it converts benchmark NetworkX
    molecules into PyG ``Data`` objects with one-hot atom and bond classes and
    uses DiGress's molecular data path.

    Important config keys
    ---------------------
    repo_root: str, optional
        Path to the extracted DiGress repository. It may point either to the
        directory that contains ``src/`` and ``configs/`` or to a parent directory
        containing a ``DiGress/`` subdirectory. If omitted, ``DIGRESS_REPO`` is
        used, then ``external/DiGress`` under the benchmark root.
    dataset / dataset_name: str
        Benchmark dataset name. ``dataset`` is injected by the benchmark scripts;
        ``dataset_name`` can override it. Supported values: ``sbm``, ``planar``,
        ``comm20``, ``qm9``, ``zinc``.
    checkpoint_path: str
        Path used by ``load`` and ``train``. Relative paths are resolved against
        the current working directory.
    data_subdir: str, optional
        Directory where DiGress raw/processed graph tensors are stored. Relative
        paths are resolved against the current working directory, not against the
        external DiGress repo. Defaults to ``outputs/digress_data/<dataset>``.
    num_epochs, batch_size, learning_rate, num_workers, gpus, seed: optional
        Common training controls mapped into the DiGress config.
    model_overrides, train_overrides, general_overrides: dict, optional
        Nested overrides applied to the corresponding DiGress config groups.

    Notes
    -----
    * Node and edge attributes from NetworkX graphs are ignored for SPECTRE
      datasets; that path uses one dummy node type and two edge classes:
      no-edge and edge.
    * For molecular datasets, ``node_label`` and ``edge_type`` are consumed as
      atom and bond classes. Bond class 0 is reserved for dense no-edge states.
    * DiGress samples by iterating all ``cfg.model.diffusion_steps``. The benchmark
      ``sampling.num_steps`` field is therefore not used to shorten sampling unless
      you explicitly set ``model_overrides.diffusion_steps`` before training.
    * Upstream SPECTRE sampling metrics are optional in this wrapper; benchmark
      metrics are computed by separate scripts.
    """

    supports_training = True
    supports_sampling = True
    supports_node_features = True
    supports_edge_features = False
    supports_node_labels = True
    supports_edge_labels = True
    supports_constraints = False
    supports_variable_size = True
    supports_featureless_graphs = True

    SPECTRE_DATASETS = {"sbm", "planar", "comm20"}
    MOLECULAR_DATASETS = {"qm9", "zinc"}
    SUPPORTED_DATASETS = SPECTRE_DATASETS | MOLECULAR_DATASETS

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self.device = config.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset_name = str(config.get("dataset_name") or config.get("dataset") or "sbm").lower()
        if self.dataset_name not in self.SUPPORTED_DATASETS:
            raise ValueError(
                f"DiGressWrapper supports only {sorted(self.SUPPORTED_DATASETS)}; "
                f"got dataset={self.dataset_name!r}."
            )
        self.is_molecular = self.dataset_name in self.MOLECULAR_DATASETS
        self.detailed_logging = bool(config.get("detailed_logging", True))
        self.log_train_every_n_steps = max(1, int(config.get("log_train_every_n_steps", 1)))
        self.log_sample_every_n_batches = max(1, int(config.get("log_sample_every_n_batches", 1)))
        self._molecular_atom_class_count: int | None = None

        default_repo_root = Path(__file__).resolve().parents[4] / "external" / "DiGress"
        repo_root = os.environ.get("DIGRESS_REPO") or config.get("repo_root") or default_repo_root
        self.repo_root = self._normalize_repo_root(Path(repo_root).expanduser())
        self.repo_src = self.repo_root / "src"

        self.checkpoint_path = Path(config.get("checkpoint_path", f"checkpoints/digress_{self.dataset_name}.ckpt"))
        self.checkpoint_path = self.checkpoint_path.expanduser().resolve()

        data_subdir = config.get("data_subdir") or str(Path("outputs") / "digress_data" / self.dataset_name)
        data_subdir = str(data_subdir).replace("${dataset}", self.dataset_name).replace("${model}", "digress")
        self.data_root = Path(data_subdir).expanduser()
        if not self.data_root.is_absolute():
            self.data_root = Path.cwd() / self.data_root
        self.data_root = self.data_root.resolve()

        self.model = None
        self.cfg = None
        self.datamodule = None
        self.dataset_infos = None
        self.repo_loaded = False
        self._imports: dict[str, Any] = {}

    @property
    def name(self) -> str:
        return "digress"

    # ------------------------------------------------------------------
    # Repository loading and DiGress config construction
    # ------------------------------------------------------------------
    def _normalize_repo_root(self, repo_root: Path) -> Path:
        repo_root = repo_root.resolve()
        if repo_root.suffix == ".zip":
            raise ValueError(
                "DiGressWrapper expects an extracted DiGress repository directory, not a zip file. "
                "Unzip it and set `repo_root` or DIGRESS_REPO to the extracted DiGress directory."
            )
        if repo_root.name == "src" and (repo_root / "utils.py").exists():
            repo_root = repo_root.parent
        if not (repo_root / "src").exists() and (repo_root / "DiGress" / "src").exists():
            repo_root = repo_root / "DiGress"
        return repo_root.resolve()

    def _ensure_repo_layout(self) -> None:
        missing = []
        for rel in ["src", "configs", "src/diffusion_model_discrete.py", "src/datasets/spectre_dataset.py"]:
            if not (self.repo_root / rel).exists():
                missing.append(rel)
        if missing:
            raise FileNotFoundError(
                "Invalid DiGress repo_root. Missing: "
                + ", ".join(missing)
                + f". repo_root={self.repo_root}. Set DIGRESS_REPO or `repo_root` in configs/models/digress.yaml."
            )

    def _ensure_repo_importable(self) -> None:
        # DiGress modules use both package-style imports (src.utils) and local
        # top-level imports (models.*, metrics.*, diffusion.*). Both paths are needed.
        for p in (str(self.repo_root), str(self.repo_src)):
            if p in sys.path:
                sys.path.remove(p)
        sys.path.insert(0, str(self.repo_root))
        sys.path.insert(0, str(self.repo_src))

    def _import_modules(self) -> None:
        if self.repo_loaded:
            return
        started_at = time.perf_counter()
        self._log("importing upstream DiGress modules dataset=%s", self.dataset_name)
        self._ensure_repo_layout()
        self._ensure_repo_importable()
        try:
            self._imports["datasets_spectre"] = importlib.import_module("src.datasets.spectre_dataset")
            self._imports["metrics_abstract"] = importlib.import_module("src.metrics.abstract_metrics")
            self._imports["extra_features"] = importlib.import_module("src.diffusion.extra_features")
            self._imports["diffusion_model_discrete"] = importlib.import_module("src.diffusion_model_discrete")
            if self.is_molecular:
                self._imports["datasets_abstract"] = importlib.import_module("src.datasets.abstract_dataset")
                self._imports["digress_utils"] = importlib.import_module("src.utils")
                self._imports["distributions"] = importlib.import_module("src.diffusion.distributions")
                self._imports["metrics_molecular_discrete"] = importlib.import_module(
                    "src.metrics.molecular_metrics_discrete"
                )
                self._imports["extra_features_molecular"] = importlib.import_module(
                    "src.diffusion.extra_features_molecular"
                )
        except ModuleNotFoundError as exc:
            missing = getattr(exc, "name", "") or str(exc)
            raise ModuleNotFoundError(
                "Could not import the upstream DiGress modules. Install DiGress dependencies "
                "(notably torch_geometric, pytorch_lightning, wandb, hydra-core/omegaconf, and optionally graph-tool) "
                f"and set DIGRESS_REPO correctly. Missing module: {missing!r}."
            ) from exc
        self._patch_spectre_dataset_process()
        self.repo_loaded = True
        self._log("imported upstream DiGress modules duration=%.3fs", time.perf_counter() - started_at)

    def _patch_spectre_dataset_process(self) -> None:
        """Patch a small upstream SPECTRE processing issue at runtime.

        The uploaded DiGress repository's ``SpectreGraphDataset.process`` appends
        each graph once before filtering/transforms and once after, effectively
        duplicating data. This benchmark wrapper replaces it with an equivalent
        one-pass implementation while leaving the external repository untouched.
        """
        spectre = self._imports["datasets_spectre"]
        if getattr(spectre.SpectreGraphDataset, "_empirical_comparison_process_patch", False):
            return

        def process(dataset_self):
            import torch_geometric.utils
            from torch_geometric.data import Data

            file_idx = {"train": 0, "val": 1, "test": 2}
            raw_dataset = torch.load(dataset_self.raw_paths[file_idx[dataset_self.split]])
            data_list = []
            for adj in raw_dataset:
                adj = torch.as_tensor(adj, dtype=torch.float)
                if adj.dim() != 2 or adj.shape[0] != adj.shape[1]:
                    raise ValueError(f"Expected square adjacency matrix, got shape={tuple(adj.shape)}")
                n = int(adj.shape[-1])
                X = torch.ones(n, 1, dtype=torch.float)
                y = torch.zeros([1, 0], dtype=torch.float)
                edge_index, _ = torch_geometric.utils.dense_to_sparse(adj)
                edge_attr = torch.zeros(edge_index.shape[-1], 2, dtype=torch.float)
                if edge_attr.numel() > 0:
                    edge_attr[:, 1] = 1
                num_nodes = n * torch.ones(1, dtype=torch.long)
                data = Data(x=X, edge_index=edge_index, edge_attr=edge_attr, y=y, n_nodes=num_nodes)
                if dataset_self.pre_filter is not None and not dataset_self.pre_filter(data):
                    continue
                if dataset_self.pre_transform is not None:
                    data = dataset_self.pre_transform(data)
                data_list.append(data)
            torch.save(dataset_self.collate(data_list), dataset_self.processed_paths[0])

        spectre.SpectreGraphDataset.process = process
        spectre.SpectreGraphDataset._empirical_comparison_process_patch = True

    def _default_cfg(self) -> Any:
        cfg_dir = self.repo_root / "configs"
        general = OmegaConf.load(cfg_dir / "general" / "general_default.yaml")
        model = OmegaConf.load(cfg_dir / "model" / "discrete.yaml")
        train = OmegaConf.load(cfg_dir / "train" / "train_default.yaml")
        dataset_cfg_path = cfg_dir / "dataset" / f"{self.dataset_name}.yaml"
        if not dataset_cfg_path.exists() and self.is_molecular:
            dataset_cfg_path = cfg_dir / "dataset" / "qm9.yaml"
        dataset = OmegaConf.load(dataset_cfg_path)

        cfg = OmegaConf.create({})
        cfg.general = general
        cfg.model = model
        cfg.train = train
        cfg.dataset = dataset

        cfg.general.name = str(self.config.get("experiment_name", f"digress_{self.dataset_name}"))
        cfg.general.wandb = str(self.config.get("wandb", "disabled"))
        cfg.general.gpus = int(self.config.get("gpus", 0 if self.device == "cpu" else 1))
        cfg.general.resume = None
        cfg.general.test_only = None
        cfg.general.evaluate_all_checkpoints = False
        cfg.general.check_val_every_n_epochs = int(self.config.get("check_val_every_n_epochs", 10))
        cfg.general.sample_every_val = int(self.config.get("sample_every_val", 10**9))
        cfg.general.samples_to_generate = int(self.config.get("samples_to_generate", 0))
        cfg.general.samples_to_save = int(self.config.get("samples_to_save", 0))
        cfg.general.chains_to_save = int(self.config.get("chains_to_save", 0))
        cfg.general.final_model_samples_to_generate = int(self.config.get("final_model_samples_to_generate", 0))
        cfg.general.final_model_samples_to_save = int(self.config.get("final_model_samples_to_save", 0))
        cfg.general.final_model_chains_to_save = int(self.config.get("final_model_chains_to_save", 0))
        cfg.general.number_chain_steps = int(self.config.get("number_chain_steps", 50))
        cfg.general.log_every_steps = int(self.config.get("log_every_steps", 50))

        cfg.train.n_epochs = int(self.config.get("num_epochs", self.config.get("n_epochs", 100)))
        cfg.train.batch_size = int(self.config.get("batch_size", 32))
        cfg.train.lr = float(self.config.get("learning_rate", self.config.get("lr", 2e-4)))
        cfg.train.num_workers = int(self.config.get("num_workers", 0))
        cfg.train.save_model = False
        cfg.train.seed = int(self.config.get("seed", 0))

        cfg.dataset.name = self.dataset_name
        if self.is_molecular:
            cfg.dataset.remove_h = bool(self.config.get("remove_h", False))
        # SpectreGraphDataModule joins DiGress repo_root with cfg.dataset.datadir.
        # Absolute paths bypass that join and keep benchmark artifacts out of the external repo.
        cfg.dataset.datadir = str(self.data_root)

        for section, target in [
            ("model_overrides", cfg.model),
            ("train_overrides", cfg.train),
            ("general_overrides", cfg.general),
            ("dataset_overrides", cfg.dataset),
        ]:
            overrides = self.config.get(section, {}) or {}
            for key, value in overrides.items():
                target[key] = value

        return cfg

    # ------------------------------------------------------------------
    # Data materialization
    # ------------------------------------------------------------------
    def _graphs_to_adj_tensors(self, graphs: list[nx.Graph], split_name: str) -> list[torch.Tensor]:
        adjs: list[torch.Tensor] = []
        for idx, graph in enumerate(graphs):
            if graph.number_of_nodes() == 0:
                raise ValueError(f"DiGressWrapper does not support empty graphs: split={split_name}, index={idx}")
            if graph.is_directed():
                raise ValueError(f"DiGressWrapper expects undirected graphs: split={split_name}, index={idx}")
            g = nx.convert_node_labels_to_integers(graph)
            adj = nx.to_numpy_array(g, dtype=np.float32)
            adj = (adj > 0).astype(np.float32)
            np.fill_diagonal(adj, 0.0)
            adj = np.maximum(adj, adj.T)
            adjs.append(torch.from_numpy(adj))
        if not adjs:
            raise ValueError(f"DiGressWrapper received an empty {split_name} split.")
        return adjs

    def _write_raw_splits(
        self,
        train_graphs: list[nx.Graph],
        val_graphs: list[nx.Graph] | None,
        test_graphs: list[nx.Graph] | None,
    ) -> None:
        self.data_root.mkdir(parents=True, exist_ok=True)
        raw_dir = self.data_root / "raw"
        processed_dir = self.data_root / "processed"
        raw_dir.mkdir(parents=True, exist_ok=True)
        if processed_dir.exists():
            shutil.rmtree(processed_dir)

        train_adjs = self._graphs_to_adj_tensors(list(train_graphs), "train")
        if val_graphs is None or len(val_graphs) == 0:
            n_val = max(1, int(round(0.1 * len(train_adjs))))
            val_adjs = [a.clone() for a in train_adjs[:n_val]]
        else:
            val_adjs = self._graphs_to_adj_tensors(list(val_graphs), "val")

        if test_graphs is None or len(test_graphs) == 0:
            test_adjs = [a.clone() for a in val_adjs]
        else:
            test_adjs = self._graphs_to_adj_tensors(list(test_graphs), "test")

        torch.save(train_adjs, raw_dir / "train.pt")
        torch.save(val_adjs, raw_dir / "val.pt")
        torch.save(test_adjs, raw_dir / "test.pt")

        metadata = {
            "dataset_name": self.dataset_name,
            "num_train": len(train_adjs),
            "num_val": len(val_adjs),
            "num_test": len(test_adjs),
            "format": "list of dense binary adjacency tensors for DiGress SPECTRE pipeline",
        }
        with open(self.data_root / "empirical_comparison_meta.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    # ------------------------------------------------------------------
    # Molecular data materialization
    # ------------------------------------------------------------------
    def _atom_class_count(self) -> int:
        if self.dataset_name == "qm9":
            return 4 if bool(self.config.get("remove_h", False)) else 5
        if self._molecular_atom_class_count is not None:
            return int(self._molecular_atom_class_count)
        stats = self.config.get("graph_attribute_stats") or {}
        values = stats.get("node_label_values") or []
        if values:
            return len(values)
        raise RuntimeError("Molecular atom class count is unknown; write molecular splits before conversion.")

    def _bond_class_count(self) -> int:
        # No bond, single, double, triple, aromatic.
        return 5

    def _graph_to_molecular_data(self, graph: nx.Graph, split_name: str, idx: int):
        from torch_geometric.data import Data

        if graph.number_of_nodes() == 0:
            raise ValueError(f"DiGressWrapper does not support empty molecules: split={split_name}, index={idx}")
        if graph.is_directed():
            raise ValueError(f"DiGressWrapper expects undirected molecules: split={split_name}, index={idx}")

        g = nx.convert_node_labels_to_integers(nx.Graph(graph), ordering="default")
        remove_h = bool(self.config.get("remove_h", False))
        if remove_h:
            h_nodes = [node for node, attrs in g.nodes(data=True) if int(attrs.get("node_label", 0)) == 0]
            g.remove_nodes_from(h_nodes)
            if g.number_of_nodes() == 0:
                raise ValueError(f"QM9 molecule became empty after remove_h=True: split={split_name}, index={idx}")
            g = nx.convert_node_labels_to_integers(g, ordering="default")

        n_atom_classes = self._atom_class_count()
        atom_indices = []
        for node in range(g.number_of_nodes()):
            atom_class = int(g.nodes[node].get("node_label", 0))
            if remove_h:
                atom_class -= 1
            if atom_class < 0 or atom_class >= n_atom_classes:
                raise ValueError(
                    f"Invalid QM9 atom class {atom_class} after remove_h={remove_h}: "
                    f"split={split_name}, index={idx}, node={node}. Expected range [0, {n_atom_classes - 1}]."
                )
            atom_indices.append(atom_class)
        x = torch.nn.functional.one_hot(
            torch.tensor(atom_indices, dtype=torch.long),
            num_classes=n_atom_classes,
        ).float()

        edge_index_values: list[list[int]] = []
        edge_type_values: list[int] = []
        for u, v, attrs in g.edges(data=True):
            if u == v:
                continue
            edge_type = int(attrs.get("edge_type", 1))
            edge_type = max(1, min(edge_type, self._bond_class_count() - 1))
            edge_index_values.extend([[int(u), int(v)], [int(v), int(u)]])
            edge_type_values.extend([edge_type, edge_type])
        if edge_index_values:
            edge_index = torch.tensor(edge_index_values, dtype=torch.long).t().contiguous()
            edge_attr = torch.nn.functional.one_hot(
                torch.tensor(edge_type_values, dtype=torch.long),
                num_classes=self._bond_class_count(),
            ).float()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, self._bond_class_count()), dtype=torch.float)

        y = torch.zeros((1, 0), dtype=torch.float)
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        data.n_nodes = torch.tensor([g.number_of_nodes()], dtype=torch.long)
        return data

    def _graphs_to_molecular_data(self, graphs: list[nx.Graph], split_name: str) -> list[Any]:
        started_at = time.perf_counter()
        data_list = [self._graph_to_molecular_data(graph, split_name, idx) for idx, graph in enumerate(graphs)]
        if not data_list:
            raise ValueError(f"DiGressWrapper received an empty {split_name} split.")
        node_counts = [int(data.x.shape[0]) for data in data_list]
        edge_counts = [int(data.edge_index.shape[1] // 2) for data in data_list]
        self._log(
            "converted molecular split=%s count=%d nodes_min=%d nodes_max=%d edges_min=%d edges_max=%d duration=%.3fs",
            split_name,
            len(data_list),
            min(node_counts),
            max(node_counts),
            min(edge_counts),
            max(edge_counts),
            time.perf_counter() - started_at,
        )
        return data_list

    def _write_molecular_splits(
        self,
        train_graphs: list[nx.Graph],
        val_graphs: list[nx.Graph] | None,
        test_graphs: list[nx.Graph] | None,
    ) -> dict[str, list[Any]]:
        started_at = time.perf_counter()
        self.data_root.mkdir(parents=True, exist_ok=True)
        if self.dataset_name != "qm9":
            all_graphs = list(train_graphs) + list(val_graphs or []) + list(test_graphs or [])
            max_label = max(
                [int(attrs.get("node_label", 0)) for graph in all_graphs for _, attrs in graph.nodes(data=True)]
                or [0]
            )
            self._molecular_atom_class_count = max_label + 1
        train_data = self._graphs_to_molecular_data(list(train_graphs), "train")
        if val_graphs is None or len(val_graphs) == 0:
            n_val = max(1, int(round(0.1 * len(train_data))))
            val_data = [data.clone() for data in train_data[:n_val]]
        else:
            val_data = self._graphs_to_molecular_data(list(val_graphs), "val")
        if test_graphs is None or len(test_graphs) == 0:
            test_data = [data.clone() for data in val_data]
        else:
            test_data = self._graphs_to_molecular_data(list(test_graphs), "test")

        splits = {"train": train_data, "val": val_data, "test": test_data}
        torch.save(splits, self.data_root / "benchmark_molecular_splits.pt")
        metadata = {
            "dataset_name": self.dataset_name,
            "num_train": len(train_data),
            "num_val": len(val_data),
            "num_test": len(test_data),
            "remove_h": bool(self.config.get("remove_h", False)),
            "num_atom_classes": self._atom_class_count(),
            "format": "dict of benchmark-converted PyG molecular Data objects for DiGress",
        }
        with open(self.data_root / "empirical_comparison_meta.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        self._log(
            "wrote molecular splits path=%s duration=%.3fs",
            self.data_root / "benchmark_molecular_splits.pt",
            time.perf_counter() - started_at,
        )
        return splits

    def _load_molecular_splits(self) -> dict[str, list[Any]]:
        path = self.data_root / "benchmark_molecular_splits.pt"
        if not path.exists():
            raise FileNotFoundError(
                f"DiGress molecular split file not found: {path}. Train first so the wrapper can persist "
                "benchmark QM9 splits, or keep data_subdir pointing at the training run's data directory."
            )
        with self._legacy_torch_load():
            splits = torch.load(path)
        return {"train": list(splits["train"]), "val": list(splits["val"]), "test": list(splits["test"])}

    # ------------------------------------------------------------------
    # Model/datamodule construction
    # ------------------------------------------------------------------
    def _build_components(self, molecular_splits: dict[str, list[Any]] | None = None) -> None:
        started_at = time.perf_counter()
        self._log("build_components_start molecular=%s", self.is_molecular)
        self._import_modules()
        self.cfg = self._default_cfg()

        TrainAbstractMetricsDiscrete = self._imports["metrics_abstract"].TrainAbstractMetricsDiscrete
        ExtraFeatures = self._imports["extra_features"].ExtraFeatures
        DummyExtraFeatures = self._imports["extra_features"].DummyExtraFeatures
        DiscreteDenoisingDiffusion = self._imports["diffusion_model_discrete"].DiscreteDenoisingDiffusion

        if self.is_molecular:
            if molecular_splits is None:
                molecular_splits = self._load_molecular_splits()
            MolecularDataModule = self._imports["datasets_abstract"].MolecularDataModule
            TrainMolecularMetricsDiscrete = self._imports["metrics_molecular_discrete"].TrainMolecularMetricsDiscrete
            ExtraMolecularFeatures = self._imports["extra_features_molecular"].ExtraMolecularFeatures

            datamodule = MolecularDataModule(self.cfg, molecular_splits)
            dataset_infos = _BenchmarkMolecularInfos(
                datamodule=datamodule,
                cfg=self.cfg,
                distribution_nodes=self._imports["distributions"].DistributionNodes,
                utils_mod=self._imports["digress_utils"],
            )
            if self.dataset_name == "qm9":
                train_metrics = TrainMolecularMetricsDiscrete(dataset_infos)
            else:
                train_metrics = TrainAbstractMetricsDiscrete()
            extra_features = DummyExtraFeatures()
            if self.dataset_name == "qm9" or bool(self.config.get("use_molecular_domain_features", False)):
                domain_features = ExtraMolecularFeatures(dataset_infos=dataset_infos)
            else:
                domain_features = DummyExtraFeatures()
            sampling_metrics = _NoOpSamplingMetrics(
                reason="benchmark wrapper disables upstream molecular sampling metrics"
            )
        else:
            SpectreGraphDataModule = self._imports["datasets_spectre"].SpectreGraphDataModule
            SpectreDatasetInfos = self._imports["datasets_spectre"].SpectreDatasetInfos
            with self._legacy_torch_load():
                datamodule = SpectreGraphDataModule(self.cfg)
                dataset_infos = SpectreDatasetInfos(datamodule, self.cfg.dataset)

            train_metrics = TrainAbstractMetricsDiscrete()
            extra_feature_cfg = self.cfg.model.extra_features
            if extra_feature_cfg is not None:
                extra_features = ExtraFeatures(extra_feature_cfg, dataset_info=dataset_infos)
            else:
                extra_features = DummyExtraFeatures()
            domain_features = DummyExtraFeatures()
            sampling_metrics = self._build_sampling_metrics(datamodule)

        dataset_infos.compute_input_output_dims(
            datamodule=datamodule,
            extra_features=extra_features,
            domain_features=domain_features,
        )

        model = DiscreteDenoisingDiffusion(
            cfg=self.cfg,
            dataset_infos=dataset_infos,
            train_metrics=train_metrics,
            sampling_metrics=sampling_metrics,
            visualization_tools=None,
            extra_features=extra_features,
            domain_features=domain_features,
        )
        self._clear_lightning_hparams(model)

        self.datamodule = datamodule
        self.dataset_infos = dataset_infos
        self.model = model
        self._log("build_components_end parameters=%d duration=%.3fs", self._count_parameters(model), time.perf_counter() - started_at)

    def _build_sampling_metrics(self, datamodule: Any) -> torch.nn.Module:
        try:
            analysis_spectre = importlib.import_module("src.analysis.spectre_utils")
            if self.dataset_name == "sbm":
                return analysis_spectre.SBMSamplingMetrics(datamodule)
            if self.dataset_name == "planar":
                return analysis_spectre.PlanarSamplingMetrics(datamodule)
            return analysis_spectre.Comm20SamplingMetrics(datamodule)
        except Exception as exc:  # graph_tool, ORCA path, or optional analysis deps may fail here.
            warnings.warn(
                "DiGress internal SPECTRE sampling metrics are disabled. "
                "Benchmark-level metrics will still be computed separately. "
                f"Reason: {type(exc).__name__}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            return _NoOpSamplingMetrics(reason=f"{type(exc).__name__}: {exc}")

    def _clear_lightning_hparams(self, model: torch.nn.Module) -> None:
        # Keep checkpoints small and avoid trying to pickle external metric objects.
        if hasattr(model, "_hparams"):
            model._hparams = {}
        if hasattr(model, "_hparams_initial"):
            model._hparams_initial = {}

    @contextlib.contextmanager
    def _legacy_torch_load(self):
        # PyTorch 2.6 changed torch.load default behavior to weights_only=True.
        # DiGress datasets store full objects/lists and require normal unpickling.
        original_load = torch.load

        def compat_load(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return original_load(*args, **kwargs)

        torch.load = compat_load
        try:
            yield
        finally:
            torch.load = original_load

    def _make_trainer(self) -> Trainer:
        assert self.cfg is not None
        use_gpu = self.cfg.general.gpus > 0 and torch.cuda.is_available()
        kwargs = dict(
            accelerator="gpu" if use_gpu else "cpu",
            devices=self.cfg.general.gpus if use_gpu else 1,
            max_epochs=self.cfg.train.n_epochs,
            enable_progress_bar=False,
            logger=[],
            callbacks=[],
            gradient_clip_val=self.cfg.train.clip_grad,
            log_every_n_steps=max(1, int(self.cfg.general.log_every_steps)),
            enable_checkpointing=False,
        )
        # Lightning uses singular `check_val_every_n_epoch`; use a guarded call so
        # the wrapper remains usable across minor Lightning versions.
        try:
            return Trainer(check_val_every_n_epoch=int(self.cfg.general.check_val_every_n_epochs), **kwargs)
        except TypeError:
            return Trainer(**kwargs)

    # ------------------------------------------------------------------
    # Public benchmark wrapper API
    # ------------------------------------------------------------------
    def load(self) -> None:
        """Load a trained DiGress checkpoint from ``checkpoint_path``."""
        started_at = time.perf_counter()
        self._log("load_start checkpoint_path=%s", self.checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"DiGress checkpoint not found: {self.checkpoint_path}. Train first with "
                "scripts/train_model.py or set `checkpoint_path` to an existing .ckpt file."
            )
        self._build_components()
        DiscreteDenoisingDiffusion = self._imports["diffusion_model_discrete"].DiscreteDenoisingDiffusion
        self.model = DiscreteDenoisingDiffusion.load_from_checkpoint(
            str(self.checkpoint_path),
            cfg=self.cfg,
            dataset_infos=self.dataset_infos,
            train_metrics=self.model.train_metrics,
            sampling_metrics=self.model.sampling_metrics,
            visualization_tools=None,
            extra_features=self.model.extra_features,
            domain_features=self.model.domain_features,
        )
        self._clear_lightning_hparams(self.model)
        self.model.eval()
        self.model.to(self.device)
        self._log("load_end duration=%.3fs", time.perf_counter() - started_at)

    def train(self, train_graphs, val_graphs=None, test_graphs=None) -> None:
        """Train DiGress on persisted benchmark splits."""
        started_at = time.perf_counter()
        seed = int(self.config.get("seed", 0))
        self._log(
            "train_start dataset=%s molecular=%s seed=%d train_count=%d val_count=%s test_count=%s",
            self.dataset_name,
            self.is_molecular,
            seed,
            len(train_graphs or []),
            None if val_graphs is None else len(val_graphs),
            None if test_graphs is None else len(test_graphs),
        )
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        if self.is_molecular:
            molecular_splits = self._write_molecular_splits(train_graphs, val_graphs, test_graphs)
            self._build_components(molecular_splits=molecular_splits)
        else:
            self._write_raw_splits(train_graphs, val_graphs, test_graphs)
            self._build_components()
        self.model.to(self.device)

        trainer = self._make_trainer()
        self._log("trainer_fit_start max_epochs=%s batch_size=%s", self.cfg.train.n_epochs, self.cfg.train.batch_size)
        trainer.fit(self.model, datamodule=self.datamodule)
        self._log("trainer_fit_end global_step=%s current_epoch=%s", trainer.global_step, trainer.current_epoch)

        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        trainer.save_checkpoint(str(self.checkpoint_path))
        self.model.eval()
        self._log("train_end checkpoint_path=%s duration=%.3fs", self.checkpoint_path, time.perf_counter() - started_at)

    def sample(self, num_graphs: int, seed: int = 0, progress_callback=None):
        started_at = time.perf_counter()
        self._log("sample_start num_graphs=%d seed=%d", num_graphs, seed)
        if self.model is None:
            self.load()

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        self.model.eval()
        self.model.to(self.device)

        remaining = int(num_graphs)
        batch_size = int(self.config.get("sample_batch_size", self.config.get("batch_size", self.cfg.train.batch_size)))
        batch_size = max(1, batch_size)
        out_graphs: list[nx.Graph] = []
        batch_id = 0

        number_chain_steps = min(int(self.cfg.general.number_chain_steps), max(1, int(self.model.T) - 1))
        with torch.no_grad():
            while remaining > 0:
                batch_started_at = time.perf_counter()
                cur_bs = min(batch_size, remaining)
                if batch_id % self.log_sample_every_n_batches == 0:
                    self._log("sample_batch_start batch_id=%d batch_size=%d remaining=%d", batch_id, cur_bs, remaining)
                samples = self.model.sample_batch(
                    batch_id=batch_id,
                    batch_size=cur_bs,
                    keep_chain=0,
                    number_chain_steps=number_chain_steps,
                    save_final=0,
                    num_nodes=None,
                )
                before = len(out_graphs)
                out_graphs.extend(self._samples_to_networkx(samples))
                update_progress(progress_callback, min(len(out_graphs), num_graphs) - min(before, num_graphs))
                remaining -= cur_bs
                if batch_id % self.log_sample_every_n_batches == 0:
                    self._log(
                        "sample_batch_end batch_id=%d generated_batch=%d generated_total=%d duration=%.3fs",
                        batch_id,
                        len(out_graphs) - before,
                        len(out_graphs),
                        time.perf_counter() - batch_started_at,
                    )
                batch_id += cur_bs

        result = out_graphs[:num_graphs]
        self._log("sample_end returned=%d duration=%.3fs", len(result), time.perf_counter() - started_at)
        return result

    # ------------------------------------------------------------------
    # Converters
    # ------------------------------------------------------------------
    def _samples_to_networkx(self, samples) -> list[nx.Graph]:
        out: list[nx.Graph] = []
        for sample in samples:
            atom_types, edge_types = sample
            atom_types = atom_types.detach().cpu().numpy()
            edge_types = edge_types.detach().cpu().numpy()
            n = int(atom_types.shape[0])
            graph = nx.Graph()
            for i in range(n):
                if self.is_molecular:
                    atom_class = int(atom_types[i])
                    graph.add_node(i, node_label=atom_class, feats=np.array([float(atom_class)], dtype=np.float32))
                else:
                    graph.add_node(i)
            for i in range(n):
                for j in range(i + 1, n):
                    edge_type = int(edge_types[i, j])
                    if edge_type > 0:
                        if self.is_molecular:
                            graph.add_edge(i, j, edge_type=edge_type)
                        else:
                            graph.add_edge(i, j)
            out.append(graph)
        return out

    def _log(self, message: str, *args: Any) -> None:
        if getattr(self, "detailed_logging", False):
            LOGGER.info("DiGressWrapper " + message, *args)

    def _count_parameters(self, model: torch.nn.Module) -> int:
        return int(sum(p.numel() for p in model.parameters()))
