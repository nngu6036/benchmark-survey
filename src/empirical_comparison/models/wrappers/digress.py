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


class _DiGressStepLogger:
    """PyTorch Lightning callback that logs detailed DiGress training progress."""

    def __init__(self, wrapper: "DiGressWrapper") -> None:
        from pytorch_lightning.callbacks import Callback

        class StepCallback(Callback):
            def __init__(self, outer: "DiGressWrapper") -> None:
                super().__init__()
                self.outer = outer
                self._epoch_started_at = 0.0
                self._batch_started_at = 0.0

            def on_fit_start(self, trainer, pl_module) -> None:  # type: ignore[override]
                self.outer._log_model_parameters(pl_module)
                self.outer._log(
                    "fit_start max_epochs=%s batches_per_epoch=%s val_batches=%s device=%s precision=%s",
                    getattr(trainer, "max_epochs", None),
                    getattr(trainer, "num_training_batches", None),
                    getattr(trainer, "num_val_batches", None),
                    getattr(pl_module, "device", None),
                    getattr(trainer, "precision", None),
                )

            def on_train_epoch_start(self, trainer, pl_module) -> None:  # type: ignore[override]
                self._epoch_started_at = time.perf_counter()
                self.outer._log(
                    "train_epoch_start epoch=%d/%d global_step=%d lr=%s",
                    int(getattr(trainer, "current_epoch", 0)) + 1,
                    int(getattr(trainer, "max_epochs", 0)),
                    int(getattr(trainer, "global_step", 0)),
                    self.outer._optimizer_lrs(trainer),
                )

            def on_train_batch_start(self, trainer, pl_module, batch, batch_idx) -> None:  # type: ignore[override]
                self._batch_started_at = time.perf_counter()
                if self.outer._should_log_train_step(batch_idx):
                    self.outer._log(
                        "train_batch_start epoch=%d batch=%d/%s global_step=%d batch=%s",
                        int(getattr(trainer, "current_epoch", 0)) + 1,
                        int(batch_idx) + 1,
                        getattr(trainer, "num_training_batches", "?"),
                        int(getattr(trainer, "global_step", 0)),
                        self.outer._summarize_batch(batch),
                    )

            def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:  # type: ignore[override]
                if self.outer._should_log_train_step(batch_idx):
                    self.outer._log(
                        "train_batch_end epoch=%d batch=%d/%s global_step=%d duration=%.3fs output=%s metrics=%s",
                        int(getattr(trainer, "current_epoch", 0)) + 1,
                        int(batch_idx) + 1,
                        getattr(trainer, "num_training_batches", "?"),
                        int(getattr(trainer, "global_step", 0)),
                        time.perf_counter() - self._batch_started_at,
                        self.outer._summarize_value(outputs),
                        self.outer._trainer_metrics(trainer),
                    )

            def on_train_epoch_end(self, trainer, pl_module) -> None:  # type: ignore[override]
                self.outer._log(
                    "train_epoch_end epoch=%d/%d global_step=%d duration=%.3fs metrics=%s",
                    int(getattr(trainer, "current_epoch", 0)) + 1,
                    int(getattr(trainer, "max_epochs", 0)),
                    int(getattr(trainer, "global_step", 0)),
                    time.perf_counter() - self._epoch_started_at,
                    self.outer._trainer_metrics(trainer),
                )

            def on_validation_epoch_start(self, trainer, pl_module) -> None:  # type: ignore[override]
                self.outer._log(
                    "validation_epoch_start epoch=%d global_step=%d",
                    int(getattr(trainer, "current_epoch", 0)) + 1,
                    int(getattr(trainer, "global_step", 0)),
                )

            def on_validation_epoch_end(self, trainer, pl_module) -> None:  # type: ignore[override]
                self.outer._log(
                    "validation_epoch_end epoch=%d global_step=%d metrics=%s",
                    int(getattr(trainer, "current_epoch", 0)) + 1,
                    int(getattr(trainer, "global_step", 0)),
                    self.outer._trainer_metrics(trainer),
                )

            def on_fit_end(self, trainer, pl_module) -> None:  # type: ignore[override]
                self.outer._log(
                    "fit_end current_epoch=%s global_step=%s metrics=%s",
                    getattr(trainer, "current_epoch", None),
                    getattr(trainer, "global_step", None),
                    self.outer._trainer_metrics(trainer),
                )

        self.callback = StepCallback(wrapper)


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

    The wrapper is intended for the benchmark's featureless synthetic datasets
    (currently ``sbm`` and ``planar``). It materializes benchmark NetworkX splits
    into DiGress's SPECTRE raw format: ``raw/train.pt``, ``raw/val.pt`` and
    ``raw/test.pt``, each storing a list of dense binary adjacency tensors.

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
        ``comm20``.
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
    * Node and edge attributes from NetworkX graphs are ignored; DiGress's SPECTRE
      pipeline uses one dummy node type and two edge classes: no-edge and edge.
    * DiGress samples by iterating all ``cfg.model.diffusion_steps``. The benchmark
      ``sampling.num_steps`` field is therefore not used to shorten sampling unless
      you explicitly set ``model_overrides.diffusion_steps`` before training.
    * Upstream SPECTRE sampling metrics are optional in this wrapper; benchmark
      metrics are computed by separate scripts.
    """

    supports_training = True
    supports_sampling = True
    supports_node_features = False
    supports_edge_features = False
    supports_constraints = False
    supports_variable_size = True
    supports_featureless_graphs = True

    SUPPORTED_DATASETS = {"sbm", "planar", "comm20"}

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self.device = config.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset_name = str(config.get("dataset_name") or config.get("dataset") or "sbm").lower()
        if self.dataset_name not in self.SUPPORTED_DATASETS:
            raise ValueError(
                f"DiGressWrapper supports only {sorted(self.SUPPORTED_DATASETS)} for the SPECTRE path; "
                f"got dataset={self.dataset_name!r}."
            )

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
        self.detailed_logging = bool(config.get("detailed_logging", True))
        self.log_config = bool(config.get("log_config", True))
        self.log_train_every_n_steps = max(1, int(config.get("log_train_every_n_steps", 1)))
        self.log_graph_details = bool(config.get("log_graph_details", False))
        self.log_sample_every_n_batches = max(1, int(config.get("log_sample_every_n_batches", 1)))
        self._log(
            "initialized dataset=%s device=%s repo_root=%s data_root=%s checkpoint_path=%s config_keys=%s",
            self.dataset_name,
            self.device,
            self.repo_root,
            self.data_root,
            self.checkpoint_path,
            sorted(self.config.keys()),
        )

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
        self._log("checking DiGress repository layout repo_root=%s", self.repo_root)
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
        self._log("updating sys.path for DiGress repo_root=%s repo_src=%s", self.repo_root, self.repo_src)
        # DiGress modules use both package-style imports (src.utils) and local
        # top-level imports (models.*, metrics.*, diffusion.*). Both paths are needed.
        for p in (str(self.repo_root), str(self.repo_src)):
            if p in sys.path:
                sys.path.remove(p)
        sys.path.insert(0, str(self.repo_root))
        sys.path.insert(0, str(self.repo_src))

    def _import_modules(self) -> None:
        if self.repo_loaded:
            self._log("DiGress modules already imported")
            return
        started_at = time.perf_counter()
        self._ensure_repo_layout()
        self._ensure_repo_importable()
        try:
            self._log("importing upstream DiGress modules")
            self._imports["datasets_spectre"] = importlib.import_module("src.datasets.spectre_dataset")
            self._imports["metrics_abstract"] = importlib.import_module("src.metrics.abstract_metrics")
            self._imports["extra_features"] = importlib.import_module("src.diffusion.extra_features")
            self._imports["diffusion_model_discrete"] = importlib.import_module("src.diffusion_model_discrete")
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
        spectre._empirical_comparison_detailed_logging = self.detailed_logging
        if getattr(spectre.SpectreGraphDataset, "_empirical_comparison_process_patch", False):
            self._log("SpectreGraphDataset.process patch already installed")
            return

        def process(dataset_self):
            import torch_geometric.utils
            from torch_geometric.data import Data

            file_idx = {"train": 0, "val": 1, "test": 2}
            raw_dataset = torch.load(dataset_self.raw_paths[file_idx[dataset_self.split]])
            data_list = []
            started_at = time.perf_counter()
            if getattr(spectre, "_empirical_comparison_detailed_logging", True):
                LOGGER.info(
                    "DiGress upstream process start split=%s raw_path=%s num_graphs=%d",
                    dataset_self.split,
                    dataset_self.raw_paths[file_idx[dataset_self.split]],
                    len(raw_dataset),
                )
            for idx, adj in enumerate(raw_dataset):
                adj = torch.as_tensor(adj, dtype=torch.float)
                if adj.dim() != 2 or adj.shape[0] != adj.shape[1]:
                    raise ValueError(f"Expected square adjacency matrix, got shape={tuple(adj.shape)}")
                n = int(adj.shape[-1])
                edge_count = int((adj > 0).sum().item() // 2)
                if getattr(spectre, "_empirical_comparison_detailed_logging", True):
                    LOGGER.info(
                        "DiGress upstream process graph split=%s index=%d nodes=%d edges=%d",
                        dataset_self.split,
                        idx,
                        n,
                        edge_count,
                    )
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
            if getattr(spectre, "_empirical_comparison_detailed_logging", True):
                LOGGER.info(
                    "DiGress upstream process end split=%s kept=%d processed_path=%s duration=%.3fs",
                    dataset_self.split,
                    len(data_list),
                    dataset_self.processed_paths[0],
                    time.perf_counter() - started_at,
                )

        spectre.SpectreGraphDataset.process = process
        spectre.SpectreGraphDataset._empirical_comparison_process_patch = True
        self._log("installed SpectreGraphDataset.process patch")

    def _default_cfg(self) -> Any:
        started_at = time.perf_counter()
        cfg_dir = self.repo_root / "configs"
        self._log("loading DiGress config files cfg_dir=%s dataset=%s", cfg_dir, self.dataset_name)
        general = OmegaConf.load(cfg_dir / "general" / "general_default.yaml")
        model = OmegaConf.load(cfg_dir / "model" / "discrete.yaml")
        train = OmegaConf.load(cfg_dir / "train" / "train_default.yaml")
        dataset = OmegaConf.load(cfg_dir / "dataset" / f"{self.dataset_name}.yaml")

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
                self._log("applying config override section=%s key=%s value=%r", section, key, value)
                target[key] = value

        if self.log_config:
            self._log("resolved DiGress config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))
        self._log("built DiGress config duration=%.3fs", time.perf_counter() - started_at)
        return cfg

    # ------------------------------------------------------------------
    # Data materialization
    # ------------------------------------------------------------------
    def _graphs_to_adj_tensors(self, graphs: list[nx.Graph], split_name: str) -> list[torch.Tensor]:
        started_at = time.perf_counter()
        self._log("converting graphs to adjacency tensors split=%s count=%d", split_name, len(graphs))
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
            if self.log_graph_details:
                self._log(
                    "converted graph split=%s index=%d nodes=%d edges=%d tensor_shape=%s",
                    split_name,
                    idx,
                    graph.number_of_nodes(),
                    graph.number_of_edges(),
                    tuple(adj.shape),
                )
        if not adjs:
            raise ValueError(f"DiGressWrapper received an empty {split_name} split.")
        node_counts = [int(a.shape[0]) for a in adjs]
        edge_counts = [int(a.sum().item() // 2) for a in adjs]
        self._log(
            "converted split=%s count=%d nodes_min=%d nodes_max=%d edges_min=%d edges_max=%d duration=%.3fs",
            split_name,
            len(adjs),
            min(node_counts),
            max(node_counts),
            min(edge_counts),
            max(edge_counts),
            time.perf_counter() - started_at,
        )
        return adjs

    def _write_raw_splits(
        self,
        train_graphs: list[nx.Graph],
        val_graphs: list[nx.Graph] | None,
        test_graphs: list[nx.Graph] | None,
    ) -> None:
        started_at = time.perf_counter()
        self._log(
            "writing raw DiGress splits data_root=%s train_count=%d val_count=%s test_count=%s",
            self.data_root,
            len(train_graphs),
            None if val_graphs is None else len(val_graphs),
            None if test_graphs is None else len(test_graphs),
        )
        self.data_root.mkdir(parents=True, exist_ok=True)
        raw_dir = self.data_root / "raw"
        processed_dir = self.data_root / "processed"
        raw_dir.mkdir(parents=True, exist_ok=True)
        if processed_dir.exists():
            self._log("removing stale DiGress processed directory path=%s", processed_dir)
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
        self._log("wrote raw train split path=%s count=%d", raw_dir / "train.pt", len(train_adjs))
        torch.save(val_adjs, raw_dir / "val.pt")
        self._log("wrote raw val split path=%s count=%d", raw_dir / "val.pt", len(val_adjs))
        torch.save(test_adjs, raw_dir / "test.pt")
        self._log("wrote raw test split path=%s count=%d", raw_dir / "test.pt", len(test_adjs))

        metadata = {
            "dataset_name": self.dataset_name,
            "num_train": len(train_adjs),
            "num_val": len(val_adjs),
            "num_test": len(test_adjs),
            "format": "list of dense binary adjacency tensors for DiGress SPECTRE pipeline",
        }
        with open(self.data_root / "empirical_comparison_meta.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        self._log(
            "finished writing raw splits metadata_path=%s duration=%.3fs",
            self.data_root / "empirical_comparison_meta.json",
            time.perf_counter() - started_at,
        )

    # ------------------------------------------------------------------
    # Model/datamodule construction
    # ------------------------------------------------------------------
    def _build_components(self) -> None:
        started_at = time.perf_counter()
        self._log("building DiGress components")
        self._import_modules()
        self.cfg = self._default_cfg()

        SpectreGraphDataModule = self._imports["datasets_spectre"].SpectreGraphDataModule
        SpectreDatasetInfos = self._imports["datasets_spectre"].SpectreDatasetInfos
        TrainAbstractMetricsDiscrete = self._imports["metrics_abstract"].TrainAbstractMetricsDiscrete
        ExtraFeatures = self._imports["extra_features"].ExtraFeatures
        DummyExtraFeatures = self._imports["extra_features"].DummyExtraFeatures
        DiscreteDenoisingDiffusion = self._imports["diffusion_model_discrete"].DiscreteDenoisingDiffusion

        with self._legacy_torch_load():
            self._log("constructing SpectreGraphDataModule")
            datamodule = SpectreGraphDataModule(self.cfg)
            self._log("constructing SpectreDatasetInfos")
            dataset_infos = SpectreDatasetInfos(datamodule, self.cfg.dataset)

        self._log("constructing TrainAbstractMetricsDiscrete")
        train_metrics = TrainAbstractMetricsDiscrete()
        extra_feature_cfg = self.cfg.model.extra_features
        if extra_feature_cfg is not None:
            self._log("constructing ExtraFeatures extra_features=%r", extra_feature_cfg)
            extra_features = ExtraFeatures(extra_feature_cfg, dataset_info=dataset_infos)
        else:
            self._log("constructing DummyExtraFeatures for extra features")
            extra_features = DummyExtraFeatures()
        domain_features = DummyExtraFeatures()
        self._log("computing input/output dimensions")
        dataset_infos.compute_input_output_dims(
            datamodule=datamodule,
            extra_features=extra_features,
            domain_features=domain_features,
        )

        sampling_metrics = self._build_sampling_metrics(datamodule)
        self._log("constructing DiscreteDenoisingDiffusion")
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
        self._log_model_parameters(model)
        self._log("built DiGress components duration=%.3fs", time.perf_counter() - started_at)

    def _build_sampling_metrics(self, datamodule: Any) -> torch.nn.Module:
        try:
            self._log("constructing SPECTRE sampling metrics dataset=%s", self.dataset_name)
            analysis_spectre = importlib.import_module("src.analysis.spectre_utils")
            if self.dataset_name == "sbm":
                return analysis_spectre.SBMSamplingMetrics(datamodule)
            if self.dataset_name == "planar":
                return analysis_spectre.PlanarSamplingMetrics(datamodule)
            return analysis_spectre.Comm20SamplingMetrics(datamodule)
        except Exception as exc:  # graph_tool, ORCA path, or optional analysis deps may fail here.
            self._log("falling back to no-op sampling metrics reason=%s: %s", type(exc).__name__, exc)
            warnings.warn(
                "DiGress internal SPECTRE sampling metrics are disabled. "
                "Benchmark-level metrics will still be computed separately. "
                f"Reason: {type(exc).__name__}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            return _NoOpSamplingMetrics(reason=f"{type(exc).__name__}: {exc}")

    def _clear_lightning_hparams(self, model: torch.nn.Module) -> None:
        self._log("clearing Lightning hparams before checkpoint serialization")
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
        callback_wrapper = _DiGressStepLogger(self)
        kwargs = dict(
            accelerator="gpu" if use_gpu else "cpu",
            devices=self.cfg.general.gpus if use_gpu else 1,
            max_epochs=self.cfg.train.n_epochs,
            enable_progress_bar=False,
            logger=[],
            callbacks=[callback_wrapper.callback],
            gradient_clip_val=self.cfg.train.clip_grad,
            log_every_n_steps=max(1, int(self.cfg.general.log_every_steps)),
            enable_checkpointing=False,
        )
        self._log(
            "creating Lightning Trainer accelerator=%s devices=%s max_epochs=%s gradient_clip_val=%s log_every_n_steps=%s",
            kwargs["accelerator"],
            kwargs["devices"],
            kwargs["max_epochs"],
            kwargs["gradient_clip_val"],
            kwargs["log_every_n_steps"],
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
        self._log("loading DiGress checkpoint path=%s", self.checkpoint_path)
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
        self._log("loaded DiGress checkpoint duration=%.3fs", time.perf_counter() - started_at)

    def train(self, train_graphs, val_graphs=None, test_graphs=None) -> None:
        """Train DiGress on persisted benchmark splits."""
        started_at = time.perf_counter()
        seed = int(self.config.get("seed", 0))
        self._log(
            "train_start seed=%d device=%s train_count=%d val_count=%s test_count=%s",
            seed,
            self.device,
            len(train_graphs),
            None if val_graphs is None else len(val_graphs),
            None if test_graphs is None else len(test_graphs),
        )
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        self._write_raw_splits(train_graphs, val_graphs, test_graphs)
        self._build_components()
        self.model.to(self.device)

        trainer = self._make_trainer()
        self._log("calling trainer.fit")
        trainer.fit(self.model, datamodule=self.datamodule)
        self._log("trainer.fit returned global_step=%s current_epoch=%s", trainer.global_step, trainer.current_epoch)

        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        self._log("saving checkpoint path=%s", self.checkpoint_path)
        trainer.save_checkpoint(str(self.checkpoint_path))
        self.model.eval()
        self._log("train_end checkpoint_path=%s duration=%.3fs", self.checkpoint_path, time.perf_counter() - started_at)

    def sample(self, num_graphs: int, seed: int = 0, progress_callback=None):
        started_at = time.perf_counter()
        self._log("sample_start requested=%d seed=%d", num_graphs, seed)
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
        self._log(
            "sample_config batch_size=%d diffusion_T=%s number_chain_steps=%d device=%s",
            batch_size,
            getattr(self.model, "T", None),
            number_chain_steps,
            self.device,
        )
        with torch.no_grad():
            while remaining > 0:
                cur_bs = min(batch_size, remaining)
                batch_started_at = time.perf_counter()
                if batch_id % self.log_sample_every_n_batches == 0:
                    self._log(
                        "sample_batch_start batch_id=%d cur_batch_size=%d remaining_before=%d generated=%d",
                        batch_id,
                        cur_bs,
                        remaining,
                        len(out_graphs),
                    )
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
                        "sample_batch_end batch_id=%d generated_batch=%d generated_total=%d remaining_after=%d duration=%.3fs",
                        batch_id,
                        len(out_graphs) - before,
                        len(out_graphs),
                        remaining,
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
        started_at = time.perf_counter()
        out: list[nx.Graph] = []
        for idx, sample in enumerate(samples):
            atom_types, edge_types = sample
            atom_types = atom_types.detach().cpu().numpy()
            edge_types = edge_types.detach().cpu().numpy()
            n = int(atom_types.shape[0])
            graph = nx.Graph()
            for i in range(n):
                graph.add_node(i)
            for i in range(n):
                for j in range(i + 1, n):
                    if int(edge_types[i, j]) > 0:
                        graph.add_edge(i, j)
            out.append(graph)
            if self.log_graph_details:
                self._log(
                    "converted sample index=%d nodes=%d edges=%d",
                    idx,
                    graph.number_of_nodes(),
                    graph.number_of_edges(),
                )
        self._log("converted samples to NetworkX count=%d duration=%.3fs", len(out), time.perf_counter() - started_at)
        return out

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------
    def _log(self, message: str, *args: Any) -> None:
        if self.detailed_logging:
            LOGGER.info("DiGressWrapper " + message, *args)

    def _should_log_train_step(self, batch_idx: int) -> bool:
        return self.detailed_logging and (int(batch_idx) % self.log_train_every_n_steps == 0)

    def _optimizer_lrs(self, trainer: Any) -> list[float]:
        lrs: list[float] = []
        for optimizer in getattr(trainer, "optimizers", []) or []:
            for group in getattr(optimizer, "param_groups", []) or []:
                if "lr" in group:
                    lrs.append(float(group["lr"]))
        return lrs

    def _trainer_metrics(self, trainer: Any) -> dict[str, float]:
        metrics: dict[str, float] = {}
        for name, value in dict(getattr(trainer, "callback_metrics", {}) or {}).items():
            try:
                if torch.is_tensor(value):
                    if value.numel() == 1:
                        metrics[str(name)] = float(value.detach().cpu().item())
                elif isinstance(value, (int, float)):
                    metrics[str(name)] = float(value)
            except Exception:
                continue
        return metrics

    def _summarize_batch(self, batch: Any) -> Any:
        return self._summarize_value(batch, max_depth=3)

    def _summarize_value(self, value: Any, max_depth: int = 2) -> Any:
        if max_depth < 0:
            return type(value).__name__
        if torch.is_tensor(value):
            summary: dict[str, Any] = {
                "type": "Tensor",
                "shape": tuple(value.shape),
                "dtype": str(value.dtype),
                "device": str(value.device),
            }
            if value.numel() == 1:
                summary["value"] = float(value.detach().cpu().item())
            return summary
        if isinstance(value, dict):
            return {str(k): self._summarize_value(v, max_depth - 1) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            items = [self._summarize_value(v, max_depth - 1) for v in list(value)[:6]]
            if len(value) > 6:
                items.append(f"... {len(value) - 6} more")
            return {"type": type(value).__name__, "len": len(value), "items": items}
        if hasattr(value, "x") and hasattr(value, "edge_index"):
            return {
                "type": type(value).__name__,
                "x": self._summarize_value(getattr(value, "x", None), max_depth - 1),
                "edge_index": self._summarize_value(getattr(value, "edge_index", None), max_depth - 1),
                "edge_attr": self._summarize_value(getattr(value, "edge_attr", None), max_depth - 1),
                "n_nodes": self._summarize_value(getattr(value, "n_nodes", None), max_depth - 1),
            }
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return type(value).__name__

    def _log_model_parameters(self, model: torch.nn.Module) -> None:
        total = 0
        trainable = 0
        for name, param in model.named_parameters():
            count = int(param.numel())
            total += count
            if param.requires_grad:
                trainable += count
            self._log(
                "parameter name=%s shape=%s dtype=%s requires_grad=%s count=%d",
                name,
                tuple(param.shape),
                param.dtype,
                param.requires_grad,
                count,
            )
        self._log("parameter_summary total=%d trainable=%d frozen=%d", total, trainable, total - trainable)
