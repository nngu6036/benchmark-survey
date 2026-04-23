from __future__ import annotations

import importlib
import json
import os
import random
import shutil
import sys
import warnings
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import torch
from omegaconf import OmegaConf
from pytorch_lightning import Trainer

from empirical_comparison.models.base import BaseGenerator


class _NoOpSamplingMetrics(torch.nn.Module):
    """Fallback used when DiGress sampling metrics dependencies are unavailable."""

    def __init__(self, reason: str) -> None:
        super().__init__()
        self.reason = reason

    def reset(self) -> None:
        return None

    def forward(self, *args, **kwargs) -> None:
        return None


class DiGressWrapper(BaseGenerator):
    """Adapter around the DiGress implementation for generic synthetic graphs.

    This wrapper targets the DiGress discrete model on SPECTRE-style graph datasets
    such as SBM and Planar. It can train from lists of NetworkX graphs by writing
    adjacency tensors into DiGress's expected raw dataset format.

    Expected config keys
    --------------------
    repo_root: str
        Path to the extracted DiGress repository root.
    checkpoint_path: str
        Where to save/load the Lightning checkpoint.
    dataset_name: str
        One of {"sbm", "planar", "comm20"}. For this project, use "sbm"
        or "planar".
    data_subdir: str, optional
        Relative or absolute path for the DiGress dataset root. Defaults to
        ``data/<dataset_name>/`` under the repo.
    experiment_name: str, optional
        Name stored in cfg.general.name.
    batch_size: int, optional
    num_epochs: int, optional
    learning_rate: float, optional
    num_workers: int, optional
    gpus: int, optional
    seed: int, optional
    model_overrides: dict, optional
        Nested overrides for cfg.model.
    train_overrides: dict, optional
        Nested overrides for cfg.train.
    general_overrides: dict, optional
        Nested overrides for cfg.general.

    Notes
    -----
    - This wrapper is written for the synthetic graph benchmark use case.
    - It assumes undirected simple graphs.
    - Node features are ignored during training because DiGress's SPECTRE data
      pipeline uses a constant one-dimensional node feature for all nodes.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self.repo_root = Path(config["repo_root"]).expanduser().resolve()
        self.repo_src = self.repo_root / "src"
        self.device = config.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = Path(config["checkpoint_path"]).expanduser().resolve()
        self.dataset_name = str(config.get("dataset_name", "sbm"))
        if self.dataset_name not in {"sbm", "planar", "comm20"}:
            raise ValueError(f"Unsupported dataset_name for DiGressWrapper: {self.dataset_name}")

        default_subdir = Path("data") / self.dataset_name
        self.data_subdir = config.get("data_subdir", str(default_subdir))
        self.data_root = Path(self.data_subdir)
        if not self.data_root.is_absolute():
            self.data_root = self.repo_root / self.data_root

        self.model = None
        self.cfg = None
        self.datamodule = None
        self.dataset_infos = None
        self.repo_loaded = False
        self._imports: dict[str, Any] = {}

    @property
    def name(self) -> str:
        return "digress"

    # ---------------------------------------------------------------------
    # Repository loading and config construction
    # ---------------------------------------------------------------------
    def _ensure_repo_importable(self) -> None:
        for p in (str(self.repo_root), str(self.repo_src)):
            if p not in sys.path:
                sys.path.insert(0, p)

    def _import_modules(self) -> None:
        if self.repo_loaded:
            return
        self._ensure_repo_importable()
        self._imports["utils"] = importlib.import_module("src.utils")
        self._imports["datasets_spectre"] = importlib.import_module("src.datasets.spectre_dataset")
        self._imports["metrics_abstract"] = importlib.import_module("src.metrics.abstract_metrics")
        self._imports["extra_features"] = importlib.import_module("src.diffusion.extra_features")
        self._imports["diffusion_model_discrete"] = importlib.import_module("src.diffusion_model_discrete")
        self.repo_loaded = True

    def _default_cfg(self) -> Any:
        cfg_dir = self.repo_root / "configs"
        base = OmegaConf.load(cfg_dir / "config.yaml")
        general = OmegaConf.load(cfg_dir / "general" / "general_default.yaml")
        model = OmegaConf.load(cfg_dir / "model" / "discrete.yaml")
        train = OmegaConf.load(cfg_dir / "train" / "train_default.yaml")
        dataset = OmegaConf.load(cfg_dir / "dataset" / f"{self.dataset_name}.yaml")

        cfg = OmegaConf.create({})
        cfg.general = general
        cfg.model = model
        cfg.train = train
        cfg.dataset = dataset

        # Keep a copy of the Hydra run config only for completeness.
        if "hydra" in base:
            cfg.hydra = base.hydra

        cfg.general.name = str(self.config.get("experiment_name", f"digress_{self.dataset_name}"))
        cfg.general.wandb = "disabled"
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

        cfg.train.n_epochs = int(self.config.get("num_epochs", 100))
        cfg.train.batch_size = int(self.config.get("batch_size", 32))
        cfg.train.lr = float(self.config.get("learning_rate", 2e-4))
        cfg.train.num_workers = int(self.config.get("num_workers", 0))
        cfg.train.save_model = bool(self.config.get("save_model", False))
        cfg.train.seed = int(self.config.get("seed", 0))

        cfg.dataset.name = self.dataset_name
        cfg.dataset.datadir = str(self.data_root)

        model_overrides = self.config.get("model_overrides", {}) or {}
        train_overrides = self.config.get("train_overrides", {}) or {}
        general_overrides = self.config.get("general_overrides", {}) or {}
        for k, v in model_overrides.items():
            cfg.model[k] = v
        for k, v in train_overrides.items():
            cfg.train[k] = v
        for k, v in general_overrides.items():
            cfg.general[k] = v

        return cfg

    # ---------------------------------------------------------------------
    # Data materialization
    # ---------------------------------------------------------------------
    def _graphs_to_adj_tensors(self, graphs: list[nx.Graph]) -> list[torch.Tensor]:
        adjs: list[torch.Tensor] = []
        for g in graphs:
            if g.number_of_nodes() == 0:
                raise ValueError("DiGressWrapper does not support empty graphs.")
            if g.is_directed():
                raise ValueError("DiGressWrapper expects undirected graphs.")
            g2 = nx.convert_node_labels_to_integers(g)
            adj = nx.to_numpy_array(g2, dtype=np.float32)
            # ensure binary simple graph adjacency
            adj = (adj > 0).astype(np.float32)
            np.fill_diagonal(adj, 0.0)
            # symmetrize defensively
            adj = np.maximum(adj, adj.T)
            adjs.append(torch.from_numpy(adj))
        return adjs

    def _write_raw_splits(
        self,
        train_graphs: list[nx.Graph],
        val_graphs: list[nx.Graph] | None,
    ) -> None:
        self.data_root.mkdir(parents=True, exist_ok=True)
        raw_dir = self.data_root / "raw"
        processed_dir = self.data_root / "processed"
        raw_dir.mkdir(parents=True, exist_ok=True)
        if processed_dir.exists():
            shutil.rmtree(processed_dir)

        train_adjs = self._graphs_to_adj_tensors(train_graphs)
        if val_graphs is None:
            n = len(train_adjs)
            n_val = max(1, int(round(0.1 * n))) if n > 2 else 1
            val_adjs = train_adjs[:n_val]
            test_adjs = train_adjs[n_val:2 * n_val] if n > 2 * n_val else train_adjs[:n_val]
            train_adjs = train_adjs[2 * n_val:] if n > 2 * n_val else train_adjs
            if len(train_adjs) == 0:
                raise ValueError("Need more graphs to derive train/val/test splits automatically.")
        else:
            val_adjs = self._graphs_to_adj_tensors(val_graphs)
            test_adjs = [a.clone() for a in val_adjs]

        torch.save(train_adjs, raw_dir / "train.pt")
        torch.save(val_adjs, raw_dir / "val.pt")
        torch.save(test_adjs, raw_dir / "test.pt")

        meta = {
            "dataset_name": self.dataset_name,
            "num_train": len(train_adjs),
            "num_val": len(val_adjs),
            "num_test": len(test_adjs),
        }
        with open(self.data_root / "empirical_comparison_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    # ---------------------------------------------------------------------
    # Model/datamodule construction
    # ---------------------------------------------------------------------
    def _build_components(self) -> None:
        self._import_modules()
        self.cfg = self._default_cfg()

        SpectreGraphDataModule = self._imports["datasets_spectre"].SpectreGraphDataModule
        SpectreDatasetInfos = self._imports["datasets_spectre"].SpectreDatasetInfos
        TrainAbstractMetricsDiscrete = self._imports["metrics_abstract"].TrainAbstractMetricsDiscrete
        ExtraFeatures = self._imports["extra_features"].ExtraFeatures
        DummyExtraFeatures = self._imports["extra_features"].DummyExtraFeatures
        DiscreteDenoisingDiffusion = self._imports["diffusion_model_discrete"].DiscreteDenoisingDiffusion

        datamodule = SpectreGraphDataModule(self.cfg)
        dataset_infos = SpectreDatasetInfos(datamodule, self.cfg.dataset)
        train_metrics = TrainAbstractMetricsDiscrete()
        if self.cfg.model.extra_features is not None:
            extra_features = ExtraFeatures(self.cfg.model.extra_features, dataset_info=dataset_infos)
        else:
            extra_features = DummyExtraFeatures()
        domain_features = DummyExtraFeatures()
        dataset_infos.compute_input_output_dims(
            datamodule=datamodule,
            extra_features=extra_features,
            domain_features=domain_features,
        )

        sampling_metrics = self._build_sampling_metrics(datamodule)

        model = DiscreteDenoisingDiffusion(
            cfg=self.cfg,
            dataset_infos=dataset_infos,
            train_metrics=train_metrics,
            sampling_metrics=sampling_metrics,
            visualization_tools=None,
            extra_features=extra_features,
            domain_features=domain_features,
        )

        self.datamodule = datamodule
        self.dataset_infos = dataset_infos
        self.model = model

    def _build_sampling_metrics(self, datamodule: Any) -> torch.nn.Module:
        try:
            analysis_spectre = importlib.import_module("src.analysis.spectre_utils")
            if self.dataset_name == "sbm":
                return analysis_spectre.SBMSamplingMetrics(datamodule)
            if self.dataset_name == "planar":
                return analysis_spectre.PlanarSamplingMetrics(datamodule)
            return analysis_spectre.Comm20SamplingMetrics(datamodule)
        except ModuleNotFoundError as exc:
            missing_name = getattr(exc, "name", "") or str(exc)
            warnings.warn(
                f"DiGress sampling metrics disabled because optional dependency "
                f"{missing_name!r} is unavailable.",
                RuntimeWarning,
                stacklevel=2,
            )
            return _NoOpSamplingMetrics(reason=str(exc))

    def _make_trainer(self) -> Trainer:
        use_gpu = self.cfg.general.gpus > 0 and torch.cuda.is_available()
        return Trainer(
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

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def load(self) -> None:
        """Load a trained DiGress checkpoint.

        This assumes the dataset root already exists and contains raw or
        processed SBM/Planar-style data compatible with the configured
        ``dataset_name``.
        """
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
        self.model.eval()
        self.model.to(self.device)

    def train(self, train_graphs, val_graphs=None) -> None:
        """Train DiGress from lists of NetworkX graphs.

        Parameters
        ----------
        train_graphs: list[nx.Graph]
            Training graphs.
        val_graphs: list[nx.Graph] | None
            Optional validation graphs. If omitted, a small validation/test split
            is derived from ``train_graphs``.
        """
        seed = int(self.config.get("seed", 0))
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        self._write_raw_splits(train_graphs, val_graphs)
        self._build_components()
        self.model.to(self.device)

        trainer = self._make_trainer()
        trainer.fit(self.model, datamodule=self.datamodule)

        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        trainer.save_checkpoint(str(self.checkpoint_path))
        self.model.eval()

    def sample(self, num_graphs: int, seed: int = 0):
        if self.model is None:
            self.load()

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        self.model.eval()
        self.model.to(self.device)
        self.model.current_epoch = 0

        remaining = int(num_graphs)
        batch_size = int(self.config.get("sample_batch_size", self.cfg.train.batch_size))
        out_graphs: list[nx.Graph] = []
        batch_id = 0

        while remaining > 0:
            cur_bs = min(batch_size, remaining)
            samples = self.model.sample_batch(
                batch_id=batch_id,
                batch_size=cur_bs,
                keep_chain=0,
                number_chain_steps=min(int(self.cfg.general.number_chain_steps), max(1, int(self.model.T) - 1)),
                save_final=0,
                num_nodes=None,
            )
            out_graphs.extend(self._samples_to_networkx(samples))
            remaining -= cur_bs
            batch_id += cur_bs

        return out_graphs

    # ---------------------------------------------------------------------
    # Converters
    # ---------------------------------------------------------------------
    def _samples_to_networkx(self, samples) -> list[nx.Graph]:
        out: list[nx.Graph] = []
        for atom_types, edge_types in samples:
            # For synthetic SPECTRE-style graphs, node types are usually a dummy category
            atom_types = atom_types.detach().cpu().numpy()
            edge_types = edge_types.detach().cpu().numpy()
            n = int(atom_types.shape[0])
            g = nx.Graph()
            for i in range(n):
                g.add_node(i, feats=np.ones(1, dtype=np.float32))
            for i in range(n):
                for j in range(i + 1, n):
                    if edge_types[i, j] > 0:
                        g.add_edge(i, j)
            out.append(g)
        return out
