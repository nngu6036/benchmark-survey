from __future__ import annotations

import importlib
import json
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import networkx as nx
import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from pytorch_lightning import Trainer

from empirical_comparison.models.base import BaseGenerator


class ConStructWrapper(BaseGenerator):
    """Adapter around the ConStruct implementation for generic synthetic graphs.

    This wrapper targets the discrete ConStruct model on SPECTRE-style synthetic
    graph datasets such as Planar and SBM. It can train from lists of NetworkX
    graphs by materializing the raw train/val/test adjacency tensors expected by
    ``ConStruct.datasets.spectre_dataset.SpectreGraphDataset``.

    Expected config keys
    --------------------
    repo_root: str
        Path to the extracted ConStruct repository root.
    checkpoint_path: str
        Where to save/load the Lightning checkpoint.
    dataset_name: str, optional
        One of {"planar", "sbm"}. Defaults to ``planar``.
    data_subdir: str, optional
        Relative or absolute dataset root. Defaults to ``data/<dataset_name>_empirical``
        under the repo root.
    experiment_name: str, optional
    batch_size: int, optional
    num_epochs: int, optional
    learning_rate: float, optional
    num_workers: int, optional
    seed: int, optional
    transition: str, optional
        Noise model transition. Defaults to ``absorbing_edges``.
    rev_proj: str | bool, optional
        Reverse projector, e.g. ``planar`` or ``False``. For generic SBM leave
        it disabled; for planar you may set ``planar``.
    model_overrides: dict, optional
    train_overrides: dict, optional
    general_overrides: dict, optional
    dataset_overrides: dict, optional

    Notes
    -----
    - This wrapper is intended for generic synthetic graph benchmarks.
    - It assumes undirected simple graphs.
    - Node features are reduced to the constant single-channel feature used by
      the upstream SPECTRE pipeline.
    - ConStruct's codebase uses Hydra path resolution internally; this wrapper
      patches that path resolution so it can be called from an external project.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        default_repo_root = Path(__file__).resolve().parents[4] / "external" / "ConStruct"
        repo_root = os.environ.get("CONSTRUCT_REPO") or config.get("repo_root") or default_repo_root
        if not repo_root:
            raise ValueError("ConStructWrapper requires `repo_root` or the CONSTRUCT_REPO environment variable.")
        self.repo_root = self._normalize_repo_root(Path(repo_root).expanduser().resolve())
        self.device = config.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = Path(config["checkpoint_path"]).expanduser().resolve()
        self.dataset_name = str(config.get("dataset_name", "planar")).lower()
        if self.dataset_name not in {"planar", "sbm"}:
            raise ValueError(
                f"Unsupported dataset_name for ConStructWrapper: {self.dataset_name}"
            )

        default_subdir = Path("data") / f"{self.dataset_name}_empirical"
        self.data_root = Path(config.get("data_subdir", default_subdir))
        if not self.data_root.is_absolute():
            self.data_root = (self.repo_root / self.data_root).resolve()

        self.repo_loaded = False
        self.mods: dict[str, Any] = {}

        self.cfg = None
        self.model = None
        self.datamodule = None
        self.dataset_infos = None
        self.val_sampling_metrics = None
        self.test_sampling_metrics = None

    @property
    def name(self) -> str:
        return "construct"

    # ------------------------------------------------------------------
    # Repo imports and config
    # ------------------------------------------------------------------
    def _normalize_repo_root(self, repo_root: Path) -> Path:
        if repo_root.name == "ConStruct" and (repo_root.parent / "configs").exists():
            repo_root = repo_root.parent
        return repo_root

    def _ensure_repo_importable(self) -> None:
        root_str = str(self.repo_root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)

    def _import_modules(self) -> None:
        if self.repo_loaded:
            return
        self._ensure_repo_importable()
        if not (self.repo_root / "ConStruct").exists():
            raise FileNotFoundError(f"ConStruct package directory not found under repo_root={self.repo_root}")
        self.mods["spectre_dataset"] = importlib.import_module(
            "ConStruct.datasets.spectre_dataset"
        )
        self.mods["sampling_metrics"] = importlib.import_module(
            "ConStruct.metrics.sampling_metrics"
        )
        self.mods["diffusion_model"] = importlib.import_module(
            "ConStruct.diffusion_model_discrete"
        )
        self.mods["utils"] = importlib.import_module("ConStruct.utils")
        self.mods["abstract_dataset"] = importlib.import_module(
            "ConStruct.datasets.abstract_dataset"
        )
        self.repo_loaded = True

    def _patch_hydra_path_resolution(self) -> None:
        """Patch the imported module-level get_original_cwd used by the datamodule.

        ConStruct's SpectreGraphDataModule computes its dataset root as
        ``Path(get_original_cwd()).parents[0] / cfg.dataset.datadir``. Returning
        ``repo_root / '_hydra_dummy'`` makes ``parents[0]`` equal ``repo_root``.
        """
        spectre_module = self.mods["spectre_dataset"]
        dummy = str((self.repo_root / "_hydra_dummy").resolve())
        spectre_module.get_original_cwd = lambda: dummy

    def _default_cfg(self):
        cfg_dir = self.repo_root / "configs"
        base = OmegaConf.load(cfg_dir / "config.yaml")
        general = OmegaConf.load(cfg_dir / "general" / "general_default.yaml")
        train = OmegaConf.load(cfg_dir / "train" / "train_default.yaml")
        model = OmegaConf.load(cfg_dir / "model" / "discrete.yaml")
        # No dedicated SBM dataset yaml is shipped, so use the planar yaml as a base
        # and override the name/datadir below.
        dataset = OmegaConf.load(cfg_dir / "dataset" / "planar.yaml")

        cfg = OmegaConf.create({})
        cfg.general = general
        cfg.train = train
        cfg.model = model
        cfg.dataset = dataset
        if "hydra" in base:
            cfg.hydra = base.hydra

        cfg.general.name = str(
            self.config.get("experiment_name", f"construct_{self.dataset_name}")
        )
        cfg.general.wandb = "disabled"
        cfg.general.resume = None
        cfg.general.test_only = None
        cfg.general.evaluate_all_checkpoints = False
        cfg.general.check_val_every_n_epochs = int(
            self.config.get("check_val_every_n_epochs", 10)
        )
        cfg.general.sample_every_val = int(
            self.config.get("sample_every_val", 10**9)
        )
        cfg.general.samples_to_generate = int(self.config.get("samples_to_generate", 0))
        cfg.general.samples_to_save = int(self.config.get("samples_to_save", 0))
        cfg.general.chains_to_save = int(self.config.get("chains_to_save", 0))
        cfg.general.final_model_samples_to_generate = int(
            self.config.get("final_model_samples_to_generate", 0)
        )
        cfg.general.final_model_samples_to_save = int(
            self.config.get("final_model_samples_to_save", 0)
        )
        cfg.general.final_model_chains_to_save = int(
            self.config.get("final_model_chains_to_save", 0)
        )
        cfg.general.log_every_steps = int(self.config.get("log_every_steps", 50))
        cfg.general.number_chain_steps = int(self.config.get("number_chain_steps", 50))
        cfg.general.faster_sampling = int(self.config.get("faster_sampling", 1))

        cfg.train.n_epochs = int(self.config.get("num_epochs", 100))
        cfg.train.batch_size = int(self.config.get("batch_size", 32))
        cfg.train.lr = float(self.config.get("learning_rate", 2e-4))
        cfg.train.num_workers = int(self.config.get("num_workers", 0))
        cfg.train.save_model = False
        cfg.train.seed = int(self.config.get("seed", 0))
        cfg.train.weight_decay = float(self.config.get("weight_decay", 1e-12))
        cfg.train.clip_grad = self.config.get("clip_grad", None)

        cfg.model.transition = str(self.config.get("transition", "absorbing_edges"))
        cfg.model.rev_proj = self.config.get(
            "rev_proj", "planar" if self.dataset_name == "planar" else False
        )
        cfg.model.diffusion_steps = int(self.config.get("diffusion_steps", 500))
        cfg.model.extra_molecular_features = False

        cfg.dataset.name = self.dataset_name
        # Must remain relative to repo_root because SpectreGraphDataModule prefixes it.
        if self.data_root.is_relative_to(self.repo_root):
            cfg.dataset.datadir = str(self.data_root.relative_to(self.repo_root))
        else:
            # Fallback: keep only final directory name under repo root.
            rel = Path("data") / self.data_root.name
            cfg.dataset.datadir = str(rel)
        cfg.dataset.adaptive_loader = bool(self.config.get("adaptive_loader", False))
        cfg.dataset.fraction = float(self.config.get("fraction", 1.0))

        for section_name in ["model_overrides", "train_overrides", "general_overrides", "dataset_overrides"]:
            for k, v in (self.config.get(section_name, {}) or {}).items():
                getattr(cfg, section_name.split("_")[0])[k] = v

        return cfg

    # ------------------------------------------------------------------
    # Data materialization
    # ------------------------------------------------------------------
    def _graphs_to_adj_tensors(self, graphs: list[nx.Graph]) -> list[torch.Tensor]:
        adjs: list[torch.Tensor] = []
        for g in graphs:
            if g.number_of_nodes() == 0:
                raise ValueError("ConStructWrapper does not support empty graphs.")
            if g.is_directed():
                raise ValueError("ConStructWrapper expects undirected graphs.")
            g2 = nx.convert_node_labels_to_integers(g)
            adj = nx.to_numpy_array(g2, dtype=np.float32)
            adj = (adj > 0).astype(np.float32)
            np.fill_diagonal(adj, 0.0)
            adj = np.maximum(adj, adj.T)
            adjs.append(torch.from_numpy(adj))
        return adjs

    def _write_raw_splits(
        self,
        train_graphs: list[nx.Graph],
        val_graphs: list[nx.Graph] | None,
    ) -> None:
        self.data_root.mkdir(parents=True, exist_ok=True)
        raw_dir = self.data_root / self.dataset_name / "raw"
        processed_dir = self.data_root / self.dataset_name / "processed"
        raw_dir.mkdir(parents=True, exist_ok=True)
        if processed_dir.exists():
            shutil.rmtree(processed_dir)

        train_adjs = self._graphs_to_adj_tensors(train_graphs)
        if val_graphs is None:
            n = len(train_adjs)
            n_val = max(1, int(round(0.1 * n))) if n > 2 else 1
            val_adjs = train_adjs[:n_val]
            test_adjs = train_adjs[n_val : 2 * n_val] if n > 2 * n_val else train_adjs[:n_val]
            train_adjs = train_adjs[2 * n_val :] if n > 2 * n_val else train_adjs
            if len(train_adjs) == 0:
                raise ValueError(
                    "Need more graphs to derive train/val/test splits automatically."
                )
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
            "source": "empirical_comparison",
        }
        with open(self.data_root / "empirical_comparison_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    # ------------------------------------------------------------------
    # Building datamodule and model
    # ------------------------------------------------------------------
    def _build_datamodule_and_metrics(self) -> None:
        self._import_modules()
        self._patch_hydra_path_resolution()
        self.cfg = self._default_cfg()

        SpectreGraphDataModule = self.mods["spectre_dataset"].SpectreGraphDataModule
        SpectreDatasetInfos = self.mods["spectre_dataset"].SpectreDatasetInfos
        SamplingMetrics = self.mods["sampling_metrics"].SamplingMetrics

        datamodule = SpectreGraphDataModule(self.cfg)
        dataset_infos = SpectreDatasetInfos(datamodule)

        val_sampling_metrics = SamplingMetrics(
            dataset_infos,
            test=False,
            train_loader=datamodule.train_dataloader(),
            val_loader=datamodule.val_dataloader(),
        )
        test_sampling_metrics = SamplingMetrics(
            dataset_infos,
            test=True,
            train_loader=datamodule.train_dataloader(),
            val_loader=datamodule.test_dataloader(),
        )

        self.datamodule = datamodule
        self.dataset_infos = dataset_infos
        self.val_sampling_metrics = val_sampling_metrics
        self.test_sampling_metrics = test_sampling_metrics

    def _build_model(self) -> None:
        if self.datamodule is None:
            self._build_datamodule_and_metrics()
        DiscreteDenoisingDiffusion = self.mods["diffusion_model"].DiscreteDenoisingDiffusion
        self.model = DiscreteDenoisingDiffusion(
            cfg=self.cfg,
            dataset_infos=self.dataset_infos,
            val_sampling_metrics=self.val_sampling_metrics,
            test_sampling_metrics=self.test_sampling_metrics,
        )

    def _build_trainer(self) -> Trainer:
        use_gpu = self.device.startswith("cuda") and torch.cuda.is_available()
        trainer = Trainer(
            accelerator="gpu" if use_gpu else "cpu",
            devices=1,
            max_epochs=int(self.cfg.train.n_epochs),
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            num_sanity_val_steps=0,
            log_every_n_steps=1,
            gradient_clip_val=self.cfg.train.clip_grad,
            check_val_every_n_epoch=int(self.cfg.general.check_val_every_n_epochs),
        )
        return trainer

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def load(self) -> None:
        self._build_datamodule_and_metrics()
        DiscreteDenoisingDiffusion = self.mods["diffusion_model"].DiscreteDenoisingDiffusion
        self.model = DiscreteDenoisingDiffusion.load_from_checkpoint(
            str(self.checkpoint_path),
            cfg=self.cfg,
            dataset_infos=self.dataset_infos,
            val_sampling_metrics=self.val_sampling_metrics,
            test_sampling_metrics=self.test_sampling_metrics,
            map_location=self.device,
        )
        self.model.eval()
        self.model.to(self.device)
        # Minimal trainer stub so sampling utilities that query trainer fields work.
        self.model._trainer = SimpleNamespace(num_devices=1, strategy=SimpleNamespace(barrier=lambda: None))

    def train(
        self,
        train_graphs: list[nx.Graph],
        val_graphs: list[nx.Graph] | None = None,
    ) -> None:
        self._write_raw_splits(train_graphs, val_graphs)
        self._build_datamodule_and_metrics()
        self._build_model()

        pl.seed_everything(int(self.cfg.train.seed))
        trainer = self._build_trainer()
        trainer.fit(self.model, datamodule=self.datamodule)

        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        trainer.save_checkpoint(str(self.checkpoint_path))
        self.model.eval()
        self.model.to(self.device)
        self.model._trainer = SimpleNamespace(num_devices=1, strategy=SimpleNamespace(barrier=lambda: None))

    def sample(self, num_graphs: int, seed: int = 0) -> list[nx.Graph]:
        if self.model is None:
            self.load()

        pl.seed_everything(seed)
        self.model.eval()
        self.model.to(self.device)
        # Ensure a trainer-like object exists for helper methods.
        if getattr(self.model, "_trainer", None) is None:
            self.model._trainer = SimpleNamespace(num_devices=1, strategy=SimpleNamespace(barrier=lambda: None))

        with torch.no_grad():
            batches = self.model.sample_n_graphs(
                samples_to_generate=int(num_graphs),
                chains_to_save=0,
                samples_to_save=0,
                test=True,
            )

        graphs: list[nx.Graph] = []
        for batch in batches:
            graphs.extend(self._placeholder_batch_to_networkx(batch))
            if len(graphs) >= num_graphs:
                break
        return graphs[:num_graphs]

    # ------------------------------------------------------------------
    # Conversions
    # ------------------------------------------------------------------
    def _placeholder_batch_to_networkx(self, batch) -> list[nx.Graph]:
        # Batch is already collapsed in sample_batch().
        graphs: list[nx.Graph] = []
        X = batch.X.detach().cpu()
        E = batch.E.detach().cpu()
        node_mask = batch.node_mask.detach().cpu()

        for i in range(X.shape[0]):
            n = int(node_mask[i].sum().item())
            if n <= 0:
                continue
            g = nx.Graph()
            for u in range(n):
                x_val = int(X[i, u].item())
                g.add_node(u, feats=np.array([float(x_val)], dtype=np.float32))
            for u in range(n):
                for v in range(u + 1, n):
                    edge_type = int(E[i, u, v].item())
                    if edge_type > 0:
                        g.add_edge(u, v, edge_type=edge_type)
            graphs.append(g)
        return graphs
