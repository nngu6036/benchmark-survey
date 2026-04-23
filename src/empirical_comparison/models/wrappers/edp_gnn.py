from __future__ import annotations

import contextlib
import importlib
import json
import logging
import pickle
import random
import shutil
import os
import sys
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import torch
from easydict import EasyDict as edict

try:
    from empirical_comparison.models.base import BaseGenerator
except Exception:  # pragma: no cover
    class BaseGenerator:
        def __init__(self, config: dict[str, Any]) -> None:
            self.config = config


class EDPGNNWrapper(BaseGenerator):
    """Adapter around the upstream EDP-GNN implementation.

    This wrapper targets the generic synthetic graph setting used by the
    original codebase and exposes a unified ``load`` / ``train`` / ``sample``
    interface for the benchmark scaffold.

    Expected config keys
    --------------------
    repo_root: str
        Path to the extracted EDP-GNN repository root.
    checkpoint_path: str
        Location of the wrapper checkpoint.
    dataset_name: str, optional
        Logical dataset name to materialize under ``repo_root/data``.
    batch_size: int, optional
    num_epochs: int, optional
    learning_rate: float, optional
    weight_decay: float, optional
    sigmas: list[float], optional
    grad_step_size: float | list[float], optional
    eps: float | list[float], optional
    step_num: int, optional
    max_node_num: int, optional
    test_split: float, optional
    seed: int, optional
    model_overrides: dict, optional
        Overrides for ``config.model.models.model_1``.
    device: str, optional

    Notes
    -----
    - This wrapper is intended for generic graph benchmarks such as SBM or
      Planar rather than molecule generation.
    - It assumes undirected simple graphs.
    - Node features are taken from the node attribute ``feature`` if present;
      otherwise constant one-dimensional features are used.
    - Sampling follows the upstream annealed Langevin routine used in
      ``sample.py``.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        default_repo_root = Path(__file__).resolve().parents[4] / "external" / "EDP-GNN"
        repo_root = os.environ.get("EDP_GNN_REPO") or config.get("repo_root") or default_repo_root
        if not repo_root:
            raise ValueError("EDPGNNWrapper requires `repo_root` or the EDP_GNN_REPO environment variable.")
        self.repo_root = self._normalize_repo_root(Path(repo_root).expanduser().resolve())
        self.checkpoint_path = Path(config["checkpoint_path"]).expanduser().resolve()
        self.device = torch.device(config.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.dataset_name = str(config.get("dataset_name", "empirical_graphs"))
        self.repo_loaded = False
        self.mods: dict[str, Any] = {}
        self.model: torch.nn.Module | None = None
        self.mcmc_sampler: Any = None
        self.edp_config: Any = None
        self.template_graphs: list[nx.Graph] = []
        self.feature_dim: int | None = None

    @property
    def name(self) -> str:
        return "edp_gnn"

    # ------------------------------------------------------------------
    # Repo imports
    # ------------------------------------------------------------------
    def _normalize_repo_root(self, repo_root: Path) -> Path:
        if repo_root.name == "utils" and (repo_root.parent / "train.py").exists():
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
        if not (self.repo_root / "train.py").exists():
            raise FileNotFoundError(f"EDP-GNN repository layout not found under repo_root={self.repo_root}")
        self.mods["train"] = importlib.import_module("train")
        self.mods["arg_helper"] = importlib.import_module("utils.arg_helper")
        self.mods["loading_utils"] = importlib.import_module("utils.loading_utils")
        self.mods["sample"] = importlib.import_module("sample")
        self.mods["stats"] = importlib.import_module("evaluation.stats")
        self.repo_loaded = True

    # ------------------------------------------------------------------
    # Dataset materialization
    # ------------------------------------------------------------------
    def _normalize_graph(self, g: nx.Graph) -> nx.Graph:
        if g.number_of_nodes() == 0:
            raise ValueError("EDPGNNWrapper does not support empty graphs.")
        if g.is_directed():
            raise ValueError("EDPGNNWrapper expects undirected graphs.")
        g2 = nx.convert_node_labels_to_integers(g.copy())
        if any(u == v for u, v in g2.edges()):
            g2.remove_edges_from(list(nx.selfloop_edges(g2)))
        # Normalize node features to upstream attribute name: `feature`.
        feat_attr = "feature"
        feats = nx.get_node_attributes(g2, feat_attr)
        if not feats:
            feats = nx.get_node_attributes(g2, "feats")
        clean: dict[int, np.ndarray] = {}
        for i in range(g2.number_of_nodes()):
            arr = feats.get(i, np.ones(1, dtype=np.float32))
            arr = np.asarray(arr, dtype=np.float32)
            if arr.ndim == 0:
                arr = arr[None]
            clean[i] = arr
        nx.set_node_attributes(g2, clean, feat_attr)
        return g2

    def _infer_feature_dim(self, graphs: list[nx.Graph]) -> int:
        for g in graphs:
            feats = nx.get_node_attributes(g, "feature") or nx.get_node_attributes(g, "feats")
            if feats:
                arr = np.asarray(feats[next(iter(feats))], dtype=np.float32)
                return int(arr.shape[0] if arr.ndim > 0 else 1)
        return 1

    def _data_dir(self) -> Path:
        return self.repo_root / "data"

    def _dataset_prefix(self) -> Path:
        return self._data_dir() / self.dataset_name

    def _materialize_dataset(self, train_graphs: list[nx.Graph], val_graphs: list[nx.Graph] | None) -> tuple[list[nx.Graph], list[nx.Graph]]:
        self._data_dir().mkdir(parents=True, exist_ok=True)
        train_graphs = [self._normalize_graph(g) for g in train_graphs]
        val_graphs = [self._normalize_graph(g) for g in (val_graphs or [])]
        all_graphs = train_graphs + val_graphs
        if not all_graphs:
            raise ValueError("No graphs provided to train().")
        prefix = self._dataset_prefix()
        with open(str(prefix) + ".pkl", "wb") as f:
            pickle.dump(all_graphs, f, protocol=pickle.HIGHEST_PROTOCOL)
        meta = {
            "dataset_name": self.dataset_name,
            "num_graphs": len(all_graphs),
            "num_train_graphs": len(train_graphs),
            "num_val_graphs": len(val_graphs),
            "max_nodes": max(g.number_of_nodes() for g in all_graphs),
            "source": "empirical_comparison",
        }
        with open(str(prefix) + ".txt", "w", encoding="utf-8") as f:
            f.write(json.dumps(meta))
        return train_graphs, val_graphs

    # ------------------------------------------------------------------
    # Config construction
    # ------------------------------------------------------------------
    def _build_config(self, train_graphs: list[nx.Graph], val_graphs: list[nx.Graph]) -> Any:
        self._import_modules()
        process_config = self.mods["arg_helper"].process_config
        max_nodes = max(g.number_of_nodes() for g in (train_graphs + val_graphs if val_graphs else train_graphs))
        feature_dim = self._infer_feature_dim(train_graphs)
        self.feature_dim = feature_dim
        split = len(val_graphs) / max(1, len(train_graphs) + len(val_graphs)) if val_graphs else float(self.config.get("test_split", 0.2))

        model_1 = {
            "dropout_p": 0.0,
            "gnn_hidden_num_list": [16, 16, 16, 16],
            "feature_nums": [16, 16, 16, 16, 16],
            "channel_num_list": [2, 4, 4, 4, 2],
            "name": "gin",
            "use_norm_layers": False,
        }
        model_1.update(dict(self.config.get("model_overrides", {})))

        cfg = edict(
            {
                "exp_dir": str(self.config.get("exp_dir", self.repo_root / "exp")),
                "exp_name": str(self.config.get("exp_name", f"{self.dataset_name}_empirical")),
                "seed": int(self.config.get("seed", 0)),
                "dataset": edict(
                    {
                        "dataset_size": int(self.config.get("dataset_size", len(train_graphs) + len(val_graphs))),
                        "max_node_num": int(self.config.get("max_node_num", max_nodes)),
                        "name": self.dataset_name,
                        "in_feature": feature_dim,
                    }
                ),
                "mcmc": edict(
                    {
                        "name": "langevin",
                        "eps": self.config.get("eps", [0.5]),
                        "fixed_node_number": True,
                        "grad_step_size": self.config.get("grad_step_size", [0.01, 0.001]),
                        "step_num": int(self.config.get("step_num", 1000)),
                    }
                ),
                "model": edict(
                    {
                        "name": "edp-gnn",
                        "models": edict({"model_1": edict(model_1)}),
                        "stack_num": int(self.config.get("stack_num", 1)),
                    }
                ),
                "sample": edict({"batch_size": int(self.config.get("sample_batch_size", self.config.get("batch_size", 32)))}),
                "test": edict({"batch_size": int(self.config.get("test_batch_size", self.config.get("batch_size", 32))), "split": split}),
                "train": edict(
                    {
                        "batch_size": int(self.config.get("batch_size", 32)),
                        "lr_dacey": float(self.config.get("lr_dacey", 0.999)),
                        "lr_init": float(self.config.get("learning_rate", 1e-3)),
                        "momentum": 0.9,
                        "max_epoch": int(self.config.get("num_epochs", 100)),
                        "sample_interval": int(self.config.get("sample_interval", 10**9)),
                        "save_interval": int(self.config.get("save_interval", int(self.config.get("num_epochs", 100)))),
                        "shuffle": True,
                        "sigmas": list(self.config.get("sigmas", [0.1, 0.2, 0.4, 0.6, 0.8, 1.6])),
                        "weight_decay": float(self.config.get("weight_decay", 0.0)),
                    }
                ),
            }
        )
        process_config(cfg, comment="empirical_comparison")
        cfg.dev = self.device
        return cfg

    # ------------------------------------------------------------------
    # Training / loading
    # ------------------------------------------------------------------
    def load(self) -> None:
        self._import_modules()
        with self._legacy_torch_load():
            ckpt = torch.load(self.checkpoint_path, map_location=self.device)
        self.edp_config = edict(ckpt["config"])
        self.edp_config.dev = self.device
        self.template_graphs = [self._restore_graph(gd) for gd in ckpt["template_graphs"]]
        self.feature_dim = int(ckpt.get("feature_dim", 1))

        get_score_model = self.mods["loading_utils"].get_score_model
        get_mc_sampler = self.mods["loading_utils"].get_mc_sampler
        with self._legacy_torch_load(), self._legacy_networkx_matrix():
            self.model = get_score_model(self.edp_config, dev=self.device)
        self.model.load_state_dict(ckpt["model_state"], strict=False)
        self.model.to(self.device)
        self.model.eval()
        self.mcmc_sampler = get_mc_sampler(self.edp_config)

    def train(self, train_graphs, val_graphs=None) -> None:
        self._import_modules()
        fit = self.mods["train"].fit
        set_seed_and_logger = self.mods["arg_helper"].set_seed_and_logger
        load_data = self.mods["arg_helper"].load_data
        get_mc_sampler = self.mods["loading_utils"].get_mc_sampler
        get_score_model = self.mods["loading_utils"].get_score_model

        train_graphs, val_graphs = self._materialize_dataset(list(train_graphs), list(val_graphs or []))
        self.template_graphs = [g.copy() for g in train_graphs]
        self.edp_config = self._build_config(train_graphs, val_graphs)

        # Lightweight logger bootstrap expected by upstream utilities.
        class _Args:
            comment = "empirical_comparison"
            log_level = "INFO"

        with self._repo_cwd(), self._legacy_networkx_matrix():
            set_seed_and_logger(self.edp_config, _Args())
            random.seed(self.edp_config.seed)
            np.random.seed(self.edp_config.seed)
            torch.manual_seed(self.edp_config.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.edp_config.seed)

            train_dl, test_dl = load_data(self.edp_config)
            self.mcmc_sampler = get_mc_sampler(self.edp_config)
            self.model = get_score_model(self.edp_config, dev=self.device)
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=float(self.edp_config.train.lr_init),
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=float(self.edp_config.train.weight_decay),
            )
            fit(
                self.model,
                optimizer,
                self.mcmc_sampler,
                train_dl,
                max_node_number=int(self.edp_config.dataset.max_node_num),
                max_epoch=int(self.edp_config.train.max_epoch),
                config=self.edp_config,
                save_interval=int(self.edp_config.train.save_interval),
                sample_interval=int(self.edp_config.train.sample_interval),
                sigma_list=list(self.edp_config.train.sigmas),
                sample_from_sigma_delta=0.0,
                test_dl=test_dl,
            )

        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_state": self.model.state_dict(),
            "config": self._to_plain_dict(self.edp_config),
            "template_graphs": [self._serialize_graph(g) for g in self.template_graphs],
            "feature_dim": self.feature_dim,
        }
        torch.save(payload, self.checkpoint_path)
        self.model.eval()

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------
    def _prepare_init_batch(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        graphs_to_tensor = self.mods["arg_helper"].graphs_to_tensor
        if not self.template_graphs:
            raise RuntimeError("No template graphs available for sampling.")
        idx = np.random.randint(0, len(self.template_graphs), size=batch_size)
        graph_list = [self.template_graphs[i] for i in idx]
        with self._legacy_networkx_matrix():
            base_adjs, base_x = graphs_to_tensor(self.edp_config, graph_list)
        base_adjs, base_x = base_adjs.to(self.device), base_x.to(self.device)
        node_flags = base_adjs.sum(-1).gt(1e-5).to(dtype=torch.float32)
        base_adjs = self.mcmc_sampler.gen_init_sample(batch_size, self.edp_config.dataset.max_node_num, node_flags=node_flags)[0]
        return base_adjs, base_x, node_flags

    def sample(self, num_graphs: int, seed: int = 0):
        self._import_modules()
        if self.model is None or self.mcmc_sampler is None or self.edp_config is None:
            self.load()
        assert self.model is not None and self.mcmc_sampler is not None and self.edp_config is not None
        adjs_to_graphs = self.mods["stats"].adjs_to_graphs

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        self.model.eval()
        sigma_list = sorted(list(self.edp_config.train.sigmas))
        batch_size = int(self.edp_config.sample.batch_size)
        step_size_ratio = self.edp_config.mcmc.grad_step_size[0] if isinstance(self.edp_config.mcmc.grad_step_size, (list, tuple)) else self.edp_config.mcmc.grad_step_size
        eps = self.edp_config.mcmc.eps[0] if isinstance(self.edp_config.mcmc.eps, (list, tuple)) else self.edp_config.mcmc.eps

        total_batch = batch_size * len(sigma_list)
        init_adjs, sample_x, sample_node_flags = self._prepare_init_batch(total_batch)
        generated: list[nx.Graph] = []
        warm_up_count = 0

        while len(generated) < num_graphs:
            step_size = step_size_ratio * torch.tensor(sigma_list, device=self.device).repeat_interleave(batch_size, dim=0)[..., None, None] ** 2
            with torch.no_grad():
                sampled_adjs, _ = self.mcmc_sampler.sample(
                    total_batch,
                    lambda x, y: self.model(sample_x, x, y),
                    max_node_num=int(self.edp_config.dataset.max_node_num),
                    step_num=None,
                    init_adjs=init_adjs,
                    init_flags=sample_node_flags,
                    is_final=False,
                    step_size=step_size,
                    eps=eps,
                )
            sampled_chunks = sampled_adjs.chunk(len(sigma_list), dim=0)
            if warm_up_count < len(sigma_list):
                warm_up_count += 1
            else:
                rounded, _ = self.mcmc_sampler.end_sample(sampled_chunks[0])[0], None
                generated.extend(adjs_to_graphs(rounded.detach().cpu().numpy()))
            new_init_adjs, new_x, new_flags = self._prepare_init_batch(batch_size)
            init_adjs = torch.cat(list(sampled_chunks[1:]) + [new_init_adjs], dim=0)
            keep = sampled_chunks[0].size(0)
            sample_x = torch.cat([sample_x[keep:], new_x], dim=0)
            sample_node_flags = torch.cat([sample_node_flags[keep:], new_flags], dim=0)

        result = []
        for g in generated[:num_graphs]:
            g2 = nx.convert_node_labels_to_integers(g)
            for i in g2.nodes():
                g2.nodes[i]["feats"] = np.ones(self.feature_dim or 1, dtype=np.float32)
            result.append(g2)
        return result

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------
    def _serialize_graph(self, g: nx.Graph) -> dict[str, Any]:
        g2 = nx.convert_node_labels_to_integers(g)
        feats = []
        feat_map = nx.get_node_attributes(g2, "feature") or nx.get_node_attributes(g2, "feats")
        for i in range(g2.number_of_nodes()):
            arr = np.asarray(feat_map.get(i, np.ones(1, dtype=np.float32)), dtype=np.float32)
            if arr.ndim == 0:
                arr = arr[None]
            feats.append(arr.tolist())
        return {"edges": list(g2.edges()), "num_nodes": g2.number_of_nodes(), "features": feats}

    def _restore_graph(self, data: dict[str, Any]) -> nx.Graph:
        g = nx.Graph()
        g.add_nodes_from(range(int(data["num_nodes"])))
        g.add_edges_from([tuple(map(int, e)) for e in data["edges"]])
        for i, feat in enumerate(data["features"]):
            g.nodes[i]["feature"] = np.asarray(feat, dtype=np.float32)
            g.nodes[i]["feats"] = np.asarray(feat, dtype=np.float32)
        return g

    def _to_plain_dict(self, obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: self._to_plain_dict(v) for k, v in obj.items()}
        if hasattr(obj, "items"):
            return {k: self._to_plain_dict(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._to_plain_dict(v) for v in obj]
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, torch.device):
            return str(obj)
        return obj

    @contextlib.contextmanager
    def _legacy_torch_load(self):
        original_load = torch.load

        def compat_load(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return original_load(*args, **kwargs)

        torch.load = compat_load
        try:
            yield
        finally:
            torch.load = original_load

    @contextlib.contextmanager
    def _repo_cwd(self):
        old_cwd = Path.cwd()
        os.chdir(self.repo_root)
        try:
            yield
        finally:
            os.chdir(old_cwd)

    @contextlib.contextmanager
    def _legacy_networkx_matrix(self):
        if hasattr(nx, "to_numpy_matrix"):
            yield
            return

        def compat_to_numpy_matrix(g, nodelist=None, dtype=None, order=None, multigraph_weight=sum, weight="weight", nonedge=0.0):
            arr = nx.to_numpy_array(
                g,
                nodelist=nodelist,
                dtype=dtype,
                order=order,
                multigraph_weight=multigraph_weight,
                weight=weight,
                nonedge=nonedge,
            )
            return np.asmatrix(arr)

        nx.to_numpy_matrix = compat_to_numpy_matrix
        try:
            yield
        finally:
            delattr(nx, "to_numpy_matrix")
