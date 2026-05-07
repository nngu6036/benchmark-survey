from __future__ import annotations

import contextlib
import importlib
import json
import logging
import os
import pickle
import random
import sys
import types
from pathlib import Path
from typing import Any, Iterable

import networkx as nx
import numpy as np

try:  # Keep the benchmark importable without optional EDP-GNN dependencies.
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

try:
    from easydict import EasyDict as edict
except ModuleNotFoundError:  # pragma: no cover
    class edict(dict):  # type: ignore[no-redef]
        """Minimal EasyDict fallback for wrapper-created configs.

        The uploaded EDP-GNN scripts depend on easydict, but this benchmark
        wrapper no longer imports those scripts. A small fallback is therefore
        enough for the config objects passed to upstream model/sampler helpers.
        """

        def __init__(self, mapping=None, **kwargs):
            super().__init__()
            mapping = {} if mapping is None else dict(mapping)
            mapping.update(kwargs)
            for key, value in mapping.items():
                self[key] = self._wrap(value)

        @classmethod
        def _wrap(cls, value):
            if isinstance(value, dict) and not isinstance(value, cls):
                return cls(value)
            if isinstance(value, list):
                return [cls._wrap(item) for item in value]
            return value

        def __getattr__(self, key):
            try:
                return self[key]
            except KeyError as exc:
                raise AttributeError(key) from exc

        def __setattr__(self, key, value):
            self[key] = self._wrap(value)

        def __delattr__(self, key):
            try:
                del self[key]
            except KeyError as exc:
                raise AttributeError(key) from exc

try:
    from empirical_comparison.models.base import BaseGenerator
except Exception:  # pragma: no cover
    class BaseGenerator:  # type: ignore[no-redef]
        supports_training = True
        supports_sampling = True
        supports_node_features = True
        supports_edge_features = False
        supports_constraints = False
        supports_variable_size = True
        supports_featureless_graphs = True

        def __init__(self, config: dict[str, Any]) -> None:
            self.config = config


LOGGER = logging.getLogger(__name__)


class EDPGNNWrapper(BaseGenerator):
    """Benchmark adapter for the uploaded EDP-GNN implementation.

    EDP-GNN is a score-based graph generator that trains a permutation-equivariant
    score network on noisy adjacency matrices and samples with annealed Langevin
    dynamics. The upstream repository was written around top-level executable
    scripts and assumes that valid nodes are the nonzero rows of an adjacency
    matrix. This wrapper keeps the benchmark interface stable while preserving
    benchmark splits, graph sizes, and optional true isolated nodes.

    What this wrapper imports from upstream
    ---------------------------------------
    * ``utils.loading_utils.get_score_model``
    * ``utils.loading_utils.get_mc_sampler``
    * ``utils.graph_utils.gen_list_of_data``

    What this wrapper intentionally avoids
    --------------------------------------
    * Upstream ``train.py`` and ``sample.py``. Those scripts trigger original
      evaluation/plotting code and assume their own 1024-sample flow.
    * Upstream ``utils.arg_helper.graphs_to_tensor``. It derives node masks from
      adjacency row sums, so true isolated nodes are silently dropped.
    * Upstream ``evaluation.stats.adjs_to_graphs``. It removes isolated nodes.

    Config highlights
    -----------------
    ``preserve_isolated_nodes`` defaults to ``True``. Set it to ``False`` only
    when you explicitly want to reproduce the original upstream convention.
    """

    supports_training = True
    supports_sampling = True
    supports_node_features = True
    supports_edge_features = False
    supports_constraints = False
    supports_variable_size = True
    supports_featureless_graphs = True

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self._require_torch()

        self.dataset_name = self._safe_dataset_name(
            str(config.get("dataset_name") or config.get("dataset") or "empirical_graphs")
        )
        default_repo_root = Path(__file__).resolve().parents[4] / "external" / "EDP-GNN"
        repo_root = os.environ.get("EDP_GNN_REPO") or config.get("repo_root") or default_repo_root
        self.repo_root = self._normalize_repo_root(Path(repo_root).expanduser().resolve())

        ckpt = config.get("checkpoint_path") or "outputs/checkpoints/{dataset}/edp_gnn.pt"
        self.checkpoint_path = self._resolve_path_template(ckpt).expanduser().resolve()

        device_name = config.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device_name)
        self.preserve_isolated_nodes = bool(config.get("preserve_isolated_nodes", True))
        self.feature_normalization = str(config.get("feature_normalization", "per_graph"))

        self.repo_loaded = False
        self.mods: dict[str, Any] = {}
        self.model: Any | None = None
        self.mcmc_sampler: Any | None = None
        self.edp_config: Any | None = None
        self.template_graphs: list[nx.Graph] = []
        self.eval_graphs: list[nx.Graph] = []
        self.feature_dim = 1
        self._shadowed_modules: dict[str, Any] = {}

    @property
    def name(self) -> str:
        return "edp_gnn"

    # ------------------------------------------------------------------
    # Optional dependency and repository handling
    # ------------------------------------------------------------------
    def _require_torch(self) -> None:
        if torch is None:  # pragma: no cover
            raise ModuleNotFoundError(
                "EDPGNNWrapper requires PyTorch. Install EDP-GNN dependencies before using this wrapper."
            )

    def _normalize_repo_root(self, repo_root: Path) -> Path:
        """Accept either the EDP-GNN root or a parent directory containing it."""
        if (repo_root / "train.py").exists() and (repo_root / "model").is_dir():
            return repo_root
        nested = repo_root / "EDP-GNN"
        if (nested / "train.py").exists() and (nested / "model").is_dir():
            return nested
        return repo_root

    def _resolve_path_template(self, value: str | os.PathLike[str]) -> Path:
        text = str(value)
        text = text.replace("${dataset}", self.dataset_name).replace("{dataset}", self.dataset_name)
        text = text.replace("${model}", "edp_gnn").replace("{model}", "edp_gnn")
        return Path(text)

    def _safe_dataset_name(self, name: str) -> str:
        return name.replace("/", "_").replace("\\", "_").replace(" ", "_")

    def _ensure_repo_importable(self) -> None:
        if not (self.repo_root / "train.py").exists() or not (self.repo_root / "utils").is_dir():
            raise FileNotFoundError(
                f"EDP-GNN repository not found at {self.repo_root}. "
                "Set `repo_root` in configs/models/edp_gnn.yaml or set EDP_GNN_REPO."
            )
        root_str = str(self.repo_root)
        if root_str in sys.path:
            sys.path.remove(root_str)
        sys.path.insert(0, root_str)

    def _is_repo_module(self, module: Any) -> bool:
        file = getattr(module, "__file__", None)
        if not file:
            return False
        try:
            return Path(file).resolve().is_relative_to(self.repo_root)
        except Exception:
            return str(Path(file).resolve()).startswith(str(self.repo_root))

    def _drop_conflicting_top_level_module(self, name: str) -> None:
        module = sys.modules.get(name)
        if module is not None and not self._is_repo_module(module):
            self._shadowed_modules.setdefault(name, module)
            del sys.modules[name]

    def _install_visual_utils_stub(self) -> None:
        """Avoid importing old upstream plotting code that the wrapper never uses."""
        module = sys.modules.get("utils.visual_utils")
        if module is not None and self._is_repo_module(module):
            del sys.modules["utils.visual_utils"]
        if "utils.visual_utils" not in sys.modules:
            stub = types.ModuleType("utils.visual_utils")

            def _noop(*args, **kwargs):
                return None

            stub.plot_graphs_adj = _noop
            stub.plot_graphs_list = _noop
            stub.plot_graphs_list_new = _noop
            stub.plot_multi_channel_numpy_adjs = _noop
            stub.plot_multi_channel_numpy_adjs_1b1 = _noop
            sys.modules["utils.visual_utils"] = stub

    def _import_modules(self) -> None:
        if self.repo_loaded:
            return
        self._require_torch()
        self._ensure_repo_importable()

        # EDP-GNN uses absolute top-level imports. Remove modules imported by
        # other wrappers before importing this repository's modules.
        for name in ["train", "sample", "utils", "model", "evaluation"]:
            self._drop_conflicting_top_level_module(name)
        self._install_visual_utils_stub()

        try:
            self.mods["loading_utils"] = importlib.import_module("utils.loading_utils")
            self.mods["graph_utils"] = importlib.import_module("utils.graph_utils")
            self.mods["langevin_mc"] = importlib.import_module("model.langevin_mc")
        except ModuleNotFoundError as exc:  # pragma: no cover - optional upstream deps
            raise ModuleNotFoundError(
                "Failed to import the uploaded EDP-GNN repository. Install its runtime dependencies "
                "such as torch, scipy, PyYAML, and set EDP_GNN_REPO correctly."
            ) from exc
        self.repo_loaded = True

    # ------------------------------------------------------------------
    # Dataset conversion
    # ------------------------------------------------------------------
    def _normalize_graph(self, graph: nx.Graph) -> nx.Graph:
        if graph.number_of_nodes() == 0:
            raise ValueError("EDPGNNWrapper does not support empty graphs.")
        if graph.is_directed():
            graph = nx.Graph(graph)
        graph = nx.convert_node_labels_to_integers(graph.copy())
        graph.remove_edges_from(list(nx.selfloop_edges(graph)))

        raw_features = nx.get_node_attributes(graph, "feature")
        if not raw_features:
            raw_features = nx.get_node_attributes(graph, "feats")

        normalized: dict[int, np.ndarray] = {}
        feature_dim: int | None = None
        for node in range(graph.number_of_nodes()):
            value = raw_features.get(node, np.ones(1, dtype=np.float32))
            arr = np.asarray(value, dtype=np.float32)
            if arr.ndim == 0:
                arr = arr[None]
            arr = arr.reshape(-1)
            feature_dim = feature_dim or int(arr.shape[0])
            if arr.shape[0] != feature_dim:
                raise ValueError("All node features must have the same dimensionality for EDP-GNN.")
            normalized[node] = arr
        nx.set_node_attributes(graph, normalized, "feature")
        nx.set_node_attributes(graph, normalized, "feats")
        return graph

    def _prepare_graph_lists(
        self,
        train_graphs: Iterable[nx.Graph],
        val_graphs: Iterable[nx.Graph] | None = None,
        test_graphs: Iterable[nx.Graph] | None = None,
    ) -> tuple[list[nx.Graph], list[nx.Graph]]:
        train = [self._normalize_graph(g) for g in train_graphs]
        val = [self._normalize_graph(g) for g in (val_graphs or [])]
        test = [self._normalize_graph(g) for g in (test_graphs or [])]
        if not train:
            raise ValueError("EDPGNNWrapper.train() requires at least one training graph.")
        eval_graphs = val or test or train[: max(1, min(len(train), max(1, len(train) // 5)))]
        self.feature_dim = self._infer_feature_dim(train + eval_graphs)
        self.template_graphs = [g.copy() for g in train]
        self.eval_graphs = [g.copy() for g in eval_graphs]
        return train, eval_graphs

    def _infer_feature_dim(self, graphs: list[nx.Graph]) -> int:
        dim: int | None = None
        for graph in graphs:
            features = nx.get_node_attributes(graph, "feature") or nx.get_node_attributes(graph, "feats")
            if not features:
                continue
            for value in features.values():
                arr = np.asarray(value, dtype=np.float32).reshape(-1)
                if dim is None:
                    dim = int(arr.shape[0])
                elif int(arr.shape[0]) != dim:
                    raise ValueError("All node features must share the same dimension across graphs.")
        return dim or 1

    def _graph_to_arrays(self, graph: nx.Graph, max_nodes: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if graph.number_of_nodes() > max_nodes:
            raise ValueError(f"Graph has {graph.number_of_nodes()} nodes, exceeding max_node_num={max_nodes}.")
        n = graph.number_of_nodes()
        nodes = list(range(n))
        adj = nx.to_numpy_array(graph, nodelist=nodes, dtype=np.float32)
        np.fill_diagonal(adj, 0.0)
        padded_adj = np.zeros((max_nodes, max_nodes), dtype=np.float32)
        padded_adj[:n, :n] = adj

        features = nx.get_node_attributes(graph, "feature") or nx.get_node_attributes(graph, "feats")
        feat = np.zeros((max_nodes, self.feature_dim), dtype=np.float32)
        for node in nodes:
            value = features.get(node, np.ones(self.feature_dim, dtype=np.float32))
            arr = np.asarray(value, dtype=np.float32).reshape(-1)
            if arr.shape[0] != self.feature_dim:
                raise ValueError("Node feature dimension changed during tensor conversion.")
            feat[node] = arr

        if self.feature_normalization == "per_graph":
            if n > 0:
                local = feat[:n]
                feat[:n] = (local - local.mean(axis=0, keepdims=True)) / (local.std(axis=0, keepdims=True) + 1e-6)
        elif self.feature_normalization in {"none", "off", "false"}:
            pass
        else:
            raise ValueError("feature_normalization must be 'per_graph' or 'none'.")

        if self.preserve_isolated_nodes:
            node_flags = np.zeros((max_nodes,), dtype=np.float32)
            node_flags[:n] = 1.0
        else:
            node_flags = (padded_adj.sum(axis=-1) > 1e-5).astype(np.float32)
            if node_flags.sum() == 0:
                node_flags[: max(1, n)] = 1.0
        return padded_adj, feat, node_flags

    def _graphs_to_tensors(self, graphs: list[nx.Graph], max_nodes: int | None = None) -> tuple[Any, Any, Any]:
        self._require_torch()
        if not graphs:
            raise ValueError("Cannot convert an empty graph list to EDP-GNN tensors.")
        if max_nodes is None:
            if self.edp_config is not None:
                max_nodes = int(self.edp_config.dataset.max_node_num)
            else:
                max_nodes = max(g.number_of_nodes() for g in graphs)
        adjs, xs, flags = zip(*(self._graph_to_arrays(g, int(max_nodes)) for g in graphs))
        return (
            torch.tensor(np.stack(adjs), dtype=torch.float32),
            torch.tensor(np.stack(xs), dtype=torch.float32),
            torch.tensor(np.stack(flags), dtype=torch.float32),
        )

    def _graphs_to_dataloader(self, graphs: list[nx.Graph], batch_size: int, shuffle: bool) -> Any:
        from torch.utils.data import DataLoader, TensorDataset

        adjs, xs, flags = self._graphs_to_tensors(graphs, int(self.edp_config.dataset.max_node_num))
        return DataLoader(TensorDataset(adjs, xs, flags), batch_size=int(batch_size), shuffle=bool(shuffle))

    def _data_dir(self) -> Path:
        data_dir = self.config.get("data_dir") or (self.repo_root / "data")
        path = self._resolve_path_template(data_dir) if isinstance(data_dir, str) else Path(data_dir)
        return path.expanduser().resolve()

    def _materialize_dataset(self, train_graphs: list[nx.Graph], eval_graphs: list[nx.Graph]) -> None:
        """Write an upstream-compatible data file for provenance/debugging."""
        self._data_dir().mkdir(parents=True, exist_ok=True)
        graph_list = [g.copy() for g in eval_graphs] + [g.copy() for g in train_graphs]
        prefix = self._data_dir() / self.dataset_name
        with open(str(prefix) + ".pkl", "wb") as handle:
            pickle.dump(graph_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
        meta = {
            "dataset_name": self.dataset_name,
            "source": "empirical_comparison",
            "ordering": "eval_graphs_then_train_graphs_for_upstream_load_data_compatibility",
            "num_train_graphs": len(train_graphs),
            "num_eval_graphs": len(eval_graphs),
            "num_graphs": len(graph_list),
            "max_nodes": max(g.number_of_nodes() for g in graph_list),
            "feature_dim": self.feature_dim,
            "preserve_isolated_nodes": self.preserve_isolated_nodes,
            "feature_normalization": self.feature_normalization,
        }
        with open(str(prefix) + ".txt", "w", encoding="utf-8") as handle:
            handle.write(json.dumps(meta, indent=2))

    # ------------------------------------------------------------------
    # Config construction
    # ------------------------------------------------------------------
    def _as_list(self, value: Any) -> list[Any]:
        if isinstance(value, (list, tuple)):
            return list(value)
        return [value]

    def _nested_get(self, *keys: str, default: Any = None) -> Any:
        current: Any = self.config
        for key in keys:
            if not isinstance(current, dict) or key not in current:
                return default
            current = current[key]
        return current

    def _build_config(self, train_graphs: list[nx.Graph], eval_graphs: list[nx.Graph]) -> Any:
        all_graphs = train_graphs + eval_graphs
        observed_max_nodes = max(g.number_of_nodes() for g in all_graphs)
        max_nodes = int(self.config.get("max_node_num") or self._nested_get("dataset", "max_node_num", default=observed_max_nodes))
        if max_nodes < observed_max_nodes:
            raise ValueError("Configured max_node_num is smaller than at least one benchmark graph.")

        model_1 = {
            "dropout_p": 0.0,
            "gnn_hidden_num_list": [16, 16, 16, 16],
            "feature_nums": [16, 16, 16, 16, 16],
            "channel_num_list": [2, 4, 4, 4, 2],
            "name": "gin",
            "use_norm_layers": False,
        }
        model_1.update(dict(self.config.get("model_overrides", {})))

        run_root = self._resolve_path_template(self.config.get("run_root", "outputs/edp_gnn_runs/{dataset}"))
        save_dir = self._resolve_path_template(self.config.get("save_dir", str(run_root))).expanduser().resolve()
        model_save_dir = self._resolve_path_template(self.config.get("model_save_dir", str(save_dir / "models"))).expanduser().resolve()
        save_dir.mkdir(parents=True, exist_ok=True)
        model_save_dir.mkdir(parents=True, exist_ok=True)

        split = len(eval_graphs) / max(1, len(train_graphs) + len(eval_graphs))
        sampling_cfg = self.config.get("sampling", {}) if isinstance(self.config.get("sampling", {}), dict) else {}

        cfg = edict(
            {
                "exp_dir": str(self._resolve_path_template(self.config.get("exp_dir", "outputs/edp_gnn_runs")).expanduser().resolve()),
                "exp_name": str(self.config.get("exp_name", self.dataset_name)),
                "save_dir": str(save_dir),
                "model_save_dir": str(model_save_dir),
                "seed": int(self.config.get("seed", 0)),
                "dataset": edict(
                    {
                        "dataset_size": int(len(train_graphs) + len(eval_graphs)),
                        "max_node_num": int(max_nodes),
                        "name": self.dataset_name,
                        "in_feature": int(self.feature_dim),
                    }
                ),
                "mcmc": edict(
                    {
                        "name": "langevin",
                        "eps": self._as_list(self.config.get("eps", sampling_cfg.get("eps", 0.5))),
                        "fixed_node_number": bool(self.config.get("fixed_node_number", True)),
                        "grad_step_size": self._as_list(self.config.get("grad_step_size", sampling_cfg.get("grad_step_size", 0.01))),
                        "step_num": int(self.config.get("step_num", sampling_cfg.get("num_steps", 1000))),
                    }
                ),
                "model": edict(
                    {
                        "name": "edp-gnn",
                        "models": edict({"model_1": edict(model_1)}),
                        "stack_num": int(self.config.get("stack_num", 1)),
                    }
                ),
                "sample": edict(
                    {"batch_size": int(self.config.get("sample_batch_size", self.config.get("batch_size", 32)))}
                ),
                "test": edict(
                    {
                        "batch_size": int(self.config.get("test_batch_size", self.config.get("batch_size", 32))),
                        "split": float(split),
                    }
                ),
                "train": edict(
                    {
                        "batch_size": int(self.config.get("batch_size", 32)),
                        # Keep the upstream misspelled key: train.py calls this lr_dacey.
                        "lr_dacey": float(self.config.get("lr_dacey", self.config.get("lr_decay", 0.999))),
                        "lr_init": float(self.config.get("learning_rate", self.config.get("lr_init", 1e-3))),
                        "momentum": float(self.config.get("momentum", 0.9)),
                        "max_epoch": int(self.config.get("num_epochs", self.config.get("max_epoch", 100))),
                        "sample_interval": int(self.config.get("sample_interval", 10**9)),
                        "save_interval": int(self.config.get("save_interval", self.config.get("num_epochs", self.config.get("max_epoch", 100)))),
                        "shuffle": bool(self.config.get("shuffle", True)),
                        "sigmas": [float(x) for x in self.config.get("sigmas", [0.1, 0.2, 0.4, 0.6, 0.8, 1.6])],
                        "weight_decay": float(self.config.get("weight_decay", 0.0)),
                    }
                ),
            }
        )
        cfg.dev = self.device
        return cfg

    # ------------------------------------------------------------------
    # Training / loading
    # ------------------------------------------------------------------
    def load(self) -> None:
        self._import_modules()
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"EDP-GNN checkpoint not found: {self.checkpoint_path}. Run train_model.py first or set checkpoint_path."
            )
        with self._legacy_torch_load():
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.edp_config = self._to_edict(checkpoint["config"])
        self.edp_config.dev = self.device
        self.feature_dim = int(checkpoint.get("feature_dim", self.edp_config.dataset.get("in_feature", 1)))
        self.preserve_isolated_nodes = bool(checkpoint.get("preserve_isolated_nodes", self.preserve_isolated_nodes))
        self.feature_normalization = str(checkpoint.get("feature_normalization", self.feature_normalization))
        self.template_graphs = [self._restore_graph(gd) for gd in checkpoint.get("template_graphs", [])]
        self.eval_graphs = [self._restore_graph(gd) for gd in checkpoint.get("eval_graphs", [])]

        get_score_model = self.mods["loading_utils"].get_score_model
        get_mc_sampler = self.mods["loading_utils"].get_mc_sampler
        with self._repo_cwd(), self._legacy_networkx_matrix():
            self.model = get_score_model(self.edp_config, dev=self.device)
            self.model.load_state_dict(checkpoint["model_state"], strict=False)
            self.model.to(self.device)
            self.model.eval()
            self.mcmc_sampler = get_mc_sampler(self.edp_config)

    def train(self, train_graphs, val_graphs=None, test_graphs=None) -> None:
        self._import_modules()
        train_list, eval_list = self._prepare_graph_lists(train_graphs, val_graphs, test_graphs)
        self._materialize_dataset(train_list, eval_list)
        self.edp_config = self._build_config(train_list, eval_list)

        self._seed_everything(int(self.edp_config.seed))
        with self._repo_cwd(), self._legacy_networkx_matrix():
            get_mc_sampler = self.mods["loading_utils"].get_mc_sampler
            get_score_model = self.mods["loading_utils"].get_score_model
            self.mcmc_sampler = get_mc_sampler(self.edp_config)
            self.model = get_score_model(self.edp_config, dev=self.device)
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=float(self.edp_config.train.lr_init),
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=float(self.edp_config.train.weight_decay),
            )
            train_dl = self._graphs_to_dataloader(
                train_list,
                batch_size=int(self.edp_config.train.batch_size),
                shuffle=bool(self.edp_config.train.shuffle),
            )
            eval_dl = self._graphs_to_dataloader(
                eval_list,
                batch_size=int(self.edp_config.test.batch_size),
                shuffle=False,
            )
            self._fit_score_model(optimizer, train_dl, eval_dl)

        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_state": self.model.state_dict(),
            "config": self._to_plain_dict(self.edp_config),
            "template_graphs": [self._serialize_graph(g) for g in self.template_graphs],
            "eval_graphs": [self._serialize_graph(g) for g in self.eval_graphs],
            "feature_dim": int(self.feature_dim),
            "preserve_isolated_nodes": bool(self.preserve_isolated_nodes),
            "feature_normalization": str(self.feature_normalization),
            "wrapper": "EDPGNNWrapper",
            "upstream_repo": str(self.repo_root),
        }
        torch.save(payload, self.checkpoint_path)
        self.model.eval()

    def _fit_score_model(self, optimizer: Any, train_dl: Any, eval_dl: Any) -> None:
        assert self.model is not None and self.edp_config is not None
        gen_list_of_data = self.mods["graph_utils"].gen_list_of_data
        sigma_list = [float(x) for x in self.edp_config.train.sigmas]
        if not sigma_list:
            raise ValueError("EDP-GNN training requires a non-empty sigmas list.")
        max_epoch = int(self.edp_config.train.max_epoch)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=float(self.edp_config.train.lr_dacey))

        for epoch in range(max_epoch):
            self.model.train()
            train_losses: list[float] = []
            for adj_b, x_b, flags_b in train_dl:
                adj_b = adj_b.to(self.device)
                x_b = x_b.to(self.device)
                flags_b = flags_b.to(self.device)
                x_rep, noise_adj_b, flags_rep, grad_log_q_noise_list = gen_list_of_data(x_b, adj_b, flags_b, sigma_list)
                optimizer.zero_grad(set_to_none=True)
                score = self.model(x=x_rep, adjs=noise_adj_b, node_flags=flags_rep)
                loss, _ = self._loss_func(score.chunk(len(sigma_list), dim=0), grad_log_q_noise_list, sigma_list)
                loss.backward()
                optimizer.step()
                train_losses.append(float(loss.detach().cpu().item()))
            scheduler.step()

            eval_losses: list[float] = []
            self.model.eval()
            with torch.no_grad():
                for adj_b, x_b, flags_b in eval_dl:
                    adj_b = adj_b.to(self.device)
                    x_b = x_b.to(self.device)
                    flags_b = flags_b.to(self.device)
                    x_rep, noise_adj_b, flags_rep, grad_log_q_noise_list = gen_list_of_data(x_b, adj_b, flags_b, sigma_list)
                    score = self.model(x=x_rep, adjs=noise_adj_b, node_flags=flags_rep)
                    loss, _ = self._loss_func(score.chunk(len(sigma_list), dim=0), grad_log_q_noise_list, sigma_list)
                    eval_losses.append(float(loss.detach().cpu().item()))

            if epoch % int(self.config.get("log_interval", 1)) == 0 or epoch == max_epoch - 1:
                train_mean = float(np.mean(train_losses)) if train_losses else float("nan")
                eval_mean = float(np.mean(eval_losses)) if eval_losses else float("nan")
                LOGGER.info(
                    "EDP-GNN epoch %04d/%04d train_loss=%.6e eval_loss=%.6e",
                    epoch + 1,
                    max_epoch,
                    train_mean,
                    eval_mean,
                )

    def _loss_func(self, score_list: tuple[Any, ...], grad_log_q_noise_list: list[Any], sigma_list: list[float]) -> tuple[Any, list[float]]:
        loss: Any = 0.0
        loss_items: list[float] = []
        for score, grad_log_q_noise, sigma in zip(score_list, grad_log_q_noise_list, sigma_list):
            cur_loss = 0.5 * sigma**2 * ((score - grad_log_q_noise) ** 2).sum(dim=[-1, -2]).mean()
            loss_items.append(float(cur_loss.detach().cpu().item()))
            loss = loss + cur_loss
        return loss, loss_items

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------
    def _prepare_init_batch(self, batch_size: int) -> tuple[Any, Any, Any]:
        assert self.edp_config is not None and self.mcmc_sampler is not None
        if not self.template_graphs:
            raise RuntimeError("No training template graphs were saved; cannot infer node-count distribution for sampling.")
        indices = np.random.randint(0, len(self.template_graphs), size=batch_size)
        graph_list = [self.template_graphs[i] for i in indices]
        _, base_x, node_flags = self._graphs_to_tensors(graph_list, int(self.edp_config.dataset.max_node_num))
        base_x = base_x.to(self.device)
        node_flags = node_flags.to(self.device)
        init_adjs = self.mcmc_sampler.gen_init_sample(
            batch_size,
            int(self.edp_config.dataset.max_node_num),
            node_flags=node_flags,
        )[0]
        return init_adjs, base_x, node_flags

    def sample(self, num_graphs: int, seed: int = 0):
        self._import_modules()
        if self.model is None or self.mcmc_sampler is None or self.edp_config is None:
            self.load()
        assert self.model is not None and self.mcmc_sampler is not None and self.edp_config is not None

        self._seed_everything(seed)
        self.model.eval()
        sigma_list = sorted([float(x) for x in self.edp_config.train.sigmas])
        if not sigma_list:
            raise ValueError("EDP-GNN config must contain at least one training sigma.")
        batch_size = int(self.edp_config.sample.batch_size)
        step_size_ratio = self.edp_config.mcmc.grad_step_size[0] if isinstance(self.edp_config.mcmc.grad_step_size, (list, tuple)) else self.edp_config.mcmc.grad_step_size
        eps = self.edp_config.mcmc.eps[0] if isinstance(self.edp_config.mcmc.eps, (list, tuple)) else self.edp_config.mcmc.eps
        step_size_ratio = float(step_size_ratio)
        eps = float(eps)

        total_batch = batch_size * len(sigma_list)
        init_adjs, sample_x, sample_node_flags = self._prepare_init_batch(total_batch)
        generated: list[nx.Graph] = []
        warm_up_count = 0

        while len(generated) < num_graphs:
            step_size = (
                step_size_ratio
                * torch.tensor(sigma_list, device=self.device).repeat_interleave(batch_size, dim=0)[..., None, None]
                ** 2
            )
            with torch.no_grad():
                sampled_adjs, _ = self.mcmc_sampler.sample(
                    total_batch,
                    lambda adjs, flags: self.model(sample_x, adjs, flags),
                    max_node_num=int(self.edp_config.dataset.max_node_num),
                    step_num=None,
                    init_adjs=init_adjs,
                    init_flags=sample_node_flags,
                    is_final=False,
                    step_size=step_size,
                    eps=eps,
                )
            sampled_chunks = sampled_adjs.chunk(len(sigma_list), dim=0)
            flag_chunks = sample_node_flags.chunk(len(sigma_list), dim=0)
            if warm_up_count < len(sigma_list):
                warm_up_count += 1
            else:
                rounded_adjs, _ = self.mcmc_sampler.end_sample(sampled_chunks[0], to_int=True)
                generated.extend(
                    self._adjs_to_graphs(
                        rounded_adjs.detach().cpu().numpy(),
                        flag_chunks[0].detach().cpu().numpy(),
                    )
                )

            new_init_adjs, new_x, new_flags = self._prepare_init_batch(batch_size)
            init_adjs = torch.cat(list(sampled_chunks[1:]) + [new_init_adjs], dim=0)
            consumed = sampled_chunks[0].size(0)
            sample_x = torch.cat([sample_x[consumed:], new_x], dim=0)
            sample_node_flags = torch.cat([sample_node_flags[consumed:], new_flags], dim=0)

        result: list[nx.Graph] = []
        for graph in generated[:num_graphs]:
            graph = nx.convert_node_labels_to_integers(graph)
            graph.remove_edges_from(list(nx.selfloop_edges(graph)))
            features = {node: np.ones(self.feature_dim, dtype=np.float32) for node in graph.nodes()}
            nx.set_node_attributes(graph, features, "feature")
            nx.set_node_attributes(graph, features, "feats")
            result.append(graph)
        return result

    def _adjs_to_graphs(self, adjs: np.ndarray, node_flags: np.ndarray | None = None) -> list[nx.Graph]:
        graph_list: list[nx.Graph] = []
        for idx, adj in enumerate(adjs):
            if node_flags is None:
                n = int((np.asarray(adj).sum(axis=-1) > 1e-5).sum())
                n = max(n, 1)
            else:
                n = int(np.asarray(node_flags[idx]).round().astype(bool).sum())
                n = max(n, 1)
            arr = np.asarray(adj[:n, :n], dtype=np.float32)
            arr = ((arr + arr.T) / 2.0 >= 0.5).astype(np.int8)
            np.fill_diagonal(arr, 0)
            graph = nx.from_numpy_array(arr)
            graph.remove_edges_from(list(nx.selfloop_edges(graph)))
            # Do not drop isolates when preserve_isolated_nodes=True. Node count
            # is part of the benchmark distribution.
            if not self.preserve_isolated_nodes:
                graph.remove_nodes_from(list(nx.isolates(graph)))
                if graph.number_of_nodes() < 1:
                    graph.add_node(0)
                graph = nx.convert_node_labels_to_integers(graph)
            graph_list.append(graph)
        return graph_list

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------
    def _serialize_graph(self, graph: nx.Graph) -> dict[str, Any]:
        graph = nx.convert_node_labels_to_integers(graph)
        features = nx.get_node_attributes(graph, "feature") or nx.get_node_attributes(graph, "feats")
        feature_list = []
        for node in range(graph.number_of_nodes()):
            arr = np.asarray(features.get(node, np.ones(self.feature_dim, dtype=np.float32)), dtype=np.float32)
            feature_list.append(arr.reshape(-1).tolist())
        return {
            "num_nodes": int(graph.number_of_nodes()),
            "edges": [(int(u), int(v)) for u, v in graph.edges()],
            "features": feature_list,
        }

    def _restore_graph(self, data: dict[str, Any]) -> nx.Graph:
        graph = nx.Graph()
        graph.add_nodes_from(range(int(data["num_nodes"])))
        graph.add_edges_from([tuple(map(int, edge)) for edge in data.get("edges", [])])
        features = {}
        for node, feature in enumerate(data.get("features", [])):
            features[node] = np.asarray(feature, dtype=np.float32).reshape(-1)
        if not features:
            features = {node: np.ones(self.feature_dim, dtype=np.float32) for node in graph.nodes()}
        nx.set_node_attributes(graph, features, "feature")
        nx.set_node_attributes(graph, features, "feats")
        return graph

    def _to_edict(self, obj: Any) -> Any:
        if isinstance(obj, dict):
            return edict({key: self._to_edict(value) for key, value in obj.items()})
        if isinstance(obj, list):
            return [self._to_edict(value) for value in obj]
        return obj

    def _to_plain_dict(self, obj: Any) -> Any:
        if isinstance(obj, dict):
            return {key: self._to_plain_dict(value) for key, value in obj.items()}
        if hasattr(obj, "items"):
            return {key: self._to_plain_dict(value) for key, value in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._to_plain_dict(value) for value in obj]
        if isinstance(obj, Path):
            return str(obj)
        if torch is not None and isinstance(obj, torch.device):
            return str(obj)
        return obj

    def _seed_everything(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @contextlib.contextmanager
    def _legacy_torch_load(self):
        original_load = torch.load

        def compat_load(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            try:
                return original_load(*args, **kwargs)
            except TypeError:
                kwargs.pop("weights_only", None)
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
        """Patch NetworkX 3.x names expected by optional upstream helpers."""
        has_to = hasattr(nx, "to_numpy_matrix")
        has_from = hasattr(nx, "from_numpy_matrix")
        has_selfloop_edges = hasattr(nx.Graph, "selfloop_edges")

        def compat_to_numpy_matrix(
            graph,
            nodelist=None,
            dtype=None,
            order=None,
            multigraph_weight=sum,
            weight="weight",
            nonedge=0.0,
        ):
            array = nx.to_numpy_array(
                graph,
                nodelist=nodelist,
                dtype=dtype,
                order=order,
                multigraph_weight=multigraph_weight,
                weight=weight,
                nonedge=nonedge,
            )
            return np.asmatrix(array)

        def compat_from_numpy_matrix(A, parallel_edges=False, create_using=None, edge_attr="weight"):
            return nx.from_numpy_array(
                np.asarray(A),
                parallel_edges=parallel_edges,
                create_using=create_using,
                edge_attr=edge_attr,
            )

        if not has_to:
            nx.to_numpy_matrix = compat_to_numpy_matrix  # type: ignore[attr-defined]
        if not has_from:
            nx.from_numpy_matrix = compat_from_numpy_matrix  # type: ignore[attr-defined]
        if not has_selfloop_edges:
            nx.Graph.selfloop_edges = lambda self, data=False, keys=False, default=None: list(  # type: ignore[attr-defined]
                nx.selfloop_edges(self, data=data)
            )
        try:
            yield
        finally:
            if not has_to:
                delattr(nx, "to_numpy_matrix")
            if not has_from:
                delattr(nx, "from_numpy_matrix")
            if not has_selfloop_edges:
                delattr(nx.Graph, "selfloop_edges")
