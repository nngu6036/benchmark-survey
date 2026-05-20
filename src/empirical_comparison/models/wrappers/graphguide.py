from __future__ import annotations

import contextlib
import importlib
import inspect
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import networkx as nx
import numpy as np
import torch

from empirical_comparison.models.base import BaseGenerator
from empirical_comparison.utils.progress import update_progress
from empirical_comparison.utils.numerics import assert_model_tensors_finite, assert_torch_grads_finite


class GraphGUIDEWrapper(BaseGenerator):
    """Benchmark adapter for the uploaded GraphGUIDE repository.

    The uploaded GraphGUIDE code implements graph generation as Bernoulli
    diffusion over the canonical upper-triangular edge vector of a graph.  The
    graph neural network is conditioned on node features and graph size; the
    generated object is the edge set.  This wrapper therefore preserves the
    benchmark train/validation/test split, converts NetworkX graphs into the
    PyG format expected by GraphGUIDE, trains the original GraphGUIDE GNN and
    diffuser objects directly, and stores node-feature templates for later
    sampling.

    The wrapper intentionally does not import GraphGUIDE's ``train_model.py``:
    that file imports Sacred and installs a FileStorageObserver at import time.
    Instead, the training loop below mirrors ``train_graph_model`` while keeping
    benchmark-owned output paths and metadata.
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
        self.model_name = str(config.get("name", "graphguide"))
        self.dataset = str(config.get("dataset", "unknown"))
        self.seed = int(config.get("seed", 42))
        self.device = torch.device(config.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))

        default_repo_root = Path(__file__).resolve().parents[4] / "external" / "GraphGUIDE"
        repo_root = os.environ.get("GRAPHGUIDE_REPO") or config.get("repo_root") or default_repo_root
        self.repo_root = self._normalize_repo_root(Path(repo_root).expanduser().resolve())
        self.repo_src = self.repo_root / "src"

        self.checkpoint_path = self._resolve_path(config.get("checkpoint_path", "outputs/checkpoints/{dataset}/graphguide.pt"))
        self.data_root = self._resolve_path(config.get("data_root", "outputs/graphguide_data/{dataset}"))
        self.run_root = self._resolve_path(config.get("run_root", "outputs/graphguide_runs/{dataset}"))

        self.node_feature_attr = str(config.get("node_feature_attr", "feats"))
        self.force_constant_features = bool(config.get("force_constant_features", False))
        self.default_node_feature_dim = int(config.get("default_node_feature_dim", 1))
        self.drop_node_features_on_output = bool(config.get("drop_node_features_on_output", False))

        self.batch_size = int(config.get("batch_size", 32))
        self.sample_batch_size = int(config.get("sample_batch_size", config.get("batch_size", 32)))
        self.num_workers = int(config.get("num_workers", 0))
        self.num_epochs = int(config.get("num_epochs", 100))
        self.learning_rate = float(config.get("learning_rate", 1e-3))
        self.weight_decay = float(config.get("weight_decay", 0.0))
        self.clip_grad = float(config.get("clip_grad", 1.0)) if config.get("clip_grad", 1.0) is not None else None
        self.val_every = int(config.get("val_every", 0))
        self.t_limit = int(config.get("t_limit", config.get("sampling", {}).get("num_steps", 1000)))

        self.model_type = str(config.get("model_type", "gat")).lower()
        self.model_kwargs = dict(config.get("model_kwargs", {}))
        self.diffuser_type = str(config.get("diffuser_type", "bernoulli_zero_skip")).lower()
        self.diffuser_kwargs = dict(config.get("diffuser_kwargs", {"a": 100, "b": 10}))

        self.template_strategy = str(config.get("template_strategy", "random")).lower()
        self.max_templates = int(config.get("max_templates", 2048))
        self.verbose_sampling = bool(config.get("verbose_sampling", False))

        if config.get("torch_num_threads") is not None:
            torch.set_num_threads(int(config["torch_num_threads"]))

        self.gg_loaded = False
        self.gg_graph_conversions = None
        self.gg_generate = None
        self.gg_gnn = None
        self.gg_digress_gnn = None
        self.gg_diffusers = None
        self.pyg_DataLoader = None
        self.pyg_from_networkx = None
        self.pyg_sort_edge_index = None

        self.model: torch.nn.Module | None = None
        self.diffuser: Any = None
        self.template_graphs: list[nx.Graph] = []
        self.graph_size_counts: dict[int, int] = {}
        self.input_dim: int | None = None
        self.train_metadata: dict[str, Any] = {}

    @property
    def name(self) -> str:
        return "graphguide"

    def _resolve_path(self, value: str | os.PathLike[str]) -> Path:
        raw = str(value)
        replacements = {"dataset": self.dataset, "model": self.model_name, "name": self.model_name}
        for key, replacement in replacements.items():
            raw = raw.replace("${" + key + "}", str(replacement))
        try:
            raw = raw.format(**replacements)
        except Exception:
            pass
        return Path(raw).expanduser().resolve()

    @staticmethod
    def _normalize_repo_root(repo_root: Path) -> Path:
        if repo_root.name == "src" and (repo_root / "model").exists():
            return repo_root.parent
        # Some zips extract as /path/GraphGUIDE/GraphGUIDE.
        if not (repo_root / "src").exists() and (repo_root / "GraphGUIDE" / "src").exists():
            return repo_root / "GraphGUIDE"
        return repo_root

    def _ensure_repo_importable(self) -> None:
        if not self.repo_src.exists():
            raise FileNotFoundError(
                f"GraphGUIDE src directory not found at {self.repo_src}. "
                "Set repo_root in configs/models/graphguide.yaml or export GRAPHGUIDE_REPO."
            )
        if str(self.repo_src) not in sys.path:
            sys.path.insert(0, str(self.repo_src))

    def _import_graphguide_modules(self) -> None:
        if self.gg_loaded:
            return
        self._ensure_repo_importable()
        try:
            from torch_geometric.loader import DataLoader as PyGDataLoader
            from torch_geometric.utils import from_networkx, sort_edge_index
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "GraphGUIDEWrapper requires torch_geometric. Install the upstream "
                "GraphGUIDE dependencies before using model=graphguide."
            ) from exc

        # Import only the upstream modules needed for the benchmark path.  Do
        # not import model.train_model because it requires Sacred at import time.
        self.gg_graph_conversions = importlib.import_module("feature.graph_conversions")
        self.gg_generate = importlib.import_module("model.generate")
        self.gg_gnn = importlib.import_module("model.gnn")
        self.gg_diffusers = importlib.import_module("model.discrete_diffusers")
        try:
            self.gg_digress_gnn = importlib.import_module("model.digress_gnn")
        except Exception:
            self.gg_digress_gnn = None

        self.pyg_DataLoader = PyGDataLoader
        self.pyg_from_networkx = from_networkx
        self.pyg_sort_edge_index = sort_edge_index
        self._patch_upstream_device()
        self.gg_loaded = True

    def _patch_upstream_device(self) -> None:
        # The GraphGUIDE repo sets module-level DEVICE at import time based only
        # on CUDA availability.  The benchmark can explicitly request CPU, so we
        # overwrite the module-level constants to avoid CPU/CUDA tensor mixing.
        device_value = str(self.device)
        for module in (
            self.gg_graph_conversions,
            self.gg_generate,
            self.gg_gnn,
            self.gg_digress_gnn,
            self.gg_diffusers,
        ):
            if module is not None:
                try:
                    setattr(module, "DEVICE", device_value)
                except Exception:
                    pass

    def _set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if self.diffuser is not None and hasattr(self.diffuser, "rng"):
            try:
                self.diffuser.rng.manual_seed(seed)
            except Exception:
                pass

    def _canonical_graph(self, graph: nx.Graph) -> nx.Graph:
        if graph.number_of_nodes() == 0:
            raise ValueError("GraphGUIDEWrapper does not support empty graphs.")
        g = nx.Graph(graph)
        g.remove_edges_from(nx.selfloop_edges(g))
        return nx.convert_node_labels_to_integers(g, ordering="sorted")

    def _node_feature_array(self, graph: nx.Graph, node: int) -> np.ndarray:
        if self.force_constant_features:
            return np.ones(self.default_node_feature_dim, dtype=np.float32)
        value = graph.nodes[node].get(self.node_feature_attr, None)
        if value is None and self.node_feature_attr != "feats":
            value = graph.nodes[node].get("feats", None)
        if value is None:
            return np.ones(self.default_node_feature_dim, dtype=np.float32)
        arr = np.asarray(value, dtype=np.float32)
        if arr.ndim == 0:
            arr = arr[None]
        return arr.astype(np.float32, copy=False)

    def _prepare_graph_features(self, graph: nx.Graph) -> nx.Graph:
        g = self._canonical_graph(graph)
        features: dict[int, np.ndarray] = {}
        dims: set[int] = set()
        for node in range(g.number_of_nodes()):
            arr = self._node_feature_array(g, node)
            features[node] = arr
            dims.add(int(arr.shape[0]))
        if len(dims) != 1:
            raise ValueError(
                f"GraphGUIDE requires a fixed node feature dimension per graph; got dimensions {sorted(dims)}."
            )
        nx.set_node_attributes(g, features, "feats")
        return g

    def _infer_input_dim(self, graphs: Iterable[nx.Graph]) -> int:
        for graph in graphs:
            if graph.number_of_nodes() == 0:
                continue
            g = self._prepare_graph_features(graph)
            return int(np.asarray(g.nodes[0]["feats"]).shape[0])
        return self.default_node_feature_dim

    def _to_pyg_data(self, graph: nx.Graph):
        self._import_graphguide_modules()
        g = self._prepare_graph_features(graph)
        data = self.pyg_from_networkx(g, group_node_attrs=["feats"])
        # from_networkx creates data.x when group_node_attrs is set.  Make sure
        # edge_index exists and is sorted even for edgeless graphs.
        if not hasattr(data, "edge_index") or data.edge_index is None:
            data.edge_index = torch.empty((2, 0), dtype=torch.long)
        data.edge_index = self.pyg_sort_edge_index(data.edge_index)
        return data

    def _make_loader(self, graphs: list[nx.Graph], shuffle: bool):
        self._import_graphguide_modules()
        pyg_graphs = [self._to_pyg_data(g) for g in graphs]
        return self.pyg_DataLoader(
            pyg_graphs,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
        )

    def _model_class(self, model_type: str | None = None):
        self._import_graphguide_modules()
        mt = str(model_type or self.model_type).lower()
        if mt == "gat":
            return self.gg_gnn.GraphLinkGAT
        if mt == "gin":
            return self.gg_gnn.GraphLinkGIN
        if mt in {"digress", "digress_gnn"}:
            if self.gg_digress_gnn is None:
                raise ValueError("GraphGUIDE model_type='digress' requested but model.digress_gnn could not be imported.")
            return self.gg_digress_gnn.DiGressGNN
        raise ValueError("Unsupported GraphGUIDE model_type={!r}. Use 'gat', 'gin', or 'digress'.".format(mt))

    def _filter_constructor_kwargs(self, cls, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Drop config keys that are not accepted by the selected upstream model.

        This lets users switch between GraphLinkGAT, GraphLinkGIN, and the
        GraphGUIDE DiGress-style transformer without having to manually delete
        stale keys from ``model_kwargs``.
        """
        try:
            signature = inspect.signature(cls.__init__)
            accepted = set(signature.parameters) - {"self"}
            has_var_kwargs = any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()
            )
        except Exception:
            return kwargs
        if has_var_kwargs:
            return kwargs
        return {k: v for k, v in kwargs.items() if k in accepted}

    def _build_model(self, creation_args: dict[str, Any] | None = None) -> torch.nn.Module:
        if self.input_dim is None:
            raise RuntimeError("input_dim is unknown; call train() first or provide it in the checkpoint/config.")
        cls = self._model_class()
        if creation_args is not None:
            args = self._filter_constructor_kwargs(cls, dict(creation_args))
            model = cls(**args)
        else:
            args = {"input_dim": int(self.input_dim), "t_limit": int(self.t_limit), **self.model_kwargs}
            args = self._filter_constructor_kwargs(cls, args)
            model = cls(**args)
        return model.to(self.device)

    def _build_diffuser(self, seed: int | None = None):
        self._import_graphguide_modules()
        diffuser_type = self.diffuser_type.lower()
        kwargs = dict(self.diffuser_kwargs)
        kwargs.setdefault("a", 100)
        kwargs.setdefault("b", 10)
        if seed is not None:
            kwargs["seed"] = seed
        elif "seed" not in kwargs:
            kwargs["seed"] = self.seed
        kwargs.setdefault("input_shape", (1,))

        cls_map = {
            "bernoulli": self.gg_diffusers.BernoulliDiffuser,
            "bernoulli_one": self.gg_diffusers.BernoulliOneDiffuser,
            "bernoulli_zero": self.gg_diffusers.BernoulliZeroDiffuser,
            "bernoulli_skip": self.gg_diffusers.BernoulliSkipDiffuser,
            "bernoulli_one_skip": self.gg_diffusers.BernoulliOneSkipDiffuser,
            "bernoulli_zero_skip": self.gg_diffusers.BernoulliZeroSkipDiffuser,
        }
        if diffuser_type not in cls_map:
            raise ValueError(f"Unsupported GraphGUIDE diffuser_type: {diffuser_type}")
        return cls_map[diffuser_type](**kwargs)

    def _graph_size_counts(self, graphs: list[nx.Graph]) -> dict[int, int]:
        counts: dict[int, int] = {}
        for graph in graphs:
            n = int(graph.number_of_nodes())
            counts[n] = counts.get(n, 0) + 1
        return counts

    def _select_template_graphs(self, train_graphs: list[nx.Graph]) -> list[nx.Graph]:
        templates = [self._prepare_graph_features(g) for g in train_graphs]
        if self.max_templates > 0 and len(templates) > self.max_templates:
            rng = np.random.default_rng(self.seed)
            indices = rng.choice(len(templates), size=self.max_templates, replace=False)
            templates = [templates[int(i)] for i in indices]
        return templates

    def _run_one_epoch(self, loader, optimizer: torch.optim.Optimizer | None) -> float:
        is_train = optimizer is not None
        self.model.train(is_train)
        losses: list[float] = []
        grad_context = contextlib.nullcontext() if is_train else torch.no_grad()
        with grad_context:
            for data in loader:
                data = data.to(self.device)
                e0, edge_batch_inds = self.gg_graph_conversions.pyg_data_to_edge_vector(data, return_batch_inds=True)
                if e0.numel() == 0:
                    continue

                graph_sizes = torch.diff(data.ptr)
                graph_times = torch.randint(
                    self.t_limit,
                    size=(graph_sizes.shape[0],),
                    device=self.device,
                ) + 1
                t_v = graph_times[data.batch].float()
                t_e = graph_times[edge_batch_inds].float()

                et, true_post = self.diffuser.forward(e0[:, None].float(), t_e)
                et = et[:, 0]
                true_post = true_post[:, 0].float()

                noisy_data = data.clone()
                noisy_data.edge_index = self.gg_graph_conversions.edge_vector_to_pyg_data(noisy_data, et)

                pred_post = self.model(noisy_data, t_v).float()
                pred_post = torch.clamp(torch.nan_to_num(pred_post, nan=0.5, posinf=1.0, neginf=0.0), 1e-6, 1 - 1e-6)
                loss = self.model.loss(pred_post, true_post)
                if not torch.isfinite(loss):
                    continue

                if is_train:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    try:
                        assert_torch_grads_finite(self.model, context=f"GraphGUIDE {self.dataset} training")
                    except FloatingPointError as exc:
                        print(f"[GraphGUIDE:{self.dataset}] skipping non-finite gradients: {exc}", flush=True)
                        optimizer.zero_grad(set_to_none=True)
                        continue
                    if self.clip_grad is not None and self.clip_grad > 0:
                        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad, error_if_nonfinite=False)
                        if torch.is_tensor(grad_norm) and not torch.isfinite(grad_norm).all():
                            print(f"[GraphGUIDE:{self.dataset}] skipping non-finite clipped gradient norm", flush=True)
                            optimizer.zero_grad(set_to_none=True)
                            continue
                    optimizer.step()
                    assert_model_tensors_finite(self.model, context=f"GraphGUIDE {self.dataset} parameters")
                losses.append(float(loss.detach().cpu().item()))
        return float(np.mean(losses)) if losses else float("nan")

    def train(self, train_graphs, val_graphs=None, test_graphs=None) -> None:
        train_graphs = list(train_graphs or [])
        val_graphs = list(val_graphs or [])
        test_graphs = list(test_graphs or [])
        if not train_graphs:
            raise ValueError("GraphGUIDEWrapper.train() requires at least one training graph.")

        self._set_seed(self.seed)
        self._import_graphguide_modules()
        self.input_dim = self._infer_input_dim(train_graphs)
        self.template_graphs = self._select_template_graphs(train_graphs)
        self.graph_size_counts = self._graph_size_counts(train_graphs)

        self.model = self._build_model()
        # Ensure non-standard models such as DiGressGNN have enough metadata for
        # benchmark checkpoint restoration even if upstream creation_args is empty.
        if not getattr(self.model, "creation_args", None):
            self.model.creation_args = {"input_dim": int(self.input_dim), "t_limit": int(self.t_limit), **self.model_kwargs}
        self.diffuser = self._build_diffuser(seed=self.seed)

        train_loader = self._make_loader(train_graphs, shuffle=True)
        val_loader = self._make_loader(val_graphs, shuffle=False) if val_graphs else None

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        self.run_root.mkdir(parents=True, exist_ok=True)
        self.data_root.mkdir(parents=True, exist_ok=True)
        self._write_provenance(train_graphs, val_graphs, test_graphs)

        best_state: dict[str, torch.Tensor] | None = None
        best_loss = float("inf")
        history: list[dict[str, Any]] = []
        start = time.perf_counter()
        for epoch in range(1, self.num_epochs + 1):
            train_loss = self._run_one_epoch(train_loader, optimizer)
            val_loss = None
            if val_loader is not None and self.val_every > 0 and (epoch % self.val_every == 0 or epoch == self.num_epochs):
                val_loss = self._run_one_epoch(val_loader, optimizer=None)
            score = val_loss if val_loss is not None and np.isfinite(val_loss) else train_loss
            if np.isfinite(score) and score < best_loss:
                best_loss = float(score)
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
            record = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
            history.append(record)
            if val_loss is None:
                print(f"[GraphGUIDE] epoch {epoch}/{self.num_epochs} train_loss={train_loss:.4f}")
            else:
                print(f"[GraphGUIDE] epoch {epoch}/{self.num_epochs} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        if best_state is not None:
            self.model.load_state_dict(best_state)
        elapsed = time.perf_counter() - start
        self.train_metadata = {
            "dataset": self.dataset,
            "model": self.model_name,
            "model_type": self.model_type,
            "diffuser_type": self.diffuser_type,
            "num_epochs": self.num_epochs,
            "runtime_seconds": elapsed,
            "history": history,
            "best_loss": best_loss,
            "split_sizes": {"train": len(train_graphs), "val": len(val_graphs), "test": len(test_graphs)},
        }
        self._save_checkpoint()
        self.model.eval()
        print(f"[GraphGUIDE] saved checkpoint to {self.checkpoint_path}")

    def _write_provenance(self, train_graphs: list[nx.Graph], val_graphs: list[nx.Graph], test_graphs: list[nx.Graph]) -> None:
        payload = {
            "dataset": self.dataset,
            "model": self.model_name,
            "node_feature_attr": self.node_feature_attr,
            "input_dim": self.input_dim,
            "graph_size_counts": {str(k): int(v) for k, v in self.graph_size_counts.items()},
            "split_sizes": {"train": len(train_graphs), "val": len(val_graphs), "test": len(test_graphs)},
            "notes": (
                "GraphGUIDE conditions edge generation on node features and graph size. "
                "The benchmark wrapper saves train-split node-feature templates for sampling."
            ),
        }
        self.data_root.mkdir(parents=True, exist_ok=True)
        with (self.data_root / "metadata.json").open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        torch.save(
            {
                "train": [self._serialize_graph(g) for g in train_graphs],
                "val": [self._serialize_graph(g) for g in val_graphs],
                "test": [self._serialize_graph(g) for g in test_graphs],
            },
            self.data_root / "benchmark_splits.pt",
        )

    def _save_checkpoint(self) -> None:
        if self.model is None:
            raise RuntimeError("No GraphGUIDE model exists to save.")
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format": "empirical_comparison.graphguide.v2",
                "model_state": self.model.state_dict(),
                "model_creation_args": getattr(self.model, "creation_args", {"input_dim": self.input_dim, "t_limit": self.t_limit}),
                "model_type": self.model_type,
                "input_dim": int(self.input_dim or 1),
                "t_limit": int(self.t_limit),
                "diffuser_type": self.diffuser_type,
                "diffuser_kwargs": dict(self.diffuser_kwargs),
                "wrapper_config": dict(self.config),
                "graph_size_counts": {str(k): int(v) for k, v in self.graph_size_counts.items()},
                "template_graphs": [self._serialize_graph(g) for g in self.template_graphs],
                "train_metadata": self.train_metadata,
            },
            self.checkpoint_path,
        )

    def load(self) -> None:
        self._import_graphguide_modules()
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"GraphGUIDE checkpoint not found: {self.checkpoint_path}. "
                "Run scripts/train_model.py first or set checkpoint_path."
            )
        with self._legacy_torch_load():
            ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)

        wrapper_cfg = ckpt.get("wrapper_config") if isinstance(ckpt, dict) else None
        if isinstance(wrapper_cfg, dict):
            # Use checkpoint defaults unless current config explicitly overrides them.
            self.model_type = str(self.config.get("model_type", ckpt.get("model_type", wrapper_cfg.get("model_type", self.model_type)))).lower()
            self.diffuser_type = str(self.config.get("diffuser_type", ckpt.get("diffuser_type", wrapper_cfg.get("diffuser_type", self.diffuser_type)))).lower()
            if "diffuser_kwargs" not in self.config:
                self.diffuser_kwargs = dict(ckpt.get("diffuser_kwargs", wrapper_cfg.get("diffuser_kwargs", self.diffuser_kwargs)))
            self.t_limit = int(self.config.get("t_limit", ckpt.get("t_limit", wrapper_cfg.get("t_limit", self.t_limit))))

        self.input_dim = int(ckpt.get("input_dim") or (ckpt.get("model_creation_args") or {}).get("input_dim") or self.config.get("input_dim") or self.default_node_feature_dim)
        creation_args = dict(ckpt.get("model_creation_args") or {})
        if not creation_args:
            creation_args = {"input_dim": int(self.input_dim), "t_limit": int(self.t_limit), **self.model_kwargs}
        creation_args.setdefault("input_dim", int(self.input_dim))
        creation_args.setdefault("t_limit", int(self.t_limit))

        self.model = self._build_model(creation_args=creation_args)
        state = ckpt.get("model_state", ckpt.get("model_state_dict"))
        if state is None:
            raise ValueError(f"Checkpoint {self.checkpoint_path} does not contain model_state.")
        self.model.load_state_dict(state)
        self.model.eval()
        self.diffuser = self._build_diffuser(seed=self.seed)

        self.graph_size_counts = {int(k): int(v) for k, v in (ckpt.get("graph_size_counts") or {}).items()}
        self.template_graphs = [self._restore_graph(g) for g in ckpt.get("template_graphs", [])]
        self.train_metadata = dict(ckpt.get("train_metadata", {}))
        if not self.template_graphs:
            self.template_graphs = self._fallback_templates_from_size_counts()

    @contextlib.contextmanager
    def _legacy_torch_load(self):
        # Kept for compatibility with wrappers/models saved before PyTorch's
        # weights_only default changed.
        original_load = torch.load

        def compat_load(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return original_load(*args, **kwargs)

        torch.load = compat_load
        try:
            yield
        finally:
            torch.load = original_load

    def _fallback_templates_from_size_counts(self) -> list[nx.Graph]:
        if not self.graph_size_counts:
            node_counts = self.config.get("sample_node_counts")
            if node_counts:
                self.graph_size_counts = {int(n): 1 for n in node_counts}
        if not self.graph_size_counts:
            min_nodes = int(self.config.get("min_nodes", 10))
            max_nodes = int(self.config.get("max_nodes", min_nodes))
            self.graph_size_counts = {n: 1 for n in range(min_nodes, max_nodes + 1)}
        templates: list[nx.Graph] = []
        for n, count in sorted(self.graph_size_counts.items()):
            for _ in range(min(max(int(count), 1), 16)):
                g = nx.empty_graph(int(n))
                for i in range(int(n)):
                    g.nodes[i]["feats"] = np.ones(int(self.input_dim or self.default_node_feature_dim), dtype=np.float32)
                templates.append(g)
        return templates

    def _choose_templates(self, num_graphs: int, seed: int) -> list[nx.Graph]:
        if not self.template_graphs:
            raise RuntimeError("GraphGUIDE sampling requires template graphs or graph_size_counts in the checkpoint.")
        if self.template_strategy == "cycle":
            return [self.template_graphs[i % len(self.template_graphs)].copy() for i in range(num_graphs)]
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(self.template_graphs), size=num_graphs, replace=True)
        return [self.template_graphs[int(i)].copy() for i in indices]

    def _prepare_initial_sample(self, template_graphs: list[nx.Graph]):
        self._import_graphguide_modules()
        pyg_graphs = [self._to_pyg_data(g) for g in template_graphs]
        batch = next(iter(self.pyg_DataLoader(pyg_graphs, batch_size=len(pyg_graphs), shuffle=False)))
        batch = batch.to(self.device)

        num_edges = int(self.gg_graph_conversions.pyg_data_to_edge_vector(batch).numel())
        if num_edges == 0:
            return batch
        t_e = torch.full((num_edges,), self.t_limit, device=self.device, dtype=torch.float32)
        prior_edges = self.diffuser.sample_prior(num_edges, t_e)
        if prior_edges.ndim > 1:
            prior_edges = prior_edges[:, 0]
        prior_edges = torch.clamp(torch.nan_to_num(prior_edges.float(), nan=0.0), 0, 1)
        prior_edges = torch.round(prior_edges)
        batch.edge_index = self.gg_graph_conversions.edge_vector_to_pyg_data(batch, prior_edges)
        return batch

    def sample(self, num_graphs: int, seed: int = 0, progress_callback=None):
        if self.model is None or self.diffuser is None:
            raise RuntimeError("Call load() or train() before sample().")
        if num_graphs <= 0:
            return []
        self._set_seed(seed)
        self.model.eval()

        chosen = self._choose_templates(num_graphs, seed)
        generated: list[nx.Graph] = []
        for start in range(0, num_graphs, self.sample_batch_size):
            batch_templates = chosen[start : start + self.sample_batch_size]
            initial_batch = self._prepare_initial_sample(batch_templates)
            if int(self.gg_graph_conversions.pyg_data_to_edge_vector(initial_batch).numel()) == 0:
                samples = initial_batch
            else:
                samples = self.gg_generate.generate_graph_samples(
                    self.model,
                    self.diffuser,
                    initial_samples=initial_batch,
                    t_start=0,
                    t_limit=self.t_limit,
                    return_all_times=False,
                    verbose=self.verbose_sampling,
                )
            batch_graphs = self.gg_graph_conversions.split_pyg_data_to_nx_graphs(samples)
            before = len(generated)
            generated.extend(self._postprocess_output_graph(g) for g in batch_graphs)
            update_progress(progress_callback, min(len(generated), num_graphs) - min(before, num_graphs))
        return generated[:num_graphs]

    def _postprocess_output_graph(self, graph: nx.Graph) -> nx.Graph:
        g = nx.Graph(graph)
        g.remove_edges_from(nx.selfloop_edges(g))
        g = nx.convert_node_labels_to_integers(g, ordering="sorted")
        if self.drop_node_features_on_output:
            for _, attrs in g.nodes(data=True):
                attrs.pop("feats", None)
                attrs.pop(self.node_feature_attr, None)
        return g

    def _serialize_graph(self, graph: nx.Graph) -> dict[str, Any]:
        g = self._prepare_graph_features(graph)
        features = []
        for i in range(g.number_of_nodes()):
            arr = np.asarray(g.nodes[i].get("feats", np.ones(self.default_node_feature_dim)), dtype=np.float32)
            if arr.ndim == 0:
                arr = arr[None]
            features.append(arr.tolist())
        return {
            "num_nodes": int(g.number_of_nodes()),
            "edges": [[int(u), int(v)] for u, v in g.edges()],
            "features": features,
        }

    def _restore_graph(self, data: dict[str, Any]) -> nx.Graph:
        g = nx.Graph()
        g.add_nodes_from(range(int(data.get("num_nodes", len(data.get("features", []))))))
        g.add_edges_from((int(u), int(v)) for u, v in data.get("edges", []))
        features = data.get("features") or []
        for i in range(g.number_of_nodes()):
            if i < len(features):
                arr = np.asarray(features[i], dtype=np.float32)
            else:
                arr = np.ones(int(self.input_dim or self.default_node_feature_dim), dtype=np.float32)
            if arr.ndim == 0:
                arr = arr[None]
            g.nodes[i]["feats"] = arr
        return g
