from __future__ import annotations

import contextlib
import importlib
import json
import math
import os
import pickle
import random
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import torch
import yaml

from empirical_comparison.utils.progress import update_progress
from torch.utils.data import DataLoader, TensorDataset


class _EasyDict(dict):
    """Tiny EasyDict replacement used to avoid depending on upstream easydict."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as exc:  # pragma: no cover - mirrors normal attribute errors.
            raise AttributeError(key) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def __delattr__(self, key: str) -> None:
        try:
            del self[key]
        except KeyError as exc:  # pragma: no cover
            raise AttributeError(key) from exc




def _ensure_easydict_fallback() -> None:
    """Provide a minimal easydict module for loading raw upstream checkpoints."""
    if "easydict" in sys.modules:
        return
    try:
        import easydict  # noqa: F401
        return
    except Exception:
        module = types.ModuleType("easydict")
        module.EasyDict = _EasyDict
        sys.modules["easydict"] = module

def _to_edict(obj: Any) -> Any:
    if isinstance(obj, _EasyDict):
        return obj
    if isinstance(obj, Mapping):
        return _EasyDict({k: _to_edict(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_edict(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_to_edict(v) for v in obj)
    return obj


def _to_plain(obj: Any) -> Any:
    if isinstance(obj, Mapping):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_plain(v) for v in obj]
    if isinstance(obj, tuple):
        return [_to_plain(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, torch.Tensor):
        if obj.ndim == 0:
            return obj.item()
        return obj.detach().cpu().tolist()
    return obj


@dataclass
class _GruMModules:
    transformer_mod: Any
    mix_mod: Any
    solver_mod: Any
    ema_mod: Any
    graph_utils_mod: Any
    node_features_mod: Any


@contextlib.contextmanager
def _pushd(path: Path) -> Iterator[None]:
    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


@contextlib.contextmanager
def _prepend_sys_path(path: Path) -> Iterator[None]:
    s = str(path)
    inserted = False
    if s not in sys.path:
        sys.path.insert(0, s)
        inserted = True
    try:
        yield
    finally:
        if inserted and s in sys.path:
            sys.path.remove(s)


@contextlib.contextmanager
def _isolated_upstream_import(project_root: Path) -> Iterator[None]:
    """Import GruM modules without leaving generic top-level names in sys.modules.

    The uploaded GruM code uses imports such as ``from utils.graph_utils import ...``
    and ``from models.transformer import ...``. Several other wrappers also load
    upstream repositories that use top-level ``utils`` or ``models`` packages. This
    context temporarily removes such modules, imports GruM, then restores the prior
    interpreter state. Module objects returned by importlib remain valid because they
    keep references to their imported globals.
    """

    prefixes = (
        "utils",
        "utils.",
        "models",
        "models.",
        "mix",
        "solver",
        "losses",
        "sampler",
        "trainer",
        "parsers",
        "parsers.",
    )
    saved = {
        name: module
        for name, module in list(sys.modules.items())
        if any(name == p.rstrip(".") or name.startswith(p) for p in prefixes)
    }
    for name in saved:
        sys.modules.pop(name, None)
    try:
        with _prepend_sys_path(project_root), _pushd(project_root):
            yield
    finally:
        for name in list(sys.modules):
            if any(name == p.rstrip(".") or name.startswith(p) for p in prefixes):
                sys.modules.pop(name, None)
        sys.modules.update(saved)


class GruMWrapper:
    """Benchmark wrapper for GruM's 2D generic-graph implementation.

    The uploaded GruM repository contains separate projects for 2D graph/molecule
    generation and 3D molecule generation. This wrapper targets ``GruM_2D`` for
    generic structure-only graphs such as SBM and planar graphs.

    Key differences from the upstream CLI path:
    - It does not call ``main.py``, ``trainer.py``, or ``sampler.py``. Those scripts
      rebuild upstream data/evaluation flows and require optional ORCA/graph-tool/RDKit
      paths that are unnecessary for this benchmark.
    - It uses GruM's real dense ``GraphTransformer``, ``DiffusionMixture``, Euler and
      Langevin solver components, and EMA implementation.
    - It preserves benchmark train/validation/test splits directly.
    - It uses explicit node masks, because upstream GruM infers valid nodes from
      nonzero adjacency rows and therefore treats genuine isolated nodes as padding.
    """

    supports_training = True
    supports_sampling = True
    supports_node_features = False
    supports_edge_features = False
    supports_constraints = False
    supports_variable_size = True
    supports_featureless_graphs = True

    def __init__(self, config: Dict[str, Any]):
        self.config: Dict[str, Any] = dict(config or {})
        default_repo_root = Path(__file__).resolve().parents[4] / "external" / "GruM"
        repo_root = os.environ.get("GRUM_REPO") or self.config.get("repo_root") or default_repo_root
        self.repo_root = self._normalize_repo_root(Path(repo_root).expanduser().resolve())
        self.project_root = self.repo_root / "GruM_2D"
        if not self.project_root.exists():
            # Also support GRUM_REPO=/path/to/GruM_2D.
            if self.repo_root.name == "GruM_2D" and (self.repo_root / "config").exists():
                self.project_root = self.repo_root
                self.repo_root = self.project_root.parent
            else:
                raise FileNotFoundError(
                    f"Could not find GruM_2D under repo_root={self.repo_root}. "
                    "Set repo_root in configs/models/grum.yaml or export GRUM_REPO."
                )

        dataset = self.config.get("dataset") or self.config.get("dataset_name") or "planar"
        self.dataset_name = str(dataset).lower()
        self.base_config_name = str(self.config.get("base_config") or self._default_base_config(self.dataset_name))

        self.checkpoint_path = self._format_path(
            self.config.get("checkpoint_path", "outputs/checkpoints/{dataset}/grum.pt")
        )
        self.run_root = self._format_path(self.config.get("run_root", "outputs/grum_runs/{dataset}"))
        self.data_root = self._format_path(self.config.get("data_root", "outputs/grum_data/{dataset}"))
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        self.run_root.mkdir(parents=True, exist_ok=True)
        self.data_root.mkdir(parents=True, exist_ok=True)

        torch_num_threads = self.config.get("torch_num_threads")
        if torch_num_threads is not None:
            try:
                torch.set_num_threads(int(torch_num_threads))
            except Exception:
                pass

        self._modules: Optional[_GruMModules] = None
        self._model: Optional[torch.nn.Module] = None
        self._ema: Optional[Any] = None
        self._loaded_config: Optional[_EasyDict] = None
        self._loaded_params: Optional[Dict[str, Any]] = None
        self._loaded_metadata: Optional[Dict[str, Any]] = None
        self._train_node_counts: Optional[List[int]] = None
        self._train_graphs: Optional[List[nx.Graph]] = None
        self._ema_copied_for_sampling = False

    @property
    def name(self) -> str:
        return "grum"

    @classmethod
    def capabilities(cls) -> Dict[str, bool]:
        return {
            "supports_training": cls.supports_training,
            "supports_sampling": cls.supports_sampling,
            "supports_node_features": cls.supports_node_features,
            "supports_edge_features": cls.supports_edge_features,
            "supports_node_labels": getattr(cls, "supports_node_labels", False),
            "supports_edge_labels": getattr(cls, "supports_edge_labels", False),
            "supports_graph_labels": getattr(cls, "supports_graph_labels", False),
            "supports_constraints": cls.supports_constraints,
            "supports_variable_size": cls.supports_variable_size,
            "supports_featureless_graphs": cls.supports_featureless_graphs,
        }

    # ------------------------------------------------------------------
    # Public benchmark API
    # ------------------------------------------------------------------
    def load(self) -> None:
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"GruM checkpoint not found: {self.checkpoint_path}. "
                "Train the wrapper first or set checkpoint_path to a compatible checkpoint."
            )
        mods = self._import_modules()
        _ensure_easydict_fallback()
        ckpt = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)

        if "benchmark_wrapper" in ckpt:
            config = _to_edict(ckpt["config"])
            params = ckpt["params"]
            state_dict = ckpt["state_dict"]
            ema_state = ckpt.get("ema")
            metadata = dict(ckpt.get("metadata", {}))
        else:
            # Best-effort compatibility with raw upstream GruM checkpoints.
            config = _to_edict(_to_plain(ckpt["config"]))
            params = _to_plain(ckpt["params"])
            state_dict = ckpt["state_dict"]
            ema_state = ckpt.get("ema")
            metadata = {
                "source": "upstream_grum_checkpoint",
                "node_counts": [],
                "preserve_isolated_nodes": False,
            }

        model = self._instantiate_model(params)
        if state_dict and "module." in next(iter(state_dict.keys())):
            state_dict = {k[7:]: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        model.to(self._device())
        model.eval()

        ema = None
        if ema_state is not None:
            ema = mods.ema_mod.ExponentialMovingAverage(model.parameters(), decay=float(config.train.ema))
            # Raw upstream EMA state stores tensors on the original device.
            ema_state = dict(ema_state)
            if "shadow_params" in ema_state:
                ema_state["shadow_params"] = [p.to(self._device()) for p in ema_state["shadow_params"]]
            ema.load_state_dict(ema_state)

        self._model = model
        self._ema = ema
        self._loaded_config = config
        self._loaded_params = params
        self._loaded_metadata = metadata
        self._ema_copied_for_sampling = False

        node_counts = metadata.get("node_counts") or metadata.get("train_node_counts") or []
        self._train_node_counts = [int(n) for n in node_counts if int(n) > 0]

        split_path = self.data_root / "benchmark_splits.pkl"
        if split_path.exists():
            with open(split_path, "rb") as f:
                splits = pickle.load(f)
            self._train_graphs = splits.get("train") or None
            if not self._train_node_counts and self._train_graphs:
                self._train_node_counts = [g.number_of_nodes() for g in self._train_graphs]

    def train(
        self,
        train_graphs: Sequence[nx.Graph],
        val_graphs: Optional[Sequence[nx.Graph]] = None,
        test_graphs: Optional[Sequence[nx.Graph]] = None,
    ) -> None:
        self._set_seed(int(self._cfg("seed", 0)))
        mods = self._import_modules()

        train_split = self._prepare_graphs(train_graphs)
        val_split = self._prepare_graphs(val_graphs) if val_graphs is not None else []
        test_split = self._prepare_graphs(test_graphs) if test_graphs is not None else []
        if not train_split:
            raise ValueError("GruMWrapper received an empty training graph split.")
        if not val_split:
            val_split = list(train_split[: max(1, min(len(train_split), len(train_split) // 10 or 1))])
        if not test_split:
            test_split = list(val_split)

        print(
            f"[GruM:{self.dataset_name}] preparing {len(train_split)} train / "
            f"{len(val_split)} val / {len(test_split)} test graphs",
            flush=True,
        )
        config = self._build_config(train_split)
        print(
            f"[GruM:{self.dataset_name}] config resolved: max_node_num={int(config.data.max_node_num)} "
            f"batch_size={int(config.data.batch_size)} epochs={int(config.train.num_epochs)} "
            f"feat_types={list(config.data.feat.type)}",
            flush=True,
        )
        self._write_provenance(train_split, val_split, test_split, config)

        print(f"[GruM:{self.dataset_name}] building training tensors", flush=True)
        train_x, train_adj, train_mask = self._graphs_to_tensors(train_split, config)
        print(
            f"[GruM:{self.dataset_name}] tensors ready: x={tuple(train_x.shape)} "
            f"adj={tuple(train_adj.shape)}",
            flush=True,
        )
        dataset = TensorDataset(train_x, train_adj, train_mask)
        loader = DataLoader(
            dataset,
            batch_size=int(config.data.batch_size),
            shuffle=True,
            num_workers=int(self._cfg("num_workers", 0)),
            drop_last=False,
        )

        params = self._model_params(config)
        model = self._instantiate_model(params).to(self._device())
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(config.train.lr),
            amsgrad=True,
            weight_decay=float(config.train.weight_decay),
        )
        scheduler = None
        if bool(config.train.lr_schedule):
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=float(config.train.lr_decay))
        ema = mods.ema_mod.ExponentialMovingAverage(model.parameters(), decay=float(config.train.ema))
        loss_fn = self._build_loss_fn(config)

        history: List[Dict[str, float]] = []
        num_epochs = int(config.train.num_epochs)
        log_every = int(self._cfg("log_every", max(1, min(50, num_epochs))))
        batch_log_every = int(self._cfg("batch_log_every", 50))
        grad_clip = float(config.train.grad_norm) if float(config.train.grad_norm) > 0 else None

        model.train()
        for epoch in range(num_epochs):
            losses_x: List[float] = []
            losses_adj: List[float] = []
            losses_total: List[float] = []
            for x, adj, mask in loader:
                x = x.to(self._device())
                adj = adj.to(self._device())
                mask = mask.to(self._device())
                if bool(config.data.perm_mix):
                    x, adj, mask = self._rand_perm(x, adj, mask)
                optimizer.zero_grad(set_to_none=True)
                loss, loss_x, loss_adj = loss_fn(model, x, adj, mask)
                loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                ema.update(model.parameters())
                losses_total.append(float(loss.detach().cpu()))
                losses_x.append(float(loss_x.detach().cpu()))
                losses_adj.append(float(loss_adj.detach().cpu()))
                if batch_log_every > 0 and (len(losses_total) % batch_log_every == 0):
                    print(
                        f"[GruM:{self.dataset_name}] epoch {epoch + 1}/{num_epochs} "
                        f"batch {len(losses_total)}/{len(loader)} "
                        f"loss={float(np.mean(losses_total)):.4e}",
                        flush=True,
                    )
            if scheduler is not None:
                scheduler.step()
            epoch_stats = {
                "epoch": epoch + 1,
                "loss": float(np.mean(losses_total)) if losses_total else math.nan,
                "loss_x": float(np.mean(losses_x)) if losses_x else math.nan,
                "loss_adj": float(np.mean(losses_adj)) if losses_adj else math.nan,
            }
            history.append(epoch_stats)
            if log_every > 0 and ((epoch + 1) % log_every == 0 or epoch == 0 or epoch + 1 == num_epochs):
                print(
                    f"[GruM:{self.dataset_name}] epoch {epoch + 1}/{num_epochs} "
                    f"loss={epoch_stats['loss']:.4e} x={epoch_stats['loss_x']:.4e} "
                    f"adj={epoch_stats['loss_adj']:.4e}",
                    flush=True,
                )

        metadata = self._metadata(train_split, val_split, test_split, config)
        metadata["history_tail"] = history[-20:]
        metadata["benchmark_wrapper_version"] = 3
        ckpt = {
            "benchmark_wrapper": "GruMWrapper",
            "epoch": num_epochs,
            "config": _to_plain(config),
            "wrapper_config": _to_plain(self.config),
            "params": _to_plain(params),
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "optimizer": optimizer.state_dict(),
            "ema": ema.state_dict(),
            "metadata": metadata,
        }
        # Keep EMA tensors on CPU for portable checkpoints.
        ckpt["ema"]["shadow_params"] = [p.detach().cpu() for p in ckpt["ema"].get("shadow_params", [])]
        torch.save(ckpt, self.checkpoint_path)

        self._model = model.eval()
        self._ema = ema
        self._loaded_config = config
        self._loaded_params = params
        self._loaded_metadata = metadata
        self._train_node_counts = metadata["node_counts"]
        self._train_graphs = train_split
        self._ema_copied_for_sampling = False

    def sample(self, num_graphs: int, seed: int = 0, progress_callback=None) -> List[nx.Graph]:
        if self._model is None or self._loaded_config is None:
            self.load()
        assert self._model is not None
        assert self._loaded_config is not None

        self._set_seed(int(seed))
        config = self._sampling_config(self._loaded_config, seed)
        model = self._model
        model.to(self._device())
        model.eval()
        if bool(self._cfg("use_ema_for_sampling", config.sample.use_ema)) and self._ema is not None:
            if not self._ema_copied_for_sampling:
                self._ema.copy_to(model.parameters())
                self._ema_copied_for_sampling = True

        max_node_num = int(config.data.max_node_num)
        feature_dim = int(config.data.max_feat_num)
        batch_size = int(config.sample.batch_size)
        graphs: List[nx.Graph] = []

        while len(graphs) < int(num_graphs):
            current_bs = min(batch_size, int(num_graphs) - len(graphs))
            masks = self._sample_node_masks(current_bs, max_node_num, seed + len(graphs)).to(self._device())
            x, adj = self._sample_batch(model, config, masks, feature_dim)
            before = len(graphs)
            graphs.extend(self._adjs_to_graphs(adj, masks))
            update_progress(progress_callback, min(len(graphs), num_graphs) - min(before, num_graphs))
        return graphs[:num_graphs]

    # ------------------------------------------------------------------
    # Upstream import and config helpers
    # ------------------------------------------------------------------
    def _normalize_repo_root(self, repo_root: Path) -> Path:
        if repo_root.name == "GruM_2D" and (repo_root / "config").exists():
            return repo_root.parent
        if (repo_root / "GruM" / "GruM_2D").exists():
            return repo_root / "GruM"
        return repo_root

    def _default_base_config(self, dataset: str) -> str:
        if dataset in {"sbm", "planar", "proteins"}:
            return dataset
        return "planar"

    def _format_path(self, value: str | os.PathLike[str]) -> Path:
        formatted = str(value).format(dataset=self.dataset_name, model="grum")
        return Path(formatted).expanduser().resolve()

    def _import_modules(self) -> _GruMModules:
        if self._modules is not None:
            return self._modules
        if not self.project_root.exists():
            raise FileNotFoundError(f"GruM_2D not found: {self.project_root}")
        with _isolated_upstream_import(self.project_root):
            transformer_mod = importlib.import_module("models.transformer")
            mix_mod = importlib.import_module("mix")
            solver_mod = importlib.import_module("solver")
            ema_mod = importlib.import_module("utils.ema")
            graph_utils_mod = importlib.import_module("utils.graph_utils")
            node_features_mod = importlib.import_module("utils.node_features")
        self._patch_node_features(node_features_mod)
        self._modules = _GruMModules(
            transformer_mod=transformer_mod,
            mix_mod=mix_mod,
            solver_mod=solver_mod,
            ema_mod=ema_mod,
            graph_utils_mod=graph_utils_mod,
            node_features_mod=node_features_mod,
        )
        return self._modules

    def _patch_node_features(self, node_features_mod: Any) -> None:
        if getattr(node_features_mod, "_benchmark_safe_eigen_patch", False):
            return

        def safe_get_eigenvalues_features(eigenvalues: torch.Tensor, k: int = 5):
            bs, n = eigenvalues.shape
            n_connected_components = (eigenvalues < 1e-5).sum(dim=-1)
            n_connected_components = torch.clamp(n_connected_components, min=1)
            max_connected = int(torch.max(n_connected_components).item())
            to_extend = max_connected + int(k) - int(n)
            if to_extend > 0:
                eigenvalues = torch.hstack(
                    (eigenvalues, 2 * torch.ones(bs, to_extend, dtype=eigenvalues.dtype, device=eigenvalues.device))
                )
            indices = (
                torch.arange(int(k), device=eigenvalues.device, dtype=torch.long).unsqueeze(0)
                + n_connected_components.unsqueeze(1)
            )
            first_k_ev = torch.gather(eigenvalues, dim=1, index=indices)
            return n_connected_components.unsqueeze(-1), first_k_ev

        def safe_get_eigenvectors_features(
            vectors: torch.Tensor,
            node_mask: torch.Tensor,
            n_connected: torch.Tensor,
            k: int = 2,
        ):
            if n_connected.ndim > 1:
                n_connected = n_connected.squeeze(-1)
            bs, n = vectors.size(0), vectors.size(1)
            n_connected = torch.clamp(n_connected.to(torch.long), min=1)
            max_connected = int(torch.max(n_connected).item())
            to_extend = max_connected + int(k) - int(n)
            if to_extend > 0:
                vectors = torch.cat(
                    (vectors, torch.zeros(bs, n, to_extend, dtype=vectors.dtype, device=vectors.device)), dim=2
                )
            indices = (
                torch.arange(int(k), device=vectors.device, dtype=torch.long).unsqueeze(0).unsqueeze(0)
                + n_connected.unsqueeze(1).unsqueeze(2)
            )
            indices = indices.expand(-1, n, -1)
            first_k_ev = torch.gather(vectors, dim=2, index=indices)
            return first_k_ev * node_mask.unsqueeze(2)

        node_features_mod.get_eigenvalues_features = safe_get_eigenvalues_features
        node_features_mod.get_eigenvectors_features = safe_get_eigenvectors_features
        node_features_mod._benchmark_safe_eigen_patch = True

    def _load_base_yaml(self) -> Dict[str, Any]:
        cfg_path = self.project_root / "config" / f"{self.base_config_name}.yaml"
        if not cfg_path.exists():
            raise FileNotFoundError(
                f"GruM base config not found: {cfg_path}. "
                "Use base_config: planar or sbm, or point repo_root to the upstream GruM repository."
            )
        with open(cfg_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _build_config(self, train_graphs: Sequence[nx.Graph]) -> _EasyDict:
        cfg = self._load_base_yaml()
        cfg.setdefault("data", {})
        cfg.setdefault("train", {})
        cfg.setdefault("model", {})
        cfg.setdefault("mix", {})
        cfg.setdefault("sampler", {})
        cfg.setdefault("sample", {})

        max_nodes_in_data = max(g.number_of_nodes() for g in train_graphs)
        cfg["data"]["data"] = self.dataset_name
        base_max_nodes = int(cfg["data"].get("max_node_num", max_nodes_in_data))
        if self.config.get("base_config") is None and self.dataset_name not in {"sbm", "planar", "proteins"}:
            base_max_nodes = max_nodes_in_data
        explicit_max_nodes = self.config.get("max_node_num")
        if explicit_max_nodes is None:
            cfg["data"]["max_node_num"] = max(base_max_nodes, max_nodes_in_data)
        else:
            cfg["data"]["max_node_num"] = int(explicit_max_nodes)
        if cfg["data"]["max_node_num"] < max_nodes_in_data:
            raise ValueError(
                f"GruM max_node_num={cfg['data']['max_node_num']} is smaller than a training graph with "
                f"{max_nodes_in_data} nodes. Increase max_node_num in configs/models/grum.yaml."
            )
        cfg["data"]["batch_size"] = int(self._cfg("batch_size", cfg["data"].get("batch_size", 32)))
        cfg["data"]["perm_mix"] = bool(self._cfg("perm_mix", cfg["data"].get("perm_mix", True)))
        base_feat = cfg["data"].get("feat", {})
        if isinstance(base_feat, Mapping):
            cfg["data"]["feat"] = dict(base_feat)
            base_feat_types = cfg["data"]["feat"].get("type", ["eig1", "eig2"])
        else:
            cfg["data"]["feat"] = {"type": base_feat}
            base_feat_types = base_feat
        feat_types = self._cfg("feat_types", base_feat_types)
        if isinstance(feat_types, str):
            feat_types = [feat_types]
        cfg["data"]["feat"]["type"] = list(feat_types)
        cfg["data"]["feat"]["scale"] = float(self._cfg("feat_scale", cfg["data"]["feat"].get("scale", 10.0)))
        cfg["data"]["feat"]["norm"] = bool(self._cfg("feat_norm", cfg["data"]["feat"].get("norm", True)))

        cfg["train"]["name"] = str(self._cfg("experiment_name", f"benchmark_{self.dataset_name}"))
        cfg["train"]["num_epochs"] = int(self._cfg("num_epochs", cfg["train"].get("num_epochs", 100)))
        cfg["train"]["save_interval"] = int(self._cfg("save_interval", cfg["train"].get("save_interval", 50)))
        cfg["train"]["reduce_mean"] = bool(self._cfg("reduce_mean", cfg["train"].get("reduce_mean", False)))
        cfg["train"]["lr"] = float(self._cfg("learning_rate", cfg["train"].get("lr", 2.0e-4)))
        cfg["train"]["lr_schedule"] = bool(self._cfg("lr_schedule", cfg["train"].get("lr_schedule", False)))
        cfg["train"]["ema"] = float(self._cfg("ema", cfg["train"].get("ema", 0.999)))
        cfg["train"]["weight_decay"] = float(self._cfg("weight_decay", cfg["train"].get("weight_decay", 1.0e-12)))
        cfg["train"]["grad_norm"] = float(self._cfg("clip_grad", cfg["train"].get("grad_norm", 1.0)))
        cfg["train"]["lr_decay"] = float(self._cfg("lr_decay", cfg["train"].get("lr_decay", 0.999)))
        cfg["train"]["eps"] = float(self._cfg("train_eps", cfg["train"].get("eps", 2.0e-3)))
        cfg["train"]["optimizer"] = str(self._cfg("optimizer", cfg["train"].get("optimizer", "adamw")))
        cfg["train"]["lambda_train"] = float(self._cfg("lambda_train", cfg["train"].get("lambda_train", 5.0)))
        cfg["train"].setdefault("loss_type", {"x": "const", "adj": "default"})
        if self.config.get("loss_type") is not None:
            cfg["train"]["loss_type"].update(dict(self.config["loss_type"]))
        cfg["train"]["use_tensorboard"] = False

        # Noise schedule overrides. The historical top-level sampling.num_steps is
        # kept for backwards compatibility with the earlier benchmark scaffold.
        old_sampling = self.config.get("sampling", {}) if isinstance(self.config.get("sampling"), Mapping) else {}
        shared_num_scales = self._cfg("num_scales", old_sampling.get("num_steps"))
        for key in ("x", "adj"):
            cfg["mix"].setdefault(key, {})
            if self.config.get(f"{key}_sigma_0") is not None:
                cfg["mix"][key]["sigma_0"] = float(self.config[f"{key}_sigma_0"])
            if self.config.get(f"{key}_sigma_1") is not None:
                cfg["mix"][key]["sigma_1"] = float(self.config[f"{key}_sigma_1"])
            if self.config.get(f"{key}_drift_coeff") is not None:
                cfg["mix"][key]["drift_coeff"] = float(self.config[f"{key}_drift_coeff"])
            n_scales = self._cfg(f"{key}_num_scales", shared_num_scales)
            if n_scales is not None:
                cfg["mix"][key]["num_scales"] = int(n_scales)

        # Optional compact model overrides are useful for smoke tests.
        model_overrides = dict(self._cfg("model_overrides", {}))
        self._deep_update(cfg["model"], model_overrides)
        if self.config.get("num_layers") is not None:
            cfg["model"]["num_layers"] = int(self.config["num_layers"])
        for section in ("hidden_mlp_dims", "hidden_dims", "input_dims"):
            if self.config.get(section) is not None:
                cfg["model"].setdefault(section, {})
                cfg["model"][section].update(dict(self.config[section]))

        cfg["sample"]["batch_size"] = int(self._cfg("sample_batch_size", cfg["sample"].get("batch_size", 32)))
        cfg["sample"]["use_ema"] = bool(self._cfg("use_ema_for_sampling", cfg["sample"].get("use_ema", True)))
        cfg["sample"]["eps"] = float(self._cfg("sample_eps", cfg["sample"].get("eps", 2.0e-3)))
        cfg["sample"]["kernel"] = str(self._cfg("kernel", cfg["sample"].get("kernel", "tv")))
        cfg["sample"]["noise_removal"] = bool(self._cfg("noise_removal", cfg["sample"].get("noise_removal", True)))
        cfg["sample"]["seed"] = int(self._cfg("seed", cfg["sample"].get("seed", 0)))
        if self.config.get("predictor") is not None:
            cfg["sampler"]["predictor"] = str(self.config["predictor"])
        if self.config.get("corrector") is not None:
            cfg["sampler"]["corrector"] = str(self.config["corrector"])
        if self.config.get("snr") is not None:
            cfg["sampler"]["snr"] = float(self.config["snr"])
        if self.config.get("scale_eps") is not None:
            cfg["sampler"]["scale_eps"] = float(self.config["scale_eps"])
        if self.config.get("corrector_steps") is not None:
            cfg["sampler"]["n_steps"] = int(self.config["corrector_steps"])

        cfg["seed"] = int(self._cfg("seed", 0))
        config = _to_edict(cfg)

        # Compute feature dimensions after masks have been defined.
        _, _, mask = self._graphs_to_adj_mask(train_graphs, int(config.data.max_node_num))
        dummy_adj = self._graphs_to_adj_only(train_graphs, int(config.data.max_node_num))
        x, feat_dim = self._compute_features(dummy_adj, mask, config.data)
        config.data.feat.dim = list(feat_dim)
        config.data.max_feat_num = int(x.shape[-1])
        return config

    def _sampling_config(self, config: _EasyDict, seed: int) -> _EasyDict:
        cfg = _to_edict(_to_plain(config))
        cfg.sample.seed = int(seed)
        cfg.sample.batch_size = int(self._cfg("sample_batch_size", cfg.sample.batch_size))
        cfg.sample.use_ema = bool(self._cfg("use_ema_for_sampling", cfg.sample.use_ema))
        cfg.sample.noise_removal = bool(self._cfg("noise_removal", cfg.sample.noise_removal))
        cfg.sample.eps = float(self._cfg("sample_eps", cfg.sample.eps))
        sample_num_scales = self._cfg("sample_num_scales")
        if sample_num_scales is not None:
            cfg.mix.x.num_scales = int(sample_num_scales)
            cfg.mix.adj.num_scales = int(sample_num_scales)
        return cfg

    def _deep_update(self, target: Dict[str, Any], updates: Mapping[str, Any]) -> None:
        for key, value in updates.items():
            if isinstance(value, Mapping) and isinstance(target.get(key), Mapping):
                self._deep_update(target[key], value)  # type: ignore[index]
            else:
                target[key] = value

    # ------------------------------------------------------------------
    # Tensor conversion and features
    # ------------------------------------------------------------------
    def _prepare_graphs(self, graphs: Optional[Sequence[nx.Graph]]) -> List[nx.Graph]:
        if graphs is None:
            return []
        out: List[nx.Graph] = []
        for g in graphs:
            if g is None:
                continue
            h = nx.Graph()
            nodes = sorted(g.nodes())
            mapping = {node: i for i, node in enumerate(nodes)}
            h.add_nodes_from(range(len(nodes)))
            for node in h.nodes:
                h.nodes[node]["feature"] = 1.0
                h.nodes[node]["feats"] = [1.0]
            for u, v in g.edges():
                if u not in mapping or v not in mapping:
                    continue
                uu, vv = mapping[u], mapping[v]
                if uu != vv:
                    h.add_edge(uu, vv)
            out.append(h)
        return out

    def _graphs_to_adj_only(self, graphs: Sequence[nx.Graph], max_node_num: int) -> torch.Tensor:
        _, adjs, _ = self._graphs_to_adj_mask(graphs, max_node_num)
        return adjs

    def _graphs_to_adj_mask(
        self, graphs: Sequence[nx.Graph], max_node_num: int
    ) -> Tuple[List[int], torch.Tensor, torch.Tensor]:
        adjs: List[np.ndarray] = []
        masks: List[np.ndarray] = []
        node_counts: List[int] = []
        for g in graphs:
            n = int(g.number_of_nodes())
            if n > max_node_num:
                raise ValueError(f"Graph with {n} nodes exceeds GruM max_node_num={max_node_num}.")
            node_counts.append(n)
            adj = np.zeros((max_node_num, max_node_num), dtype=np.float32)
            if n > 0:
                a = nx.to_numpy_array(g, nodelist=list(range(n)), dtype=np.float32)
                a = np.maximum(a, a.T)
                np.fill_diagonal(a, 0.0)
                adj[:n, :n] = a
            mask = np.zeros((max_node_num,), dtype=np.float32)
            mask[:n] = 1.0
            adjs.append(adj)
            masks.append(mask)
        return node_counts, torch.tensor(np.stack(adjs), dtype=torch.float32), torch.tensor(np.stack(masks), dtype=torch.float32)

    def _graphs_to_tensors(self, graphs: Sequence[nx.Graph], config: _EasyDict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _, adjs, masks = self._graphs_to_adj_mask(graphs, int(config.data.max_node_num))
        x, feat_dim = self._compute_features(adjs, masks, config.data)
        # Keep the already-computed dimensions stable, but allow this method to be
        # called before _build_config has inserted data.feat.dim.
        if not getattr(config.data.feat, "dim", None):
            config.data.feat.dim = list(feat_dim)
            config.data.max_feat_num = int(x.shape[-1])
        return x, adjs, masks

    def _compute_features(
        self, adjs: torch.Tensor, masks: torch.Tensor, data_config: _EasyDict
    ) -> Tuple[torch.Tensor, List[int]]:
        mods = self._import_modules()
        features: List[torch.Tensor] = []
        dims: List[int] = []
        feat_types = list(data_config.feat.type)
        for feat_type in feat_types:
            ft = str(feat_type).lower()
            if ft in {"const", "constant", "ones", "one"}:
                feat = torch.ones((*adjs.shape[:2], 1), dtype=adjs.dtype, device=adjs.device)
                feat = feat * masks.unsqueeze(-1)
            elif ft in {"deg", "degree"}:
                num_classes = int(self._cfg("degree_num_classes", data_config.max_node_num))
                degrees = adjs.sum(dim=-1).long().clamp(min=0, max=num_classes - 1)
                feat = torch.nn.functional.one_hot(degrees, num_classes=num_classes).to(torch.float32)
                feat = feat * masks.unsqueeze(-1)
            elif ft.startswith("eig"):
                idx_text = ft.replace("eig", "")
                idx = int(idx_text) if idx_text else 1
                idx = max(1, idx)
                eig = mods.node_features_mod.EigenFeatures(idx)(adjs, masks)
                feat = eig[..., -1:] * float(data_config.feat.scale)
                feat = torch.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
                feat = feat * masks.unsqueeze(-1)
            else:
                raise NotImplementedError(
                    f"GruM feature type {feat_type!r} is not supported. Use const, degree/deg, eig1, eig2, ..."
                )
            features.append(feat)
            dims.append(int(feat.shape[-1]))
        if not features:
            feat = torch.ones((*adjs.shape[:2], 1), dtype=adjs.dtype, device=adjs.device) * masks.unsqueeze(-1)
            features = [feat]
            dims = [1]
        x = torch.cat(features, dim=-1)
        return x, dims

    # ------------------------------------------------------------------
    # Model, loss, sampling
    # ------------------------------------------------------------------
    def _model_params(self, config: _EasyDict) -> Dict[str, Any]:
        input_dims = {"X": int(config.data.max_feat_num), "E": int(config.model.input_dims.E), "y": int(config.model.input_dims.y) + 1}
        output_dims = {"X": int(config.data.max_feat_num), "E": 1, "y": 0}
        return {
            "model_type": str(config.model.type),
            "n_layers": int(config.model.num_layers),
            "hidden_mlp_dims": _to_plain(config.model.hidden_mlp_dims),
            "hidden_dims": _to_plain(config.model.hidden_dims),
            "input_dims": input_dims,
            "output_dims": output_dims,
            "feat_dict": _to_plain(config.data.feat),
        }

    def _instantiate_model(self, params: Mapping[str, Any]) -> torch.nn.Module:
        mods = self._import_modules()
        params = dict(params)
        model_type = params.pop("model_type", params.pop("type", "transformer"))
        if str(model_type) != "transformer":
            raise ValueError(
                f"GruMWrapper currently supports the generic-graph GraphTransformer only, got {model_type!r}."
            )
        params["hidden_mlp_dims"] = _to_edict(params["hidden_mlp_dims"])
        params["hidden_dims"] = _to_edict(params["hidden_dims"])
        params["input_dims"] = _to_edict(params["input_dims"])
        params["output_dims"] = _to_edict(params["output_dims"])
        params["feat_dict"] = _to_edict(params["feat_dict"])
        return mods.transformer_mod.GraphTransformer(**params)

    def _load_mix(self, cfg_mix: _EasyDict) -> Any:
        mods = self._import_modules()
        return mods.mix_mod.DiffusionMixture(
            bridge=str(cfg_mix.type),
            drift_coeff=float(cfg_mix.drift_coeff),
            sigma_0=float(cfg_mix.sigma_0),
            sigma_1=float(cfg_mix.sigma_1),
            N=int(cfg_mix.num_scales),
        )

    def _build_loss_fn(self, config: _EasyDict):
        mix_x = self._load_mix(config.mix.x)
        mix_adj = self._load_mix(config.mix.adj)
        reduce_mean = bool(config.train.reduce_mean)
        eps = float(config.train.eps)
        lambda_train = float(config.train.lambda_train)
        loss_type = config.train.loss_type

        def reduce(values: torch.Tensor) -> torch.Tensor:
            flat = values.reshape(values.shape[0], -1)
            return flat.mean(dim=-1) if reduce_mean else flat.sum(dim=-1)

        def loss_coeff(sde: Any, t: torch.Tensor, kind: str) -> torch.Tensor:
            if kind == "default":
                return sde.loss_coeff(t)
            if kind == "const":
                return torch.ones_like(t)
            raise NotImplementedError(f"GruM loss type {kind!r} is not supported.")

        def loss_fn(model: torch.nn.Module, x: torch.Tensor, adj: torch.Tensor, mask: torch.Tensor):
            sde_x = mix_x.bridge(x)
            sde_adj = mix_adj.bridge(adj)
            t = torch.rand(adj.shape[0], device=adj.device) * (sde_adj.T - eps)
            x0 = sde_x.prior_sampling(x.shape, x.device)
            adj0 = sde_adj.prior_sampling_sym(adj.shape, adj.device)
            x0 = self._mask_x(x0, mask)
            adj0 = self._mask_adj(adj0, mask)
            mean_x, std_x = sde_x.marginal_prob(x0, t)
            xt = self._mask_x(mean_x + std_x[:, None, None] * self._noise_like(x, mask, sym=False), mask)
            mean_adj, std_adj = sde_adj.marginal_prob(adj0, t)
            adjt = self._mask_adj(mean_adj + std_adj[:, None, None] * self._noise_like(adj, mask, sym=True), mask)
            pred_x, pred_adj = model(xt, adjt, t.unsqueeze(-1), mask)
            pred_x = torch.nan_to_num(pred_x, nan=0.0, posinf=1.0, neginf=-1.0)
            pred_adj = torch.nan_to_num(pred_adj, nan=0.0, posinf=1.0, neginf=0.0)
            coeff_x = loss_coeff(sde_x, t, str(loss_type.x))[:, None, None]
            coeff_adj = loss_coeff(sde_adj, t, str(loss_type.adj))[:, None, None]
            node_loss_mask = mask.unsqueeze(-1)
            edge_loss_mask = mask.unsqueeze(1) * mask.unsqueeze(2)
            losses_x = torch.square((pred_x - x) * coeff_x) * node_loss_mask * 0.5
            losses_adj = torch.square((pred_adj - adj) * coeff_adj) * edge_loss_mask * 0.5
            per_graph_x = reduce(losses_x)
            per_graph_adj = reduce(losses_adj)
            loss_x = per_graph_x.mean()
            loss_adj = per_graph_adj.mean()
            return loss_x + lambda_train * loss_adj, loss_x, loss_adj

        return loss_fn

    def _sample_batch(
        self,
        model: torch.nn.Module,
        config: _EasyDict,
        masks: torch.Tensor,
        feature_dim: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mods = self._import_modules()
        mix = _EasyDict({"x": self._load_mix(config.mix.x), "adj": self._load_mix(config.mix.adj)})
        sampler = config.sampler
        predictor_obj = mods.solver_mod.load_predictor(str(sampler.predictor), mix, self._safe_drift_fn(mix, model))
        corrector_obj = mods.solver_mod.load_corrector(
            str(sampler.corrector),
            mix,
            self._safe_drift_fn(mix, model),
            float(sampler.snr),
            float(sampler.scale_eps),
            int(sampler.n_steps),
        )
        bsz = int(masks.shape[0])
        n = int(config.data.max_node_num)
        device = self._device()
        x = mix.x.bridge(0).prior_sampling((bsz, n, feature_dim), device)
        adj = mix.adj.bridge(0).prior_sampling_sym((bsz, n, n), device)
        x = self._mask_x(x, masks)
        adj = self._mask_adj(adj, masks)
        steps = int(mix.adj.N)
        T = float(mix.adj.bridge(0).T)
        eps = float(config.sample.eps)
        timesteps = torch.linspace(0.0, T - eps, steps, device=device)
        x_mean, adj_mean = x, adj
        with torch.no_grad():
            for t in timesteps:
                vec_t = torch.ones(bsz, device=device) * t
                y = vec_t.unsqueeze(-1)
                x, x_mean, adj, adj_mean = corrector_obj.update_fn(x, adj, y, masks, vec_t)
                x, x_mean, adj, adj_mean = predictor_obj.update_fn(x, adj, y, masks, vec_t)
                x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
                adj = torch.nan_to_num(adj, nan=0.0, posinf=1.0, neginf=0.0)
                x_mean = torch.nan_to_num(x_mean, nan=0.0, posinf=1.0, neginf=-1.0)
                adj_mean = torch.nan_to_num(adj_mean, nan=0.0, posinf=1.0, neginf=0.0)
        if bool(config.sample.noise_removal):
            return x_mean.detach().cpu(), adj_mean.detach().cpu()
        return x.detach().cpu(), adj.detach().cpu()

    def _safe_drift_fn(self, mix: _EasyDict, model: torch.nn.Module):
        model.eval()

        def get_drift_from_pred(mix_obj: Any, pred: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            bridge = mix_obj.bridge(0)
            if "BB" in str(mix_obj.bridge_type):
                drift = bridge.drift_time_scaled(t)[:, None, None] * (pred - z)
            elif "OU" in str(mix_obj.bridge_type):
                var = bridge.variance(t)
                a_t1 = bridge.a_ou(t, torch.ones_like(t))
                gamma = var * a_t1 * bridge.a_over_v(t)
                drift = (bridge.alpha_t(t) * var)[:, None, None] * z + gamma[:, None, None] * (
                    pred / a_t1[:, None, None] - z
                )
            else:
                raise NotImplementedError(f"GruM bridge type {mix_obj.bridge_type!r} is not supported.")
            return torch.nan_to_num(drift, nan=0.0, posinf=1.0, neginf=-1.0)

        def drift_fn(x: torch.Tensor, adj: torch.Tensor, y: torch.Tensor, flags: torch.Tensor, t: torch.Tensor):
            pred_x, pred_adj = model(x, adj, y, flags)
            pred_x = torch.nan_to_num(pred_x, nan=0.0, posinf=1.0, neginf=-1.0)
            pred_adj = torch.nan_to_num(pred_adj, nan=0.0, posinf=1.0, neginf=0.0)
            return get_drift_from_pred(mix.x, pred_x, x, t), get_drift_from_pred(mix.adj, pred_adj, adj, t)

        return drift_fn

    # ------------------------------------------------------------------
    # Masking, permutation, graph conversion
    # ------------------------------------------------------------------
    def _mask_x(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return x * mask[:, :, None]

    def _mask_adj(self, adj: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        out = adj * mask[:, None, :] * mask[:, :, None]
        n = out.shape[-1]
        diag = torch.eye(n, dtype=out.dtype, device=out.device).unsqueeze(0)
        return out * (1.0 - diag)

    def _noise_like(self, tensor: torch.Tensor, mask: torch.Tensor, sym: bool) -> torch.Tensor:
        z = torch.randn_like(tensor)
        if sym:
            z = torch.triu(z, diagonal=1)
            z = z + z.transpose(-1, -2)
            return self._mask_adj(z, mask)
        return self._mask_x(z, mask)

    def _rand_perm(
        self, x: torch.Tensor, adj: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, n = mask.shape
        out_x = x.clone()
        out_adj = adj.clone()
        for b in range(bsz):
            valid = torch.where(mask[b] > 0.5)[0]
            if valid.numel() <= 1:
                continue
            perm_valid = valid[torch.randperm(valid.numel(), device=valid.device)]
            perm = torch.arange(n, device=mask.device)
            perm[valid] = perm_valid
            out_x[b] = x[b, perm]
            out_adj[b] = adj[b][perm][:, perm]
        return out_x, out_adj, mask

    def _sample_node_masks(self, batch_size: int, max_node_num: int, seed: int) -> torch.Tensor:
        counts = self._train_node_counts or []
        if not counts:
            min_nodes = int(self._cfg("min_nodes", 1))
            max_nodes = int(self._cfg("max_nodes", max_node_num))
            counts = list(range(max(1, min_nodes), max(1, max_nodes) + 1))
        rng = random.Random(int(seed))
        masks = torch.zeros((batch_size, max_node_num), dtype=torch.float32)
        for i in range(batch_size):
            n = max(1, min(max_node_num, int(rng.choice(counts))))
            masks[i, :n] = 1.0
        return masks

    def _adjs_to_graphs(self, adjs: torch.Tensor, masks: torch.Tensor) -> List[nx.Graph]:
        threshold = float(self._cfg("quantize_threshold", 0.5))
        drop_isolates = bool(self._cfg("drop_isolates_on_output", False))
        graphs: List[nx.Graph] = []
        adjs_np = adjs.detach().cpu().numpy()
        masks_np = masks.detach().cpu().numpy()
        for adj, mask in zip(adjs_np, masks_np):
            n = int(np.round(mask).astype(bool).sum())
            n = max(1, n)
            a = np.asarray(adj[:n, :n], dtype=np.float32)
            a = (a + a.T) / 2.0
            a = (a >= threshold).astype(np.float32)
            np.fill_diagonal(a, 0.0)
            g = nx.from_numpy_array(a)
            g.remove_edges_from(nx.selfloop_edges(g))
            if drop_isolates:
                g.remove_nodes_from(list(nx.isolates(g)))
                if g.number_of_nodes() == 0:
                    g.add_node(0)
                g = nx.convert_node_labels_to_integers(g)
            else:
                g.add_nodes_from(range(n))
            for node in g.nodes:
                g.nodes[node]["feature"] = 1.0
                g.nodes[node]["feats"] = [1.0]
            graphs.append(g)
        return graphs

    # ------------------------------------------------------------------
    # Metadata and utility methods
    # ------------------------------------------------------------------
    def _metadata(
        self,
        train_graphs: Sequence[nx.Graph],
        val_graphs: Sequence[nx.Graph],
        test_graphs: Sequence[nx.Graph],
        config: _EasyDict,
    ) -> Dict[str, Any]:
        node_counts = [int(g.number_of_nodes()) for g in train_graphs]
        edge_counts = [int(g.number_of_edges()) for g in train_graphs]
        return {
            "model": "grum",
            "dataset": self.dataset_name,
            "repo_root": str(self.repo_root),
            "project_root": str(self.project_root),
            "base_config": self.base_config_name,
            "preserve_isolated_nodes": bool(self._cfg("preserve_isolated_nodes", True)),
            "split_sizes": {"train": len(train_graphs), "val": len(val_graphs), "test": len(test_graphs)},
            "node_counts": node_counts,
            "edge_counts": edge_counts,
            "max_node_num": int(config.data.max_node_num),
            "feat_types": list(config.data.feat.type),
            "feat_dim": list(config.data.feat.dim),
            "max_feat_num": int(config.data.max_feat_num),
        }

    def _write_provenance(
        self,
        train_graphs: Sequence[nx.Graph],
        val_graphs: Sequence[nx.Graph],
        test_graphs: Sequence[nx.Graph],
        config: _EasyDict,
    ) -> None:
        splits = {"train": list(train_graphs), "val": list(val_graphs), "test": list(test_graphs)}
        with open(self.data_root / "benchmark_splits.pkl", "wb") as f:
            pickle.dump(splits, f)
        # Also write the filename the upstream GruM dataloader expects. The wrapper
        # does not read it, but it is useful when debugging against the upstream repo.
        with open(self.data_root / f"{self.dataset_name}.pkl", "wb") as f:
            pickle.dump((list(train_graphs), list(val_graphs), list(test_graphs)), f)
        metadata = self._metadata(train_graphs, val_graphs, test_graphs, config)
        with open(self.data_root / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(_to_plain(metadata), f, indent=2, sort_keys=True)
        with open(self.data_root / "resolved_grum_config.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(_to_plain(config), f, sort_keys=False)

    def _cfg(self, key: str, default: Any = None) -> Any:
        value = self.config.get(key, default)
        return default if value is None else value

    def _set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:
            pass

    def _device(self) -> torch.device:
        requested = self.config.get("device")
        if requested is None or str(requested).lower() in {"", "auto", "null"}:
            requested = "cuda" if torch.cuda.is_available() else "cpu"
        requested = str(requested)
        if requested.startswith("cuda") and not torch.cuda.is_available():
            requested = "cpu"
        return torch.device(requested)
