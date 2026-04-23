from __future__ import annotations

import contextlib
import importlib
import os
import pickle
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import networkx as nx
import torch
import yaml


@dataclass
class _LoadedModules:
    trainer_mod: Any
    loader_mod: Any
    graph_utils_mod: Any


@contextlib.contextmanager
def _pushd(path: Path):
    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


@contextlib.contextmanager
def _prepend_sys_path(path: Path):
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


class GruMWrapper:
    """Starter wrapper for the GruM 2D implementation on generic graph datasets.

    This wrapper targets the generic-graph pipeline in ``GruM_2D`` for datasets such as
    ``planar``, ``sbm``, and optionally ``proteins``. It exposes a simple unified API:
    ``load()``, ``train()``, and ``sample()``.

    Important notes:
    - The wrapper is designed for undirected simple graphs.
    - It materializes an upstream-compatible ``data/<dataset>.pkl`` file containing
      ``(train_graphs, val_graphs, test_graphs)``.
    - Node attributes from the benchmark graphs are *not* used by GruM's generic graph
      pipeline; the upstream code derives node features from graph structure (eigenfeatures).
      The wrapper therefore only ensures each node has a dummy ``feature`` attribute.
    - Sampling uses the upstream predictor/corrector sampler and returns NetworkX graphs.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = dict(config)
        default_repo_root = Path(__file__).resolve().parents[4] / "external" / "GruM"
        repo_root = os.environ.get("GRUM_REPO") or config.get("repo_root") or default_repo_root
        if not repo_root:
            raise ValueError("GruMWrapper requires `repo_root` or the GRUM_REPO environment variable.")
        self.repo_root = self._normalize_repo_root(Path(repo_root).expanduser().resolve())
        self.repo_src = self.repo_root / "src"
        self.project_root = self.repo_root / "GruM_2D"
        if not self.project_root.exists():
            raise FileNotFoundError(f"GruM_2D not found under repo_root: {self.project_root}")

        self.dataset_name = str(self.config.get("dataset_name", "planar")).lower()
        self.base_config_name = str(self.config.get("base_config", self.dataset_name))
        self.checkpoint_path = Path(self.config["checkpoint_path"]).expanduser().resolve()
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        self._modules: Optional[_LoadedModules] = None
        self._loaded_model: Optional[torch.nn.Module] = None
        self._loaded_ema: Optional[Any] = None
        self._loaded_config: Optional[Any] = None
        self._loaded_ckpt: Optional[Dict[str, Any]] = None
        self._train_graphs: Optional[List[nx.Graph]] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def name(self) -> str:
        return "grum"

    def load(self) -> None:
        with self._legacy_torch_load():
            ckpt = torch.load(self.checkpoint_path, map_location=self._device_id())
        mods = self._import_modules()
        with self._legacy_torch_load(), _pushd(self.project_root), _prepend_sys_path(self.project_root):
            model = mods.loader_mod.load_model_from_ckpt(ckpt["params"], ckpt["state_dict"], self._device())
            ema = None
            if "ema" in ckpt:
                ema = mods.loader_mod.load_ema_from_ckpt(model, ckpt["ema"], ckpt["config"].train.ema)
                ema.copy_to(model.parameters())
            model.eval()

        self._loaded_ckpt = ckpt
        self._loaded_config = ckpt["config"]
        self._loaded_model = model
        self._loaded_ema = ema

        # Best-effort recovery of the training graph list for init flags during sampling.
        data_file = self.project_root / "data" / f"{self._loaded_config.data.data}.pkl"
        if data_file.exists():
            with open(data_file, "rb") as f:
                train_graphs, _, _ = pickle.load(f)
            self._train_graphs = train_graphs

    def train(
        self,
        train_graphs: Sequence[nx.Graph],
        val_graphs: Optional[Sequence[nx.Graph]] = None,
        test_graphs: Optional[Sequence[nx.Graph]] = None,
    ) -> None:
        mods = self._import_modules()
        prepared_train = self._prepare_graphs(train_graphs)
        prepared_val = self._prepare_graphs(val_graphs) if val_graphs is not None else None
        prepared_test = self._prepare_graphs(test_graphs) if test_graphs is not None else None

        train_split, val_split, test_split = self._resolve_splits(prepared_train, prepared_val, prepared_test)
        self._write_dataset_pickle(train_split, val_split, test_split)
        self._train_graphs = train_split

        config = self._build_config(train_split)

        with _pushd(self.project_root), _prepend_sys_path(self.project_root):
            trainer = mods.trainer_mod.Trainer(config)
            ckpt_name = trainer.train(self._timestamp_name())
            produced = self.project_root / "checkpoints" / config.data.data / f"{ckpt_name}.pth"
            if not produced.exists():
                # Try to find the final checkpoint if the upstream script saved a suffixed name.
                candidates = sorted((self.project_root / "checkpoints" / config.data.data).glob(f"{ckpt_name}*.pth"))
                if not candidates:
                    raise FileNotFoundError(f"Could not locate produced GruM checkpoint for prefix {ckpt_name}")
                produced = candidates[-1]

        shutil.copy2(produced, self.checkpoint_path)
        self.load()

    def sample(self, num_graphs: int, seed: int = 0) -> List[nx.Graph]:
        if self._loaded_model is None or self._loaded_config is None or self._loaded_ckpt is None:
            self.load()
        if self._train_graphs is None:
            raise RuntimeError("Training graph templates are required for GruM sampling (init flags).")

        assert self._loaded_model is not None
        assert self._loaded_config is not None

        mods = self._import_modules()
        config = self._loaded_config
        config.sample.seed = int(seed)
        config.sample.batch_size = int(self.config.get("sample_batch_size", config.sample.batch_size))
        config.sample.use_ema = bool(self.config.get("use_ema_for_sampling", True))
        config.sample.noise_removal = bool(self.config.get("noise_removal", True))
        config.sample.eps = float(self.config.get("sample_eps", config.sample.eps))
        config.sampler.predictor = str(self.config.get("predictor", config.sampler.predictor))
        config.sampler.corrector = str(self.config.get("corrector", config.sampler.corrector))
        config.sampler.snr = float(self.config.get("snr", config.sampler.snr))
        config.sampler.scale_eps = float(self.config.get("scale_eps", config.sampler.scale_eps))
        config.sampler.n_steps = int(self.config.get("corrector_steps", config.sampler.n_steps))

        with _pushd(self.project_root), _prepend_sys_path(self.project_root):
            mods.loader_mod.load_seed(seed)
            sampling_fn = mods.loader_mod.load_sampling_fn(config, config.sampler, config.sample, self._device())

            graph_list: List[nx.Graph] = []
            rounds = (num_graphs + config.sample.batch_size - 1) // config.sample.batch_size
            device0 = self._device0()
            model = self._loaded_model
            model.eval()
            for _ in range(rounds):
                init_flags = mods.graph_utils_mod.get_init_flags(
                    self._train_graphs, config, config.sample.batch_size
                ).to(device0)
                with torch.no_grad():
                    _, adj, _ = sampling_fn(model, init_flags)
                adj_int = mods.graph_utils_mod.quantize(adj)
                graph_list.extend(mods.graph_utils_mod.adjs_to_graphs(adj_int, True))

        out = []
        for g in graph_list[:num_graphs]:
            g2 = nx.convert_node_labels_to_integers(g)
            for n in g2.nodes():
                g2.nodes[n]["feats"] = [1.0]
                g2.nodes[n]["feature"] = 1.0
            out.append(g2)
        return out

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _normalize_repo_root(self, repo_root: Path) -> Path:
        if repo_root.name == "GruM_2D" and (repo_root / "config").exists():
            repo_root = repo_root.parent
        return repo_root

    def _import_modules(self) -> _LoadedModules:
        if self._modules is not None:
            return self._modules

        if not self.project_root.exists():
            raise FileNotFoundError(f"GruM_2D not found under repo_root: {self.project_root}")
        with _prepend_sys_path(self.project_root), _pushd(self.project_root):
            node_features_mod = importlib.import_module("utils.node_features")
            trainer_mod = importlib.import_module("trainer")
            loader_mod = importlib.import_module("utils.loader")
            graph_utils_mod = importlib.import_module("utils.graph_utils")
        self._patch_node_features(node_features_mod)
        self._modules = _LoadedModules(
            trainer_mod=trainer_mod,
            loader_mod=loader_mod,
            graph_utils_mod=graph_utils_mod,
        )
        return self._modules

    def _patch_node_features(self, node_features_mod: Any) -> None:
        if getattr(node_features_mod, "_empirical_eigen_patch", False):
            return
        original = node_features_mod.get_eigenvalues_features

        def safe_get_eigenvalues_features(eigenvalues, k=5):
            ev = eigenvalues
            bs, n = ev.shape
            n_connected_components = (ev < 1e-5).sum(dim=-1)
            n_connected_components = torch.clamp(n_connected_components, min=1)

            to_extend = int(torch.max(n_connected_components).item()) + k - n
            if to_extend > 0:
                eigenvalues = torch.hstack(
                    (eigenvalues, 2 * torch.ones(bs, to_extend).type_as(eigenvalues))
                )
            indices = (
                torch.arange(k).type_as(eigenvalues).long().unsqueeze(0)
                + n_connected_components.unsqueeze(1)
            )
            first_k_ev = torch.gather(eigenvalues, dim=1, index=indices)
            return n_connected_components.unsqueeze(-1), first_k_ev

        node_features_mod.get_eigenvalues_features = safe_get_eigenvalues_features
        node_features_mod._empirical_eigen_patch = True

    def _build_config(self, train_graphs: Sequence[nx.Graph]):
        cfg_path = self.project_root / "config" / f"{self.base_config_name}.yaml"
        if not cfg_path.exists():
            raise FileNotFoundError(f"GruM config not found: {cfg_path}")
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        # Dataset-level overrides
        cfg.setdefault("data", {})
        cfg["data"]["data"] = self.dataset_name
        cfg["data"]["batch_size"] = int(self.config.get("batch_size", cfg["data"].get("batch_size", 32)))
        cfg["data"]["max_node_num"] = int(
            self.config.get("max_node_num", max(g.number_of_nodes() for g in train_graphs))
        )
        cfg["data"]["perm_mix"] = bool(self.config.get("perm_mix", cfg["data"].get("perm_mix", True)))
        if self.config.get("feat_types") is not None:
            cfg["data"].setdefault("feat", {})
            cfg["data"]["feat"]["type"] = list(self.config["feat_types"])
        if self.config.get("feat_scale") is not None:
            cfg["data"].setdefault("feat", {})
            cfg["data"]["feat"]["scale"] = float(self.config["feat_scale"])

        # Training overrides
        cfg.setdefault("train", {})
        cfg["train"]["name"] = str(self.config.get("experiment_name", f"grum_{self.dataset_name}"))
        cfg["train"]["num_epochs"] = int(self.config.get("num_epochs", cfg["train"].get("num_epochs", 100)))
        cfg["train"]["save_interval"] = int(self.config.get("save_interval", cfg["train"].get("save_interval", 50)))
        cfg["train"]["lr"] = float(self.config.get("learning_rate", cfg["train"].get("lr", 2e-4)))
        cfg["train"]["use_tensorboard"] = bool(self.config.get("use_tensorboard", False))
        cfg["train"]["ema"] = float(self.config.get("ema", cfg["train"].get("ema", 0.999)))
        cfg["train"]["eps"] = float(self.config.get("train_eps", cfg["train"].get("eps", 2e-3)))
        cfg["train"]["lambda_train"] = float(self.config.get("lambda_train", cfg["train"].get("lambda_train", 5)))

        # Mix overrides
        for key in ("x", "adj"):
            if self.config.get(f"{key}_sigma_0") is not None:
                cfg["mix"][key]["sigma_0"] = float(self.config[f"{key}_sigma_0"])
            if self.config.get(f"{key}_sigma_1") is not None:
                cfg["mix"][key]["sigma_1"] = float(self.config[f"{key}_sigma_1"])
            if self.config.get(f"{key}_num_scales") is not None:
                cfg["mix"][key]["num_scales"] = int(self.config[f"{key}_num_scales"])
            if self.config.get(f"{key}_drift_coeff") is not None:
                cfg["mix"][key]["drift_coeff"] = float(self.config[f"{key}_drift_coeff"])

        # Model overrides
        model_overrides = dict(self.config.get("model_overrides", {}))
        for section, updates in model_overrides.items():
            if isinstance(updates, dict) and section in cfg["model"]:
                cfg["model"][section].update(updates)
        for k, v in model_overrides.items():
            if k not in cfg["model"] or not isinstance(v, dict):
                cfg["model"][k] = v

        # Sampler/sample overrides for later use.
        cfg.setdefault("sample", {})
        cfg["sample"]["batch_size"] = int(self.config.get("sample_batch_size", cfg["sample"].get("batch_size", 40)))
        cfg["sample"]["use_ema"] = bool(self.config.get("use_ema_for_sampling", cfg["sample"].get("use_ema", True)))
        cfg["sample"]["seed"] = int(self.config.get("seed", 0))
        cfg["sample"]["noise_removal"] = bool(self.config.get("noise_removal", cfg["sample"].get("noise_removal", True)))
        cfg.setdefault("sampler", {})
        if self.config.get("predictor") is not None:
            cfg["sampler"]["predictor"] = self.config["predictor"]
        if self.config.get("corrector") is not None:
            cfg["sampler"]["corrector"] = self.config["corrector"]
        if self.config.get("snr") is not None:
            cfg["sampler"]["snr"] = float(self.config["snr"])
        if self.config.get("scale_eps") is not None:
            cfg["sampler"]["scale_eps"] = float(self.config["scale_eps"])
        if self.config.get("corrector_steps") is not None:
            cfg["sampler"]["n_steps"] = int(self.config["corrector_steps"])

        # Convert to EasyDict using upstream dependency.
        with _prepend_sys_path(self.project_root):
            easydict = importlib.import_module("easydict")
        edict = easydict.EasyDict

        def _to_edict(obj):
            if isinstance(obj, dict):
                return edict({k: _to_edict(v) for k, v in obj.items()})
            if isinstance(obj, list):
                return [_to_edict(v) for v in obj]
            return obj

        config = _to_edict(cfg)
        config.seed = int(self.config.get("seed", 0))
        return config

    def _write_dataset_pickle(
        self,
        train_graphs: Sequence[nx.Graph],
        val_graphs: Sequence[nx.Graph],
        test_graphs: Sequence[nx.Graph],
    ) -> Path:
        data_dir = self.project_root / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        path = data_dir / f"{self.dataset_name}.pkl"
        with open(path, "wb") as f:
            pickle.dump((list(train_graphs), list(val_graphs), list(test_graphs)), f)
        return path

    def _prepare_graphs(self, graphs: Optional[Sequence[nx.Graph]]) -> Optional[List[nx.Graph]]:
        if graphs is None:
            return None
        out: List[nx.Graph] = []
        for g in graphs:
            if g is None:
                continue
            h = nx.Graph()
            nodes = sorted(g.nodes())
            mapping = {n: i for i, n in enumerate(nodes)}
            h.add_nodes_from(range(len(nodes)))
            for i in range(len(nodes)):
                h.nodes[i]["feature"] = 1.0
                h.nodes[i]["feats"] = [1.0]
            for u, v in g.edges():
                uu, vv = mapping[u], mapping[v]
                if uu != vv:
                    h.add_edge(uu, vv)
            out.append(h)
        return out

    def _resolve_splits(
        self,
        train_graphs: Sequence[nx.Graph],
        val_graphs: Optional[Sequence[nx.Graph]],
        test_graphs: Optional[Sequence[nx.Graph]],
    ) -> Tuple[List[nx.Graph], List[nx.Graph], List[nx.Graph]]:
        train_graphs = list(train_graphs)
        if val_graphs is not None and test_graphs is not None:
            return train_graphs, list(val_graphs), list(test_graphs)
        if val_graphs is not None and test_graphs is None:
            return train_graphs, list(val_graphs), list(val_graphs)

        # Create deterministic 80/10/10 split from the provided train graphs.
        rng = random.Random(int(self.config.get("seed", 0)))
        idx = list(range(len(train_graphs)))
        rng.shuffle(idx)
        n = len(idx)
        n_val = max(1, int(0.1 * n)) if n >= 10 else max(1, n // 5)
        n_test = n_val
        val_idx = idx[:n_val]
        test_idx = idx[n_val:n_val + n_test]
        train_idx = idx[n_val + n_test:]
        if not train_idx:
            train_idx = idx[:-2] if len(idx) > 2 else idx[:1]
        train = [train_graphs[i] for i in train_idx]
        val = [train_graphs[i] for i in val_idx]
        test = [train_graphs[i] for i in test_idx] if test_idx else list(val)
        return train, val, test

    def _timestamp_name(self) -> str:
        import time
        return str(self.config.get("run_name", time.strftime("%b%d-%H:%M:%S", time.gmtime())))

    def _device(self):
        return self._import_modules().loader_mod.load_device()

    def _device_id(self) -> str:
        device = self._device()
        return f"cuda:{device[0]}" if isinstance(device, list) else str(device)

    def _device0(self):
        device = self._device()
        return f"cuda:{device[0]}" if isinstance(device, list) else device

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
