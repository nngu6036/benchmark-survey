from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F

from empirical_comparison.models.base import BaseGenerator
from empirical_comparison.utils.progress import update_progress


@dataclass
class _DisCoDatasetMeta:
    dataset: str
    n_node_type: int
    n_edge_type: int
    max_n_nodes: int
    n_node_distribution: list[float]
    edge_marginal: list[float]
    node_marginal: list[float]
    input_dims: dict[str, int]
    output_dims: dict[str, int]
    hidden_mlp_dims: dict[str, int]
    hidden_dims: dict[str, int]
    backbone: str
    n_layers: int
    n_dim: int
    aux_features: dict[str, bool]


class DisCoWrapper(BaseGenerator):
    """Benchmark adapter for the uploaded DisCo repository.

    The upstream repository trains on SPECTRE-style synthetic graph datasets via
    ``train_spectre.py``.  That script also constructs its own random splits and
    imports optional evaluation dependencies.  This wrapper uses DisCo's real
    dense neural backbones, continuous-time forward diffusion, auxiliary
    features, and tau-leaping sampler, but controls the benchmark data path
    itself so that the persisted benchmark train/val/test splits are preserved.

    Supported benchmark setting
    ---------------------------
    - undirected, simple, featureless NetworkX graphs;
    - one constant node feature channel;
    - two edge classes: 0 = no edge, 1 = edge;
    - variable graph sizes sampled from the training graph-size distribution.
    """

    supports_training = True
    supports_sampling = True
    supports_node_features = False
    supports_edge_features = False
    supports_node_labels = True
    supports_edge_labels = True
    supports_graph_labels = False
    supports_constraints = False
    supports_variable_size = True
    supports_featureless_graphs = True

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        default_repo_root = Path(__file__).resolve().parents[4] / "external" / "DisCo"
        repo_root = os.environ.get("DISCO_REPO") or config.get("repo_root") or default_repo_root
        self.repo_root = self._normalize_repo_root(Path(repo_root).expanduser().resolve())

        self.dataset = str(config.get("dataset") or config.get("dataset_name") or "sbm").lower()
        self.model_name = str(config.get("name", "disco"))
        self.seed = int(config.get("seed", 42))
        self.device = torch.device(config.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
        if self.device.type == "cpu":
            # DisCo performs many small dense tensor operations during sampling.
            # Restricting CPU threads avoids rare OpenMP/BLAS stalls in repeated
            # tau-leaping calls and is usually faster for small benchmark graphs.
            torch.set_num_threads(int(config.get("torch_num_threads", 1)))

        self.checkpoint_path = self._resolve_path(config.get("checkpoint_path", "outputs/checkpoints/{dataset}/disco.pt"))
        self.run_root = self._resolve_path(config.get("run_root", "outputs/disco_runs/{dataset}"))
        self.data_root = self._resolve_path(config.get("data_root", "outputs/disco_data/{dataset}"))

        self.repo_loaded = False
        self.mods: dict[str, Any] = {}

        self.model: torch.nn.Module | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.diffuser: Any = None
        self.sampler: Any = None
        self.add_auxiliary_feature: Any = None
        self.n_node_distribution: torch.distributions.Categorical | None = None
        self.meta: _DisCoDatasetMeta | None = None
        self.best_metric: float | None = None

    @property
    def name(self) -> str:
        return "disco"

    # ------------------------------------------------------------------
    # Path/config helpers
    # ------------------------------------------------------------------
    def _resolve_path(self, value: str | os.PathLike[str]) -> Path:
        raw = str(value)
        replacements = {
            "dataset": self.dataset,
            "model": self.model_name,
            "name": self.model_name,
        }
        for k, v in replacements.items():
            raw = raw.replace("${" + k + "}", v)
        try:
            raw = raw.format(**replacements)
        except Exception:
            pass
        return Path(raw).expanduser().resolve()

    def _normalize_repo_root(self, repo_root: Path) -> Path:
        """Accept either the DisCo root or a parent containing a DisCo folder."""
        candidates = [repo_root]
        if (repo_root / "DisCo").exists():
            candidates.append(repo_root / "DisCo")
        if repo_root.name == "loader" and repo_root.parent.exists():
            candidates.append(repo_root.parent)

        for candidate in candidates:
            if (candidate / "train_spectre.py").exists() and (candidate / "forward_diff.py").exists():
                return candidate
        # Defer the explicit FileNotFoundError until the wrapper is actually used.
        return repo_root

    def _validate_repo_root(self) -> None:
        required = [
            "forward_diff.py",
            "sampling.py",
            "auxiliary_features.py",
            "digress_models.py",
            "models.py",
        ]
        missing = [p for p in required if not (self.repo_root / p).exists()]
        if missing:
            raise FileNotFoundError(
                "DisCo repository layout was not found. Set DISCO_REPO or "
                "configs/models/disco.yaml:repo_root to the extracted DisCo directory. "
                f"repo_root={self.repo_root}; missing={missing}"
            )

    # ------------------------------------------------------------------
    # Import upstream DisCo modules without importing the upstream dataset/eval path
    # ------------------------------------------------------------------
    @staticmethod
    def _batch_symmetrize(tensor: torch.Tensor) -> torch.Tensor:
        """Same semantics as upstream utils.batch_symmetrize, without PyG imports."""
        if tensor.ndim != 3 or tensor.shape[1] != tensor.shape[2]:
            raise ValueError(f"Expected a batched square tensor, got shape={tuple(tensor.shape)}")
        n = tensor.shape[1]
        upper = torch.ones((n, n), device=tensor.device, dtype=torch.bool).triu()
        out = tensor.clone()
        out.transpose(1, 2)[:, upper] = out[:, upper]
        return out

    @staticmethod
    def _assert_correctly_masked(variable: torch.Tensor, node_mask: torch.Tensor) -> None:
        # Upstream assertion helper.  Keep it strict because masking errors can
        # silently corrupt graph-size handling.
        if variable.numel() == 0:
            return
        assert (variable * (1 - node_mask.long())).abs().max().item() < 1e-4, "Variables not masked properly."

    class _PlaceHolder:
        """Minimal upstream-compatible placeholder used by DisCo modules."""

        def __init__(self, X: torch.Tensor, E: torch.Tensor, y: torch.Tensor | None):
            self.X = X
            self.E = E
            self.y = y

        def type_as(self, x: torch.Tensor):
            self.X = self.X.type_as(x)
            self.E = self.E.type_as(x)
            if self.y is not None:
                self.y = self.y.type_as(x)
            return self

        def mask(self, node_mask: torch.Tensor, collapse: bool = False):
            x_mask = node_mask.unsqueeze(-1)
            e_mask1 = x_mask.unsqueeze(2)
            e_mask2 = x_mask.unsqueeze(1)
            if collapse:
                self.X = torch.argmax(self.X, dim=-1)
                self.E = torch.argmax(self.E, dim=-1)
                self.X[node_mask == 0] = -1
                self.E[(e_mask1 * e_mask2).squeeze(-1) == 0] = -1
            else:
                self.X = self.X * x_mask
                self.E = self.E * e_mask1 * e_mask2
                assert torch.allclose(self.E, torch.transpose(self.E, 1, 2))
            return self

    def _make_utils_shim(self) -> types.ModuleType:
        shim = types.ModuleType("utils")
        shim.batch_symmetrize = self._batch_symmetrize
        shim.assert_correctly_masked = self._assert_correctly_masked
        shim.PlaceHolder = self._PlaceHolder
        return shim


    class _SafeTauLeaping:
        """Numerically guarded version of upstream ``sampling.TauLeaping``.

        The uploaded implementation can produce extremely large reverse rates
        when a tiny or partially trained model is sampled.  Very large Poisson
        rates can make smoke tests appear to hang.  This class preserves the
        upstream update equations but applies ``nan_to_num`` and a configurable
        rate clip before Poisson sampling.
        """

        def __init__(
            self,
            n_node_type: int,
            n_edge_type: int,
            *,
            num_steps: int,
            min_t: float,
            add_auxiliary_feature: Any,
            device: str,
            rate_clip: float = 1000.0,
        ) -> None:
            self.n_node_type = int(n_node_type)
            self.n_edge_type = int(n_edge_type)
            self.num_steps = int(num_steps)
            self.min_t = float(min_t)
            self.add_auxiliary_feature = add_auxiliary_feature
            self.device = device
            self.rate_clip = float(rate_clip)
            self.diffuse_node = self.n_node_type > 1

        def _sanitize_rates(self, rates: torch.Tensor) -> torch.Tensor:
            rates = torch.nan_to_num(rates, nan=0.0, posinf=self.rate_clip, neginf=0.0)
            return rates.clamp(min=0.0, max=self.rate_clip)

        @staticmethod
        def _poisson(rate: torch.Tensor) -> torch.Tensor:
            # CPU torch.poisson can be unexpectedly slow/hang for some rate
            # patterns in the sandbox.  NumPy gives a stable fallback and is
            # acceptable here because graph benchmarks are sampled in batches.
            if rate.device.type == "cpu":
                arr = rate.detach().cpu().numpy()
                return torch.from_numpy(np.random.poisson(arr).astype(np.float32)).to(rate.device)
            return torch.poisson(rate)

        @torch.no_grad()
        def sample(self, diffuser, model, n_node: torch.Tensor, trajectory: bool = False):
            N = int(n_node.shape[0])
            n_node_type = self.n_node_type
            n_edge_type = self.n_edge_type
            min_t = self.min_t
            num_steps = self.num_steps
            eps_ratio = 1e-9
            device = self.device
            rate_clip = self.rate_clip

            n_node_max = int(torch.max(n_node).item())
            node_mask = torch.arange(n_node_max, device=device).unsqueeze(0).expand(N, -1)
            node_mask = node_mask < n_node.unsqueeze(1)
            graphs = []

            E = diffuser.get_initial_samples(N * n_node_max * n_node_max, "edge").view(N, n_node_max, n_node_max)
            X = diffuser.get_initial_samples(N * n_node_max, "node").view(N, n_node_max)
            ts = np.concatenate((np.linspace(1.0, min_t, num_steps), np.array([0.0])))

            diag_mask = torch.eye(E.shape[1], dtype=torch.bool, device=E.device).unsqueeze(0).expand(E.shape[0], -1, -1)
            E[diag_mask] = 0
            E = DisCoWrapper._batch_symmetrize(E)
            E = E * node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
            if trajectory:
                graphs.append([X, E])

            for idx, t in enumerate(ts[:-1]):
                h = float(ts[idx] - ts[idx + 1])
                t_tensor = float(t) * torch.ones((N,), device=device)
                E_qt0, X_qt0 = diffuser.transition(t_tensor)
                E_rate, X_rate = diffuser.rate(t_tensor)

                X_t_one_hot = F.one_hot(X, num_classes=n_node_type).float()
                E_t_one_hot = F.one_hot(E, num_classes=n_edge_type).float()
                X_t, E_t, y_t = self.add_auxiliary_feature(X_t_one_hot, E_t_one_hot, node_mask)
                y_t = torch.cat([y_t, t_tensor.unsqueeze(-1)], dim=-1)
                X_p0t, E_p0t = model(X_t, E_t, y_t, node_mask)

                D = n_node_max * n_node_max
                E_p0t = F.softmax(E_p0t.view(N, D, n_edge_type), dim=-1)
                E_qt0_denom = E_qt0[
                    torch.arange(N, device=device).repeat_interleave(D * n_edge_type),
                    torch.arange(n_edge_type, device=device).repeat(N * D),
                    E.long().flatten().repeat_interleave(n_edge_type),
                ].view(N, D, n_edge_type) + eps_ratio
                forward_rates = E_rate[
                    torch.arange(N, device=device).repeat_interleave(D * n_edge_type),
                    torch.arange(n_edge_type, device=device).repeat(N * D),
                    E.long().flatten().repeat_interleave(n_edge_type),
                ].view(N, D, n_edge_type)
                reverse_rates = forward_rates * ((E_p0t / E_qt0_denom) @ E_qt0)
                reverse_rates[
                    torch.arange(N, device=device).repeat_interleave(D),
                    torch.arange(D, device=device).repeat(N),
                    E.long().flatten(),
                ] = 0.0
                reverse_rates = self._sanitize_rates(reverse_rates)
                diffs = torch.arange(n_edge_type, device=device).view(1, 1, n_edge_type) - E.view(N, D, 1)
                jump_nums = self._poisson(reverse_rates * h)
                overall_jump = torch.sum(jump_nums * diffs, dim=2)
                E_new = torch.clamp(E.view(N, D) + overall_jump.long(), min=0, max=n_edge_type - 1)

                if self.diffuse_node:
                    Dn = n_node_max
                    X_p0t = F.softmax(X_p0t.view(N, Dn, n_node_type), dim=-1)
                    X_qt0_denom = X_qt0[
                        torch.arange(N, device=device).repeat_interleave(Dn * n_node_type),
                        torch.arange(n_node_type, device=device).repeat(N * Dn),
                        X.long().flatten().repeat_interleave(n_node_type),
                    ].view(N, Dn, n_node_type) + eps_ratio
                    forward_rates = X_rate[
                        torch.arange(N, device=device).repeat_interleave(Dn * n_node_type),
                        torch.arange(n_node_type, device=device).repeat(N * Dn),
                        X.long().flatten().repeat_interleave(n_node_type),
                    ].view(N, Dn, n_node_type)
                    reverse_rates = forward_rates * ((X_p0t / X_qt0_denom) @ X_qt0)
                    reverse_rates[
                        torch.arange(N, device=device).repeat_interleave(Dn),
                        torch.arange(Dn, device=device).repeat(N),
                        X.long().flatten(),
                    ] = 0.0
                    reverse_rates = self._sanitize_rates(reverse_rates)
                    diffs = torch.arange(n_node_type, device=device).view(1, 1, n_node_type) - X.view(N, Dn, 1)
                    jump_nums = self._poisson(reverse_rates * h)
                    overall_jump = torch.sum(jump_nums * diffs, dim=2)
                    X_new = torch.clamp(X.view(N, Dn) + overall_jump.long(), min=0, max=n_node_type - 1)

                E = E_new.view(N, n_node_max, n_node_max)
                if self.diffuse_node:
                    X = X_new.view(N, n_node_max)

                diag_mask = torch.eye(E.shape[1], dtype=torch.bool, device=E.device).unsqueeze(0).expand(E.shape[0], -1, -1)
                E[diag_mask] = 0
                E = DisCoWrapper._batch_symmetrize(E)
                E = E * node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
                if trajectory:
                    graphs.append([X, E])

            X_t_one_hot = F.one_hot(X, num_classes=n_node_type).float()
            E_t_one_hot = F.one_hot(E, num_classes=n_edge_type).float()
            X_t, E_t, y_t = self.add_auxiliary_feature(X_t_one_hot, E_t_one_hot, node_mask)
            y_t = torch.cat([y_t, min_t * torch.ones((N,), device=device).unsqueeze(-1)], dim=-1)
            X_p_0gt, E_p_0gt = model(X_t, E_t, y_t, node_mask)
            E_p_0gt = torch.max(F.softmax(E_p_0gt, dim=-1), dim=-1)[1]
            X_p_0gt = torch.max(F.softmax(X_p_0gt, dim=-1), dim=-1)[1]

            diag_mask = torch.eye(E_p_0gt.shape[1], dtype=torch.bool, device=E_p_0gt.device).unsqueeze(0).expand(E_p_0gt.shape[0], -1, -1)
            E_p_0gt[diag_mask] = 0
            E_p_0gt = DisCoWrapper._batch_symmetrize(E_p_0gt)
            E_p_0gt = E_p_0gt * node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
            X_p_0gt = X_p_0gt * node_mask

            if trajectory:
                graphs.append([X_p_0gt, E_p_0gt])
                return X_p_0gt, E_p_0gt, node_mask, graphs
            return X_p_0gt, E_p_0gt, node_mask

    @staticmethod
    def _load_module(module_name: str, path: Path):
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not create import spec for {module_name} at {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

    @contextlib.contextmanager
    def _temporary_modules(self, overrides: dict[str, types.ModuleType]) -> Iterator[None]:
        sentinel = object()
        old = {name: sys.modules.get(name, sentinel) for name in overrides}
        sys.modules.update(overrides)
        try:
            yield
        finally:
            for name, value in old.items():
                if value is sentinel:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = value  # type: ignore[assignment]

    @contextlib.contextmanager
    def _temporary_sys_path(self, path: Path) -> Iterator[None]:
        path_str = str(path)
        inserted = False
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
            inserted = True
        try:
            yield
        finally:
            if inserted:
                with contextlib.suppress(ValueError):
                    sys.path.remove(path_str)

    def _unique_prefix(self) -> str:
        digest = hashlib.sha1(str(self.repo_root).encode("utf-8")).hexdigest()[:10]
        return f"_benchmark_disco_{digest}"

    def _import_modules(self) -> None:
        if self.repo_loaded:
            return
        self._validate_repo_root()

        prefix = self._unique_prefix()
        utils_shim = self._make_utils_shim()
        aux_mod = self._load_module(f"{prefix}_auxiliary_features", self.repo_root / "auxiliary_features.py")

        # The uploaded repository uses top-level imports such as
        # ``from utils import *`` and ``from auxiliary_features import *``.
        # We temporarily provide only the pieces needed by the dense model path,
        # avoiding torch_geometric/dataset/evaluation imports.
        with self._temporary_sys_path(self.repo_root), self._temporary_modules(
            {"utils": utils_shim, "auxiliary_features": aux_mod}
        ):
            self.mods["forward_diff"] = self._load_module(f"{prefix}_forward_diff", self.repo_root / "forward_diff.py")
            self.mods["sampling"] = self._load_module(f"{prefix}_sampling", self.repo_root / "sampling.py")
            self.mods["digress_models"] = self._load_module(f"{prefix}_digress_models", self.repo_root / "digress_models.py")
            self.mods["models"] = self._load_module(f"{prefix}_models", self.repo_root / "models.py")
            self.mods["aux"] = aux_mod

        self.repo_loaded = True

    # ------------------------------------------------------------------
    # Dense graph conversion and dataset statistics
    # ------------------------------------------------------------------
    @staticmethod
    def _clean_graph(g: nx.Graph) -> nx.Graph:
        if g.number_of_nodes() == 0:
            raise ValueError("DisCoWrapper does not support empty graphs for training.")
        if g.is_directed():
            raise ValueError("DisCoWrapper expects undirected graphs.")
        h = nx.Graph()
        for node, attrs in g.nodes(data=True):
            h.add_node(node, **dict(attrs))
        for u, v, attrs in g.edges(data=True):
            if u != v:
                h.add_edge(u, v, **dict(attrs))
        h = nx.convert_node_labels_to_integers(h, ordering="default")
        return h

    def _prepare_graphs(self, graphs: Iterable[nx.Graph]) -> list[nx.Graph]:
        cleaned = [self._clean_graph(g) for g in graphs]
        if not cleaned:
            raise ValueError("DisCoWrapper received an empty graph list.")
        return cleaned

    @staticmethod
    def _node_count_distribution(graphs: list[nx.Graph]) -> tuple[int, list[float]]:
        counts = [g.number_of_nodes() for g in graphs]
        max_n = max(counts)
        probs = np.zeros(max_n + 1, dtype=np.float64)
        for n in counts:
            probs[n] += 1.0
        probs /= probs.sum()
        return max_n, probs.tolist()

    @staticmethod
    def _edge_marginal(graphs: list[nx.Graph], n_edge_type: int = 2) -> list[float]:
        counts = np.zeros(max(2, int(n_edge_type)), dtype=np.float64)
        for g in graphs:
            n = g.number_of_nodes()
            E = np.zeros((n, n), dtype=np.int64)
            for u, v, attrs in g.edges(data=True):
                et = max(1, min(int(attrs.get("edge_type", 1)), len(counts) - 1))
                E[int(u), int(v)] = et
                E[int(v), int(u)] = et
            for u in range(n):
                for v in range(n):
                    counts[int(E[u, v])] += 1.0
        if counts.sum() <= 0:
            counts[0] = 1.0
        return (counts / counts.sum()).tolist()

    def _graphs_to_dense_batch(self, graphs: list[nx.Graph]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = len(graphs)
        max_nodes = max(g.number_of_nodes() for g in graphs)
        n_node_type = int(getattr(self, "_tmp_n_node_type", getattr(self.meta, "n_node_type", 1) if self.meta is not None else 1))
        n_edge_type = int(getattr(self, "_tmp_n_edge_type", getattr(self.meta, "n_edge_type", 2) if self.meta is not None else 2))
        X = torch.zeros((batch_size, max_nodes, n_node_type), dtype=torch.float32, device=self.device)
        E = torch.zeros((batch_size, max_nodes, max_nodes, n_edge_type), dtype=torch.float32, device=self.device)
        node_mask = torch.zeros((batch_size, max_nodes), dtype=torch.bool, device=self.device)

        for i, graph in enumerate(graphs):
            n = graph.number_of_nodes()
            node_mask[i, :n] = True
            for node in range(n):
                label = max(0, min(int(graph.nodes[node].get("node_label", 0)), n_node_type - 1))
                X[i, node, label] = 1.0
            E[i, :n, :n, 0] = 1.0
            for u, v, attrs in graph.edges(data=True):
                if u == v:
                    continue
                et = max(1, min(int(attrs.get("edge_type", 1)), n_edge_type - 1))
                E[i, int(u), int(v), 0] = 0.0
                E[i, int(v), int(u), 0] = 0.0
                E[i, int(u), int(v), et] = 1.0
                E[i, int(v), int(u), et] = 1.0
        return X, E, node_mask

    def _iter_batches(self, graphs: list[nx.Graph], *, shuffle: bool) -> Iterator[list[nx.Graph]]:
        batch_size = max(1, int(self.config.get("batch_size", 32)))
        indices = list(range(len(graphs)))
        if shuffle:
            rng = random.Random(self.seed)
            rng.shuffle(indices)
        for start in range(0, len(indices), batch_size):
            yield [graphs[i] for i in indices[start : start + batch_size]]

    def _add_mask_idx(
        self,
        X_idx: torch.Tensor | None,
        E_idx: torch.Tensor,
        n_node_type: int,
        n_edge_type: int,
        node_mask: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        X_idx_masked = None
        if X_idx is not None:
            X_idx_masked = torch.masked_fill(X_idx, ~node_mask, value=n_node_type)
        E_mask = ~(node_mask.unsqueeze(-1) * node_mask.unsqueeze(-2))
        E_idx_masked = torch.masked_fill(E_idx, E_mask, value=n_edge_type)
        return X_idx_masked, E_idx_masked

    @staticmethod
    def _ce_loss(pred_y: torch.Tensor, true_y: torch.Tensor) -> torch.Tensor:
        pred_y = pred_y.flatten(end_dim=-2)
        true_y = true_y.flatten()
        return F.cross_entropy(pred_y, true_y, reduction="mean", ignore_index=pred_y.shape[-1])

    def _write_provenance_dataset(
        self,
        train_graphs: list[nx.Graph],
        val_graphs: list[nx.Graph] | None,
        test_graphs: list[nx.Graph] | None,
        meta: _DisCoDatasetMeta,
    ) -> None:
        if not bool(self.config.get("save_provenance_dataset", True)):
            return
        self.data_root.mkdir(parents=True, exist_ok=True)

        def to_adj_list(graphs: list[nx.Graph] | None) -> list[torch.Tensor]:
            if not graphs:
                return []
            out: list[torch.Tensor] = []
            for graph in graphs:
                adj = nx.to_numpy_array(graph, dtype=np.float32)
                adj = (adj > 0).astype(np.float32)
                np.fill_diagonal(adj, 0.0)
                adj = np.maximum(adj, adj.T)
                out.append(torch.from_numpy(adj))
            return out

        payload = {
            "train": to_adj_list(train_graphs),
            "val": to_adj_list(val_graphs),
            "test": to_adj_list(test_graphs),
            "metadata": meta.__dict__,
            "note": "Benchmark-preserved splits for the DisCo wrapper; not consumed by upstream train_spectre.py.",
        }
        torch.save(payload, self.data_root / "benchmark_splits.pt")
        with open(self.data_root / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta.__dict__, f, indent=2)

    # ------------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------------
    def _aux_feature_flags(self) -> dict[str, bool]:
        return {
            "cycle_fea": bool(self.config.get("cycle_fea", True)),
            "eigen_fea": bool(self.config.get("eigen_fea", True)),
            "rwpe_fea": bool(self.config.get("rwpe_fea", False)),
            "global_fea": bool(self.config.get("global_fea", True)),
        }

    def _build_meta_from_graphs(
        self,
        train_graphs: list[nx.Graph],
        val_graphs: list[nx.Graph] | None = None,
        test_graphs: list[nx.Graph] | None = None,
    ) -> _DisCoDatasetMeta:
        self._import_modules()
        all_graphs = train_graphs + (val_graphs or []) + (test_graphs or [])
        max_n_nodes_all = max(g.number_of_nodes() for g in all_graphs)
        _, node_probs = self._node_count_distribution(train_graphs)
        if len(node_probs) <= max_n_nodes_all:
            node_probs = node_probs + [0.0] * (max_n_nodes_all + 1 - len(node_probs))

        n_node_type = max(1, max(int(g.nodes[n].get("node_label", 0)) for g in all_graphs for n in g.nodes()) + 1)
        n_edge_type = max(2, max([1] + [int(attrs.get("edge_type", 1)) for g in all_graphs for _, _, attrs in g.edges(data=True)]) + 1)
        edge_marginal = self._edge_marginal(train_graphs, n_edge_type=n_edge_type)
        node_counts = np.zeros(n_node_type, dtype=np.float64)
        for g in train_graphs:
            for node in g.nodes():
                node_counts[int(g.nodes[node].get("node_label", 0))] += 1.0
        if node_counts.sum() <= 0:
            node_counts[0] = 1.0
        node_marginal = (node_counts / node_counts.sum()).tolist()
        self._tmp_n_node_type = n_node_type
        self._tmp_n_edge_type = n_edge_type

        aux_flags = self._aux_feature_flags()
        AuxFeatures = self.mods["aux"].AuxFeatures
        aux = AuxFeatures(
            [aux_flags["cycle_fea"], aux_flags["eigen_fea"], aux_flags["rwpe_fea"], aux_flags["global_fea"]],
            max_n_nodes_all,
        )
        X0, E0, mask0 = self._graphs_to_dense_batch([train_graphs[0]])
        with torch.no_grad():
            X_t, E_t, y_t = aux(X0, E0, mask0)
        input_dims = {"X": int(X_t.shape[-1]), "E": int(E_t.shape[-1]), "y": int(y_t.shape[-1] + 1)}
        output_dims = {"X": n_node_type, "E": n_edge_type, "y": 0}

        n_layers = int(self.config.get("n_layers", 5))
        n_dim = int(self.config.get("n_dim", 128))
        backbone = str(self.config.get("backbone", "GT"))
        hidden_mlp_dims, hidden_dims = self._hidden_dims(backbone=backbone, n_dim=n_dim)

        return _DisCoDatasetMeta(
            dataset=self.dataset,
            n_node_type=n_node_type,
            n_edge_type=n_edge_type,
            max_n_nodes=max_n_nodes_all,
            n_node_distribution=[float(x) for x in node_probs],
            edge_marginal=[float(x) for x in edge_marginal],
            node_marginal=node_marginal,
            input_dims=input_dims,
            output_dims=output_dims,
            hidden_mlp_dims=hidden_mlp_dims,
            hidden_dims=hidden_dims,
            backbone=backbone,
            n_layers=n_layers,
            n_dim=n_dim,
            aux_features=aux_flags,
        )

    def _hidden_dims(self, *, backbone: str, n_dim: int) -> tuple[dict[str, int], dict[str, int]]:
        if backbone == "GT":
            hidden_mlp_dims = {"X": 128, "E": 64, "y": 128}
            hidden_dims = {
                "dx": 256,
                "de": 64,
                "dy": 64,
                "n_head": 8,
                "dim_ffX": 256,
                "dim_ffE": 64,
                "dim_ffy": 256,
            }
            override = self.config.get("graph_transformer_hidden_dims") or {}
            hidden_dims.update({k: int(v) for k, v in override.items()})
            override_mlp = self.config.get("graph_transformer_hidden_mlp_dims") or {}
            hidden_mlp_dims.update({k: int(v) for k, v in override_mlp.items()})
            return hidden_mlp_dims, hidden_dims
        if backbone == "MPNN":
            return {"X": n_dim, "E": n_dim, "y": n_dim}, {
                "dx": n_dim,
                "de": n_dim,
                "dy": n_dim,
                "n_head": int(self.config.get("n_head", 8)),
                "dim_ffX": n_dim,
                "dim_ffE": n_dim,
                "dim_ffy": n_dim,
            }
        raise ValueError(f"Unsupported DisCo backbone={backbone!r}. Expected 'GT' or 'MPNN'.")

    def _build_components(self, meta: _DisCoDatasetMeta) -> None:
        self._import_modules()
        self.meta = meta
        ForwardDiffusion = self.mods["forward_diff"].ForwardDiffusion
        UpstreamTauLeaping = self.mods["sampling"].TauLeaping
        AuxFeatures = self.mods["aux"].AuxFeatures
        GraphTransformer = self.mods["digress_models"].GraphTransformer
        MPNN = self.mods["models"].MPNN

        aux_flags = meta.aux_features
        self.add_auxiliary_feature = AuxFeatures(
            [aux_flags["cycle_fea"], aux_flags["eigen_fea"], aux_flags["rwpe_fea"], aux_flags["global_fea"]],
            meta.max_n_nodes,
        )

        edge_marginal = torch.tensor(meta.edge_marginal, dtype=torch.float32, device=self.device)
        edge_marginal = edge_marginal / edge_marginal.sum().clamp_min(1e-12)
        node_marginal = torch.tensor(meta.node_marginal, dtype=torch.float32, device=self.device)
        node_marginal = node_marginal / node_marginal.sum().clamp_min(1e-12)

        self.diffuser = ForwardDiffusion(
            meta.n_node_type,
            meta.n_edge_type,
            forward_type=str(self.config.get("diff_type", "marginal")),
            node_marginal=node_marginal,
            edge_marginal=edge_marginal,
            device=str(self.device),
            time_exponential=float(self.config.get("beta", 2.0)),
            time_base=float(self.config.get("alpha", 0.8)),
        )

        if meta.backbone == "GT":
            self.model = GraphTransformer(
                n_layers=meta.n_layers,
                input_dims=meta.input_dims,
                hidden_mlp_dims=meta.hidden_mlp_dims,
                hidden_dims=meta.hidden_dims,
                output_dims=meta.output_dims,
            ).to(self.device)
        elif meta.backbone == "MPNN":
            self.model = MPNN(
                n_layers=meta.n_layers,
                input_dims=meta.input_dims,
                hidden_dims=meta.n_dim,
                output_dims=meta.output_dims,
                dropout=float(self.config.get("dropout", 0.1)),
            ).to(self.device)
        else:
            raise ValueError(f"Unsupported DisCo backbone={meta.backbone!r}")

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(self.config.get("learning_rate", self.config.get("lr", 2e-4))),
            amsgrad=True,
            weight_decay=float(self.config.get("weight_decay", self.config.get("wd", 5e-12))),
        )

        sampling_steps = int(self.config.get("sampling_steps", self.config.get("sampling", {}).get("num_steps", 50)))
        min_time = float(self.config.get("min_time", 0.01))
        if str(self.config.get("sampler_backend", "safe")).lower() == "upstream":
            self.sampler = UpstreamTauLeaping(
                meta.n_node_type,
                meta.n_edge_type,
                num_steps=sampling_steps,
                min_t=min_time,
                add_auxiliary_feature=self.add_auxiliary_feature,
                device=str(self.device),
                BAR=bool(self.config.get("BAR", False)),
            )
        else:
            self.sampler = self._SafeTauLeaping(
                meta.n_node_type,
                meta.n_edge_type,
                num_steps=sampling_steps,
                min_t=min_time,
                add_auxiliary_feature=self.add_auxiliary_feature,
                device=str(self.device),
                rate_clip=float(self.config.get("sampler_rate_clip", 1000.0)),
            )

        probs = torch.tensor(meta.n_node_distribution, dtype=torch.float32)
        if probs.sum() <= 0:
            raise ValueError("Invalid DisCo node-count distribution: all probabilities are zero.")
        probs = probs / probs.sum()
        self.n_node_distribution = torch.distributions.Categorical(probs=probs)

    # ------------------------------------------------------------------
    # Training / validation / checkpointing
    # ------------------------------------------------------------------
    def _run_epoch(self, graphs: list[nx.Graph], *, train: bool) -> tuple[float, float]:
        if self.model is None or self.optimizer is None or self.diffuser is None or self.add_auxiliary_feature is None:
            raise RuntimeError("DisCo components have not been built.")
        self.model.train(train)
        min_time = float(self.config.get("min_time", 0.01))
        edge_loss_weight = float(self.config.get("edge_loss_weight", 5.0))
        include_node_feature = bool(getattr(self.diffuser, "diffuse_node", False))

        total_loss = 0.0
        total_acc = 0.0
        batches = 0
        iterator = self._iter_batches(graphs, shuffle=train and bool(self.config.get("shuffle", True)))
        context = contextlib.nullcontext() if train else torch.no_grad()
        with context:
            for batch_graphs in iterator:
                X_0, E_0, node_mask = self._graphs_to_dense_batch(batch_graphs)
                ts = torch.rand((E_0.shape[0],), device=self.device) * (1.0 - min_time) + min_time

                X_t_idx, E_t_idx = self.diffuser.forward_diffusion(X_0, E_0, ts)
                X_t_one_hot = X_t_idx
                E_t_one_hot = F.one_hot(E_t_idx, num_classes=self.meta.n_edge_type).float()
                X_t, E_t, y_t = self.add_auxiliary_feature(X_t_one_hot, E_t_one_hot, node_mask)
                y_t = torch.cat([y_t, ts.unsqueeze(-1)], dim=-1)

                pred_X_0, pred_E_0 = self.model(X_t, E_t, y_t, node_mask)

                X_0_idx = torch.max(X_0, dim=-1)[1].long()
                E_0_idx = torch.max(E_0, dim=-1)[1].long()
                X_0_idx_masked, E_0_idx_masked = self._add_mask_idx(
                    X_0_idx, E_0_idx, self.meta.n_node_type, self.meta.n_edge_type, node_mask
                )

                loss_E = self._ce_loss(pred_E_0, E_0_idx_masked)
                if include_node_feature and X_0_idx_masked is not None:
                    loss_X = self._ce_loss(pred_X_0, X_0_idx_masked)
                else:
                    loss_X = torch.tensor(0.0, device=self.device)
                loss = loss_X + edge_loss_weight * loss_E

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    clip_value = self.config.get("gradient_clip_value")
                    if clip_value is not None:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(clip_value))
                    self.optimizer.step()

                with torch.no_grad():
                    E_mask = node_mask.unsqueeze(-1) * node_mask.unsqueeze(-2)
                    pred_edge = torch.argmax(pred_E_0, dim=-1)
                    correct = (pred_edge[E_mask] == E_0_idx[E_mask]).float()
                    edge_acc = float(correct.mean().item()) if correct.numel() else float("nan")

                total_loss += float(loss.item())
                total_acc += edge_acc
                batches += 1

        if batches == 0:
            return float("nan"), float("nan")
        return total_loss / batches, total_acc / batches

    def _checkpoint_payload(self, epoch: int, train_loss: float, val_loss: float | None) -> dict[str, Any]:
        if self.model is None or self.meta is None:
            raise RuntimeError("Cannot checkpoint before model/meta are built.")
        return {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict() if self.optimizer is not None else None,
            "config": self.config,
            "dataset": self.dataset,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "metadata": self.meta.__dict__,
        }

    def _save_checkpoint(self, epoch: int, train_loss: float, val_loss: float | None) -> None:
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._checkpoint_payload(epoch, train_loss, val_loss), self.checkpoint_path)

    def train(self, train_graphs, val_graphs=None, test_graphs=None) -> None:
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        train_clean = self._prepare_graphs(train_graphs)
        val_clean = self._prepare_graphs(val_graphs) if val_graphs else None
        test_clean = self._prepare_graphs(test_graphs) if test_graphs else None

        meta = self._build_meta_from_graphs(train_clean, val_clean, test_clean)
        self._write_provenance_dataset(train_clean, val_clean, test_clean, meta)
        self._build_components(meta)

        num_epochs = int(self.config.get("num_epochs", self.config.get("epochs", 100)))
        log_interval = max(1, int(self.config.get("log_interval", 1)))
        best_score = float("inf")
        best_epoch = 0

        for epoch in range(1, num_epochs + 1):
            train_loss, train_edge_acc = self._run_epoch(train_clean, train=True)
            val_loss = None
            val_edge_acc = None
            if val_clean:
                val_loss, val_edge_acc = self._run_epoch(val_clean, train=False)
                score = val_loss
            else:
                score = train_loss

            if score < best_score or epoch == 1:
                best_score = score
                best_epoch = epoch
                self._save_checkpoint(epoch, train_loss, val_loss)

            if epoch % log_interval == 0 or epoch == 1 or epoch == num_epochs:
                msg = f"[DisCo] epoch {epoch}/{num_epochs} train_loss={train_loss:.4f} train_edge_acc={train_edge_acc:.4f}"
                if val_loss is not None:
                    msg += f" val_loss={val_loss:.4f} val_edge_acc={val_edge_acc:.4f}"
                print(msg)

        self.best_metric = best_score
        if self.checkpoint_path.exists():
            self.load()
        print(f"[DisCo] best_epoch={best_epoch} best_loss={best_score:.4f} checkpoint={self.checkpoint_path}")

    def load(self) -> None:
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"DisCo checkpoint not found: {self.checkpoint_path}. Run scripts/train_model.py first or set checkpoint_path."
            )
        try:
            ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        except TypeError:  # PyTorch < 2.6
            ckpt = torch.load(self.checkpoint_path, map_location=self.device)
        meta_raw = ckpt.get("metadata")
        if not meta_raw:
            raise ValueError(f"Checkpoint {self.checkpoint_path} does not contain DisCo metadata.")
        meta = _DisCoDatasetMeta(**meta_raw)
        self._build_components(meta)
        assert self.model is not None
        self.model.load_state_dict(ckpt["model_state_dict"])
        if self.optimizer is not None and ckpt.get("optimizer_state_dict") is not None:
            with contextlib.suppress(Exception):
                self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.model.eval()

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample(self, num_graphs: int, seed: int = 0, progress_callback=None):
        if self.model is None:
            self.load()
        if self.model is None or self.sampler is None or self.diffuser is None or self.n_node_distribution is None:
            raise RuntimeError("DisCo components are not ready for sampling.")

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        self.model.eval()
        sample_batch_size = max(1, int(self.config.get("sample_batch_size", self.config.get("batch_size", 32))))
        graphs: list[nx.Graph] = []

        remaining = int(num_graphs)
        while remaining > 0:
            batch = min(sample_batch_size, remaining)
            n_node = self.n_node_distribution.sample((batch,)).to(self.device)
            # Guard against a degenerate distribution with zero-size samples.
            if torch.any(n_node <= 0):
                replacement = torch.clamp(n_node, min=1)
                n_node = replacement

            X_sample, E, _node_mask = self.sampler.sample(self.diffuser, self.model, n_node)
            before = len(graphs)
            for i in range(batch):
                n = int(n_node[i].item())
                edge_mat = E[i, :n, :n].detach().cpu().numpy().astype(np.int64)
                adj = (edge_mat > 0).astype(np.int64)
                np.fill_diagonal(adj, 0)
                adj = np.maximum(adj, adj.T)
                graph = nx.Graph()
                x_arr = X_sample[i, :n].detach().cpu().numpy().astype(np.int64) if X_sample is not None else np.zeros(n, dtype=np.int64)
                for v in range(n):
                    label = int(max(x_arr[v], 0))
                    graph.add_node(v, node_label=label, feats=np.array([float(label)], dtype=np.float32))
                for u in range(n):
                    for v in range(u + 1, n):
                        et = int(max(edge_mat[u, v], edge_mat[v, u]))
                        if et > 0:
                            graph.add_edge(u, v, edge_type=et)
                graphs.append(graph)
            update_progress(progress_callback, min(len(graphs), num_graphs) - min(before, num_graphs))
            remaining -= batch

        return graphs
