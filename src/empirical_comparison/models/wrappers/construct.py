from __future__ import annotations

import json
import math
import os
import random
import shutil
import sys
import time
import types
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F

from empirical_comparison.models.base import BaseGenerator
from empirical_comparison.utils.logging import get_logger
from empirical_comparison.utils.progress import update_progress


LOGGER = get_logger(__name__)


class _ConfigNode(types.SimpleNamespace):
    """Small OmegaConf-like namespace used by upstream ConStruct modules."""

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, key, value)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def to_dict(self) -> dict[str, Any]:
        return _namespace_to_dict(self)


def _to_config_node(value: Any) -> Any:
    if isinstance(value, dict):
        return _ConfigNode(**{k: _to_config_node(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_to_config_node(v) for v in value]
    return value


def _namespace_to_dict(value: Any) -> Any:
    if isinstance(value, _ConfigNode) or isinstance(value, types.SimpleNamespace):
        return {k: _namespace_to_dict(v) for k, v in vars(value).items()}
    if isinstance(value, dict):
        return {k: _namespace_to_dict(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_namespace_to_dict(v) for v in value]
    return value


def _deep_update(dst: dict[str, Any], src: dict[str, Any] | None) -> dict[str, Any]:
    if not src:
        return dst
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            _deep_update(dst[key], value)
        else:
            dst[key] = value
    return dst


def _safe_symmetric_eigh(matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Run eigh with a CPU/double fallback for rare ill-conditioned batches."""
    matrix = torch.nan_to_num(matrix)
    matrix = (matrix + matrix.transpose(-1, -2)) / 2
    try:
        return torch.linalg.eigh(matrix)
    except RuntimeError as exc:
        if "linalg.eigh" not in str(exc):
            raise

    device = matrix.device
    dtype = matrix.dtype
    n = matrix.shape[-1]
    eye = torch.eye(n, dtype=torch.float64, device="cpu")
    eigvals: list[torch.Tensor] = []
    eigvectors: list[torch.Tensor] = []

    for item in matrix.detach().to(device="cpu", dtype=torch.float64):
        item = (item + item.transpose(-1, -2)) / 2
        last_error: RuntimeError | None = None
        for attempt in range(6):
            shift = 0.0 if attempt == 0 else 1.0e-8 * (10 ** (attempt - 1))
            try:
                vals, vecs = torch.linalg.eigh(item + shift * eye)
                if shift:
                    vals = vals - shift
                eigvals.append(vals.to(device=device, dtype=dtype))
                eigvectors.append(vecs.to(device=device, dtype=dtype))
                break
            except RuntimeError as err:
                last_error = err
        else:
            raise RuntimeError("torch.linalg.eigh failed after CPU/double fallback with jitter") from last_error

    return torch.stack(eigvals, dim=0), torch.stack(eigvectors, dim=0)


def _finite_tensor(tensor: torch.Tensor, limit: float | None = None) -> torch.Tensor:
    tensor = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
    if limit is not None:
        tensor = torch.clamp(tensor, min=-limit, max=limit)
    return tensor


class _PlaceHolder:
    """Minimal ConStruct-compatible dense graph container.

    The upstream ConStruct modules only need ``mask``, ``copy``, ``collapse`` and
    ``device_as`` along the benchmark path.  Providing this shim lets the
    wrapper use ConStruct's real Transformer, noise model, auxiliary features,
    and projector code without importing the upstream PyG datamodule.
    """

    def __init__(
        self,
        X: torch.Tensor | None,
        E: torch.Tensor | None,
        y: torch.Tensor | None,
        charges: torch.Tensor | None = None,
        t_int: torch.Tensor | None = None,
        t: torch.Tensor | None = None,
        node_mask: torch.Tensor | None = None,
    ) -> None:
        self.X = X
        self.E = E
        self.y = y
        self.charges = charges
        self.t_int = t_int
        self.t = t
        self.node_mask = node_mask

    def device_as(self, x: torch.Tensor):
        device = x.device
        self.X = self.X.to(device) if self.X is not None else None
        self.E = self.E.to(device) if self.E is not None else None
        self.y = self.y.to(device) if self.y is not None else None
        self.charges = self.charges.to(device) if self.charges is not None else None
        self.t_int = self.t_int.to(device) if self.t_int is not None else None
        self.t = self.t.to(device) if self.t is not None else None
        self.node_mask = self.node_mask.to(device) if self.node_mask is not None else None
        return self

    def mask(self, node_mask: torch.Tensor | None = None):
        if node_mask is None:
            if self.node_mask is None:
                raise ValueError("node_mask must be provided for masking.")
            node_mask = self.node_mask
        else:
            self.node_mask = node_mask
        bs, n = node_mask.shape
        x_mask = node_mask.unsqueeze(-1)
        e_mask1 = x_mask.unsqueeze(2)
        e_mask2 = x_mask.unsqueeze(1)
        diag_mask = (
            ~torch.eye(n, dtype=torch.bool, device=node_mask.device)
            .unsqueeze(0)
            .expand(bs, -1, -1)
            .unsqueeze(-1)
        )
        if self.X is not None:
            self.X = self.X * x_mask
        if self.charges is not None and self.charges.numel() > 0:
            self.charges = self.charges * x_mask
        if self.E is not None:
            self.E = self.E * e_mask1 * e_mask2 * diag_mask
            if not torch.allclose(self.E, self.E.transpose(1, 2), atol=1e-5):
                raise AssertionError("Dense edge tensor is not symmetric after masking.")
        return self

    def collapse(self, collapse_charges: torch.Tensor | None = None):
        copy = self.copy()
        copy.X = torch.argmax(self.X, dim=-1)
        if self.charges is not None and self.charges.numel() > 0:
            if collapse_charges is not None:
                copy.charges = collapse_charges.to(self.charges.device)[torch.argmax(self.charges, dim=-1)]
            else:
                copy.charges = torch.argmax(self.charges, dim=-1)
        else:
            copy.charges = self.X.new_zeros(self.X.shape[:-1], dtype=torch.long)
        copy.E = torch.argmax(self.E, dim=-1)
        if self.node_mask is not None:
            x_mask = self.node_mask.unsqueeze(-1)
            e_mask = (x_mask.unsqueeze(2) * x_mask.unsqueeze(1)).squeeze(-1)
            copy.X[self.node_mask == 0] = -1
            copy.charges[self.node_mask == 0] = 1000
            copy.E[e_mask == 0] = -1
        return copy

    def copy(self):
        return _PlaceHolder(
            X=self.X,
            E=self.E,
            y=self.y,
            charges=self.charges,
            t_int=self.t_int,
            t=self.t,
            node_mask=self.node_mask,
        )

    def split(self):
        if self.node_mask is None:
            raise ValueError("Cannot split a PlaceHolder without node_mask.")
        graphs = []
        for i in range(self.X.shape[0]):
            n = int(self.node_mask[i].sum().item())
            graphs.append(
                _PlaceHolder(
                    X=self.X[i, :n],
                    E=self.E[i, :n, :n],
                    y=self.y[i] if self.y is not None and self.y.numel() else None,
                    charges=self.charges[i, :n] if self.charges is not None else None,
                    node_mask=None,
                )
            )
        return graphs


class _NodeDistribution:
    def __init__(self, probs: Iterable[float]) -> None:
        p = torch.as_tensor(list(probs), dtype=torch.float32)
        if p.numel() == 0 or p.sum() <= 0:
            raise ValueError("Node-count distribution must contain at least one positive entry.")
        self.probs = p / p.sum()

    def sample_n(self, n: int, device: torch.device | str) -> torch.Tensor:
        idx = torch.multinomial(self.probs, num_samples=int(n), replacement=True)
        return idx.to(device=device, dtype=torch.long)

    def log_prob(self, n_nodes: torch.Tensor) -> torch.Tensor:
        probs = self.probs.to(n_nodes.device)
        n_nodes = n_nodes.clamp(min=0, max=len(probs) - 1).long()
        return torch.log(probs[n_nodes] + 1e-12)


@dataclass
class _ConStructDatasetMeta:
    dataset: str
    num_train: int
    num_val: int
    num_test: int
    max_n_nodes: int
    n_node_distribution: list[float]
    atom_types: list[float]
    edge_types: list[float]
    num_atom_types: int = 1
    num_edge_types: int = 2
    num_charge_types: int = 0


class _ConStructDatasetInfos:
    """Dataset-info object with the attributes expected by ConStruct modules."""

    def __init__(self, meta: _ConStructDatasetMeta, placeholder_cls: type[_PlaceHolder]) -> None:
        self.is_molecular = False
        self.is_tls = False
        self.dataset_name = meta.dataset
        self.max_n_nodes = int(meta.max_n_nodes)
        self.num_atom_types = int(meta.num_atom_types)
        self.num_edge_types = int(meta.num_edge_types)
        self.num_charge_types = int(meta.num_charge_types)
        self.atom_types = torch.as_tensor(meta.atom_types, dtype=torch.float32)
        self.edge_types = torch.as_tensor(meta.edge_types, dtype=torch.float32)
        self.charge_types = torch.zeros((0,), dtype=torch.float32)
        self.charges_marginals = torch.zeros((0,), dtype=torch.float32)
        self.n_nodes = torch.as_tensor(meta.n_node_distribution, dtype=torch.float32)
        self.nodes_dist = _NodeDistribution(meta.n_node_distribution)
        self.input_dims = placeholder_cls(X=self.num_atom_types, charges=0, E=self.num_edge_types, y=1)
        self.output_dims = placeholder_cls(X=self.num_atom_types, charges=0, E=self.num_edge_types, y=0)
        self.collapse_charges = None
        self.statistics = {}

    def to_one_hot(self, pl: _PlaceHolder) -> _PlaceHolder:
        pl.X = F.one_hot(pl.X.long(), num_classes=self.num_atom_types).float()
        pl.E = F.one_hot(pl.E.long(), num_classes=self.num_edge_types).float()
        pl.charges = pl.X.new_zeros((*pl.X.shape[:-1], 0))
        return pl.mask(pl.node_mask)


class _DenseConStructModel(torch.nn.Module):
    """Dense benchmark model assembled from ConStruct's upstream components."""

    def __init__(self, cfg: _ConfigNode, dataset_infos: _ConStructDatasetInfos, mods: dict[str, Any]) -> None:
        super().__init__()
        self.cfg = cfg
        self.dataset_infos = dataset_infos
        self.T = int(cfg.model.diffusion_steps)
        self.nodes_dist = dataset_infos.nodes_dist
        self.PlaceHolder = mods["PlaceHolder"]

        ExtraFeatures = mods["ExtraFeatures"]
        DummyExtraFeatures = mods["DummyExtraFeatures"]
        GraphTransformer = mods["GraphTransformer"]
        noise_mod = mods["noise_model"]

        if bool(cfg.model.extra_features):
            self.extra_features = ExtraFeatures(cfg=cfg, dataset_infos=dataset_infos)
            self.input_dims = self.extra_features.update_input_dims(dataset_infos.input_dims)
        else:
            self.extra_features = DummyExtraFeatures()
            self.input_dims = dataset_infos.input_dims
        self.domain_features = DummyExtraFeatures()
        self.output_dims = dataset_infos.output_dims

        self.model = GraphTransformer(
            input_dims=self.input_dims,
            n_layers=int(cfg.model.n_layers),
            hidden_mlp_dims=dict(cfg.model.hidden_mlp_dims.to_dict() if hasattr(cfg.model.hidden_mlp_dims, "to_dict") else cfg.model.hidden_mlp_dims),
            hidden_dims=dict(cfg.model.hidden_dims.to_dict() if hasattr(cfg.model.hidden_dims, "to_dict") else cfg.model.hidden_dims),
            output_dims=self.output_dims,
            dropout=float(cfg.model.dropout),
            dropout_in_and_out=bool(cfg.model.dropout_in_and_out),
        )

        transition = str(cfg.model.transition)
        if transition == "uniform":
            self.noise_model = noise_mod.DiscreteUniformTransition(output_dims=self.output_dims, cfg=cfg)
        elif transition == "marginal":
            self.noise_model = noise_mod.MarginalTransition(
                x_marginals=dataset_infos.atom_types,
                e_marginals=dataset_infos.edge_types,
                charges_marginals=dataset_infos.charges_marginals,
                y_classes=self.output_dims.y,
                cfg=cfg,
            )
        elif transition == "absorbing":
            self.noise_model = noise_mod.AbsorbingTransition(cfg=cfg, output_dims=self.output_dims)
        elif transition == "absorbing_edges":
            self.noise_model = noise_mod.AbsorbingEdgesTransition(
                cfg=cfg,
                x_marginals=dataset_infos.atom_types,
                e_marginals=dataset_infos.edge_types,
                charges_marginals=dataset_infos.charges_marginals,
                y_classes=self.output_dims.y,
            )
        else:
            raise ValueError(f"Unsupported ConStruct transition: {transition}")

    def forward(self, z_t: _PlaceHolder) -> _PlaceHolder:
        if z_t.node_mask is None:
            raise ValueError("ConStruct forward requires node_mask.")
        extra_features = self.extra_features(z_t)
        extra_domain_features = self.domain_features(z_t)
        model_input = z_t.copy()
        model_input.X = _finite_tensor(
            torch.cat((z_t.X, z_t.charges, extra_features.X, extra_domain_features.X), dim=2).float(),
            limit=1.0e4,
        )
        model_input.E = _finite_tensor(
            torch.cat((z_t.E, extra_features.E, extra_domain_features.E), dim=3).float(),
            limit=1.0e4,
        )
        model_input.y = _finite_tensor(
            torch.hstack((z_t.y, extra_features.y, extra_domain_features.y, z_t.t)).float(),
            limit=1.0e4,
        )
        return self.model(model_input)

    def loss_on_clean_batch(self, clean_data: _PlaceHolder, log: bool = False) -> tuple[torch.Tensor, dict[str, float]]:
        z_t = self.noise_model.apply_noise(clean_data)
        pred = self.forward(z_t)
        node_mask = clean_data.node_mask
        bs, n = node_mask.shape
        lambdas = list(self.cfg.model.lambda_train)

        total = clean_data.X.new_tensor(0.0)
        logs: dict[str, float] = {}

        if pred.X.numel() and clean_data.X.numel():
            x_logits = pred.X[node_mask]
            x_target = torch.argmax(clean_data.X[node_mask], dim=-1)
            if x_logits.numel() > 0:
                x_loss = F.cross_entropy(x_logits, x_target, reduction="mean")
                total = total + float(lambdas[0]) * x_loss
                logs["train_loss/X_CE"] = float(x_loss.detach().cpu())

        if clean_data.charges is not None and clean_data.charges.numel() > 0:
            c_logits = pred.charges[node_mask]
            c_target = torch.argmax(clean_data.charges[node_mask], dim=-1)
            c_loss = F.cross_entropy(c_logits, c_target, reduction="mean")
            total = total + float(lambdas[1]) * c_loss
            logs["train_loss/charges_CE"] = float(c_loss.detach().cpu())

        if pred.E.numel() and clean_data.E.numel():
            diag_mask = ~torch.eye(n, device=node_mask.device, dtype=torch.bool).unsqueeze(0).expand(bs, -1, -1)
            edge_mask = diag_mask & node_mask.unsqueeze(1) & node_mask.unsqueeze(2)
            e_logits = pred.E[edge_mask]
            e_target = torch.argmax(clean_data.E[edge_mask], dim=-1)
            if e_logits.numel() > 0:
                e_loss = F.cross_entropy(e_logits, e_target, reduction="mean")
                total = total + float(lambdas[2]) * e_loss
                logs["train_loss/E_CE"] = float(e_loss.detach().cpu())

        if pred.y is not None and clean_data.y is not None and clean_data.y.numel() > 0:
            y_target = torch.argmax(clean_data.y, dim=-1)
            y_loss = F.cross_entropy(pred.y, y_target, reduction="mean")
            total = total + float(lambdas[3]) * y_loss
            logs["train_loss/y_CE"] = float(y_loss.detach().cpu())

        logs["train_loss/batch_loss"] = float(total.detach().cpu())
        return total, logs if log else {}

    @torch.no_grad()
    def sample_batch(
        self,
        n_nodes: list[int],
        faster_sampling: int = 1,
        rev_proj: str | bool | None = None,
        projector_classes: dict[str, Any] | None = None,
    ) -> _PlaceHolder:
        device = next(self.parameters()).device
        n_nodes_tensor = torch.as_tensor(n_nodes, dtype=torch.long, device=device)
        batch_size = int(n_nodes_tensor.numel())
        n_max = int(n_nodes_tensor.max().item())
        arange = torch.arange(n_max, device=device).unsqueeze(0).expand(batch_size, -1)
        node_mask = arange < n_nodes_tensor.unsqueeze(1)
        z_t = self.noise_model.sample_limit_dist(node_mask=node_mask)

        projector = None
        if rev_proj and str(rev_proj).lower() not in {"false", "none", "0"}:
            if str(self.cfg.model.transition) != "absorbing_edges":
                raise ValueError("ConStruct reverse projectors require transition='absorbing_edges'.")
            if projector_classes is None or str(rev_proj) not in projector_classes:
                raise ValueError(f"Unknown ConStruct reverse projector: {rev_proj}")
            projector = projector_classes[str(rev_proj)](z_t)

        step = max(1, int(faster_sampling))
        for s_int in reversed(range(0, self.T, step)):
            s_array = torch.full((batch_size, 1), int(s_int), dtype=torch.long, device=device)
            pred = self.forward(z_t)
            z_s = self.noise_model.sample_zs_from_zt_and_pred(z_t=z_t, pred=pred, s_int=s_array)
            if projector is not None:
                projector.project(z_s)
            z_t = z_s
        return z_t.collapse(self.dataset_infos.collapse_charges)

    @torch.no_grad()
    def sample_n_graphs(
        self,
        samples_to_generate: int,
        sample_batch_size: int,
        faster_sampling: int,
        rev_proj: str | bool | None,
        projector_classes: dict[str, Any] | None,
        progress_callback=None,
    ) -> list[_PlaceHolder]:
        if samples_to_generate <= 0:
            return []
        device = next(self.parameters()).device
        batches = []
        remaining = int(samples_to_generate)
        while remaining > 0:
            cur = min(int(sample_batch_size), remaining)
            n_nodes = self.nodes_dist.sample_n(cur, device=device).tolist()
            # Node count zero has probability zero in normal benchmark data, but
            # guard against malformed checkpoints/configs.
            n_nodes = [max(1, int(n)) for n in n_nodes]
            batches.append(
                self.sample_batch(
                    n_nodes=n_nodes,
                    faster_sampling=faster_sampling,
                    rev_proj=rev_proj,
                    projector_classes=projector_classes,
                )
            )
            update_progress(progress_callback, cur)
            remaining -= cur
        return batches


class ConStructWrapper(BaseGenerator):
    """Benchmark adapter for the uploaded ConStruct repository.

    The original ConStruct entry point uses Hydra, PyTorch Lightning, PyG
    SPECTRE datamodules, optional graph-tool/ORCA metrics, and generated-graph
    visualization.  Those paths are useful for the paper codebase but are too
    brittle for a benchmark wrapper that must preserve externally persisted
    train/validation/test splits.

    This wrapper therefore assembles ConStruct's real dense components directly:
    ``GraphTransformer``, discrete noise transitions, auxiliary spectral/cycle
    features, and optional reverse projectors.  It owns the NetworkX -> dense
    tensor conversion, checkpoint metadata, train loop, and sample conversion so
    that benchmark splits and isolated nodes are preserved exactly.
    """

    supports_training = True
    supports_sampling = True
    supports_node_features = False
    supports_edge_features = False
    supports_node_labels = True
    supports_edge_labels = True
    supports_graph_labels = False
    supports_constraints = True
    supports_variable_size = True
    supports_featureless_graphs = True

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        default_repo_root = Path(__file__).resolve().parents[4] / "external" / "ConStruct"
        repo_root = os.environ.get("CONSTRUCT_REPO") or config.get("repo_root") or default_repo_root
        self.repo_root = self._normalize_repo_root(Path(repo_root).expanduser().resolve())

        self.dataset = str(config.get("dataset") or config.get("dataset_name") or "planar").lower()
        self.model_name = str(config.get("name", "construct"))
        self.seed = int(config.get("seed", 42))
        self.device = torch.device(config.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
        if self.device.type == "cpu":
            torch.set_num_threads(int(config.get("torch_num_threads", 1)))

        self.checkpoint_path = self._resolve_path(config.get("checkpoint_path", "outputs/checkpoints/{dataset}/construct.pt"))
        self.run_root = self._resolve_path(config.get("run_root", "outputs/construct_runs/{dataset}"))
        self.data_root = self._resolve_path(config.get("data_root", "outputs/construct_data/{dataset}"))

        self.mods: dict[str, Any] = {}
        self.repo_loaded = False
        self.cfg_dict = self._make_cfg_dict()
        self.cfg = _to_config_node(self.cfg_dict)
        self.meta: _ConStructDatasetMeta | None = None
        self.dataset_infos: _ConStructDatasetInfos | None = None
        self.model: _DenseConStructModel | None = None
        self.projector_classes: dict[str, Any] | None = None
        self.detailed_logging = bool(config.get("detailed_logging", True))
        self.log_train_every_n_steps = max(1, int(config.get("log_train_every_n_steps", 1)))
        self.log_epoch_every = max(1, int(config.get("log_epoch_every", 1)))

    @property
    def name(self) -> str:
        return "construct"

    # ------------------------------------------------------------------
    # Path/config helpers
    # ------------------------------------------------------------------
    def _resolve_path(self, value: str | os.PathLike[str]) -> Path:
        raw = str(value)
        replacements = {"dataset": self.dataset, "model": self.model_name, "name": self.model_name}
        for k, v in replacements.items():
            raw = raw.replace("${" + k + "}", v)
        try:
            raw = raw.format(**replacements)
        except Exception:
            pass
        return Path(raw).expanduser().resolve()

    def _normalize_repo_root(self, repo_root: Path) -> Path:
        """Accept either the repository root or its nested ConStruct package root."""
        candidates = [repo_root]
        if (repo_root / "ConStruct").exists():
            candidates.append(repo_root / "ConStruct")
        if repo_root.name == "ConStruct" and (repo_root / "configs").exists() and (repo_root / "ConStruct").exists():
            candidates.insert(0, repo_root)
        if repo_root.name == "ConStruct" and (repo_root.parent / "configs").exists():
            candidates.append(repo_root.parent)
        for candidate in candidates:
            if (candidate / "ConStruct" / "models" / "transformer_model.py").exists() and (candidate / "configs").exists():
                return candidate
        return repo_root

    def _validate_repo_root(self) -> None:
        required = [
            "ConStruct/models/transformer_model.py",
            "ConStruct/diffusion/noise_model.py",
            "ConStruct/diffusion/extra_features.py",
            "ConStruct/projector/projector_utils.py",
        ]
        missing = [p for p in required if not (self.repo_root / p).exists()]
        if missing:
            raise FileNotFoundError(
                "ConStruct repository layout was not found. Set CONSTRUCT_REPO or "
                "configs/models/construct.yaml:repo_root to the extracted ConStruct directory. "
                f"repo_root={self.repo_root}; missing={missing}"
            )

    def _make_cfg_dict(self) -> dict[str, Any]:
        sampling_cfg = self.config.get("sampling", {}) or {}
        diffusion_steps = int(self.config.get("diffusion_steps", self.config.get("T", 500)))
        sampling_steps = int(self.config.get("sampling_steps", sampling_cfg.get("num_steps", diffusion_steps)))
        faster_sampling = max(1, int(math.ceil(diffusion_steps / max(1, sampling_steps))))

        hidden_mlp_dims = self.config.get("hidden_mlp_dims", {"X": 256, "E": 128, "y": 128})
        hidden_dims = self.config.get(
            "hidden_dims",
            {"dx": 256, "de": 64, "dy": 64, "n_head": 8, "dim_ffX": 256, "dim_ffE": 128, "dim_ffy": 128},
        )

        rev_proj_value = self.config.get("rev_proj", None)
        if rev_proj_value is None:
            rev_proj_value = "planar" if self.dataset == "planar" else False

        molecular_like = self.dataset in {"qm9", "zinc"}
        disable_structural_for_molecular = bool(self.config.get("disable_structural_features_for_molecular", True))
        use_structural_features = not (molecular_like and disable_structural_for_molecular)

        cfg = {
            "general": {
                "name": self.config.get("experiment_name", f"construct_{self.dataset}"),
                "wandb": "disabled",
                "log_every_steps": int(self.config.get("log_every_steps", 50)),
                "faster_sampling": faster_sampling,
                "number_chain_steps": int(self.config.get("number_chain_steps", 50)),
            },
            "train": {
                "n_epochs": int(self.config.get("num_epochs", self.config.get("n_epochs", 100))),
                "batch_size": int(self.config.get("batch_size", 32)),
                "lr": float(self.config.get("learning_rate", self.config.get("lr", 2e-4))),
                "weight_decay": float(self.config.get("weight_decay", 1e-12)),
                "clip_grad": 1.0
                if self.config.get("clip_grad", 1.0) is None
                else self.config.get("clip_grad", 1.0),
                "seed": int(self.config.get("seed", self.seed)),
                "num_workers": int(self.config.get("num_workers", 0)),
            },
            "model": {
                "transition": str(self.config.get("transition", "absorbing_edges")),
                "diffusion_steps": diffusion_steps,
                "n_layers": int(self.config.get("n_layers", 5)),
                "extra_features": bool(self.config.get("extra_features", True)) and use_structural_features,
                "eigenfeatures": bool(self.config.get("eigenfeatures", True)) and use_structural_features,
                "max_degree": int(self.config.get("max_degree", 10)),
                "num_eigenvectors": int(self.config.get("num_eigenvectors", 5)),
                "num_eigenvalues": int(self.config.get("num_eigenvalues", 9)),
                "num_degree": int(self.config.get("num_degree", 10)),
                "extra_molecular_features": False,
                "hidden_mlp_dims": hidden_mlp_dims,
                "hidden_dims": hidden_dims,
                "lambda_train": list(self.config.get("lambda_train", [1, 2, 5, 0])),
                "nu": dict(self.config.get("nu", {"x": 1, "c": 1, "e": 1, "y": 1})),
                "rev_proj": rev_proj_value,
                "dropout": float(self.config.get("dropout", 0.1)),
                "dropout_in_and_out": bool(self.config.get("dropout_in_and_out", False)),
                "cycle_features": bool(self.config.get("cycle_features", True)) and use_structural_features,
            },
            "dataset": {
                "name": self.dataset,
                "adaptive_loader": False,
                "fraction": 1.0,
                "nodes_dist_source": str(self.config.get("nodes_dist_source", "train")),
            },
            "sampling": {
                "sample_batch_size": int(self.config.get("sample_batch_size", sampling_cfg.get("batch_size", self.config.get("batch_size", 32)))),
                "sampling_steps": sampling_steps,
                "faster_sampling": faster_sampling,
            },
        }
        _deep_update(cfg["model"], self.config.get("model_overrides"))
        _deep_update(cfg["train"], self.config.get("train_overrides"))
        _deep_update(cfg["general"], self.config.get("general_overrides"))
        _deep_update(cfg["dataset"], self.config.get("dataset_overrides"))
        _deep_update(cfg["sampling"], self.config.get("sampling_overrides"))
        return cfg

    # ------------------------------------------------------------------
    # Import upstream components with a tiny ConStruct.utils shim
    # ------------------------------------------------------------------
    def _ensure_repo_importable(self) -> None:
        for path in (self.repo_root, self.repo_root / "ConStruct"):
            path_str = str(path)
            if path_str not in sys.path:
                sys.path.insert(0, path_str)

    def _install_utils_shim(self) -> None:
        # Avoid importing upstream ConStruct.utils, which immediately imports
        # torch_geometric and torchmetrics.  The components used here only need
        # PlaceHolder plus a few optional no-op helpers.
        shim = types.ModuleType("ConStruct.utils")
        shim.PlaceHolder = _PlaceHolder
        shim.create_folders = lambda *args, **kwargs: None
        shim.setup_wandb = lambda cfg: cfg
        sys.modules["ConStruct.utils"] = shim
        try:
            construct_pkg = __import__("ConStruct")
            setattr(construct_pkg, "utils", shim)
        except Exception:
            pass
        self.mods["PlaceHolder"] = _PlaceHolder

    def _patch_extra_features_bool_bug(self, extra_mod: Any) -> None:
        # Uploaded ConStruct has: self.eigenfeatures = (cfg.model.eigenfeatures,)
        # which is always truthy.  Patch so eigenfeatures=False really disables
        # eigenvalue/eigenvector features.
        ExtraFeatures = extra_mod.ExtraFeatures
        if getattr(ExtraFeatures, "_empirical_bool_patch", False):
            return
        AdjacencyFeatures = extra_mod.AdjacencyFeatures
        EigenFeatures = extra_mod.EigenFeatures

        def fixed_init(self, cfg, dataset_infos):
            use_eigen = bool(cfg.model.eigenfeatures)
            max_degree = int(cfg.model.max_degree)
            if max_degree < 6:
                raise ValueError(
                    "ConStruct auxiliary cycle/path features require max_degree >= 6 "
                    "because the upstream feature code computes powers A^1..A^6."
                )
            self.eigenfeatures = use_eigen
            self.max_n_nodes = dataset_infos.max_n_nodes
            self.adj_features = AdjacencyFeatures(
                num_degree=cfg.model.num_degree,
                max_degree=max_degree,
                cycle_features=cfg.model.cycle_features,
            )
            if use_eigen:
                self.eigenfeatures = EigenFeatures(
                    num_eigenvectors=cfg.model.num_eigenvectors,
                    num_eigenvalues=cfg.model.num_eigenvalues,
                )

        def fixed_update_input_dims(self, input_dims):
            # The upstream variable name ``y_cycles`` is misleading: the
            # AdjacencyFeatures object actually returns graph-level cycle
            # counts, degree distribution, node marginal, and edge marginal.
            # ExtraFeatures.__call__ then prepends normalized graph size and,
            # when enabled, appends eigenvalue features.  The uploaded
            # update_input_dims is inconsistent with that tensor layout because
            # of the tuple-bool bug in __init__; this corrected version matches
            # the actual tensors emitted by __call__.
            base_x = input_dims.X
            base_e = input_dims.E
            if self.eigenfeatures:
                input_dims.y += self.eigenfeatures.num_eigenvalues + 1
                input_dims.X += self.eigenfeatures.num_eigenvectors + 1
            input_dims.X += 3  # node-level 3/4/5-cycle features
            input_dims.y += 1  # normalized graph size
            input_dims.y += 4  # graph-level 3/4/5/6-cycle features
            input_dims.y += self.adj_features.num_degree + 2
            input_dims.y += base_x
            input_dims.y += base_e
            input_dims.E += self.adj_features.max_degree
            input_dims.E += 1
            return input_dims

        ExtraFeatures.__init__ = fixed_init
        ExtraFeatures.update_input_dims = fixed_update_input_dims
        ExtraFeatures._empirical_bool_patch = True

    def _patch_extra_features_eigh_fallback(self, extra_mod: Any) -> None:
        EigenFeatures = extra_mod.EigenFeatures
        if getattr(EigenFeatures, "_empirical_eigh_patch", False):
            return

        def robust_compute_features(self, noisy_data):
            E_t = noisy_data.E
            mask = noisy_data.node_mask
            A = E_t[..., 1:].sum(dim=-1).float() * mask.unsqueeze(1) * mask.unsqueeze(2)
            L = self.compute_laplacian(A, normalize=False)
            mask_diag = 2 * L.shape[-1] * torch.eye(A.shape[-1], device=A.device).type_as(L).unsqueeze(0)
            mask_diag = mask_diag * (~mask.unsqueeze(1)) * (~mask.unsqueeze(2))
            L = L * mask.unsqueeze(1) * mask.unsqueeze(2) + mask_diag

            eigvals, eigvectors = _safe_symmetric_eigh(L)
            eigvals = eigvals.type_as(A) / torch.sum(mask, dim=1, keepdim=True)
            eigvals = torch.clamp(eigvals, min=0)
            eigvectors = eigvectors * mask.unsqueeze(2) * mask.unsqueeze(1)

            n_connected_comp, batch_eigenvalues = self.eigenvalues_features(
                eigenvalues=eigvals,
                num_eigenvalues=self.num_eigenvalues,
            )
            evector_feat = self.eigenvector_features(
                vectors=eigvectors,
                node_mask=noisy_data.node_mask,
                n_connected=n_connected_comp,
                num_eigenvectors=self.num_eigenvectors,
            )

            evalue_feat = torch.hstack((n_connected_comp, batch_eigenvalues))
            return evalue_feat, evector_feat

        EigenFeatures.compute_features = robust_compute_features
        EigenFeatures._empirical_eigh_patch = True

    def _import_modules(self) -> None:
        if self.repo_loaded:
            return
        self._validate_repo_root()
        self._ensure_repo_importable()
        self._install_utils_shim()

        import importlib

        transformer_mod = importlib.import_module("models.transformer_model")
        noise_mod = importlib.import_module("diffusion.noise_model")
        extra_mod = importlib.import_module("ConStruct.diffusion.extra_features")
        self._patch_extra_features_bool_bug(extra_mod)
        self._patch_extra_features_eigh_fallback(extra_mod)

        self.mods["GraphTransformer"] = transformer_mod.GraphTransformer
        self.mods["noise_model"] = noise_mod
        self.mods["ExtraFeatures"] = extra_mod.ExtraFeatures
        self.mods["DummyExtraFeatures"] = extra_mod.DummyExtraFeatures
        self.repo_loaded = True

    def _import_projectors(self) -> dict[str, Any]:
        if self.projector_classes is not None:
            return self.projector_classes
        self._import_modules()
        import importlib

        try:
            projector_mod = importlib.import_module("ConStruct.projector.projector_utils")
            self.projector_classes = {
                "planar": projector_mod.PlanarProjector,
                "tree": projector_mod.TreeProjector,
                "lobster": projector_mod.LobsterProjector,
            }
        except Exception as exc:
            warnings.warn(
                f"ConStruct reverse projector import failed; sampling will run without projection unless requested: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            self.projector_classes = {}
        return self.projector_classes

    # ------------------------------------------------------------------
    # Dataset materialization and dense conversion
    # ------------------------------------------------------------------
    def _graphs_to_adj_arrays(self, graphs: list[nx.Graph]) -> list[dict[str, np.ndarray]]:
        arrays: list[dict[str, np.ndarray]] = []
        for idx, g in enumerate(graphs):
            if g.number_of_nodes() == 0:
                raise ValueError(f"ConStructWrapper does not support empty graphs; graph index={idx}.")
            if g.is_directed():
                raise ValueError("ConStructWrapper expects undirected simple graphs.")
            g2 = nx.convert_node_labels_to_integers(nx.Graph(g), ordering="default")
            n = g2.number_of_nodes()
            X = np.zeros((n,), dtype=np.int64)
            for node in range(n):
                X[node] = int(g2.nodes[node].get("node_label", 0))
            E = np.zeros((n, n), dtype=np.int64)
            for u, v, attrs in g2.edges(data=True):
                if u == v:
                    continue
                edge_type = max(int(attrs.get("edge_type", 1)), 1)
                E[int(u), int(v)] = edge_type
                E[int(v), int(u)] = edge_type
            arrays.append({"X": X, "E": E})
        return arrays

    def _split_arrays_for_meta(
        self,
        train_graphs: list[nx.Graph],
        val_graphs: list[nx.Graph] | None,
        test_graphs: list[nx.Graph] | None,
    ) -> tuple[list[dict[str, np.ndarray]], list[dict[str, np.ndarray]], list[dict[str, np.ndarray]]]:
        train = self._graphs_to_adj_arrays(train_graphs)
        val = self._graphs_to_adj_arrays(val_graphs or [])
        test = self._graphs_to_adj_arrays(test_graphs or [])
        if not val:
            n_val = max(1, int(round(0.1 * len(train)))) if len(train) > 2 else 1
            val = [{"X": a["X"].copy(), "E": a["E"].copy()} for a in train[:n_val]]
        if not test:
            test = [{"X": a["X"].copy(), "E": a["E"].copy()} for a in val]
        return train, val, test

    def _build_meta(self, train: list[dict[str, np.ndarray]], val: list[dict[str, np.ndarray]], test: list[dict[str, np.ndarray]]) -> _ConStructDatasetMeta:
        if len(train) == 0:
            raise ValueError("ConStructWrapper requires at least one training graph.")
        source = str(self.cfg_dict.get("dataset", {}).get("nodes_dist_source", "train"))
        size_arrays = train if source == "train" else train + val + test
        max_n = max(int(a["X"].shape[0]) for a in train + val + test)
        counts = np.zeros(max_n + 1, dtype=np.float64)
        for item in size_arrays:
            counts[int(item["X"].shape[0])] += 1
        if counts.sum() == 0:
            counts[max_n] = 1
        n_node_distribution = (counts / counts.sum()).tolist()

        max_node_label = max(int(item["X"].max()) if item["X"].size else 0 for item in train + val + test)
        num_atom_types = max(1, max_node_label + 1)
        node_counts = np.zeros(num_atom_types, dtype=np.float64)
        max_edge_type = max(int(item["E"].max()) if item["E"].size else 0 for item in train + val + test)
        num_edge_types = max(2, max_edge_type + 1)
        edge_counts = np.zeros(num_edge_types, dtype=np.float64)
        for item in train:
            X = item["X"]
            E = item["E"]
            n = int(X.shape[0])
            for value in X:
                node_counts[int(value)] += 1.0
            for u in range(n):
                for v in range(n):
                    if u == v:
                        continue
                    edge_counts[int(E[u, v])] += 1.0
        if node_counts.sum() == 0:
            node_counts[0] = 1.0
        if edge_counts.sum() == 0:
            edge_counts[0] = 1.0
        atom_types = (node_counts / node_counts.sum()).tolist()
        edge_types = (edge_counts / edge_counts.sum()).tolist()

        return _ConStructDatasetMeta(
            dataset=self.dataset,
            num_train=len(train),
            num_val=len(val),
            num_test=len(test),
            max_n_nodes=max_n,
            n_node_distribution=n_node_distribution,
            atom_types=atom_types,
            edge_types=edge_types,
            num_atom_types=num_atom_types,
            num_edge_types=num_edge_types,
        )

    def _write_provenance(self, train: list[dict[str, np.ndarray]], val: list[dict[str, np.ndarray]], test: list[dict[str, np.ndarray]], meta: _ConStructDatasetMeta) -> None:
        self.data_root.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "train": [{"X": torch.from_numpy(a["X"]), "E": torch.from_numpy(a["E"])} for a in train],
                "val": [{"X": torch.from_numpy(a["X"]), "E": torch.from_numpy(a["E"])} for a in val],
                "test": [{"X": torch.from_numpy(a["X"]), "E": torch.from_numpy(a["E"])} for a in test],
            },
            self.data_root / "benchmark_splits.pt",
        )
        with open(self.data_root / "metadata.json", "w", encoding="utf-8") as f:
            json.dump({"meta": asdict(meta), "cfg": self.cfg_dict}, f, indent=2)

    def _adj_arrays_to_batches(self, arrays: list[dict[str, np.ndarray]], batch_size: int, shuffle: bool) -> Iterator[_PlaceHolder]:
        indices = list(range(len(arrays)))
        if shuffle:
            random.shuffle(indices)
        for start in range(0, len(indices), int(batch_size)):
            selected = [arrays[i] for i in indices[start : start + int(batch_size)]]
            yield self._adj_arrays_to_dense_batch(selected)

    def _adj_arrays_to_dense_batch(self, arrays: list[dict[str, np.ndarray]]) -> _PlaceHolder:
        if self.dataset_infos is None:
            raise RuntimeError("dataset_infos is not initialized.")
        bs = len(arrays)
        n_max = max(int(a["X"].shape[0]) for a in arrays)
        X_int = torch.zeros((bs, n_max), dtype=torch.long)
        E_int = torch.zeros((bs, n_max, n_max), dtype=torch.long)
        node_mask = torch.zeros((bs, n_max), dtype=torch.bool)
        for i, item in enumerate(arrays):
            X_arr = item["X"].astype(np.int64)
            E_arr = item["E"].astype(np.int64)
            n = int(X_arr.shape[0])
            node_mask[i, :n] = True
            X_int[i, :n] = torch.from_numpy(X_arr)
            E_int[i, :n, :n] = torch.from_numpy(E_arr)
            E_int[i].fill_diagonal_(0)
        pl = _PlaceHolder(X=X_int, E=E_int, charges=None, y=torch.zeros((bs, 0)), node_mask=node_mask)
        return self.dataset_infos.to_one_hot(pl).device_as(torch.empty(1, device=self.device))

    def _placeholder_batch_to_networkx(self, batch: _PlaceHolder) -> list[nx.Graph]:
        graphs: list[nx.Graph] = []
        X = batch.X.detach().cpu()
        E = batch.E.detach().cpu()
        node_mask = batch.node_mask.detach().cpu() if batch.node_mask is not None else (X >= 0)
        for i in range(X.shape[0]):
            n = int(node_mask[i].sum().item())
            if n <= 0:
                continue
            g = nx.Graph()
            for u in range(n):
                node_label = max(int(X[i, u].item()), 0)
                g.add_node(u, node_label=node_label, feats=np.array([float(node_label)], dtype=np.float32))
            for u in range(n):
                for v in range(u + 1, n):
                    edge_type = int(E[i, u, v].item())
                    if edge_type > 0:
                        g.add_edge(u, v, edge_type=edge_type)
            graphs.append(g)
        return graphs

    # ------------------------------------------------------------------
    # Model build/load/save
    # ------------------------------------------------------------------
    def _build_model_from_meta(self, meta: _ConStructDatasetMeta) -> None:
        self._import_modules()
        self.meta = meta
        self.dataset_infos = _ConStructDatasetInfos(meta, _PlaceHolder)
        self.model = _DenseConStructModel(self.cfg, self.dataset_infos, self.mods).to(self.device)

    def _save_checkpoint(self) -> None:
        if self.model is None or self.meta is None:
            raise RuntimeError("Cannot save ConStruct checkpoint before model/meta are initialized.")
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "meta": asdict(self.meta),
                "cfg": self.cfg_dict,
                "wrapper": "ConStructWrapper",
            },
            self.checkpoint_path,
        )

    def load(self) -> None:
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"ConStruct checkpoint not found: {self.checkpoint_path}. Train the model first or set checkpoint_path."
            )
        ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        self.cfg_dict = ckpt.get("cfg", self.cfg_dict)
        self.cfg = _to_config_node(self.cfg_dict)
        meta = _ConStructDatasetMeta(**ckpt["meta"])
        self._build_model_from_meta(meta)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

    def train(self, train_graphs, val_graphs=None, test_graphs=None) -> None:
        started_at = time.perf_counter()
        self._log(
            "train_start dataset=%s train=%d val=%s test=%s epochs=%s batch_size=%s extra_features=%s eigenfeatures=%s cycle_features=%s",
            self.dataset,
            len(train_graphs or []),
            None if val_graphs is None else len(val_graphs),
            None if test_graphs is None else len(test_graphs),
            self.cfg.train.n_epochs,
            self.cfg.train.batch_size,
            self.cfg.model.extra_features,
            self.cfg.model.eigenfeatures,
            self.cfg.model.cycle_features,
        )
        self._import_modules()
        self._log("modules_imported duration=%.2fs", time.perf_counter() - started_at)
        random.seed(int(self.cfg.train.seed))
        np.random.seed(int(self.cfg.train.seed))
        torch.manual_seed(int(self.cfg.train.seed))

        train_arrays, val_arrays, test_arrays = self._split_arrays_for_meta(train_graphs, val_graphs, test_graphs)
        self._log(
            "splits_converted train=%d val=%d test=%d duration=%.2fs",
            len(train_arrays),
            len(val_arrays),
            len(test_arrays),
            time.perf_counter() - started_at,
        )
        meta = self._build_meta(train_arrays, val_arrays, test_arrays)
        self._log(
            "meta_built max_n_nodes=%d atom_types=%d edge_types=%d duration=%.2fs",
            meta.max_n_nodes,
            meta.num_atom_types,
            meta.num_edge_types,
            time.perf_counter() - started_at,
        )
        self._write_provenance(train_arrays, val_arrays, test_arrays, meta)
        self._log("provenance_written data_root=%s duration=%.2fs", self.data_root, time.perf_counter() - started_at)
        self._build_model_from_meta(meta)
        self._log("model_built parameters=%d duration=%.2fs", self._count_parameters(self.model), time.perf_counter() - started_at)

        if self.model is None:
            raise RuntimeError("ConStruct model was not initialized.")
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(self.cfg.train.lr),
            weight_decay=float(self.cfg.train.weight_decay),
            amsgrad=True,
        )
        clip_grad = self.cfg.train.clip_grad
        n_epochs = int(self.cfg.train.n_epochs)
        batch_size = int(self.cfg.train.batch_size)
        self.model.train()
        last_train_loss = None
        last_val_loss = None
        successful_updates = 0
        for epoch in range(n_epochs):
            epoch_started_at = time.perf_counter()
            self._log("epoch_start epoch=%d/%d", epoch + 1, n_epochs)
            epoch_losses: list[float] = []
            skipped_batches = 0
            for batch_idx, batch in enumerate(self._adj_arrays_to_batches(train_arrays, batch_size=batch_size, shuffle=True)):
                batch_started_at = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                try:
                    loss, _ = self.model.loss_on_clean_batch(batch, log=False)
                except AssertionError as exc:
                    skipped_batches += 1
                    self._log(
                        "train_batch_skipped epoch=%d/%d batch=%d reason=forward_assertion error=%s duration=%.2fs",
                        epoch + 1,
                        n_epochs,
                        batch_idx + 1,
                        str(exc),
                        time.perf_counter() - batch_started_at,
                    )
                    continue
                if not torch.isfinite(loss):
                    skipped_batches += 1
                    self._log(
                        "train_batch_skipped epoch=%d/%d batch=%d reason=nonfinite_loss duration=%.2fs",
                        epoch + 1,
                        n_epochs,
                        batch_idx + 1,
                        time.perf_counter() - batch_started_at,
                    )
                    continue
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    float(clip_grad),
                    error_if_nonfinite=False,
                )
                if not torch.isfinite(grad_norm):
                    optimizer.zero_grad(set_to_none=True)
                    skipped_batches += 1
                    self._log(
                        "train_batch_skipped epoch=%d/%d batch=%d reason=nonfinite_grad grad_norm=%s duration=%.2fs",
                        epoch + 1,
                        n_epochs,
                        batch_idx + 1,
                        grad_norm.detach().cpu().item() if grad_norm.numel() else grad_norm,
                        time.perf_counter() - batch_started_at,
                    )
                    continue
                optimizer.step()
                successful_updates += 1
                epoch_losses.append(float(loss.detach().cpu()))
                if batch_idx % self.log_train_every_n_steps == 0:
                    self._log(
                        "train_batch epoch=%d/%d batch=%d loss=%.6f X_shape=%s E_shape=%s duration=%.2fs",
                        epoch + 1,
                        n_epochs,
                        batch_idx + 1,
                        float(loss.detach().cpu()),
                        tuple(batch.X.shape),
                        tuple(batch.E.shape),
                        time.perf_counter() - batch_started_at,
                    )
            last_train_loss = float(np.mean(epoch_losses)) if epoch_losses else None

            if val_arrays and ((epoch + 1) % int(self.config.get("val_every", 10)) == 0 or epoch == n_epochs - 1):
                self.model.eval()
                val_losses = []
                with torch.no_grad():
                    for batch in self._adj_arrays_to_batches(val_arrays, batch_size=batch_size, shuffle=False):
                        val_loss, _ = self.model.loss_on_clean_batch(batch, log=False)
                        val_losses.append(float(val_loss.detach().cpu()))
                last_val_loss = float(np.mean(val_losses)) if val_losses else None
                self.model.train()
            if (epoch + 1) % self.log_epoch_every == 0 or epoch == 0 or epoch + 1 == n_epochs:
                self._log(
                    "epoch_end epoch=%d/%d train_loss=%s val_loss=%s duration=%.2fs",
                    epoch + 1,
                    n_epochs,
                    last_train_loss,
                    last_val_loss,
                    time.perf_counter() - epoch_started_at,
                )
            if skipped_batches:
                self._log("epoch_skipped_batches epoch=%d/%d skipped=%d", epoch + 1, n_epochs, skipped_batches)

        if successful_updates == 0:
            raise FloatingPointError("ConStruct training completed without any finite optimizer updates; refusing to save an unusable checkpoint.")
        self.model.eval()
        self._save_checkpoint()
        self._log("checkpoint_saved path=%s total_duration=%.2fs", self.checkpoint_path, time.perf_counter() - started_at)
        self.run_root.mkdir(parents=True, exist_ok=True)
        with open(self.run_root / "train_summary.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "dataset": self.dataset,
                    "num_epochs": n_epochs,
                    "batch_size": batch_size,
                    "last_train_loss": last_train_loss,
                    "successful_updates": successful_updates,
                    "last_val_loss": last_val_loss,
                    "checkpoint_path": str(self.checkpoint_path),
                    "meta": asdict(meta),
                },
                f,
                indent=2,
            )

    def _log(self, message: str, *args: Any) -> None:
        if getattr(self, "detailed_logging", False):
            LOGGER.info("ConStructWrapper " + message, *args)

    def _count_parameters(self, model: torch.nn.Module | None) -> int:
        if model is None:
            return 0
        return int(sum(p.numel() for p in model.parameters()))

    def sample(self, num_graphs: int, seed: int = 0, progress_callback=None):
        if self.model is None:
            self.load()
        if self.model is None:
            raise RuntimeError("ConStruct model is not loaded.")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        self.model.eval().to(self.device)

        rev_proj = self.cfg.model.rev_proj
        projector_classes = None
        if rev_proj and str(rev_proj).lower() not in {"false", "none", "0"}:
            projector_classes = self._import_projectors()

        with torch.no_grad():
            batches = self.model.sample_n_graphs(
                samples_to_generate=int(num_graphs),
                sample_batch_size=int(self.cfg.sampling.sample_batch_size),
                faster_sampling=int(self.cfg.sampling.faster_sampling),
                rev_proj=rev_proj,
                projector_classes=projector_classes,
                progress_callback=progress_callback,
            )
        graphs: list[nx.Graph] = []
        for batch in batches:
            graphs.extend(self._placeholder_batch_to_networkx(batch))
            if len(graphs) >= num_graphs:
                break
        return graphs[:num_graphs]
