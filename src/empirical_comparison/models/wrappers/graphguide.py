
from __future__ import annotations

import importlib
import os
import random
import sys
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.utils import from_networkx, sort_edge_index

from empirical_comparison.models.base import BaseGenerator


class GraphGUIDEWrapper(BaseGenerator):
    """Adapter around the GraphGUIDE implementation.

    This wrapper assumes the upstream repository is available locally and uses
    its original model, diffuser, graph conversion, and sampling code.

    Expected config keys
    --------------------
    repo_root: str
        Path to the extracted GraphGUIDE repository root.
    checkpoint_path: str
        Path to save/load the GraphGUIDE model checkpoint.
    batch_size: int
    num_epochs: int
    learning_rate: float
    t_limit: int
    model_type: str
        One of {"gat", "gin"}. Defaults to "gat".
    model_kwargs: dict
        Extra kwargs forwarded to the model constructor.
    diffuser_type: str
        One of {"bernoulli", "bernoulli_one", "bernoulli_zero",
        "bernoulli_skip", "bernoulli_one_skip", "bernoulli_zero_skip"}.
        Defaults to "bernoulli_zero".
    diffuser_kwargs: dict
        Extra kwargs for the diffuser. For Bernoulli-based diffusers this
        should usually include ``a`` and ``b``.
    device: str
        Optional explicit device. Defaults to cuda if available.

    Notes
    -----
    GraphGUIDE models *edge generation conditioned on node features and graph
    size*. During sampling, this wrapper reuses node-feature templates from the
    training set and regenerates the edge set from the diffusion prior.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self.device = config.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
        default_repo_root = Path(__file__).resolve().parents[4] / "external" / "GraphGUIDE"
        repo_root = os.environ.get("GRAPHGUIDE_REPO") or config.get("repo_root") or default_repo_root
        if not repo_root:
            raise ValueError("GraphGUIDEWrapper requires `repo_root` or the GRAPHGUIDE_REPO environment variable.")
        self.repo_root = self._normalize_repo_root(Path(repo_root).expanduser().resolve())
        self.repo_src = self.repo_root / "src"
        self.checkpoint_path = Path(config["checkpoint_path"]).expanduser().resolve()

        self.gg_loaded = False
        self.gg_model_util = None
        self.gg_graph_conversions = None
        self.gg_generate = None
        self.gg_gnn = None
        self.gg_diffusers = None

        self.model: torch.nn.Module | None = None
        self.diffuser: Any = None
        self.template_graphs: list[nx.Graph] = []
        self.input_dim: int | None = None

    @property
    def name(self) -> str:
        return "graphguide"

    def _normalize_repo_root(self, repo_root: Path) -> Path:
        if repo_root.name == "src" and (repo_root / "model").exists():
            repo_root = repo_root.parent
        return repo_root

    def _ensure_repo_importable(self) -> None:
        if str(self.repo_src) not in sys.path:
            sys.path.insert(0, str(self.repo_src))

    def _import_graphguide_modules(self) -> None:
        if self.gg_loaded:
            return
        self._ensure_repo_importable()
        if not self.repo_src.exists():
            raise FileNotFoundError(f"GraphGUIDE src directory not found under repo_root={self.repo_root}")
        self.gg_model_util = importlib.import_module("model.util")
        self.gg_graph_conversions = importlib.import_module("feature.graph_conversions")
        self.gg_generate = importlib.import_module("model.generate")
        self.gg_gnn = importlib.import_module("model.gnn")
        self.gg_diffusers = importlib.import_module("model.discrete_diffusers")
        self.gg_loaded = True

    def _to_pyg_data(self, g: nx.Graph):
        g2 = g.copy()
        if g2.number_of_nodes() == 0:
            raise ValueError("GraphGUIDEWrapper does not support empty graphs.")

        feat_dict = nx.get_node_attributes(g2, "feats")
        if not feat_dict:
            # Fall back to constant one-dimensional features.
            nx.set_node_attributes(g2, {i: np.ones(1, dtype=np.float32) for i in g2.nodes()}, "feats")
        else:
            # Normalize feature format to float32 numpy arrays.
            clean = {}
            for i in range(g2.number_of_nodes()):
                if i not in feat_dict:
                    raise ValueError("Node features under attribute 'feats' must exist for all nodes.")
                arr = np.asarray(feat_dict[i], dtype=np.float32)
                if arr.ndim == 0:
                    arr = arr[None]
                clean[i] = arr
            nx.set_node_attributes(g2, clean, "feats")

        data = from_networkx(g2, group_node_attrs=["feats"])
        data.edge_index = sort_edge_index(data.edge_index)
        return data.to(self.device)

    def _make_loader(self, graphs, shuffle: bool) -> DataLoader:
        pyg_graphs = [self._to_pyg_data(g) for g in graphs]
        return DataLoader(
            pyg_graphs,
            batch_size=int(self.config.get("batch_size", 32)),
            shuffle=shuffle,
            num_workers=int(self.config.get("num_workers", 0)),
        )

    def _infer_input_dim(self, graphs) -> int:
        for g in graphs:
            feats = nx.get_node_attributes(g, "feats")
            if feats:
                first = np.asarray(feats[next(iter(feats))])
                return int(first.shape[0] if first.ndim > 0 else 1)
        return 1

    def _build_model(self) -> torch.nn.Module:
        self._import_graphguide_modules()
        model_type = str(self.config.get("model_type", "gat")).lower()
        model_kwargs = dict(self.config.get("model_kwargs", {}))
        t_limit = int(self.config.get("t_limit", 100))

        if self.input_dim is None:
            raise RuntimeError("input_dim is unknown; call train() first or set it before load().")

        if model_type == "gat":
            cls = self.gg_gnn.GraphLinkGAT
        elif model_type == "gin":
            cls = self.gg_gnn.GraphLinkGIN
        else:
            raise ValueError(f"Unsupported GraphGUIDE model_type: {model_type}")

        return cls(input_dim=self.input_dim, t_limit=t_limit, **model_kwargs).to(self.device)

    def _build_diffuser(self):
        self._import_graphguide_modules()
        diffuser_type = str(self.config.get("diffuser_type", "bernoulli_zero")).lower()
        diffuser_kwargs = dict(self.config.get("diffuser_kwargs", {"a": 100, "b": 10}))
        input_shape = (1,)

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
        return cls_map[diffuser_type](input_shape=input_shape, **diffuser_kwargs)

    def load(self) -> None:
        """Load a trained GraphGUIDE model from checkpoint_path."""
        self._import_graphguide_modules()

        if self.input_dim is None:
            input_dim = self.config.get("input_dim")
            if input_dim is None:
                raise ValueError(
                    "GraphGUIDEWrapper.load() needs config['input_dim'] unless train() was run before load()."
                )
            self.input_dim = int(input_dim)

        model_type = str(self.config.get("model_type", "gat")).lower()
        cls = self.gg_gnn.GraphLinkGAT if model_type == "gat" else self.gg_gnn.GraphLinkGIN
        self.model = self.gg_model_util.load_model(cls, str(self.checkpoint_path)).to(self.device)
        self.model.eval()
        self.diffuser = self._build_diffuser()

    def train(self, train_graphs, val_graphs=None) -> None:
        """Train GraphGUIDE on a list of NetworkX graphs.

        train_graphs and val_graphs should be lists of undirected NetworkX
        graphs with node features stored under the node attribute ``feats``.
        """
        self._import_graphguide_modules()
        self.input_dim = self._infer_input_dim(train_graphs)
        self.model = self._build_model()
        self.diffuser = self._build_diffuser()

        loader = self._make_loader(train_graphs, shuffle=True)
        self.template_graphs = [g.copy() for g in train_graphs]

        num_epochs = int(self.config.get("num_epochs", 30))
        learning_rate = float(self.config.get("learning_rate", 1e-3))
        t_limit = int(self.config.get("t_limit", 100))
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.model.train()
        torch.set_grad_enabled(True)

        for epoch in range(num_epochs):
            losses: list[float] = []
            for data in loader:
                data = data.to(self.device)
                e0, edge_batch_inds = self.gg_graph_conversions.pyg_data_to_edge_vector(
                    data, return_batch_inds=True
                )

                graph_sizes = torch.diff(data.ptr)
                graph_times = torch.randint(
                    t_limit,
                    size=(graph_sizes.shape[0],),
                    device=self.device,
                ) + 1
                t_v = graph_times[data.batch].float()
                t_e = graph_times[edge_batch_inds].float()

                et, true_post = self.diffuser.forward(e0[:, None], t_e)
                et = et[:, 0]
                true_post = true_post[:, 0]
                data.edge_index = self.gg_graph_conversions.edge_vector_to_pyg_data(data, et)

                pred_post = self.model(data, t_v)
                loss = self.model.loss(pred_post, true_post)
                if torch.isnan(loss):
                    continue
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                losses.append(float(loss.item()))

            mean_loss = float(np.mean(losses)) if losses else float("nan")
            print(f"[GraphGUIDE] epoch {epoch + 1}/{num_epochs} loss={mean_loss:.4f}")

        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        self.gg_model_util.save_model(self.model, str(self.checkpoint_path))
        self.model.eval()

    def _prepare_initial_sample(self, template_graphs: list[nx.Graph]):
        """Create initial PyG batch at time t_limit from graph templates.

        GraphGUIDE generates edge sets conditioned on node features and graph
        size. We therefore preserve node features from templates and sample the
        initial edge vector from the diffuser prior.
        """
        pyg_graphs = [self._to_pyg_data(g) for g in template_graphs]
        batch = next(iter(DataLoader(pyg_graphs, batch_size=len(pyg_graphs), shuffle=False)))
        batch = batch.to(self.device)

        num_edges = int(self.gg_graph_conversions.pyg_data_to_edge_vector(batch).shape[0])
        t_limit = int(self.config.get("t_limit", 100))
        t_e = torch.full((num_edges,), t_limit, device=self.device)
        prior_edges = self.diffuser.sample_prior(num_edges, t_e)[:, 0] if self.diffuser.sample_prior(num_edges, t_e).ndim == 2 else self.diffuser.sample_prior(num_edges, t_e)
        # sample_prior may return float or binary; edge_vector_to_pyg_data only needs 0/1-like values.
        batch.edge_index = self.gg_graph_conversions.edge_vector_to_pyg_data(batch, prior_edges)
        return batch

    def sample(self, num_graphs: int, seed: int = 0):
        if self.model is None or self.diffuser is None:
            raise RuntimeError("Call load() or train() before sample().")
        if not self.template_graphs:
            raise RuntimeError(
                "GraphGUIDEWrapper.sample() requires template_graphs from training. "
                "Store a few training graphs or set them manually before sampling."
            )

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        chosen = [self.template_graphs[i % len(self.template_graphs)].copy() for i in range(num_graphs)]
        initial_batch = self._prepare_initial_sample(chosen)
        t_limit = int(self.config.get("t_limit", 100))

        samples = self.gg_generate.generate_graph_samples(
            self.model,
            self.diffuser,
            initial_samples=initial_batch,
            t_start=0,
            t_limit=t_limit,
            return_all_times=False,
            verbose=False,
        )
        return self.gg_graph_conversions.split_pyg_data_to_nx_graphs(samples)
