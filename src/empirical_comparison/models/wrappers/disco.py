from __future__ import annotations

import importlib
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.utils import dense_to_sparse, from_networkx

try:
    from empirical_comparison.models.base import BaseGenerator
except Exception:  # pragma: no cover
    class BaseGenerator:
        def __init__(self, config: dict[str, Any]) -> None:
            self.config = config


class DisCoWrapper(BaseGenerator):
    """Adapter around the DisCo implementation for generic synthetic datasets.

    This wrapper targets the SPECTRE-style synthetic benchmark setting used by
    the original DisCo codebase, in particular SBM / Planar / Community.

    Expected config keys
    --------------------
    repo_root: str
        Path to the extracted DisCo repository root.
    checkpoint_path: str
        Where to save / load the wrapper checkpoint.
    dataset_name: str
        One of {"sbm", "planar", "community"}. Defaults to "sbm".
    data_subdir: str, optional
        Relative or absolute path to the dataset root used by DisCo's loader.
        Defaults to ``data/<dataset_name>_empirical`` under the repo root.
    batch_size: int, optional
    num_epochs: int, optional
    learning_rate: float, optional
    weight_decay: float, optional
    backbone: str, optional
        One of {"GT", "MPNN"}. Defaults to "GT".
    n_layers: int, optional
    n_dim: int, optional
    dropout: float, optional
    diff_type: str, optional
        One of {"uniform", "marginal"}. Defaults to "marginal".
    beta: float, optional
    alpha: float, optional
    min_time: float, optional
    sampling_steps: int, optional
    num_workers: int, optional
    device: str, optional

    Notes
    -----
    - This wrapper is intended for generic graph benchmarks only.
    - It assumes undirected simple graphs.
    - As in the upstream implementation, node features are reduced to a single
      constant feature channel for SPECTRE-style datasets.
    - We store our own checkpoint containing the model state, config summary,
      and dataset metadata, since the upstream training script does not save one
      by default.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self.repo_root = Path(config["repo_root"]).expanduser().resolve()
        self.device = torch.device(config.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.checkpoint_path = Path(config["checkpoint_path"]).expanduser().resolve()
        self.dataset_name = str(config.get("dataset_name", "sbm")).lower()
        if self.dataset_name not in {"sbm", "planar", "community"}:
            raise ValueError(f"Unsupported dataset_name for DisCoWrapper: {self.dataset_name}")

        default_subdir = Path("data") / f"{self.dataset_name}_empirical"
        self.data_root = Path(config.get("data_subdir", default_subdir))
        if not self.data_root.is_absolute():
            self.data_root = (self.repo_root / self.data_root).resolve()

        self.repo_loaded = False
        self.mods: dict[str, Any] = {}

        self.model: torch.nn.Module | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.diffuser: Any = None
        self.sampler: Any = None
        self.add_auxiliary_feature: Any = None
        self.dataset_info: Any = None
        self.n_node_distribution = None
        self.n_node_type: int | None = None
        self.n_edge_type: int | None = None
        self.max_n_nodes: int | None = None
        self.train_graphs: list[nx.Graph] = []

    @property
    def name(self) -> str:
        return "disco"

    # ------------------------------------------------------------------
    # Repo imports
    # ------------------------------------------------------------------
    def _ensure_repo_importable(self) -> None:
        root_str = str(self.repo_root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)

    def _import_modules(self) -> None:
        if self.repo_loaded:
            return
        self._ensure_repo_importable()
        self.mods["load_spectre"] = importlib.import_module("loader.load_spectre_data")
        self.mods["dataset_info"] = importlib.import_module("dataset_info")
        self.mods["forward_diff"] = importlib.import_module("forward_diff")
        self.mods["digress_models"] = importlib.import_module("digress_models")
        self.mods["models"] = importlib.import_module("models")
        self.mods["losses"] = importlib.import_module("losses")
        self.mods["sampling"] = importlib.import_module("sampling")
        self.mods["aux"] = importlib.import_module("auxiliary_features")
        self.mods["utils"] = importlib.import_module("utils")
        self.repo_loaded = True

    # ------------------------------------------------------------------
    # Data materialization
    # ------------------------------------------------------------------
    def _graphs_to_adj_tensors(self, graphs: list[nx.Graph]) -> list[torch.Tensor]:
        adjs: list[torch.Tensor] = []
        for g in graphs:
            if g.number_of_nodes() == 0:
                raise ValueError("DisCoWrapper does not support empty graphs.")
            if g.is_directed():
                raise ValueError("DisCoWrapper expects undirected graphs.")
            g2 = nx.convert_node_labels_to_integers(g)
            adj = nx.to_numpy_array(g2, dtype=np.float32)
            adj = (adj > 0).astype(np.float32)
            np.fill_diagonal(adj, 0.0)
            adj = np.maximum(adj, adj.T)
            adjs.append(torch.from_numpy(adj))
        return adjs

    def _raw_filename(self) -> str:
        return {
            "sbm": "sbm_200.pt",
            "planar": "planar_64_200.pt",
            "community": "community_12_21_100.pt",
        }[self.dataset_name]

    def _write_raw_dataset(self, graphs: list[nx.Graph]) -> None:
        self.data_root.mkdir(parents=True, exist_ok=True)
        raw_dir = self.data_root / self.dataset_name / "raw"
        processed_dir = self.data_root / self.dataset_name / "processed"
        raw_dir.mkdir(parents=True, exist_ok=True)
        if processed_dir.exists():
            shutil.rmtree(processed_dir)

        adjs = self._graphs_to_adj_tensors(graphs)
        # Upstream loader expects a tuple and only uses the first element.
        payload = (adjs, None, None, None, None, None, None, None)
        torch.save(payload, raw_dir / self._raw_filename())

        meta = {
            "dataset_name": self.dataset_name,
            "num_graphs": len(adjs),
            "source": "empirical_comparison",
        }
        with open(self.data_root / "empirical_comparison_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    # ------------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------------
    def _build_components(self) -> None:
        self._import_modules()
        SpectreDataset = self.mods["load_spectre"].SpectreDataset
        get_dataset_info = self.mods["dataset_info"].get_dataset_info
        ForwardDiffusion = self.mods["forward_diff"].ForwardDiffusion
        GraphTransformer = self.mods["digress_models"].GraphTransformer
        MPNN = self.mods["models"].MPNN
        TauLeaping = self.mods["sampling"].TauLeaping
        AuxFeatures = self.mods["aux"].AuxFeatures
        to_dense = self.mods["utils"].to_dense

        root_arg = str(self.data_root)
        train_set = SpectreDataset(root=root_arg, name=self.dataset_name, split="train")
        val_set = SpectreDataset(root=root_arg, name=self.dataset_name, split="val")
        test_set = SpectreDataset(root=root_arg, name=self.dataset_name, split="test")

        self.train_loader = DataLoader(
            train_set,
            batch_size=int(self.config.get("batch_size", 4)),
            shuffle=True,
            num_workers=int(self.config.get("num_workers", 0)),
        )
        self.valid_loader = DataLoader(
            val_set,
            batch_size=int(self.config.get("batch_size", 4)),
            shuffle=False,
            num_workers=int(self.config.get("num_workers", 0)),
        )
        self.test_loader = DataLoader(
            test_set,
            batch_size=int(self.config.get("batch_size", 4)),
            shuffle=False,
            num_workers=int(self.config.get("num_workers", 0)),
        )

        self.n_node_type = int(train_set[0].x.shape[-1])
        self.n_edge_type = int(train_set[0].edge_attr.shape[-1])

        dataset_info = get_dataset_info(self.dataset_name)
        dataset_info.n_edge_type = self.n_edge_type
        dataset_info.n_node_type = self.n_node_type
        self.dataset_info = dataset_info
        self.max_n_nodes = int(dataset_info.max_n_nodes)

        E_marginal = torch.tensor(dataset_info.E_marginal, dtype=torch.float32, device=self.device)
        X_marginal = torch.tensor(dataset_info.X_marginal, dtype=torch.float32, device=self.device)
        self.n_node_distribution = torch.distributions.categorical.Categorical(
            torch.tensor(dataset_info.n_node_distribution)
        )

        cycle_fea = bool(self.config.get("cycle_fea", 1))
        eigen_fea = bool(self.config.get("eigen_fea", 1))
        rwpe_fea = bool(self.config.get("rwpe_fea", 0))
        global_fea = bool(self.config.get("global_fea", 1))
        aux_feas = [cycle_fea, eigen_fea, rwpe_fea, global_fea]
        self.add_auxiliary_feature = AuxFeatures(aux_feas, self.max_n_nodes)

        diffuser = ForwardDiffusion(
            self.n_node_type,
            self.n_edge_type,
            forward_type=str(self.config.get("diff_type", "marginal")),
            node_marginal=X_marginal,
            edge_marginal=E_marginal,
            device=str(self.device),
            time_exponential=float(self.config.get("beta", 2.0)),
            time_base=float(self.config.get("alpha", 0.8)),
        )
        self.diffuser = diffuser

        example_data = train_set[0]
        X_t, E_t, y_t = self.add_auxiliary_feature(*to_dense(example_data))
        X_dim = int(X_t.shape[-1])
        E_dim = int(E_t.shape[-1])
        y_dim = int(y_t.shape[-1] + 1)

        n_layers = int(self.config.get("n_layers", 5))
        n_dim = int(self.config.get("n_dim", 128))
        backbone = str(self.config.get("backbone", "GT"))
        input_dims = {"X": X_dim, "E": E_dim, "y": y_dim}
        output_dims = {"X": self.n_node_type, "E": self.n_edge_type, "y": 0}

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
            model = GraphTransformer(
                n_layers=n_layers,
                input_dims=input_dims,
                hidden_mlp_dims=hidden_mlp_dims,
                hidden_dims=hidden_dims,
                output_dims=output_dims,
            ).to(self.device)
        elif backbone == "MPNN":
            model = MPNN(
                n_layers=n_layers,
                input_dims=input_dims,
                hidden_dims=n_dim,
                output_dims=output_dims,
            ).to(self.device)
        else:
            raise ValueError(f"Unsupported DisCo backbone: {backbone}")
        self.model = model

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(self.config.get("learning_rate", 2e-4)),
            amsgrad=True,
            weight_decay=float(self.config.get("weight_decay", 5e-12)),
        )

        self.sampler = TauLeaping(
            self.n_node_type,
            self.n_edge_type,
            num_steps=int(self.config.get("sampling_steps", 50)),
            min_t=float(self.config.get("min_time", 0.01)),
            add_auxiliary_feature=self.add_auxiliary_feature,
            device=str(self.device),
            BAR=bool(self.config.get("BAR", False)),
        )

    # ------------------------------------------------------------------
    # Training / loading / sampling
    # ------------------------------------------------------------------
    def _checkpoint_payload(self, epoch: int | None = None) -> dict[str, Any]:
        return {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict() if self.model is not None else None,
            "optimizer_state_dict": self.optimizer.state_dict() if self.optimizer is not None else None,
            "config": self.config,
            "dataset_name": self.dataset_name,
            "n_node_type": self.n_node_type,
            "n_edge_type": self.n_edge_type,
        }

    def load(self) -> None:
        self._build_components()
        ckpt = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        if self.optimizer is not None and ckpt.get("optimizer_state_dict") is not None:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.model.eval()

    def train(self, train_graphs, val_graphs=None) -> None:
        # val_graphs unused because upstream loader derives split from raw file.
        self.train_graphs = [g.copy() for g in train_graphs]
        self._write_raw_dataset(train_graphs)
        self._build_components()
        self.model.train()

        CELoss = self.mods["losses"].CELoss
        to_dense = self.mods["utils"].to_dense
        add_mask_idx = self.mods["utils"].add_mask_idx

        num_epochs = int(self.config.get("num_epochs", 100))
        min_time = float(self.config.get("min_time", 0.01))
        edge_weight = float(self.config.get("edge_loss_weight", 5.0))
        include_node_feature = getattr(self.diffuser, "diffuse_node", False)

        best_loss = float("inf")
        for epoch in range(num_epochs):
            losses: list[float] = []
            for data in self.train_loader:
                data = data.to(self.device)
                X_0, E_0, node_mask = to_dense(data)
                ts = torch.rand((E_0.shape[0],), device=self.device) * (1.0 - min_time) + min_time

                X_t_idx, E_t_idx = self.diffuser.forward_diffusion(X_0, E_0, ts)
                X_t_one_hot = X_t_idx
                E_t_one_hot = F.one_hot(E_t_idx, num_classes=self.n_edge_type).float()
                X_t, E_t, y_t = self.add_auxiliary_feature(X_t_one_hot, E_t_one_hot, node_mask)
                y_t = torch.cat([y_t, ts.unsqueeze(-1)], dim=-1)

                pred_X_0, pred_E_0 = self.model(X_t, E_t, y_t, node_mask)

                X_0_idx = torch.max(X_0, dim=-1)[1].long()
                E_0_idx = torch.max(E_0, dim=-1)[1].long()
                X_0_idx_masked, E_0_idx_masked = add_mask_idx(
                    X_0_idx, E_0_idx, self.n_node_type, self.n_edge_type, node_mask
                )

                loss_E = CELoss(pred_E_0, E_0_idx_masked)
                loss_X = CELoss(pred_X_0, X_0_idx_masked) if include_node_feature else 0.0
                loss = loss_X + edge_weight * loss_E

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                losses.append(float(loss.item()))

            mean_loss = float(np.mean(losses)) if losses else float("nan")
            print(f"[DisCo] epoch {epoch + 1}/{num_epochs} loss={mean_loss:.4f}")
            if mean_loss < best_loss:
                best_loss = mean_loss
                self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(self._checkpoint_payload(epoch=epoch + 1), self.checkpoint_path)

        self.model.eval()

    @torch.no_grad()
    def sample(self, num_graphs: int, seed: int = 0):
        if self.model is None:
            self.load()
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        n_node = self.n_node_distribution.sample((num_graphs,)).to(self.device)
        X, E, node_mask = self.sampler.sample(self.diffuser, self.model, n_node)

        graphs: list[nx.Graph] = []
        for i in range(num_graphs):
            n = int(n_node[i].item())
            if n <= 0:
                graphs.append(nx.Graph())
                continue
            edge_mat = E[i, :n, :n].detach().cpu().numpy()
            # edge type 1 corresponds to present edge in synthetic datasets
            adj = (edge_mat == 1).astype(np.int64)
            np.fill_diagonal(adj, 0)
            adj = np.maximum(adj, adj.T)
            g = nx.from_numpy_array(adj)
            for v in g.nodes():
                g.nodes[v]["feats"] = np.ones(1, dtype=np.float32)
            graphs.append(g)
        return graphs
