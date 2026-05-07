from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar


class BaseGenerator(ABC):
    """Minimal model-wrapper interface used by the benchmark pipeline."""

    # Capability flags make assumptions explicit in benchmark reports.
    supports_training: ClassVar[bool] = True
    supports_sampling: ClassVar[bool] = True
    supports_node_features: ClassVar[bool] = False
    supports_edge_features: ClassVar[bool] = False
    supports_node_labels: ClassVar[bool] = False
    supports_edge_labels: ClassVar[bool] = False
    supports_graph_labels: ClassVar[bool] = False
    supports_constraints: ClassVar[bool] = False
    supports_variable_size: ClassVar[bool] = True
    supports_featureless_graphs: ClassVar[bool] = True

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    @abstractmethod
    def load(self) -> None:
        """Load a checkpoint or external model state."""
        raise NotImplementedError

    @abstractmethod
    def train(self, train_graphs, val_graphs=None, test_graphs=None) -> None:
        """Train or adapt the wrapped generator."""
        raise NotImplementedError

    @abstractmethod
    def sample(self, num_graphs: int, seed: int = 0):
        """Return a list of generated NetworkX graphs."""
        raise NotImplementedError

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @classmethod
    def capabilities(cls) -> dict[str, bool]:
        return {
            "supports_training": cls.supports_training,
            "supports_sampling": cls.supports_sampling,
            "supports_node_features": cls.supports_node_features,
            "supports_edge_features": cls.supports_edge_features,
            "supports_node_labels": cls.supports_node_labels,
            "supports_edge_labels": cls.supports_edge_labels,
            "supports_graph_labels": cls.supports_graph_labels,
            "supports_constraints": cls.supports_constraints,
            "supports_variable_size": cls.supports_variable_size,
            "supports_featureless_graphs": cls.supports_featureless_graphs,
        }
