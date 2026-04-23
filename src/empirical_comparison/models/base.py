from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any

class BaseGenerator(ABC):
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    @abstractmethod
    def load(self) -> None: ...
    @abstractmethod
    def train(self, train_graphs, val_graphs=None) -> None: ...
    @abstractmethod
    def sample(self, num_graphs: int, seed: int = 0): ...
    @property
    @abstractmethod
    def name(self) -> str: ...
