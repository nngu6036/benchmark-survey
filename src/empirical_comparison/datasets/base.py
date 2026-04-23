from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

class BaseDatasetBuilder(ABC):
    def __init__(self, config: dict[str, Any], root: str | Path = "data/processed") -> None:
        self.config = config
        self.root = Path(root)

    @abstractmethod
    def build(self):
        raise NotImplementedError
