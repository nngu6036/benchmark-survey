from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Iterator, Mapping, MutableMapping, TypeVar, Generic, Any

T = TypeVar("T")


@dataclass(frozen=True)
class RegistrySpec:
    module: str
    class_name: str
    kind: str
    optional_dependency_hint: str | None = None


class LazyRegistry(Mapping[str, type]):
    """Dictionary-like registry that imports classes only when requested.

    This keeps dataset-only and metric-only commands usable without installing
    every model-specific dependency (e.g. torch_geometric, hydra, lightning).
    """

    def __init__(self, specs: dict[str, RegistrySpec]) -> None:
        self._specs = dict(specs)
        self._cache: dict[str, type] = {}

    def __getitem__(self, key: str) -> type:
        if key not in self._specs:
            raise KeyError(f"Unknown registry key {key!r}. Available: {sorted(self._specs)}")
        if key not in self._cache:
            spec = self._specs[key]
            try:
                module = importlib.import_module(spec.module)
                cls = getattr(module, spec.class_name)
            except ModuleNotFoundError as exc:
                hint = f" Hint: {spec.optional_dependency_hint}" if spec.optional_dependency_hint else ""
                raise ModuleNotFoundError(
                    f"Could not import {spec.kind} '{key}' from {spec.module}.{spec.class_name}." + hint
                ) from exc
            except AttributeError as exc:
                raise AttributeError(
                    f"Registry entry '{key}' points to missing class {spec.class_name!r} in {spec.module!r}."
                ) from exc
            self._cache[key] = cls
        return self._cache[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._specs)

    def __len__(self) -> int:
        return len(self._specs)

    def specs(self) -> dict[str, RegistrySpec]:
        return dict(self._specs)

    def describe(self) -> dict[str, dict[str, Any]]:
        return {k: spec.__dict__.copy() for k, spec in self._specs.items()}


DATASET_REGISTRY = LazyRegistry(
    {
        "sbm": RegistrySpec("empirical_comparison.datasets.sbm", "SBMDatasetBuilder", "dataset"),
        "planar": RegistrySpec("empirical_comparison.datasets.planar", "PlanarDatasetBuilder", "dataset"),
    }
)

MODEL_REGISTRY = LazyRegistry(
    {
        "dummy": RegistrySpec("empirical_comparison.models.wrappers.dummy", "DummyGraphGenerator", "model"),
        "graphguide": RegistrySpec(
            "empirical_comparison.models.wrappers.graphguide",
            "GraphGUIDEWrapper",
            "model",
            "Install the GraphGUIDE wrapper dependencies and set GRAPHGUIDE_REPO.",
        ),
        "digress": RegistrySpec(
            "empirical_comparison.models.wrappers.digress",
            "DiGressWrapper",
            "model",
            "Install the DiGress wrapper dependencies and set DIGRESS_REPO.",
        ),
        "construct": RegistrySpec(
            "empirical_comparison.models.wrappers.construct",
            "ConStructWrapper",
            "model",
            "Install the ConStruct wrapper dependencies and set CONSTRUCT_REPO.",
        ),
        "edp_gnn": RegistrySpec(
            "empirical_comparison.models.wrappers.edp_gnn",
            "EDPGNNWrapper",
            "model",
            "Install the EDP-GNN wrapper dependencies and set EDP_GNN_REPO.",
        ),
        "disco": RegistrySpec(
            "empirical_comparison.models.wrappers.disco",
            "DisCoWrapper",
            "model",
            "Install the DisCo wrapper dependencies and set DISCO_REPO.",
        ),
        "grum": RegistrySpec(
            "empirical_comparison.models.wrappers.grum",
            "GruMWrapper",
            "model",
            "Install the GruM wrapper dependencies and set GRUM_REPO.",
        ),
    }
)


def get_dataset_builder(name: str) -> type:
    return DATASET_REGISTRY[name]


def get_model_class(name: str) -> type:
    return MODEL_REGISTRY[name]


def available_datasets() -> list[str]:
    return sorted(DATASET_REGISTRY.keys())


def available_models() -> list[str]:
    return sorted(MODEL_REGISTRY.keys())
