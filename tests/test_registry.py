import pytest

from empirical_comparison import registry
from empirical_comparison.registry import DATASET_REGISTRY, MODEL_REGISTRY, LazyRegistry, RegistrySpec

def test_registries_present():
    assert "sbm" in DATASET_REGISTRY
    assert "planar" in DATASET_REGISTRY
    assert "digress" in MODEL_REGISTRY


def test_registry_preserves_attribute_error_during_module_import(monkeypatch):
    fake_registry = LazyRegistry(
        {"broken": RegistrySpec("fake_broken_module", "MissingClass", "model")}
    )

    def raise_attribute_error(module_name):
        raise AttributeError("torch has no attribute UntypedStorage")

    monkeypatch.setattr(registry.importlib, "import_module", raise_attribute_error)

    with pytest.raises(AttributeError, match="UntypedStorage"):
        fake_registry["broken"]
