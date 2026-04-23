from empirical_comparison.registry import DATASET_REGISTRY, MODEL_REGISTRY

def test_registries_present():
    assert "sbm" in DATASET_REGISTRY
    assert "planar" in DATASET_REGISTRY
    assert "digress" in MODEL_REGISTRY
