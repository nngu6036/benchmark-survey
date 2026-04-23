from __future__ import annotations
from empirical_comparison.registry import MODEL_REGISTRY

def sample_graphs(model_name: str, model_cfg: dict, num_graphs: int, seed: int = 0):
    model = MODEL_REGISTRY[model_name](model_cfg)
    model.load()
    return model.sample(num_graphs=num_graphs, seed=seed)
