import networkx as nx
import pytest
import torch

from empirical_comparison.metrics.molecular.rdkit_validity import molecular_quality_metrics
from empirical_comparison.models.wrappers.grum import GruMWrapper


def test_grum_qm9_postprocess_assigns_valence_valid_labels():
    wrapper = object.__new__(GruMWrapper)
    wrapper.config = {}
    wrapper.dataset_name = "qm9"
    wrapper._last_sampling_diagnostics = {}

    graph = nx.star_graph(5)
    processed = wrapper._qm9_constrained_postprocess_graphs([graph], seed=7)[0]

    assert max(dict(processed.degree()).values()) <= 4
    assert all("node_label" in data for _, data in processed.nodes(data=True))
    assert all(data.get("edge_type") == 1 for _, _, data in processed.edges(data=True))

    metrics = molecular_quality_metrics([processed], [], dataset="qm9")
    assert metrics["validity_rate"] == 1.0


def test_grum_qm9_label_sampler_respects_degree_valence():
    wrapper = object.__new__(GruMWrapper)
    wrapper.config = {}

    for degree in range(5):
        label = wrapper._sample_qm9_label_for_degree(degree, __import__("random").Random(3))
        assert wrapper._QM9_VALENCE_BY_LABEL[label] >= degree


def test_grum_uses_observed_max_node_count_when_unset(monkeypatch):
    wrapper = object.__new__(GruMWrapper)
    wrapper.config = {
        "dataset": "sbm",
        "max_node_num": None,
        "batch_size": 32,
        "sample_batch_size": 32,
        "feat_types": ["const"],
        "num_epochs": 1,
    }
    wrapper.dataset_name = "sbm"

    monkeypatch.setattr(
        wrapper,
        "_load_base_yaml",
        lambda: {
            "data": {"max_node_num": 192, "batch_size": 32, "feat": {"type": ["const"], "scale": 1.0, "norm": False}},
            "train": {},
            "model": {},
            "mix": {},
            "sampler": {},
            "sample": {},
        },
    )
    monkeypatch.setattr(wrapper, "_compute_features", lambda adj, mask, data_config: (torch.ones((*adj.shape[:2], 1)), [1]))

    graphs = [nx.empty_graph(64)]
    cfg = wrapper._build_config(graphs)

    assert int(cfg.data.max_node_num) == 64


def test_grum_explicit_max_node_count_is_still_honored(monkeypatch):
    wrapper = object.__new__(GruMWrapper)
    wrapper.config = {
        "dataset": "sbm",
        "max_node_num": 80,
        "batch_size": 32,
        "sample_batch_size": 32,
        "feat_types": ["const"],
        "num_epochs": 1,
    }
    wrapper.dataset_name = "sbm"

    monkeypatch.setattr(
        wrapper,
        "_load_base_yaml",
        lambda: {
            "data": {"max_node_num": 192, "batch_size": 32, "feat": {"type": ["const"], "scale": 1.0, "norm": False}},
            "train": {},
            "model": {},
            "mix": {},
            "sampler": {},
            "sample": {},
        },
    )
    monkeypatch.setattr(wrapper, "_compute_features", lambda adj, mask, data_config: (torch.ones((*adj.shape[:2], 1)), [1]))

    cfg = wrapper._build_config([nx.empty_graph(64)])

    assert int(cfg.data.max_node_num) == 80


def test_grum_rejects_explicit_max_node_count_below_data(monkeypatch):
    wrapper = object.__new__(GruMWrapper)
    wrapper.config = {"dataset": "sbm", "max_node_num": 32}
    wrapper.dataset_name = "sbm"

    monkeypatch.setattr(
        wrapper,
        "_load_base_yaml",
        lambda: {"data": {"max_node_num": 192}, "train": {}, "model": {}, "mix": {}, "sampler": {}, "sample": {}},
    )

    with pytest.raises(ValueError, match="smaller than a training graph"):
        wrapper._build_config([nx.empty_graph(64)])
