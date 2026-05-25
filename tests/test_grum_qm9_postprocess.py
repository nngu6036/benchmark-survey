import networkx as nx

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
