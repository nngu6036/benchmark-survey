import networkx as nx

from empirical_comparison.metrics.molecular.rdkit_validity import (
    atomic_number_from_graph_label,
    molecular_quality_metrics,
)


def test_qm9_generated_atomic_numbers_survive_evaluation_canonicalization():
    graph = nx.Graph()
    graph.add_node(0, node_label=5)

    atomic_number = atomic_number_from_graph_label(
        graph,
        0,
        node_label_attr="node_label",
        raw_node_label_values=["0", "1", "2", "3", "4", "6"],
        dataset="qm9",
    )

    assert atomic_number == 6


def test_molecular_quality_uses_fingerprints_when_smiles_are_unavailable():
    ethane = nx.Graph()
    ethane.add_node(0, node_label=1)
    ethane.add_node(1, node_label=1)
    ethane.add_edge(0, 1, edge_type=1)

    metrics = molecular_quality_metrics([ethane], [], dataset="qm9")

    assert metrics["validity_rate"] == 1.0
    assert metrics["uniqueness_rate"] == 1.0


def test_zinc_without_atom_mapping_falls_back_to_labelled_graph_quality():
    graph = nx.Graph()
    graph.add_node(0, node_label=3)
    graph.add_node(1, node_label=4)
    graph.add_edge(0, 1, edge_type=1)

    metrics = molecular_quality_metrics([graph], [], dataset="zinc")

    assert metrics["validity_rate"] == 1.0
    assert metrics["uniqueness_rate"] == 1.0
    assert metrics["rdkit_validity_rate"] == 0.0
    assert metrics["validity_backend"] == "categorical_graph_fingerprint_fallback"
