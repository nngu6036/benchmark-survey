import networkx as nx

from empirical_comparison.models.wrappers.graphguide import GraphGUIDEWrapper


def test_graphguide_pyg_conversion_graph_drops_set_metadata():
    graph = nx.Graph()
    graph.graph["partition"] = [{0, 1}]
    graph.add_node(0, feats=[1.0], block={0})
    graph.add_node(1, feats=[1.0], block={1})
    graph.add_edge(0, 1, edge_set={0, 1})

    clean = GraphGUIDEWrapper._graph_for_pyg_conversion(graph)

    assert clean.graph == {}
    assert clean.nodes[0]["feats"].tolist() == [1.0]
    assert "block" not in clean.nodes[0]
    assert clean.edges[0, 1] == {}
