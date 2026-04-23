import networkx as nx
from empirical_comparison.metrics.descriptor.descriptors import degree_histogram

def test_degree_histogram_runs():
    g = nx.path_graph(5)
    hist = degree_histogram(g)
    assert hist.ndim == 1
