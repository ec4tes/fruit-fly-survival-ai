"""Basic directed graph diagnostics, including isolates."""

import networkx as nx

from .graph import ConnectomeGraph


def graph_statistics(graph: ConnectomeGraph) -> dict:
    """Return serializable counts and per-neuron in/out degree."""
    network = graph.networkx()
    return {
        "neuron_count": graph.n,
        "edge_count": len(graph.edges),
        "average_degree": 2 * len(graph.edges) / graph.n,
        "density": nx.density(network),
        "weak_components": nx.number_weakly_connected_components(network),
        "strong_components": nx.number_strongly_connected_components(network),
        "in_degree": {str(n): d for n, d in network.in_degree()},
        "out_degree": {str(n): d for n, d in network.out_degree()},
        "data_kind": graph.provenance["data_kind"],
    }
