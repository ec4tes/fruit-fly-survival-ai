"""Deterministic subgraph selection with explicit connectivity guarantees."""

import networkx as nx
import numpy as np

from .graph import ConnectomeGraph
from .preprocessing import filter_neurons


def select_subgraph(
    graph: ConnectomeGraph,
    size: int,
    strategy: str = "connected_component",
    seed: int = 2026,
    seed_neurons: list | None = None,
    biological_filters: dict | None = None,
) -> ConnectomeGraph:
    """Select induced edges; connected/BFS modes traverse the weak undirected projection."""
    if strategy == "biologically_selected":
        if not biological_filters or graph.provenance["data_kind"] != "real":
            raise ValueError("Biological selection needs real data and explicit metadata filters")
        graph = filter_neurons(graph, biological_filters)
    if size < 1 or size > graph.n:
        raise ValueError(f"Requested {size} neurons, but dataset contains {graph.n}")
    network = graph.networkx()
    if strategy == "random":
        nodes = np.random.default_rng(seed).choice(sorted(network), size, replace=False).tolist()
    elif strategy == "high_degree":
        nodes = sorted(network, key=lambda node: (-network.degree(node), node))[:size]
    elif strategy in ("connected_component", "bfs", "biologically_selected"):
        undirected = network.to_undirected()
        components = sorted(nx.connected_components(undirected), key=lambda c: (-len(c), min(c)))
        if strategy == "bfs":
            if not seed_neurons or any(node not in network for node in seed_neurons):
                raise ValueError("BFS requires valid seed_neurons (original body IDs)")
            component = next(c for c in components if seed_neurons[0] in c)
            if not set(seed_neurons) <= component:
                raise ValueError("BFS seeds must belong to the same weak component")
            # BFS from first seed; later seeds must fit within the selected connected expansion.
            root = seed_neurons[0]
        else:
            component = components[0]
            root = min(component, key=lambda node: (-network.degree(node), node))
        if len(component) < size:
            raise ValueError(
                f"Largest/requested weak component has {len(component)} < {size} neurons"
            )
        traversal = list(nx.bfs_tree(undirected.subgraph(component), root, sort_neighbors=sorted))
        nodes = traversal[:size]
        if strategy == "bfs" and not set(seed_neurons) <= set(nodes):
            raise ValueError("Increase subgraph_size to include all requested BFS seeds")
    else:
        raise ValueError(f"Unknown subgraph strategy: {strategy}")
    nodes = sorted(nodes)
    neurons = graph.neurons.set_index("neuron_id").loc[nodes].reset_index()
    edges = graph.edges[
        graph.edges.source_neuron.isin(nodes) & graph.edges.target_neuron.isin(nodes)
    ]
    provenance = {
        **graph.provenance,
        "selection": {
            "strategy": strategy,
            "size": size,
            "seed": seed,
            "seed_neurons": seed_neurons,
        },
    }
    return ConnectomeGraph(neurons, edges.reset_index(drop=True), provenance)
