"""Directed degree-preserving rewiring for a clearly labeled non-biological control."""

import logging

import numpy as np
import pandas as pd

from ..connectome.graph import ConnectomeGraph


def random_sparse_control(
    graph: ConnectomeGraph, seed: int, swaps_per_edge: int = 10
) -> ConnectomeGraph:
    """Swap directed edge targets, preserving exact in/out degree and edge count.

    Reject parallel edges and new self-loops; existing self-loops remain eligible.
    Connectivity and higher-order motifs are not preserved or guaranteed mixed.
    """
    rng = np.random.default_rng(seed)
    pairs = list(zip(graph.edges.source_neuron, graph.edges.target_neuron, strict=True))
    weights = graph.edges.weight.to_numpy().copy()
    occupied = set(pairs)
    goal = swaps_per_edge * len(pairs)
    successes = 0
    for _ in range(goal * 20):
        if successes >= goal or len(pairs) < 2:
            break
        a, b = rng.choice(len(pairs), 2, replace=False)
        u, v = pairs[a]
        x, y = pairs[b]
        if u == y or x == v or (u, y) in occupied or (x, v) in occupied:
            continue
        occupied.remove((u, v))
        occupied.remove((x, y))
        occupied.update([(u, y), (x, v)])
        pairs[a], pairs[b] = (u, y), (x, v)
        # Keep each target's incoming weight multiset and total strength unchanged.
        weights[a], weights[b] = weights[b], weights[a]
        successes += 1
    if successes < goal:
        logging.getLogger(__name__).warning(
            "Rewiring reached %d/%d accepted swaps", successes, goal
        )
    table = pd.DataFrame(pairs, columns=["source_neuron", "target_neuron"])
    table["weight"] = weights
    return ConnectomeGraph(
        graph.neurons.copy(),
        table,
        {
            **graph.provenance,
            "parent_data_kind": graph.provenance["data_kind"],
            "data_kind": "random_control",
            "rewire_seed": seed,
            "accepted_swaps": successes,
            "requested_swaps": goal,
        },
    )
