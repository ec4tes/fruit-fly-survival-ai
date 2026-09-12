"""Metadata filtering and incoming-strength normalization."""

import numpy as np
import torch

from .graph import ConnectomeGraph


def filter_neurons(graph: ConnectomeGraph, filters: dict) -> ConnectomeGraph:
    """Filter exact metadata values (e.g. type/instance/region); reject absent fields."""
    table = graph.neurons
    for key, values in filters.items():
        if key not in table.columns:
            raise ValueError(f"Biological selection requires missing metadata column: {key}")
        table = table[table[key].isin(values if isinstance(values, list) else [values])]
    if table.empty:
        raise ValueError("No neurons match the requested biological metadata")
    ids = set(table.neuron_id)
    edges = graph.edges[graph.edges.source_neuron.isin(ids) & graph.edges.target_neuron.isin(ids)]
    return ConnectomeGraph(
        table.reset_index(drop=True),
        edges.reset_index(drop=True),
        {**graph.provenance, "filters": filters},
    )


def normalize_weights(
    source: np.ndarray, target: np.ndarray, weights: np.ndarray, neuron_count: int
) -> np.ndarray:
    """Normalize each target's incoming synapse counts to sum to one."""
    totals = np.bincount(target, weights=weights, minlength=neuron_count)
    return (weights / np.maximum(totals[target], 1e-12)).astype(np.float32)


def build_sparse_adjacency(graph: ConnectomeGraph) -> torch.Tensor:
    """Return COO A[target, source], suitable for A @ h, without an NxN allocation."""
    source, target, weights = graph.arrays()
    weights = normalize_weights(source, target, weights, graph.n)
    return torch.sparse_coo_tensor(
        np.stack([target, source]), weights, (graph.n, graph.n), check_invariants=True
    ).coalesce()
