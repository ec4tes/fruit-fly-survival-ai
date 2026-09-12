"""Deterministic sensory/readout mapping around graph-constrained dynamics."""

from __future__ import annotations

import networkx as nx
import numpy as np
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

from ..connectome.graph import ConnectomeGraph
from ..connectome.preprocessing import normalize_weights
from .connectome_layer import ConnectomeLayer


def graph_spec(graph: ConnectomeGraph, config: dict, allow_unreachable: bool = False) -> dict:
    """Serialize sparse arrays and documented input/output IDs into the checkpoint."""
    source, target, weights = graph.arrays()
    ids = graph.neurons.neuron_id.tolist()
    index = {node: i for i, node in enumerate(ids)}
    out_degree = np.bincount(source, minlength=graph.n)
    in_degree = np.bincount(target, minlength=graph.n)
    if config["input_neurons"] is None:
        inputs = np.argsort(-out_degree, kind="stable")[: config["input_count"]].tolist()
    else:
        inputs = [index[node] for node in config["input_neurons"]]
    network = nx.DiGraph()
    network.add_nodes_from(range(graph.n))
    network.add_edges_from(zip(source, target, strict=True))
    reachable, frontier = set(inputs), set(inputs)
    for _ in range(config["dynamics_steps"] - 1):
        frontier = {n for node in frontier for n in network.successors(node)} - reachable
        reachable |= frontier
    if config["output_neurons"] is None:
        candidates = reachable - set(inputs)
        outputs = sorted(candidates, key=lambda n: (-in_degree[n], n))[: config["output_count"]]
    else:
        outputs = [index[node] for node in config["output_neurons"]]
    if (
        not inputs
        or not outputs
        or len(set(inputs)) != len(inputs)
        or len(set(outputs)) != len(outputs)
        or set(inputs) & set(outputs)
    ):
        raise ValueError("Input/output mappings must be nonempty, unique and disjoint")
    if config["output_neurons"] is None and len(outputs) != config["output_count"]:
        raise ValueError(
            "Insufficient reachable output neurons; increase dynamics_steps or reduce counts"
        )
    if not allow_unreachable and not set(outputs) <= reachable:
        raise ValueError("Output neurons must be reachable during the configured dynamics_steps")
    return {
        "neuron_count": graph.n,
        "source": source.tolist(),
        "target": target.tolist(),
        "weights": normalize_weights(source, target, weights, graph.n).tolist(),
        "input_indices": inputs,
        "output_indices": outputs,
        "input_body_ids": [int(ids[n]) for n in inputs],
        "output_body_ids": [int(ids[n]) for n in outputs],
        "reachable_output_count": len(set(outputs) & reachable),
        "dynamics_steps": config["dynamics_steps"],
        "train_edge_weights": config["train_edge_weights"],
        "gain": config["recurrent_gain"],
    }


class ConnectomeNetwork(BaseFeaturesExtractor):
    """SB3 extractor using recurrent microsteps, reset at every environment decision.

    ``evolve`` accepts/returns explicit state for future sequence-aware trainers.
    ``forward`` intentionally does not persist state across shuffled PPO minibatches.
    """

    def __init__(self, observation_space, spec: dict):
        super().__init__(observation_space, features_dim=len(spec["output_indices"]))
        self.spec = spec
        self.register_buffer("input_indices", torch.tensor(spec["input_indices"], dtype=torch.long))
        self.register_buffer(
            "output_indices", torch.tensor(spec["output_indices"], dtype=torch.long)
        )
        self.encoder = nn.Linear(observation_space.shape[0], len(spec["input_indices"]))
        self.layer = ConnectomeLayer(
            spec["neuron_count"],
            spec["source"],
            spec["target"],
            spec["weights"],
            spec["train_edge_weights"],
            spec["gain"],
        )

    def evolve(
        self, observations: torch.Tensor, state: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Inject observations and run K cyclic sparse updates; return readout and h_K."""
        shape = (observations.shape[0], self.spec["neuron_count"])
        if state is None:
            state = observations.new_zeros(shape)
        elif tuple(state.shape) != shape:
            raise ValueError(f"State must have shape {shape}")
        sensory = observations.new_zeros(shape).index_add(
            1, self.input_indices, self.encoder(observations)
        )
        for _ in range(self.spec["dynamics_steps"]):
            state = self.layer(state, sensory)
        return state[:, self.output_indices], state

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Return selected output neuron activations; SB3 adds linear actor/value heads."""
        return self.evolve(observations)[0]

    def set_damage(
        self, fraction: float, seed: int, kind: str = "neuron", strategy: str = "random"
    ) -> None:
        """Apply a fixed lesion to graph neurons/edges, including mapped neurons."""
        self.layer.set_damage(fraction, seed, kind, strategy)

    def clear_damage(self) -> None:
        """Restore intact graph activations."""
        self.layer.clear_damage()
