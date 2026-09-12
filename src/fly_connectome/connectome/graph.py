"""A small, explicit edge-table format with mandatory data provenance."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd


@dataclass
class ConnectomeGraph:
    """Directed weighted graph with original body IDs and provenance metadata."""

    neurons: pd.DataFrame
    edges: pd.DataFrame
    provenance: dict

    def __post_init__(self):
        required = {"source_neuron", "target_neuron", "weight"}
        if "neuron_id" not in self.neurons or not required <= set(self.edges):
            raise ValueError("Expected neuron_id and source_neuron,target_neuron,weight columns")
        if self.neurons.empty or self.neurons.neuron_id.duplicated().any():
            raise ValueError("Neurons must be nonempty and uniquely identified")
        if self.neurons.neuron_id.isna().any():
            raise ValueError("Null neuron IDs are invalid")
        identifiers = set(self.neurons.neuron_id)
        if not (set(self.edges.source_neuron) | set(self.edges.target_neuron)) <= identifiers:
            raise ValueError("Edge endpoints must exist in neurons.csv")
        weights = self.edges.weight.to_numpy(dtype=float)
        if not np.isfinite(weights).all() or (weights <= 0).any():
            raise ValueError("Connection weights must be finite and positive")
        if self.edges.duplicated(["source_neuron", "target_neuron"]).any():
            raise ValueError("Aggregate duplicate edges before constructing a graph")
        if self.provenance.get("data_kind") not in ("real", "synthetic", "random_control"):
            raise ValueError("Explicit data_kind provenance is required")

    @property
    def n(self) -> int:
        """Number of neurons, including isolated neurons."""
        return len(self.neurons)

    def networkx(self) -> nx.DiGraph:
        """Build a directed graph for selection/statistics, never for model forward."""
        graph = nx.DiGraph()
        graph.add_nodes_from(self.neurons.neuron_id.tolist())
        graph.add_weighted_edges_from(self.edges.itertuples(index=False, name=None))
        return graph

    def arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Map original IDs to deterministic contiguous indices in neuron table order."""
        lookup = {node: index for index, node in enumerate(self.neurons.neuron_id)}
        return (
            self.edges.source_neuron.map(lookup).to_numpy(dtype=np.int64),
            self.edges.target_neuron.map(lookup).to_numpy(dtype=np.int64),
            self.edges.weight.to_numpy(dtype=np.float32),
        )


def sha256(path: Path) -> str:
    """Hash a file in bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_processed_graph(graph: ConnectomeGraph, directory: str | Path) -> None:
    """Save portable CSVs and a manifest with integrity hashes."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    graph.neurons.to_csv(directory / "neurons.csv", index=False)
    graph.edges.to_csv(directory / "edges.csv", index=False)
    manifest = dict(graph.provenance)
    manifest["sha256"] = {name: sha256(directory / name) for name in ("neurons.csv", "edges.csv")}
    (directory / "metadata.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def load_connectome(directory: str | Path) -> ConnectomeGraph:
    """Load and verify a local dataset; never silently fabricate missing data."""
    directory = Path(directory)
    if not (directory / "metadata.json").is_file():
        raise FileNotFoundError(
            f"No graph manifest at {directory}. Run download or use offline.yaml."
        )
    provenance = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    if set(provenance.get("sha256", {})) != {"neurons.csv", "edges.csv"}:
        raise ValueError("Manifest must contain integrity hashes for both CSV files")
    for name, digest in provenance.get("sha256", {}).items():
        if name not in ("neurons.csv", "edges.csv") or sha256(directory / name) != digest:
            raise ValueError(f"Dataset checksum mismatch: {name}")
    neurons, edges = (pd.read_csv(directory / name) for name in ("neurons.csv", "edges.csv"))
    return ConnectomeGraph(neurons, edges[["source_neuron", "target_neuron", "weight"]], provenance)
