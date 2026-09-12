"""Readable views of small graph samples, never full large connectomes."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

from ..connectome.graph import ConnectomeGraph


def plot_connectome(graph: ConnectomeGraph, output: Path, limit: int = 100, seed: int = 42) -> Path:
    """Render at most 200 nodes; color is total degree within the displayed sample."""
    if not 1 <= limit <= 200:
        raise ValueError("Graph visualization limit must be between 1 and 200")
    network = graph.networkx()
    component = max(nx.connected_components(network.to_undirected()), key=len)
    root = min(component, key=lambda node: (-network.degree(node), node))
    nodes = list(nx.bfs_tree(network.to_undirected(), root, sort_neighbors=sorted))[:limit]
    view = network.subgraph(nodes)
    fig, ax = plt.subplots(figsize=(9, 7), layout="constrained")
    positions = nx.spring_layout(view, seed=seed)
    nx.draw_networkx_edges(view, positions, ax=ax, alpha=0.18, arrows=True, arrowsize=7)
    colors = nx.draw_networkx_nodes(
        view,
        positions,
        ax=ax,
        node_size=50,
        node_color=[view.degree(n) for n in view],
        cmap="viridis",
    )
    fig.colorbar(colors, ax=ax, label="Sample total degree")
    ax.set_title(f"{graph.provenance['data_kind'].upper()} graph — {len(view)} displayed neurons")
    ax.axis("off")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output
