"""Reproduce the explicitly non-biological offline fixture (no external data)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from fly_connectome.connectome.graph import ConnectomeGraph, save_processed_graph


def main() -> None:
    """Generate a weakly connected directed ring plus seeded random edges."""
    size, seed = 512, 2026
    rng = np.random.default_rng(seed)
    edges = {(i, (i + 1) % size) for i in range(size)}
    while len(edges) < size * 6:
        source, target = map(int, rng.integers(size, size=2))
        if source != target:
            edges.add((source, target))
    table = pd.DataFrame(
        [(s, t, int(rng.integers(1, 21))) for s, t in sorted(edges)],
        columns=["source_neuron", "target_neuron", "weight"],
    )
    graph = ConnectomeGraph(
        pd.DataFrame({"neuron_id": range(size)}),
        table,
        {
            "data_kind": "synthetic",
            "source": "local deterministic generator",
            "generator": "scripts/make_synthetic_sample.py",
            "seed": seed,
            "description": "SYNTHETIC FALLBACK. Ring plus random edges; NOT biological data.",
            "license": "MIT",
        },
    )
    target = Path(__file__).resolve().parents[1] / "data/sample/synthetic"
    save_processed_graph(graph, target)
    (target / "README.md").write_text(
        "# SYNTHETIC FALLBACK — not biological data\n\n"
        "512 artificial nodes and 3072 directed edges. Used solely for offline tests and "
        "pipeline demonstrations. No connection was measured in a fly. Reproduce using "
        "`python scripts/make_synthetic_sample.py`. Manifest records the seed and hashes.\n",
        encoding="utf-8",
    )
    for directory in ["data/raw", "data/processed", "results", "notebooks"]:
        folder = target.parents[2] / directory
        folder.mkdir(parents=True, exist_ok=True)
        (folder / ".gitkeep").touch()
    assert json.loads((target / "metadata.json").read_text())["data_kind"] == "synthetic"


if __name__ == "__main__":
    main()
