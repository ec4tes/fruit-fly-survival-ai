"""Bounded real-data extraction with explicit caching and opt-in synthetic fallback."""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ..utils.config import resolve_path
from .client import create_client
from .graph import ConnectomeGraph, load_connectome, save_processed_graph

LOGGER = logging.getLogger(__name__)


def download_connectome(
    output: Path,
    size: int = 2000,
    seed_ids: list[int] | None = None,
    neuron_type: str | None = None,
    force: bool = False,
    allow_synthetic_fallback: bool = False,
) -> ConnectomeGraph:
    """Expand measured weak connectivity around seed bodies, then fetch induced edges.

    Queries are bounded in returned candidate count. Metadata and original IDs are
    retained. A complete local cache is usable without any API credentials.
    """
    if size < 1:
        raise ValueError("Download size must be positive")
    if (output / "metadata.json").exists() and not force:
        LOGGER.info("Using cached connectome: %s", output)
        return load_connectome(output)
    try:
        from neuprint import NeuronCriteria, fetch_adjacencies, fetch_neurons

        client = create_client()
        if seed_ids:
            selected = [int(node) for node in seed_ids]
        elif neuron_type:
            neurons, _ = fetch_neurons(
                NeuronCriteria(type=neuron_type, client=client), client=client
            )
            selected = sorted(neurons.bodyId.astype(int).tolist())[:size]
            if not selected:
                raise ValueError("No neurons match --neuron-type")
        else:
            table = client.fetch_custom(
                "MATCH (n:Neuron) WHERE n.pre > 0 AND n.post > 0 "
                "RETURN n.bodyId AS bodyId ORDER BY (n.pre + n.post) DESC, n.bodyId LIMIT 1"
            )
            selected = [int(table.iloc[0].bodyId)]
        if len(selected) > size:
            raise ValueError("More seed IDs than requested download size")
        frontier = selected.copy()
        while len(selected) < size and frontier:
            # Only integer IDs generated above are interpolated into Cypher.
            query = (
                f"MATCH (n:Neuron)-[e:ConnectsTo]-(m:Neuron) WHERE n.bodyId IN {frontier} "
                f"AND NOT m.bodyId IN {selected} RETURN m.bodyId AS bodyId, "
                "sum(e.weight) AS strength ORDER BY strength DESC, bodyId "
                f"LIMIT {size - len(selected)}"
            )
            table = client.fetch_custom(query)
            frontier = sorted(set(table.bodyId.astype(int)) - set(selected))
            selected.extend(frontier)
        neurons, _ = fetch_neurons(selected, client=client)
        _, connections = fetch_adjacencies(
            selected,
            selected,
            include_nonprimary=False,
            properties=["type", "instance"],
            client=client,
        )
        # Default primary ROI rows plus NotPrimary form a partition: sum each pair once.
        edges = connections.groupby(["bodyId_pre", "bodyId_post"], as_index=False).weight.sum()
        edges = edges.rename(
            columns={"bodyId_pre": "source_neuron", "bodyId_post": "target_neuron"}
        )
        edges = edges[edges.weight > 0]
        columns = [
            name
            for name in ("bodyId", "type", "instance", "pre", "post", "status")
            if name in neurons
        ]
        neuron_table = neurons[columns].rename(columns={"bodyId": "neuron_id"})
        # List-valued ROI annotations are encoded explicitly, rather than mislabeled as regions.
        if "inputRois" in neurons:
            neuron_table["input_rois"] = neurons.inputRois.apply(lambda v: "|".join(v))
        if "outputRois" in neurons:
            neuron_table["output_rois"] = neurons.outputRois.apply(lambda v: "|".join(v))
        graph = ConnectomeGraph(
            neuron_table.sort_values("neuron_id").reset_index(drop=True),
            pd.DataFrame(edges),
            {
                "data_kind": "real",
                "source": "neuPrint",
                "server": os.environ.get("NEUPRINT_SERVER", "https://neuprint.janelia.org"),
                "dataset": os.environ.get("NEUPRINT_DATASET", "male-cns:v1.0"),
                "retrieved_utc": datetime.now(timezone.utc).isoformat(),
                "seed_ids": seed_ids,
                "neuron_type": neuron_type,
                "requested_size": size,
                "weight_semantics": "synapse count; summed non-overlapping primary ROI rows",
                "selection_bias": "bounded weak expansion ranked by connection strength",
                "license_note": "Consult the source dataset terms and citation requirements",
            },
        )
        if graph.n < size:
            LOGGER.warning("Expansion returned %d neurons, fewer than requested %d", graph.n, size)
        save_processed_graph(graph, output)
        LOGGER.info("Cached REAL graph: %d neurons, %d edges", graph.n, len(graph.edges))
        return graph
    except Exception as error:
        # Never include the upstream exception text: HTTP errors may expose request headers.
        if allow_synthetic_fallback:
            LOGGER.warning(
                "Real download failed (%s). Explicit SYNTHETIC fallback is active; "
                "no biological conclusions are valid. Real cache was not overwritten.",
                type(error).__name__,
            )
            return load_connectome(resolve_path("data/sample/synthetic"))
        raise RuntimeError(
            f"neuPrint download failed ({type(error).__name__}). Check NEUPRINT_TOKEN, "
            "NEUPRINT_SERVER, NEUPRINT_DATASET and seed IDs. Existing local data is retained. "
            "For offline software checks use configs/offline.yaml; fallback is never silent."
        ) from None
