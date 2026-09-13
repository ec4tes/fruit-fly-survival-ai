"""Graph validation, connected selection, normalization and provenance protection."""

import networkx as nx
import numpy as np
import pandas as pd
import pytest
import torch

from fly_connectome.connectome.downloader import download_connectome
from fly_connectome.connectome.graph import ConnectomeGraph, load_connectome, save_processed_graph
from fly_connectome.connectome.preprocessing import build_sparse_adjacency, filter_neurons
from fly_connectome.connectome.selector import select_subgraph
from fly_connectome.connectome.statistics import graph_statistics
from fly_connectome.models.random_sparse import random_sparse_control
from fly_connectome.training.train import prepare_graph
from fly_connectome.utils.config import load_config, resolve_path


@pytest.fixture
def graph():
    return load_connectome(resolve_path("data/sample/synthetic"))


def test_load_and_statistics(graph):
    assert graph.n == 512 and len(graph.edges) == 3072
    assert graph.provenance["data_kind"] == "synthetic"
    stats = graph_statistics(graph)
    assert stats["weak_components"] == 1 and stats["average_degree"] == 12


@pytest.mark.parametrize("strategy", ["random", "high_degree", "connected_component", "bfs"])
def test_subgraph(graph, strategy):
    sub = select_subgraph(graph, 64, strategy, seed=123, seed_neurons=[0])
    assert sub.n == 64
    again = select_subgraph(graph, 64, strategy, seed=123, seed_neurons=[0])
    pd.testing.assert_frame_equal(sub.edges, again.edges)
    if strategy in ("connected_component", "bfs"):
        assert nx.is_weakly_connected(sub.networkx())


def test_invalid_subgraph(graph):
    with pytest.raises(ValueError):
        select_subgraph(graph, 1000)
    with pytest.raises(ValueError):
        select_subgraph(graph, 20, "biologically_selected", biological_filters={"type": "fake"})
    with pytest.raises(ValueError):
        filter_neurons(graph, {"absent_region": "AL"})


def test_real_metadata_selection():
    # Artificial table tagged real only to test metadata dispatch, never used for experiments.
    graph = ConnectomeGraph(
        pd.DataFrame({"neuron_id": [1, 2, 3], "type": ["A", "A", "B"]}),
        pd.DataFrame([(1, 2, 4), (2, 3, 2)], columns=["source_neuron", "target_neuron", "weight"]),
        {"data_kind": "real", "source": "mock test only"},
    )
    sub = select_subgraph(graph, 2, "biologically_selected", biological_filters={"type": "A"})
    assert sub.neurons.neuron_id.tolist() == [1, 2]


def test_sparse_orientation(graph):
    graph = select_subgraph(graph, 32)
    adjacency = build_sparse_adjacency(graph)
    assert adjacency.layout == torch.sparse_coo
    sums = torch.sparse.sum(adjacency, dim=1).to_dense()
    assert torch.all((sums == 0) | torch.isclose(sums, torch.ones_like(sums)))


def test_roundtrip_and_tamper(graph, tmp_path):
    save_processed_graph(graph, tmp_path)
    pd.testing.assert_frame_equal(graph.edges, load_connectome(tmp_path).edges)
    with (tmp_path / "edges.csv").open("a") as stream:
        stream.write("0,1,2\n")
    with pytest.raises(ValueError, match="checksum"):
        load_connectome(tmp_path)


def test_checksum_is_line_ending_independent(graph, tmp_path):
    """A Git checkout may use CRLF on Windows and LF on Linux."""
    save_processed_graph(graph, tmp_path)
    for name in ("neurons.csv", "edges.csv"):
        path = tmp_path / name
        path.write_bytes(path.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
    assert load_connectome(tmp_path).n == graph.n


def test_synthetic_opt_in():
    config = load_config("configs/offline.yaml")
    config["model"]["allow_synthetic"] = False
    with pytest.raises(ValueError, match="Non-biological"):
        prepare_graph(config)


def test_download_cache_works_without_token(graph, tmp_path, monkeypatch):
    monkeypatch.delenv("NEUPRINT_TOKEN", raising=False)
    save_processed_graph(graph, tmp_path)
    assert download_connectome(tmp_path).n == 512


def test_downloader_failure_and_explicit_fallback(tmp_path, monkeypatch, caplog):
    monkeypatch.delenv("NEUPRINT_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="NEUPRINT_TOKEN"):
        download_connectome(tmp_path)
    fallback = download_connectome(tmp_path, allow_synthetic_fallback=True)
    assert fallback.provenance["data_kind"] == "synthetic"
    assert "SYNTHETIC" in caplog.text and not (tmp_path / "metadata.json").exists()


def test_degree_preserving_control(graph):
    sub = select_subgraph(graph, 64)
    control = random_sparse_control(sub, 12, swaps_per_edge=2)
    a, b = sub.networkx(), control.networkx()
    assert dict(a.in_degree()) == dict(b.in_degree())
    assert dict(a.out_degree()) == dict(b.out_degree())
    assert len(a.edges) == len(b.edges) and set(a.edges) != set(b.edges)
    assert control.provenance["data_kind"] == "random_control"
    for node in sub.neurons.neuron_id:
        assert sorted(sub.edges.loc[sub.edges.target_neuron == node, "weight"]) == sorted(
            control.edges.loc[control.edges.target_neuron == node, "weight"]
        )


def test_downloader_mocked_real_contract(tmp_path, monkeypatch):
    """Validate API table order and ROI aggregation with no server access."""
    import neuprint

    import fly_connectome.connectome.downloader as downloader

    class MockClient:
        def fetch_custom(self, query):
            assert "ConnectsTo" in query
            return pd.DataFrame({"bodyId": [2, 3], "strength": [10, 5]})

    def neurons(ids, client):
        assert set(ids) == {1, 2, 3}
        return pd.DataFrame(
            {"bodyId": [1, 2, 3], "type": ["A", "B", "C"], "inputRois": [["AL"], [], []]}
        ), pd.DataFrame()

    def adjacency(sources, targets, **kwargs):
        assert set(sources) == set(targets) == {1, 2, 3}
        assert kwargs["include_nonprimary"] is False
        return pd.DataFrame(), pd.DataFrame(
            {
                "bodyId_pre": [1, 1, 2],
                "bodyId_post": [2, 2, 3],
                "roi": ["AL", "NotPrimary", "AL"],
                "weight": [4, 6, 5],
            }
        )

    monkeypatch.setattr(downloader, "create_client", lambda: MockClient())
    monkeypatch.setattr(neuprint, "fetch_neurons", neurons)
    monkeypatch.setattr(neuprint, "fetch_adjacencies", adjacency)
    graph = download_connectome(tmp_path, size=3, seed_ids=[1])
    assert graph.edges.weight.tolist() == [10, 5]
    assert graph.neurons.type.tolist() == ["A", "B", "C"]
    assert graph.provenance["data_kind"] == "real"  # mocked contract, not experimental data
    assert load_connectome(tmp_path).n == 3


def test_bad_graph_weights():
    with pytest.raises(ValueError, match="positive"):
        ConnectomeGraph(
            pd.DataFrame({"neuron_id": [1, 2]}),
            pd.DataFrame([(1, 2, np.nan)], columns=["source_neuron", "target_neuron", "weight"]),
            {"data_kind": "synthetic"},
        )
