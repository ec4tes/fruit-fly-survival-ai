"""Sparse propagation, gradients, recurrence and exact persistent lesion semantics."""

import numpy as np
import pytest
import torch
from gymnasium import spaces

from fly_connectome.models.baseline import BaselineNetwork
from fly_connectome.models.connectome_layer import ConnectomeLayer
from fly_connectome.models.connectome_network import ConnectomeNetwork, graph_spec
from fly_connectome.training.train import prepare_graph
from fly_connectome.utils.config import load_config


@pytest.fixture
def network():
    config = load_config("configs/smoke.yaml")
    spec = graph_spec(prepare_graph(config), config["model"])
    return ConnectomeNetwork(spaces.Box(-1.0, 1.0, (13,), np.float32), spec)


def test_forward_and_gradients(network):
    observations = torch.randn(4, 13)
    outputs = network(observations)
    assert outputs.shape == (4, 16) and torch.isfinite(outputs).all()
    outputs.square().sum().backward()
    assert network.layer.edge_weights.grad is not None
    assert network.layer.edge_weights.grad.abs().sum() > 0
    assert network.encoder.weight.grad.abs().sum() > 0
    assert not any(parameter.shape == (64, 64) for parameter in network.parameters())


def test_direction_and_no_extra_edges():
    layer = ConnectomeLayer(3, [0], [1], [1.0], gain=1.0)
    actual = layer(torch.tensor([[1.0, 0.0, 0.0]]), torch.zeros(1, 3))
    expected = torch.tensor([[0.0, np.tanh(1.0), 0.0]], dtype=torch.float32)
    torch.testing.assert_close(actual, expected)


def test_explicit_recurrent_state_and_batch_isolation(network):
    x = torch.randn(1, 13)
    y, state = network.evolve(x)
    z, _ = network.evolve(x, state)
    assert not torch.allclose(y, z)
    torch.testing.assert_close(network(x), network(torch.cat([x, torch.randn_like(x)]))[:1])
    torch.testing.assert_close(network(x), network(x))


def test_fixed_edge_weights():
    layer = ConnectomeLayer(2, [0, 1], [1, 0], [1.0, 1.0], train_edge_weights=False)
    assert not layer.edge_weights.requires_grad and layer.bias.requires_grad


@pytest.mark.parametrize("kind", ["neuron", "edge"])
def test_damage_reproducible_and_clear(network, kind):
    x = torch.randn(2, 13)
    intact = network(x).detach()
    network.set_damage(0.5, 42, kind)
    mask = network.layer.neuron_mask if kind == "neuron" else network.layer.edge_mask
    assert (mask == 0).sum() == int(mask.numel() * 0.5)
    damaged = network(x).detach()
    network.set_damage(0.5, 42, kind)
    torch.testing.assert_close(network(x), damaged)
    network.clear_damage()
    torch.testing.assert_close(network(x), intact)


def test_total_neuron_damage(network):
    network.set_damage(1.0, 42)
    assert torch.count_nonzero(network(torch.randn(2, 13))) == 0


def test_high_degree_damage():
    layer = ConnectomeLayer(4, [0, 0, 0], [1, 2, 3], [1, 1, 1])
    layer.set_damage(0.25, 5, strategy="high_degree")
    assert layer.neuron_mask.tolist() == [0, 1, 1, 1]


def test_baseline_damage():
    model = BaselineNetwork(spaces.Box(-1.0, 1.0, (13,), np.float32), [8, 4])
    x = torch.randn(2, 13)
    intact = model(x).detach()
    model.set_damage(0.5, 8)
    assert sum(int((layer.neuron_mask == 0).sum()) for layer in model.layers) == 6
    model.set_damage(1.0, 8)
    assert torch.count_nonzero(model(x)) == 0
    model.clear_damage()
    torch.testing.assert_close(model(x), intact)
    model.set_damage(0.2, 8, "edge")
    assert sum(int((layer.edge_mask == 0).sum()) for layer in model.layers) == int(
        (13 * 8 + 8 * 4) * 0.2
    )
