"""Directed edge-only recurrent computation, with no dense NxN parameter."""

import numpy as np
import torch
from torch import nn


class ConnectomeLayer(nn.Module):
    """One artificial tanh update using the supplied directed connectivity only."""

    def __init__(
        self,
        neuron_count: int,
        source: list,
        target: list,
        weights: list,
        train_edge_weights: bool = True,
        gain: float = 0.8,
    ):
        super().__init__()
        self.neuron_count = neuron_count
        self.gain = gain
        self.register_buffer("source", torch.as_tensor(source, dtype=torch.long))
        self.register_buffer("target", torch.as_tensor(target, dtype=torch.long))
        self.edge_weights = nn.Parameter(
            torch.as_tensor(weights, dtype=torch.float32), requires_grad=train_edge_weights
        )
        self.bias = nn.Parameter(torch.zeros(neuron_count))
        self.register_buffer("neuron_mask", torch.ones(neuron_count))
        self.register_buffer("edge_mask", torch.ones(len(source)))

    def forward(self, state: torch.Tensor, sensory: torch.Tensor) -> torch.Tensor:
        """Compute tanh(gain * A @ masked_state + sensory + bias), then lesion."""
        state = state * self.neuron_mask
        messages = state[:, self.source] * (self.edge_weights * self.edge_mask)
        recurrent = torch.zeros_like(state).index_add(1, self.target, messages)
        return torch.tanh(self.gain * recurrent + sensory + self.bias) * self.neuron_mask

    def set_damage(
        self, fraction: float, seed: int, kind: str = "neuron", strategy: str = "random"
    ) -> None:
        """Replace masks, supporting random edges or random/high-degree neurons."""
        if not 0 <= fraction <= 1 or kind not in ("neuron", "edge"):
            raise ValueError("Invalid damage fraction or kind")
        if strategy not in ("random", "high_degree") or (kind == "edge" and strategy != "random"):
            raise ValueError("Unsupported damage strategy")
        self.clear_damage()
        mask = self.neuron_mask if kind == "neuron" else self.edge_mask
        if strategy == "high_degree":
            degree = torch.bincount(
                torch.cat([self.source, self.target]), minlength=self.neuron_count
            )
            indices = np.argsort(-degree.cpu().numpy(), kind="stable")
        else:
            indices = np.random.default_rng(seed).permutation(mask.numel())
        with torch.no_grad():
            mask[
                torch.as_tensor(indices[: int(fraction * len(indices))].copy(), device=mask.device)
            ] = 0

    def clear_damage(self) -> None:
        """Restore the intact network without changing its learned parameters."""
        self.neuron_mask.fill_(1)
        self.edge_mask.fill_(1)
