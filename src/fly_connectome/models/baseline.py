"""Shared dense MLP extractor with inspectable neuron/weight lesion masks."""

import numpy as np
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn
from torch.nn import functional as functional


class MaskedDense(nn.Module):
    """Linear+tanh layer; masks affect inference but do not overwrite trained weights."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.register_buffer("neuron_mask", torch.ones(out_features))
        self.register_buffer("edge_mask", torch.ones(out_features, in_features))

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Evaluate masked weights and zero damaged hidden activations."""
        return (
            torch.tanh(
                functional.linear(
                    observations, self.linear.weight * self.edge_mask, self.linear.bias
                )
            )
            * self.neuron_mask
        )


class BaselineNetwork(BaseFeaturesExtractor):
    """Configurable MLP with a global fraction of hidden units damaged per trial."""

    def __init__(self, observation_space, layers: list[int]):
        if not layers or min(layers) < 1:
            raise ValueError("MLP layers must be nonempty positive integers")
        super().__init__(observation_space, features_dim=layers[-1])
        sizes = [observation_space.shape[0], *layers]
        self.layers = nn.ModuleList([MaskedDense(a, b) for a, b in zip(sizes, sizes[1:])])

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Extract hidden features for shared linear policy/value readouts."""
        for layer in self.layers:
            observations = layer(observations)
        return observations

    def clear_damage(self) -> None:
        """Remove lesions without changing learned parameters."""
        for layer in self.layers:
            layer.neuron_mask.fill_(1)
            layer.edge_mask.fill_(1)

    def set_damage(
        self, fraction: float, seed: int, kind: str = "neuron", strategy: str = "random"
    ) -> None:
        """Damage a global fraction; degree ranking uses hidden-layer incident edges."""
        if not 0 <= fraction <= 1 or kind not in ("neuron", "edge"):
            raise ValueError("Invalid damage fraction or kind")
        if strategy not in ("random", "high_degree") or (kind == "edge" and strategy != "random"):
            raise ValueError("Unsupported damage strategy")
        self.clear_damage()
        masks = [
            layer.neuron_mask if kind == "neuron" else layer.edge_mask for layer in self.layers
        ]
        total = sum(mask.numel() for mask in masks)
        if strategy == "random":
            order = np.random.default_rng(seed).permutation(total)
        else:
            degrees = []
            for i, layer in enumerate(self.layers):
                # Readout edges excluded, matching graph-only degree selection.
                outgoing = self.layers[i + 1].linear.out_features if i + 1 < len(self.layers) else 0
                degrees.extend([layer.linear.in_features + outgoing] * layer.linear.out_features)
            # Random tie-breaking avoids privileging a neuron's array index.
            rng = np.random.default_rng(seed)
            order = np.lexsort((rng.random(total), -np.asarray(degrees)))
        damaged = np.zeros(total, dtype=bool)
        damaged[order[: int(fraction * total)]] = True
        offset = 0
        with torch.no_grad():
            for mask in masks:
                count = mask.numel()
                mask.copy_(
                    torch.as_tensor(~damaged[offset : offset + count], device=mask.device).reshape(
                        mask.shape
                    )
                )
                offset += count
