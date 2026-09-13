"""Behavioral checks distinguish completed optimization from demonstrated foraging."""

import logging

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


def learning_health(episodes: pd.DataFrame) -> dict:
    """Describe observed collapse without declaring convergence or statistical superiority."""
    columns = ["distance_travelled", "food_collected"]
    if (
        episodes.empty
        or not set(columns).issubset(episodes.columns)
        or not np.isfinite(episodes[columns].to_numpy()).all()
    ):
        raise ValueError("Learning diagnostics need nonempty finite episode measurements")
    stationary = episodes.distance_travelled < 1e-8
    foraged = episodes.food_collected > 0
    status = (
        "stationary_policy"
        if stationary.all()
        else "no_foraging_observed"
        if not foraged.any()
        else "foraging_observed"
    )
    return {
        "status": status,
        "episodes": len(episodes),
        "mean_food": float(episodes.food_collected.mean()),
        "mean_distance": float(episodes.distance_travelled.mean()),
        "stationary_episode_fraction": float(stationary.mean()),
        "foraging_episode_fraction": float(foraged.mean()),
        "interpretation": "Behavioral smoke check only; not convergence or robustness evidence",
    }


def log_learning_health(health: dict, label: str) -> None:
    """Warn explicitly when training has not produced any foraging behavior."""
    method = LOGGER.info if health["status"] == "foraging_observed" else LOGGER.warning
    method(
        "Learning check %s: %s; mean food=%.2f, distance=%.1f, stationary episodes=%.0f%%",
        label,
        health["status"],
        health["mean_food"],
        health["mean_distance"],
        100 * health["stationary_episode_fraction"],
    )


class ReferencePolicy:
    """Non-learning controls used to distinguish foraging from accidental exploration."""

    def __init__(self, kind: str):
        if kind not in ("idle", "left", "right", "random"):
            raise ValueError("Unknown reference policy")
        self.kind = kind
        self.reset_episode(0)

    def reset_episode(self, seed: int) -> None:
        """Use an independent, episode-local action RNG."""
        self.rng = np.random.default_rng(np.random.SeedSequence([seed, 271]))

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> tuple[int, None]:
        """Return a constant or seeded uniformly sampled action."""
        action = (
            int(self.rng.integers(5))
            if self.kind == "random"
            else {"idle": 0, "left": 2, "right": 3}[self.kind]
        )
        return action, None
