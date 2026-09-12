"""Paired episode evaluation with independent observation-noise RNG."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..environment import FruitFlySurvivalEnv
from ..environment.sensors import OBSERVATION_NAMES


def noisy_observation(
    observation: np.ndarray, level: float, rng: np.random.Generator
) -> np.ndarray:
    """Gaussian sigma = level * feature range; clip to each feature's valid bounds."""
    lower = np.array(
        [-1 if "sin" in name or "cos" in name else 0 for name in OBSERVATION_NAMES],
        dtype=np.float32,
    )
    upper = np.ones_like(lower)
    return np.clip(
        observation + rng.normal(size=observation.shape) * level * (upper - lower), lower, upper
    ).astype(np.float32)


def evaluate(
    model, config: dict, seeds: list[int], noise: float = 0.0, map_seeds: list[int] | None = None
) -> pd.DataFrame:
    """Run deterministic policies on explicitly paired RNG seeds and map layouts."""
    if not seeds or not 0 <= noise <= 1:
        raise ValueError("Evaluation needs seeds and noise in [0, 1]")
    rows = []
    env = FruitFlySurvivalEnv(config)
    try:
        for index, seed in enumerate(seeds):
            noise_rng = np.random.default_rng(np.random.SeedSequence([seed, 991]))
            options = {"map_seed": map_seeds[index % len(map_seeds)]} if map_seeds else None
            obs, _ = env.reset(seed=seed, options=options)
            reward_sum, length = 0.0, 0
            while True:
                sensed = noisy_observation(obs, noise, noise_rng) if noise else obs
                action, _ = model.predict(sensed, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(int(action))
                reward_sum += reward
                length += 1
                if terminated or truncated:
                    break
            rows.append(
                {
                    "eval_seed": seed,
                    "episode_reward": reward_sum,
                    "episode_length": length,
                    "terminated": terminated,
                    "truncated": truncated,
                    **info,
                }
            )
    finally:
        env.close()
    return pd.DataFrame(rows)
