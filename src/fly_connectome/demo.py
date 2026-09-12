"""Instant heuristic demonstration, explicitly not a trained research agent."""

import logging
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from .environment import FruitFlySurvivalEnv


class HeuristicAgent:
    """Turn toward food, steer from imminent obstacles, and flee nearby predators."""

    def __init__(self, config: dict):
        self.config = config

    def predict(self, observation: np.ndarray, deterministic: bool = True):
        """Implement SB3-compatible prediction for rendering/evaluation."""
        cfg = self.config["environment"]
        angle = float(np.arctan2(observation[1], observation[2]))
        danger = observation[3] * np.hypot(cfg["width"], cfg["height"]) < (
            cfg["predator_detection_radius"] * 0.6
        )
        if observation[6] < 0.12:
            return (2 if observation[7] >= observation[8] else 3), None
        if danger:
            angle = float(np.arctan2(-observation[4], -observation[5]))
        if abs(angle) > cfg["turn_rate"] * cfg["dt"] / 2:
            return (3 if angle > 0 else 2), None
        return (4 if danger and observation[9] > 0.3 else 1), None


def run_demo(
    config: dict,
    headless: bool = False,
    steps: int = 1000,
    checkpoint: Path | None = None,
    snapshot: Path | None = None,
) -> dict:
    """Run a bounded demo, optionally saving an actual rendered RGB frame."""
    env = FruitFlySurvivalEnv(config, render_mode="rgb_array" if headless else "human")
    policy = PPO.load(checkpoint, device="cpu") if checkpoint else HeuristicAgent(config)
    logging.getLogger(__name__).info("Demo policy: %s", checkpoint or "untrained heuristic")
    observation, info = env.reset(seed=config["seed"])
    try:
        for _ in range(steps):
            action, _ = policy.predict(observation, deterministic=True)
            observation, _, terminated, truncated, info = env.step(int(action))
            if env._renderer is not None and env._renderer.quit_requested:
                break
            if terminated or truncated:
                observation, _ = env.reset()
        if snapshot:
            import pygame

            frame = (
                env.render()
                if headless
                else np.transpose(pygame.surfarray.array3d(env._renderer.surface), (1, 0, 2))
            )
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            pygame.image.save(
                pygame.surfarray.make_surface(frame.transpose(1, 0, 2)), str(snapshot)
            )
    finally:
        env.close()
    return info
