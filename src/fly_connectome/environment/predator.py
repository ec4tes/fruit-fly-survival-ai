"""Local pursuit with obstacle-aware steering and bounded wandering."""

import numpy as np

from .world import World


class Predator:
    """Pursue visible targets inside the detection radius, otherwise wander."""

    def __init__(self, position: np.ndarray, heading: float):
        self.position = position
        self.heading = heading

    def displacement(
        self, target: np.ndarray, world: World, rng: np.random.Generator
    ) -> np.ndarray:
        """Choose the clearest local steering direction toward the desired heading."""
        cfg = world.config
        speed_scale, sensing_scale = cfg["difficulty_scales"][cfg["difficulty"]]
        delta = target - self.position
        if np.linalg.norm(delta) <= cfg["predator_detection_radius"] * sensing_scale:
            desired = float(np.arctan2(delta[1], delta[0]))
        else:
            desired = self.heading + rng.uniform(-1, 1) * cfg["predator_wander_turn"]
        angles = desired + np.deg2rad(cfg["predator_steering_angles"])
        scores = [
            world.sense(self.position, a, cfg["predator_radius"])
            - cfg["predator_steering_penalty"]
            * abs(float((a - desired + np.pi) % (2 * np.pi) - np.pi))
            for a in angles
        ]
        self.heading = float(angles[int(np.argmax(scores))])
        return (
            np.array([np.cos(self.heading), np.sin(self.heading)])
            * cfg["predator_speed"]
            * speed_scale
            * cfg["dt"]
        )
