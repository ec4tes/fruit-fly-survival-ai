"""Event rewards plus bounded potential shaping with terminal correction."""

import numpy as np


def potential(env) -> float:
    """Bounded proximity, safety and optional food-heading potential of the state."""
    target = min(env.food, key=lambda p: np.linalg.norm(p - env.agent.position))
    delta = target - env.agent.position
    distance = np.linalg.norm(delta)
    food = -distance / np.linalg.norm(env.world.size)
    angle = np.arctan2(delta[1], delta[0]) - env.agent.heading
    heading = float(np.cos(angle))
    danger_distance = np.linalg.norm(env.predator.position - env.agent.position)
    radius = (
        env.config["predator_detection_radius"]
        * env.config["difficulty_scales"][env.config["difficulty"]][1]
    )
    danger = -max(0.0, 1 - danger_distance / max(radius, 1e-9))
    return float(
        env.reward_config["food_potential"] * food
        + env.reward_config["danger_potential"] * danger
        + env.reward_config.get("heading_potential", 0.0) * heading
    )


def compute_reward(
    config: dict,
    previous: float,
    current: float,
    food: int,
    collision: bool,
    reason: str,
    terminated: bool,
) -> float:
    """Use gamma*Phi(s')-Phi(s); true terminal potential is zero, timeouts bootstrap."""
    reward = config["survival"] + config["food"] * food
    reward += config["collision"] * collision
    reward += config["capture"] if reason == "capture" else 0
    reward += config["starvation"] if reason == "starvation" else 0
    return float(reward + config["gamma"] * (0 if terminated else current) - previous)
