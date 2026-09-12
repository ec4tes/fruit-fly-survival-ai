"""Event rewards plus bounded potential shaping with terminal correction."""

import numpy as np


def potential(env) -> float:
    """Food proximity and predator safety potential of the complete state."""
    distance = min(np.linalg.norm(p - env.agent.position) for p in env.food)
    food = -distance / np.linalg.norm(env.world.size)
    danger_distance = np.linalg.norm(env.predator.position - env.agent.position)
    radius = env.config["predator_detection_radius"]
    danger = -max(0.0, 1 - danger_distance / max(radius, 1e-9))
    return float(
        env.reward_config["food_potential"] * food + env.reward_config["danger_potential"] * danger
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
