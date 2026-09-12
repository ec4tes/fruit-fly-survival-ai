"""Fixed-size normalized observations; sin/cos avoid angular discontinuities."""

import numpy as np

OBSERVATION_NAMES = (
    "food_distance",
    "food_sin",
    "food_cos",
    "predator_distance",
    "predator_sin",
    "predator_cos",
    "obstacle_front",
    "obstacle_left",
    "obstacle_right",
    "energy",
    "speed",
    "heading_sin",
    "heading_cos",
)


def observe(env) -> np.ndarray:
    """Return a 13-vector in [-1, 1] from current world state."""
    agent, cfg = env.agent, env.config
    food = min(env.food, key=lambda p: np.linalg.norm(p - agent.position))
    values = []
    for target in (food, env.predator.position):
        delta = target - agent.position
        angle = np.arctan2(delta[1], delta[0]) - agent.heading
        values.extend(
            [np.linalg.norm(delta) / np.linalg.norm(env.world.size), np.sin(angle), np.cos(angle)]
        )
    values.extend(
        env.world.sense(agent.position, agent.heading + offset, cfg["agent_radius"])
        for offset in (0, -np.pi / 4, np.pi / 4)
    )
    values.extend(
        [
            agent.energy / cfg["max_energy"],
            agent.velocity / (cfg["agent_speed"] * cfg["sprint_multiplier"]),
            np.sin(agent.heading),
            np.cos(agent.heading),
        ]
    )
    return np.clip(np.asarray(values, dtype=np.float32), -1, 1)
