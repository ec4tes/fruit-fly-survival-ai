"""Agent state and discrete action semantics in logical world units."""

from dataclasses import dataclass

import numpy as np


@dataclass
class Agent:
    """Mutable state for one episode; velocity is scalar speed per second."""

    position: np.ndarray
    heading: float
    energy: float
    velocity: float = 0.0
    alive: bool = True
    age: int = 0
    food_collected: int = 0
    distance_travelled: float = 0.0
    energy_spent: float = 0.0

    def act(self, action: int, config: dict) -> np.ndarray:
        """Apply idle/forward/left/right/sprint and return proposed displacement."""
        dt = config["dt"]
        if action in (2, 3):
            self.heading += (1 if action == 3 else -1) * config["turn_rate"] * dt
        self.heading = float((self.heading + np.pi) % (2 * np.pi) - np.pi)
        self.velocity = config["agent_speed"] if action in (1, 4) else 0.0
        if action == 4:
            self.velocity *= config["sprint_multiplier"]
        cost_key = {1: "move_energy_cost", 4: "sprint_energy_cost"}.get(action, "idle_energy_cost")
        cost = config[cost_key] * dt
        self.energy = max(0.0, self.energy - cost)
        self.energy_spent += cost
        self.age += 1
        return np.array([np.cos(self.heading), np.sin(self.heading)]) * self.velocity * dt
