"""Gymnasium survival task with reproducible maps and physical contact events."""

from __future__ import annotations

from copy import deepcopy

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..utils.config import load_config
from .agent import Agent
from .predator import Predator
from .reward import compute_reward, potential
from .sensors import OBSERVATION_NAMES, observe
from .world import World


class FruitFlySurvivalEnv(gym.Env):
    """One predator, replenishing food and energy-limited discrete control."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, config: dict | None = None, render_mode: str | None = None):
        full = load_config() if config is None else deepcopy(config)
        self.config, self.reward_config = full["environment"], full["reward"]
        if render_mode not in (None, "none", *self.metadata["render_modes"]):
            raise ValueError(f"Unsupported render mode: {render_mode}")
        self.render_mode = None if render_mode == "none" else render_mode
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(-1.0, 1.0, (len(OBSERVATION_NAMES),), np.float32)
        self._renderer = None
        self._done = True

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """Reset episode RNG; map layout uses a separate deterministic seed."""
        super().reset(seed=seed)
        options = options or {}
        seeds = self.config["map_seeds"]
        self.map_seed = int(
            options.get(
                "map_seed",
                self.np_random.choice(seeds) if seeds else self.np_random.integers(0, 2**31),
            )
        )
        self.world = World(self.config, np.random.default_rng(self.map_seed))
        position = self.world.sample(self.np_random, self.config["agent_radius"])
        self.agent = Agent(
            position, float(self.np_random.uniform(-np.pi, np.pi)), self.config["initial_energy"]
        )
        self.predator = Predator(
            self.world.sample(
                self.np_random,
                self.config["predator_radius"],
                position,
                self.config["predator_spawn_distance"],
            ),
            float(self.np_random.uniform(-np.pi, np.pi)),
        )
        self.food = [self._spawn_food() for _ in range(self.config["food_count"])]
        self.collision_count = 0
        self.encounters = 0
        self.escapes = 0
        self._in_danger = False
        self.total_reward = 0.0
        self.current_reward = 0.0
        self._done = False
        return observe(self), self._info("")

    def _spawn_food(self) -> np.ndarray:
        return self.world.sample(
            self.np_random,
            self.config["food_radius"],
            self.agent.position,
            self.config["agent_radius"] + self.config["food_radius"],
            self.config["food_distribution"],
        )

    def step(self, action: int):
        """Integrate both bodies in small synchronized increments and emit events."""
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action!r}; expected integer 0..4")
        if self._done:
            raise RuntimeError("Call reset before step, including after episode end")
        cfg, agent = self.config, self.agent
        old_potential = potential(self)
        displacement = agent.act(int(action), cfg)
        pred_displacement = self.predator.displacement(agent.position, self.world, self.np_random)
        count = max(
            1,
            int(
                np.ceil(
                    max(np.linalg.norm(displacement), np.linalg.norm(pred_displacement))
                    / min(
                        cfg["collision_substep"],
                        cfg["agent_radius"],
                        cfg["food_radius"],
                        cfg["predator_radius"],
                    )
                )
            ),
        )
        collision, captured, eaten = False, False, set()
        start = agent.position.copy()
        for _ in range(count):
            agent.position, hit = self.world.move(
                agent.position, displacement / count, cfg["agent_radius"]
            )
            collision |= hit
            self.predator.position, _ = self.world.move(
                self.predator.position, pred_displacement / count, cfg["predator_radius"]
            )
            for index, position in enumerate(self.food):
                if np.linalg.norm(agent.position - position) <= (
                    cfg["agent_radius"] + cfg["food_radius"]
                ):
                    eaten.add(index)
            if np.linalg.norm(agent.position - self.predator.position) <= (
                cfg["agent_radius"] + cfg["predator_radius"]
            ):
                captured = True
                break
        travelled = float(np.linalg.norm(agent.position - start))
        agent.distance_travelled += travelled
        agent.velocity = travelled / cfg["dt"]
        agent.food_collected += len(eaten)
        agent.energy = min(cfg["max_energy"], agent.energy + cfg["food_energy_gain"] * len(eaten))
        for index in sorted(eaten):
            self.food[index] = self._spawn_food()
        self.collision_count += int(collision)
        danger = np.linalg.norm(agent.position - self.predator.position) < (
            cfg["predator_detection_radius"] * cfg["difficulty_scales"][cfg["difficulty"]][1]
        )
        if danger and not self._in_danger:
            self.encounters += 1
        if self._in_danger and not danger and not captured:
            self.escapes += 1
        self._in_danger = bool(danger)
        reason = "capture" if captured else "starvation" if agent.energy <= 0 else ""
        terminated = bool(reason)
        truncated = bool(agent.age >= cfg["max_steps"] and not terminated)
        if truncated:
            reason = "timeout"
        agent.alive = not terminated
        self._done = terminated or truncated
        reward = compute_reward(
            self.reward_config,
            old_potential,
            potential(self),
            len(eaten),
            collision,
            reason,
            terminated,
        )
        self.current_reward = reward
        self.total_reward += reward
        info = self._info(reason)
        if self.render_mode == "human":
            self.render()
        return observe(self), reward, terminated, truncated, info

    def _info(self, reason: str) -> dict:
        agent = self.agent
        return {
            "survival_time": agent.age * self.config["dt"],
            "food_collected": agent.food_collected,
            "energy_remaining": agent.energy,
            "collision_count": self.collision_count,
            "distance_travelled": agent.distance_travelled,
            "energy_spent": agent.energy_spent,
            "energy_efficiency": agent.food_collected / max(agent.energy_spent, 1e-9),
            "predator_encounters": self.encounters,
            "predator_escapes": self.escapes,
            "predator_escape_rate": self.escapes / self.encounters if self.encounters else 0.0,
            "predator_capture": int(reason == "capture"),
            "termination_reason": reason,
            "map_seed": self.map_seed,
        }

    def render(self):
        """Lazily create Pygame resources; headless training never imports Pygame."""
        if self.render_mode is None:
            return None
        if self._renderer is None:
            from ..visualization.renderer import Renderer

            self._renderer = Renderer(self.config, self.render_mode)
        return self._renderer.draw(self)

    def close(self) -> None:
        """Release Pygame resources; safe to call repeatedly."""
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
