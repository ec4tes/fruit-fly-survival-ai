"""Recursive YAML configuration with repository-relative resource paths."""

from __future__ import annotations

from copy import deepcopy
from math import isfinite
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[3]


def merge(base: dict, override: dict) -> dict:
    """Deeply merge dictionaries without mutating either argument."""
    result = deepcopy(base)
    for key, value in override.items():
        result[key] = (
            merge(result[key], value)
            if isinstance(value, dict) and isinstance(result.get(key), dict)
            else deepcopy(value)
        )
    return result


def resolve_path(path: str | Path) -> Path:
    """Resolve repository resource paths independently of the working directory."""
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def load_config(path: str | Path = "configs/default.yaml", _seen: set | None = None) -> dict:
    """Load inherited YAML, detect cycles and validate experiment settings."""
    path = resolve_path(path).resolve()
    seen = set() if _seen is None else set(_seen)
    if path in seen:
        raise ValueError(f"Cyclic config inheritance: {path}")
    seen.add(path)
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Config must be a YAML mapping")
    parent = config.pop("extends", None)
    if parent:
        config = merge(load_config(path.parent / parent, seen), config)
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    """Fail early on unsupported or physically invalid settings."""
    env, train, model = (config[k] for k in ("environment", "training", "model"))

    def finite_values(value):
        if isinstance(value, dict):
            return all(finite_values(item) for item in value.values())
        if isinstance(value, list):
            return all(finite_values(item) for item in value)
        return not isinstance(value, (int, float)) or isfinite(value)

    if not finite_values(config):
        raise ValueError("Configuration numeric values must be finite")
    positive = [
        "width",
        "height",
        "dt",
        "max_steps",
        "initial_energy",
        "max_energy",
        "agent_radius",
        "predator_radius",
        "food_radius",
        "sensor_range",
        "collision_substep",
        "agent_speed",
        "spawn_attempts",
        "food_count",
    ]
    for key in positive:
        if env[key] <= 0:
            raise ValueError(f"environment.{key} must be positive")
    if env["initial_energy"] > env["max_energy"]:
        raise ValueError("initial_energy cannot exceed max_energy")
    for key in ("max_steps", "spawn_attempts", "food_count", "obstacle_count"):
        if not isinstance(env[key], int) or env[key] < (0 if key == "obstacle_count" else 1):
            raise ValueError(f"environment.{key} must be a valid integer count")
    for key in ("timesteps", "n_envs", "n_steps", "n_epochs", "batch_size"):
        if not isinstance(train[key], int) or train[key] < 1:
            raise ValueError(f"training.{key} must be a positive integer")
    if train["learning_rate"] <= 0 or not 0 < train["gamma"] <= 1:
        raise ValueError("Invalid PPO learning rate or gamma")
    if not 0 <= train["gae_lambda"] <= 1:
        raise ValueError("gae_lambda must be in [0, 1]")
    if env["predator_speed"] < 0 or env["predator_detection_radius"] <= 0:
        raise ValueError("Predator speed must be nonnegative and sensing radius positive")
    if env["sprint_multiplier"] < 1 or env["turn_rate"] <= 0:
        raise ValueError("Sprint multiplier must be >= 1 and turn_rate positive")
    if not env["predator_steering_angles"]:
        raise ValueError("Predator needs at least one steering angle")
    if (
        len(env["obstacle_size"]) != 2
        or env["obstacle_size"][0] <= 0
        or env["obstacle_size"][0] > env["obstacle_size"][1]
    ):
        raise ValueError("obstacle_size must contain positive ordered min/max sizes")
    for key in ["idle_energy_cost", "move_energy_cost", "sprint_energy_cost"]:
        if env[key] <= 0:
            raise ValueError("Positive energy drain is required to prevent infinite idling")
    if env["difficulty"] not in env["difficulty_scales"]:
        raise ValueError("Unknown difficulty")
    if env["food_distribution"] not in ("uniform", "clustered"):
        raise ValueError("Unknown food distribution")
    if config["device"] not in ("cpu", "cuda", "auto"):
        raise ValueError("device must be cpu, cuda or auto")
    if model["kind"] not in ("baseline", "connectome", "fixed", "random_sparse"):
        raise ValueError("Unknown model kind")
    if min(model["subgraph_size"], model["dynamics_steps"]) < 1:
        raise ValueError("Graph size and dynamics_steps must be positive")
    for key in ("subgraph_size", "dynamics_steps", "input_count", "output_count"):
        if not isinstance(model[key], int) or model[key] < 1:
            raise ValueError(f"model.{key} must be a positive integer")
    if config["torch_threads"] < 1:
        raise ValueError("torch_threads must be positive")
    rollout = train["n_steps"] * train["n_envs"]
    if train["batch_size"] < 2 or rollout < train["batch_size"]:
        raise ValueError("PPO requires 2 <= batch_size <= rollout size")
    if rollout % train["batch_size"]:
        raise ValueError("batch_size must divide rollout size")
    if config["reward"]["gamma"] != train["gamma"]:
        raise ValueError("Potential shaping gamma must match PPO gamma")
    exp = config["experiments"]
    for seeds in (
        exp["train_seeds"],
        exp["eval_seeds"],
        exp["damage_seeds"],
        exp["unseen_map_seeds"],
        env["map_seeds"],
    ):
        if seeds is None:
            continue
        if (
            not seeds
            or len(seeds) != len(set(seeds))
            or any(not isinstance(s, int) or not 0 <= s < 2**32 for s in seeds)
        ):
            raise ValueError("Seed lists must be nonempty, unique uint32 integers")
    if set(env["map_seeds"] or []) & set(exp["unseen_map_seeds"]):
        raise ValueError("Training and unseen map seeds must be disjoint")
    if set(exp["train_seeds"]) & set(exp["eval_seeds"]):
        raise ValueError("Training and evaluation RNG seeds must be disjoint")
    for key in ("noise_levels", "neuron_damage_levels", "edge_damage_levels"):
        if any(not 0 <= level <= 1 for level in exp[key]):
            raise ValueError(f"{key} must contain fractions in [0, 1]")


def save_config(config: dict, path: Path) -> None:
    """Save the complete resolved configuration beside an experiment."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
