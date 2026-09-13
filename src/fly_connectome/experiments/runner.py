"""Paired-seed experiment orchestration; no fabricated or pretrained results."""

from __future__ import annotations

import hashlib
import json
import logging
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from stable_baselines3 import PPO

from ..training.diagnostics import learning_health, log_learning_health
from ..training.evaluation import evaluate
from ..training.train import train_model
from ..utils.config import resolve_path, save_config
from ..utils.seeding import resolve_device

EXPERIMENTS = (
    "comparison",
    "generalization",
    "sensor_noise",
    "neuron_damage",
    "edge_damage",
    "ablation",
)
LOGGER = logging.getLogger(__name__)


def conditions(name: str, config: dict) -> list[dict]:
    """Enumerate independent perturbations, including an intact paired reference."""
    settings = config["experiments"]
    if name == "generalization":
        return [
            {"condition": value, "level": 0.0, "damage_seed": -1, "strategy": "none"}
            for value in (
                "in_distribution",
                "unseen_layout",
                "unseen_food",
                "unseen_speed",
                "combined",
            )
        ]
    if name == "sensor_noise":
        return [
            {"condition": "sensor_noise", "level": value, "damage_seed": -1, "strategy": "none"}
            for value in sorted({0.0, *settings["noise_levels"]})
        ]
    if name in ("neuron_damage", "edge_damage"):
        strategies = ("random", "high_degree") if name == "neuron_damage" else ("random",)
        return [
            {"condition": name, "level": value, "damage_seed": seed, "strategy": strategy}
            for strategy in strategies
            for seed in settings["damage_seeds"]
            for value in sorted({0.0, *settings[f"{name}_levels"]})
        ]
    return [{"condition": "in_distribution", "level": 0.0, "damage_seed": -1, "strategy": "none"}]


def add_retention(frame: pd.DataFrame) -> pd.DataFrame:
    """Use mean intact performance per trained model/lesion seed as the denominator.

    Reward retention is undefined when the reference reward is non-positive.
    Raw rewards and absolute changes remain available instead of misleading ratios.
    """
    keys = ["model", "train_seed", "strategy", "damage_seed"]
    metrics = ["episode_reward", "survival_time", "food_collected"]
    reference = frame[frame.level == 0].groupby(keys)[metrics].mean().add_suffix("_intact")
    result = frame.merge(reference, on=keys, validate="many_to_one")
    for metric in metrics:
        denominator = result[f"{metric}_intact"]
        result[f"{metric}_retention_pct"] = np.where(
            denominator > 0, 100 * result[metric] / denominator.replace(0, np.nan), np.nan
        )
        result[f"{metric}_change"] = result[metric] - denominator
    return result


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate episodes/lesion replicates within training seed before cross-seed SEM."""
    keys = ["experiment", "model", "condition", "level", "strategy", "data_kind"]
    metrics = [
        "episode_reward",
        "survival_time",
        "food_collected",
        "predator_capture",
        "energy_efficiency",
        "collision_count",
    ]
    metrics += [key for key in frame if key.endswith("retention_pct") or key.endswith("_change")]
    metrics += [key for key in ("distance_travelled", "stationary_fraction") if key in frame]
    by_seed = frame.groupby([*keys, "train_seed"], dropna=False)[metrics].mean()
    summary = by_seed.groupby(keys, dropna=False).agg(["mean", "std", "sem", "count"])
    summary.columns = ["_".join(column) for column in summary.columns]
    return summary.reset_index()


def run_experiments(config: dict, experiment: str = "all") -> Path:
    """Train matched agents, evaluate perturbations, and write raw/summary CSVs."""
    if experiment != "all" and experiment not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment: {experiment}")
    names = EXPERIMENTS if experiment == "all" else (experiment,)
    kinds = (
        ("baseline", "connectome", "fixed", "random_sparse")
        if "ablation" in names
        else ("baseline", "connectome")
    )
    output = resolve_path(config["output_dir"])
    saved_config = output / "experiment_config.yaml"
    if saved_config.exists() and yaml.safe_load(saved_config.read_text(encoding="utf-8")) != config:
        raise ValueError(
            "Output directory contains a different experiment protocol. "
            "Choose a new output_dir in YAML to avoid mixing results."
        )
    if "generalization" in names and (
        config["environment"]["fixed_obstacles"] is not None
        or not config["environment"]["map_seeds"]
    ):
        raise ValueError(
            "Strict unseen-map experiments require procedural maps and a finite "
            "training map seed list; use comparison for fixed maps."
        )
    csv_dir = output / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, output / "experiment_config.yaml")
    settings = config["experiments"]
    checkpoints = {}
    for kind in kinds:
        for seed in settings["train_seeds"]:
            checkpoints[(kind, seed)] = train_model(
                config, kind, seed, settings["reuse_checkpoints"]
            )
    health_rows = []
    for name in names:
        frames = []
        selected_kinds = kinds if name == "ablation" else ("baseline", "connectome")
        for kind in selected_kinds:
            for seed in settings["train_seeds"]:
                checkpoint = checkpoints[(kind, seed)]
                manifest = json.loads((checkpoint.parent / "run.json").read_text())
                model = PPO.load(checkpoint, device=resolve_device(config["device"]))
                for condition in conditions(name, config):
                    env_config = deepcopy(config)
                    maps = config["environment"]["map_seeds"]
                    label = condition["condition"]
                    if label in ("unseen_layout", "combined"):
                        maps = settings["unseen_map_seeds"]
                    if label in ("unseen_food", "combined"):
                        env_config["environment"]["food_distribution"] = settings[
                            "generalization_food_distribution"
                        ]
                    if label in ("unseen_speed", "combined"):
                        env_config["environment"]["predator_speed"] *= settings[
                            "generalization_speed_multiplier"
                        ]
                    extractor = model.policy.features_extractor
                    extractor.clear_damage()
                    if name in ("neuron_damage", "edge_damage"):
                        extractor.set_damage(
                            condition["level"],
                            condition["damage_seed"],
                            "neuron" if name == "neuron_damage" else "edge",
                            condition["strategy"],
                        )
                    frame = evaluate(
                        model,
                        env_config,
                        settings["eval_seeds"],
                        condition["level"] if name == "sensor_noise" else 0.0,
                        maps,
                    )
                    for key, value in condition.items():
                        frame[key] = value
                    frame["experiment"], frame["model"], frame["train_seed"] = name, kind, seed
                    frame["data_kind"] = manifest["provenance"]["data_kind"]
                    frame["parent_data_kind"] = manifest["provenance"].get("parent_data_kind", "")
                    frame["checkpoint_fingerprint"] = manifest["fingerprint"]
                    frame["trainable_parameters"] = manifest["trainable_parameters"]
                    frame["train_wall_seconds"] = manifest["wall_seconds"]
                    frame["train_timesteps"] = manifest["actual_timesteps"]
                    frames.append(frame)
                    LOGGER.info(
                        "Evaluated %s/%s seed=%d %s level=%.2f",
                        name,
                        kind,
                        seed,
                        condition["strategy"],
                        condition["level"],
                    )
                extractor.clear_damage()
        combined = pd.concat(frames, ignore_index=True)
        # Diagnose intact policies only; damaged failures are the measured outcome.
        intact = combined[
            (combined.level == 0)
            & combined.condition.isin(
                ["in_distribution", "sensor_noise", "neuron_damage", "edge_damage"]
            )
        ]
        for (kind, seed), episodes in intact.groupby(["model", "train_seed"]):
            episodes = episodes.drop_duplicates(subset=["eval_seed"])
            health = learning_health(episodes)
            log_learning_health(health, f"{name}/{kind}/seed{seed}/intact")
            health_rows.append(
                {"experiment": name, "model": kind, "train_seed": int(seed), **health}
            )
        if name in ("sensor_noise", "neuron_damage", "edge_damage"):
            combined = add_retention(combined)
        combined.to_csv(csv_dir / f"{name}.csv", index=False)
        summarize(combined).to_csv(csv_dir / f"{name}_summary.csv", index=False)
        pd.DataFrame(health_rows).to_csv(csv_dir / "learning_health.csv", index=False)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiments": list(names),
        "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
        "checkpoints": {f"{k}:{s}": str(p) for (k, s), p in checkpoints.items()},
        "uncertainty": "SEM across independent training seeds after within-seed averaging",
        "scope": "Software smoke test"
        if config["training"]["timesteps"] < 1000
        else "Exploratory experiment; inspect provenance before interpreting",
    }
    (output / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return csv_dir
