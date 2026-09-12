"""Actual short PPO updates, checkpoint round-trips and paired evaluation."""

import json

import numpy as np
import pandas as pd
import pytest
from stable_baselines3 import PPO

from fly_connectome.demo import HeuristicAgent
from fly_connectome.environment import FruitFlySurvivalEnv
from fly_connectome.experiments.runner import add_retention, conditions, summarize
from fly_connectome.training.evaluation import evaluate, noisy_observation
from fly_connectome.training.train import train_model
from fly_connectome.utils.config import load_config


@pytest.mark.parametrize("kind", ["baseline", "connectome", "fixed", "random_sparse"])
def test_ppo_training_roundtrip(tmp_path, kind):
    config = load_config("configs/smoke.yaml")
    config["output_dir"] = str(tmp_path)
    config["training"]["timesteps"] = 64
    path = train_model(config, kind=kind)
    model = PPO.load(path, device="cpu")
    obs, _ = FruitFlySurvivalEnv(config).reset(seed=12)
    assert 0 <= int(model.predict(obs, deterministic=True)[0]) < 5
    metrics = pd.read_csv(path.parent / "episodes.csv")
    assert len(metrics) > 0 and "survival_time" in metrics
    run = json.loads((path.parent / "run.json").read_text())
    assert run["actual_timesteps"] == 64 and run["trainable_parameters"] > 0
    model.policy.features_extractor.set_damage(0.2, 5)
    damaged = model.policy.features_extractor
    model.save(tmp_path / "damaged.zip")
    restored = PPO.load(tmp_path / "damaged.zip")
    for a, b in zip(damaged.buffers(), restored.policy.features_extractor.buffers(), strict=True):
        np.testing.assert_array_equal(a.cpu().numpy(), b.cpu().numpy())
    if kind == "random_sparse":
        from fly_connectome.models.connectome_network import graph_spec
        from fly_connectome.training.train import prepare_graph

        reference = graph_spec(prepare_graph(config), config["model"])
        assert model.policy.features_extractor.spec["input_body_ids"] == reference["input_body_ids"]
        assert (
            model.policy.features_extractor.spec["output_body_ids"] == reference["output_body_ids"]
        )


def test_paired_evaluation_noise_seed():
    config = load_config("configs/smoke.yaml")
    policy = HeuristicAgent(config)
    a = evaluate(policy, config, [101, 102], noise=0.1, map_seeds=[17, 18])
    b = evaluate(policy, config, [101, 102], noise=0.1, map_seeds=[17, 18])
    pd.testing.assert_frame_equal(a, b)
    assert a.map_seed.tolist() == [17, 18]
    obs = np.zeros(13, np.float32)
    noisy = noisy_observation(obs, 1.0, np.random.default_rng(1))
    assert noisy.min() >= -1 and noisy.max() <= 1


def test_training_seed_reproducibility_with_vector_envs(tmp_path):
    import torch

    config = load_config("configs/smoke.yaml")
    config["training"].update(timesteps=64, n_envs=2, n_steps=32)
    config["environment"]["max_steps"] = 32
    config["output_dir"] = str(tmp_path / "a")
    a = PPO.load(train_model(config, kind="baseline"), device="cpu")
    config["output_dir"] = str(tmp_path / "b")
    b = PPO.load(train_model(config, kind="baseline"), device="cpu")
    for x, y in zip(a.policy.parameters(), b.policy.parameters(), strict=True):
        torch.testing.assert_close(x, y, rtol=0, atol=0)


def test_negative_reward_retention_not_misleading():
    rows = pd.DataFrame(
        {
            "model": ["baseline"] * 2,
            "train_seed": [1, 1],
            "strategy": ["random"] * 2,
            "damage_seed": [1, 1],
            "level": [0, 0.5],
            "episode_reward": [-10, -20],
            "survival_time": [20, 10],
            "food_collected": [0, 0],
        }
    )
    result = add_retention(rows)
    assert result.episode_reward_retention_pct.isna().all()
    assert result.food_collected_retention_pct.isna().all()
    assert result.survival_time_retention_pct.tolist() == [100, 50]


def test_conditions_include_intact_and_distinct_ood():
    config = load_config("configs/smoke.yaml")
    for experiment in ["sensor_noise", "neuron_damage", "edge_damage"]:
        assert any(c["level"] == 0 for c in conditions(experiment, config))
    assert len(conditions("generalization", config)) == 5


def test_summary_uses_training_seeds_as_replicates():
    frame = pd.DataFrame(
        [
            {
                "experiment": "comparison",
                "model": "baseline",
                "condition": "in_distribution",
                "level": 0,
                "strategy": "none",
                "data_kind": "not_applicable",
                "train_seed": seed,
                "episode_reward": reward,
                "survival_time": 1,
                "food_collected": 1,
                "predator_capture": 0,
                "energy_efficiency": 1,
                "collision_count": 0,
            }
            for seed, reward in [(1, 1), (1, 3), (2, 10), (2, 10)]
        ]
    )
    summary = summarize(frame)
    assert summary.episode_reward_mean.iloc[0] == 6
    assert summary.episode_reward_count.iloc[0] == 2
