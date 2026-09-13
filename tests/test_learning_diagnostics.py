"""Regression tests for meaningful behavior and legacy checkpoint semantics."""

import numpy as np
import pandas as pd
import pytest
import torch

from fly_connectome.environment import FruitFlySurvivalEnv
from fly_connectome.environment.agent import Agent
from fly_connectome.environment.reward import potential
from fly_connectome.training.diagnostics import ReferencePolicy, learning_health
from fly_connectome.training.evaluation import evaluate
from fly_connectome.utils.config import load_config, validate_config


@pytest.mark.parametrize("action", [2, 3])
def test_moving_turns_have_translation_and_movement_cost(action):
    cfg = load_config()["environment"]
    cfg["turn_speed_fraction"] = 0.65
    cfg["idle_energy_cost"] = 1
    cfg["move_energy_cost"] = 3
    agent = Agent(np.array([30.0, 30.0]), 0.0, 100.0)
    delta = agent.act(action, cfg)
    assert np.linalg.norm(delta) == pytest.approx(cfg["agent_speed"] * 0.65 * cfg["dt"])
    assert delta[1] < 0 if action == 2 else delta[1] > 0
    assert agent.energy_spent == pytest.approx(3 * cfg["dt"])


def test_legacy_turn_semantics_are_preserved():
    cfg = load_config()["environment"]
    cfg.pop("turn_speed_fraction", None)
    agent = Agent(np.array([30.0, 30.0]), 0.0, 100.0)
    np.testing.assert_array_equal(agent.act(2, cfg), np.zeros(2))
    assert agent.energy_spent == pytest.approx(cfg["idle_energy_cost"] * cfg["dt"])


def test_behavior_counters_measure_actual_movement():
    cfg = load_config()
    cfg["environment"].update(obstacle_count=0, predator_speed=0, turn_speed_fraction=0.65)
    env = FruitFlySurvivalEnv(cfg)
    env.reset(seed=12)
    env.agent.position = np.array([30.0, 30.0])
    env.predator.position = np.array([90.0, 60.0])
    env.step(0)
    _, _, _, _, info = env.step(1)
    assert info["action_0_count"] == info["action_1_count"] == 1
    assert sum(info[f"action_{i}_count"] for i in range(5)) == env.agent.age
    assert info["stationary_fraction"] == 0.5
    _, info = env.reset(seed=12)
    assert info["stationary_fraction"] == 0
    assert sum(info[f"action_{i}_count"] for i in range(5)) == 0


def test_heading_is_bounded_state_potential():
    cfg = load_config()
    cfg["reward"].update(food_potential=0, danger_potential=0, heading_potential=2)
    env = FruitFlySurvivalEnv(cfg)
    env.reset(seed=12)
    env.food = [env.agent.position + [10, 0]]
    env.agent.heading = 0
    assert potential(env) == pytest.approx(2)
    env.agent.heading = np.pi
    assert potential(env) == pytest.approx(-2)


@pytest.mark.parametrize(
    "distances,food,status",
    [
        ([0, 0], [0, 0], "stationary_policy"),
        ([10, 20], [0, 0], "no_foraging_observed"),
        ([10, 20], [1, 0], "foraging_observed"),
    ],
)
def test_health_does_not_confuse_movement_with_foraging(distances, food, status):
    result = learning_health(
        pd.DataFrame({"distance_travelled": distances, "food_collected": food})
    )
    assert result["status"] == status
    assert result["episodes"] == 2


def test_empty_or_nonfinite_health_is_not_a_success():
    for frame in [
        pd.DataFrame(),
        pd.DataFrame({"distance_travelled": [np.nan], "food_collected": [0]}),
    ]:
        with pytest.raises(ValueError):
            learning_health(frame)


def test_development_seed_leakage_rejected():
    cfg = load_config()
    cfg["training"]["validation_seeds"] = cfg["experiments"]["eval_seeds"]
    with pytest.raises(ValueError, match="disjoint"):
        validate_config(cfg)


@pytest.mark.parametrize("fraction", [-0.1, 1.1])
def test_invalid_turn_speed(fraction):
    cfg = load_config()
    cfg["environment"]["turn_speed_fraction"] = fraction
    with pytest.raises(ValueError, match="turn_speed_fraction"):
        validate_config(cfg)


def test_stochastic_evaluation_is_paired_and_restores_torch_rng():
    class SampledPolicy:
        def predict(self, observation, deterministic=True):
            return (0 if deterministic else int(torch.randint(5, (1,)).item())), None

    cfg = load_config("configs/smoke.yaml")
    before = torch.random.get_rng_state().clone()
    first = evaluate(SampledPolicy(), cfg, [301, 302], deterministic=False)
    torch.testing.assert_close(torch.random.get_rng_state(), before, rtol=0, atol=0)
    second = evaluate(SampledPolicy(), cfg, [301, 302], deterministic=False)
    pd.testing.assert_frame_equal(first, second)
    assert first.policy_mode.eq("stochastic").all()
    assert (first.action_0_count < first.episode_length).all()
    deterministic = evaluate(SampledPolicy(), cfg, [301, 302])
    assert deterministic.action_0_count.equals(deterministic.episode_length)


def test_reference_controls_reset_actions_per_episode():
    cfg = load_config("configs/smoke.yaml")
    policy = ReferencePolicy("random")
    first = evaluate(policy, cfg, [301, 302])
    second = evaluate(policy, cfg, [302])
    pd.testing.assert_frame_equal(first.iloc[[1]].reset_index(drop=True), second)
    idle = evaluate(ReferencePolicy("idle"), cfg, [301])
    assert idle.stationary_fraction.iloc[0] == 1


def test_heading_potential_telescopes_in_discounted_closed_cycle():
    from fly_connectome.environment.reward import compute_reward

    cfg = load_config()["reward"]
    cfg.update(survival=0, food=0, collision=0, capture=0, starvation=0)
    potentials = [2.0, 0.0, -2.0, 0.0, 2.0]
    total = sum(
        cfg["gamma"] ** t * compute_reward(cfg, a, b, 0, False, "", False)
        for t, (a, b) in enumerate(zip(potentials[:-1], potentials[1:], strict=True))
    )
    assert total == pytest.approx(cfg["gamma"] ** 4 * potentials[-1] - potentials[0])


def test_predator_safety_potential_matches_difficulty_detection_radius():
    cfg = load_config()
    cfg["environment"]["difficulty"] = "hard"
    cfg["reward"].update(food_potential=0, heading_potential=0, danger_potential=1)
    env = FruitFlySurvivalEnv(cfg)
    env.reset(seed=12)
    radius = cfg["environment"]["predator_detection_radius"]
    scale = cfg["environment"]["difficulty_scales"]["hard"][1]
    env.predator.position = env.agent.position + [radius, 0]
    assert potential(env) == pytest.approx(-(1 - 1 / scale))
