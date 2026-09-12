"""Offline tests for the task's API and causal event semantics."""

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from fly_connectome.environment import FruitFlySurvivalEnv
from fly_connectome.environment.world import Obstacle
from fly_connectome.utils.config import load_config


@pytest.fixture
def env():
    config = load_config()
    config["environment"].update(obstacle_count=0, predator_speed=0, food_count=1)
    task = FruitFlySurvivalEnv(config)
    task.reset(seed=7)
    task.agent.position = np.array([40.0, 30.0])
    task.agent.heading = 0.0
    task.predator.position = np.array([90.0, 60.0])
    task.food = [np.array([70.0, 50.0])]
    yield task
    task.close()


def test_gymnasium_checker():
    check_env(FruitFlySurvivalEnv(), skip_render_check=True)


def test_reset_observation(env):
    obs, info = env.reset(seed=42)
    assert obs.shape == (13,) and obs.dtype == np.float32
    assert env.observation_space.contains(obs)
    assert info["food_collected"] == 0


def test_step(env):
    before = env.agent.position.copy()
    obs, reward, terminated, truncated, _ = env.step(1)
    assert env.observation_space.contains(obs)
    assert isinstance(reward, float) and not terminated and not truncated
    assert np.linalg.norm(env.agent.position - before) > 0


@pytest.mark.parametrize("action", [-1, 5, 1.5, "forward", None])
def test_invalid_action(env, action):
    with pytest.raises(ValueError):
        env.step(action)


def test_food_collection(env):
    env.food[0] = env.agent.position + [1.6, 0]
    env.agent.energy = 40
    _, reward, terminated, _, info = env.step(1)
    assert info["food_collected"] == 1 and reward > 15 and not terminated
    assert env.agent.energy > 40
    assert env.world.valid(env.food[0], env.config["food_radius"])


def test_energy_depletion(env):
    env.agent.energy = 0.01
    _, reward, terminated, truncated, info = env.step(0)
    assert terminated and not truncated and reward < -40
    assert info["termination_reason"] == "starvation"
    with pytest.raises(RuntimeError):
        env.step(0)


def test_capture(env):
    env.predator.position = env.agent.position.copy()
    _, _, terminated, truncated, info = env.step(0)
    assert terminated and not truncated and info["predator_capture"] == 1


def test_obstacle_and_no_tunneling(env):
    env.world.obstacles = [Obstacle(43, 20, 0.1, 20)]
    env.config["agent_speed"] = 100
    _, _, _, _, info = env.step(4)
    assert env.agent.position[0] < 43 - env.config["agent_radius"]
    assert info["collision_count"] == 1


def test_timeout_is_truncation(env):
    env.config["max_steps"] = 1
    _, _, terminated, truncated, info = env.step(0)
    assert truncated and not terminated and info["termination_reason"] == "timeout"


def test_seed_reproducibility():
    a, b = FruitFlySurvivalEnv(), FruitFlySurvivalEnv()
    np.testing.assert_array_equal(a.reset(seed=12)[0], b.reset(seed=12)[0])
    for action in [1, 3, 4, 0, 2] * 5:
        x, y = a.step(action), b.step(action)
        np.testing.assert_array_equal(x[0], y[0])
        assert x[1:] == y[1:]
        if x[2] or x[3]:
            break


def test_spawn_clearance():
    env = FruitFlySurvivalEnv()
    for seed in range(10):
        env.reset(seed=seed)
        for position in env.food:
            assert env.world.valid(position, env.config["food_radius"])
        assert env.world.valid(env.agent.position, env.config["agent_radius"])


def test_rgb_render(env):
    env.render_mode = "rgb_array"
    frame = env.render()
    assert frame.shape == (700, 1000, 3) and frame.dtype == np.uint8


def test_potential_shaping_telescopes():
    from fly_connectome.environment.reward import compute_reward

    config = {"survival": 0, "food": 0, "collision": 0, "capture": 0, "starvation": 0, "gamma": 0.9}
    potentials = [-0.4, -0.2, -0.5, -0.4]
    total = sum(
        config["gamma"] ** t * compute_reward(config, a, b, 0, False, "", False)
        for t, (a, b) in enumerate(zip(potentials, potentials[1:]))
    )
    assert total == pytest.approx(-potentials[0] + 0.9**3 * potentials[-1])
    assert compute_reward(config, -0.4, -0.8, 0, False, "capture", True) == 0.4


def test_corner_spawn_and_swept_collision_are_consistent(env):
    env.world.obstacles = [Obstacle(43, 30, 5, 5)]
    # This corner is outside the true circle collision, but inside the conservative box.
    assert not env.world.valid(np.array([42.3, 29.3]), 0.8)


def test_render_resolution_does_not_change_physics():
    config = load_config()
    a = FruitFlySurvivalEnv(config)
    config["environment"].update(render_width=500, render_height=350)
    b = FruitFlySurvivalEnv(config)
    np.testing.assert_array_equal(a.reset(seed=30)[0], b.reset(seed=30)[0])
    for action in (1, 4, 2, 3):
        np.testing.assert_array_equal(a.step(action)[0], b.step(action)[0])
