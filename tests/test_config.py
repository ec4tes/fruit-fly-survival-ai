"""Validate all shipped configurations and inheritance errors."""

from copy import deepcopy

import pytest

from fly_connectome.utils.config import ROOT, load_config, merge, validate_config


@pytest.mark.parametrize("path", sorted((ROOT / "configs").glob("*.yaml")))
def test_shipped_configs(path):
    config = load_config(path)
    assert config["environment"]["width"] == 100


def test_merge_not_mutating():
    base = {"x": {"a": 1, "b": 2}}
    merged = merge(base, {"x": {"a": 7}})
    assert merged == {"x": {"a": 7, "b": 2}} and base["x"]["a"] == 1


def test_cycle(tmp_path):
    (tmp_path / "a.yaml").write_text("extends: b.yaml")
    (tmp_path / "b.yaml").write_text("extends: a.yaml")
    with pytest.raises(ValueError, match="Cyclic"):
        load_config(tmp_path / "a.yaml")


def test_no_seed_leakage():
    config = deepcopy(load_config())
    config["experiments"]["unseen_map_seeds"] = [11]
    with pytest.raises(ValueError, match="disjoint"):
        validate_config(config)


def test_reward_gamma_must_match():
    config = load_config()
    config["reward"]["gamma"] = 0.9
    with pytest.raises(ValueError, match="gamma"):
        validate_config(config)
