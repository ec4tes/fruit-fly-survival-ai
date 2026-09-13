"""One PPO implementation and budget for every architecture."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
import platform
import time
from copy import deepcopy
from pathlib import Path

import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from ..connectome.graph import load_connectome, save_processed_graph
from ..connectome.selector import select_subgraph
from ..connectome.statistics import graph_statistics
from ..environment import FruitFlySurvivalEnv
from ..models.baseline import BaselineNetwork
from ..models.connectome_network import ConnectomeNetwork, graph_spec
from ..models.random_sparse import random_sparse_control
from ..utils.config import resolve_path, save_config, validate_config
from ..utils.seeding import resolve_device, seed_everything
from .callbacks import EpisodeMetricsCallback
from .diagnostics import ReferencePolicy, learning_health, log_learning_health
from .evaluation import evaluate

LOGGER = logging.getLogger(__name__)


def prepare_graph(config: dict):
    """Require explicit synthetic opt-in and retain selected original body IDs."""
    settings = config["model"]
    graph = load_connectome(resolve_path(settings["graph_path"]))
    if graph.provenance["data_kind"] != "real":
        if not settings["allow_synthetic"]:
            raise ValueError("Non-biological dataset blocked: explicitly use configs/offline.yaml")
        LOGGER.warning(
            "SYNTHETIC/CONTROL source: software demonstration only; not biological evidence"
        )
    graph = select_subgraph(
        graph,
        settings["subgraph_size"],
        settings["selection"],
        settings["selection_seed"],
        settings["seed_neurons"],
        settings["biological_filters"],
    )
    return graph


def train_model(
    config: dict, kind: str | None = None, seed: int | None = None, reuse: bool = False
) -> Path:
    """Train or reuse a fingerprinted checkpoint; save configuration and provenance."""
    config = deepcopy(config)
    if kind is not None:
        config["model"]["kind"] = kind
    if seed is not None:
        config["seed"] = seed
    kind, seed = config["model"]["kind"], config["seed"]
    if kind == "fixed":
        config["model"]["train_edge_weights"] = False
    validate_config(config)
    seed_everything(seed, config["torch_threads"])
    graph = None if kind == "baseline" else prepare_graph(config)
    policy_kwargs = {"net_arch": [], "ortho_init": False}
    if graph is None:
        policy_kwargs.update(
            features_extractor_class=BaselineNetwork,
            features_extractor_kwargs={"layers": config["model"]["mlp_layers"]},
        )
        provenance = {"data_kind": "not_applicable", "architecture": "dense_mlp"}
        spec = None
    else:
        if kind == "random_sparse":
            reference_spec = graph_spec(graph, config["model"])
            config["model"]["input_neurons"] = reference_spec["input_body_ids"]
            config["model"]["output_neurons"] = reference_spec["output_body_ids"]
            graph = random_sparse_control(graph, seed)
        spec = graph_spec(graph, config["model"], allow_unreachable=kind == "random_sparse")
        policy_kwargs.update(
            features_extractor_class=ConnectomeNetwork, features_extractor_kwargs={"spec": spec}
        )
        provenance = graph.provenance
    versions = {
        name: importlib.metadata.version(name)
        for name in ("torch", "gymnasium", "stable-baselines3", "numpy", "neuprint-python")
    }
    code_digest = hashlib.sha256()
    source_root = Path(__file__).resolve().parents[1]
    for source_file in sorted(source_root.rglob("*.py")):
        code_digest.update(source_file.relative_to(source_root).as_posix().encode())
        code_digest.update(source_file.read_bytes())
    fingerprint = hashlib.sha256(
        json.dumps(
            {
                "config": config,
                "graph": spec,
                "provenance": provenance,
                "source_sha256": code_digest.hexdigest(),
                "versions": versions,
                "schema": 1,
            },
            sort_keys=True,
            default=str,
        ).encode()
    ).hexdigest()
    folder = resolve_path(config["output_dir"]) / "models" / f"{kind}_seed{seed}_{fingerprint[:12]}"
    checkpoint = folder / "model.zip"
    if reuse and checkpoint.exists() and (folder / "run.json").exists():
        LOGGER.info("Reusing %s", checkpoint)
        return checkpoint
    folder.mkdir(parents=True, exist_ok=True)
    save_config(config, folder / "config.yaml")
    if graph is not None:
        save_processed_graph(graph, folder / "graph")
        (folder / "graph_stats.json").write_text(json.dumps(graph_statistics(graph), indent=2))
    env = DummyVecEnv(
        [lambda: Monitor(FruitFlySurvivalEnv(config)) for _ in range(config["training"]["n_envs"])]
    )
    settings = config["training"]
    arguments = {
        key: settings[key]
        for key in (
            "n_steps",
            "batch_size",
            "n_epochs",
            "learning_rate",
            "gamma",
            "gae_lambda",
            "clip_range",
            "ent_coef",
            "vf_coef",
            "max_grad_norm",
        )
    }
    callback = EpisodeMetricsCallback(folder / "episodes.csv", kind, seed)
    started = time.perf_counter()
    try:
        model = PPO(
            "MlpPolicy",
            env,
            policy_kwargs=policy_kwargs,
            seed=seed,
            device=resolve_device(config["device"]),
            verbose=0,
            tensorboard_log=str(folder / "tensorboard") if settings["tensorboard"] else None,
            **arguments,
        )
        model.learn(
            total_timesteps=settings["timesteps"],
            callback=callback,
            log_interval=settings["log_interval"],
        )
        model.save(checkpoint)
        training_seconds = time.perf_counter() - started
        validation_health = {}
        if settings.get("validation_seeds"):
            frames = []
            for deterministic in (True, False):
                frame = evaluate(
                    model,
                    config,
                    settings["validation_seeds"],
                    map_seeds=config["environment"]["map_seeds"],
                    deterministic=deterministic,
                )
                mode = "deterministic" if deterministic else "stochastic"
                validation_health[mode] = learning_health(frame)
                log_learning_health(validation_health[mode], f"{kind}/seed{seed}/{mode}")
                frames.append(frame)
            validation = pd.concat(frames, ignore_index=True)
            validation["model"] = kind
            validation["train_seed"] = seed
            validation["data_kind"] = provenance["data_kind"]
            validation["checkpoint_fingerprint"] = fingerprint
            validation.to_csv(folder / "validation.csv", index=False)
            references = []
            for reference in ("idle", "left", "right", "random"):
                frame = evaluate(
                    ReferencePolicy(reference),
                    config,
                    settings["validation_seeds"],
                    map_seeds=config["environment"]["map_seeds"],
                )
                frame["reference_policy"] = reference
                frame["checkpoint_fingerprint"] = fingerprint
                references.append(frame)
            pd.concat(references, ignore_index=True).to_csv(
                folder / "reference_validation.csv", index=False
            )
            (folder / "learning_health.json").write_text(
                json.dumps(validation_health, indent=2), encoding="utf-8"
            )
        manifest = {
            "fingerprint": fingerprint,
            "source_sha256": code_digest.hexdigest(),
            "model": kind,
            "train_seed": seed,
            "requested_timesteps": settings["timesteps"],
            "actual_timesteps": model.num_timesteps,
            "wall_seconds": training_seconds,
            "validation_health": validation_health,
            "provenance": provenance,
            "parameters": sum(p.numel() for p in model.policy.parameters()),
            "trainable_parameters": sum(
                p.numel() for p in model.policy.parameters() if p.requires_grad
            ),
            "python": platform.python_version(),
            "device": str(model.device),
            "versions": versions,
        }
        (folder / "run.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    finally:
        callback.close()
        env.close()
    LOGGER.info("Saved %s", checkpoint)
    return checkpoint
