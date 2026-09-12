"""Episode-level CSV and TensorBoard metrics from actual environment outcomes."""

import csv
import logging
import time
from pathlib import Path

from stable_baselines3.common.callbacks import BaseCallback

METRICS = [
    "survival_time",
    "food_collected",
    "energy_remaining",
    "collision_count",
    "predator_escape_rate",
    "predator_encounters",
    "predator_escapes",
    "predator_capture",
    "energy_efficiency",
    "energy_spent",
    "distance_travelled",
    "termination_reason",
    "map_seed",
]


class EpisodeMetricsCallback(BaseCallback):
    """Stream metrics to disk and SB3's active logger on episode completion."""

    def __init__(self, path: Path, model_kind: str, seed: int):
        super().__init__()
        self.path, self.model_kind, self.seed = path, model_kind, seed

    def _on_training_start(self) -> None:
        self.started = time.perf_counter()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(
            self.stream,
            fieldnames=[
                "model",
                "train_seed",
                "timesteps",
                "wall_seconds",
                "episode_reward",
                "episode_length",
                *METRICS,
            ],
        )
        self.writer.writeheader()

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            if "episode" not in info:
                continue
            row = {
                "model": self.model_kind,
                "train_seed": self.seed,
                "timesteps": self.num_timesteps,
                "wall_seconds": time.perf_counter() - self.started,
                "episode_reward": info["episode"]["r"],
                "episode_length": info["episode"]["l"],
                **{key: info[key] for key in METRICS},
            }
            self.writer.writerow(row)
            self.stream.flush()
            for key in METRICS:
                if isinstance(info[key], (int, float)):
                    self.logger.record_mean(f"survival/{key}", info[key])
        return True

    def _on_rollout_end(self) -> None:
        logging.getLogger(__name__).info(
            "%s seed=%d steps=%d", self.model_kind, self.seed, self.num_timesteps
        )

    def _on_training_end(self) -> None:
        self.close()

    def close(self) -> None:
        """Close the CSV stream even if training is interrupted."""
        if hasattr(self, "stream") and not self.stream.closed:
            self.stream.close()
