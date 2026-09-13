"""Plots from real CSV observations only, with independent-seed uncertainty."""

import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def generate_plots(output: Path) -> list[Path]:
    """Generate training, comparison, generalization, noise, lesion and ablation plots."""
    plot_dir = output / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    written = []
    plt.style.use("seaborn-v0_8-whitegrid")
    for path in sorted((output / "csv").glob("*_summary.csv")):
        frame = pd.read_csv(path)
        name = path.stem.removesuffix("_summary")
        if frame.empty:
            continue
        synthetic = frame.data_kind.isin(["synthetic"]).any()
        title = name.replace("_", " ").title() + (" — SYNTHETIC SOFTWARE DEMO" if synthetic else "")
        for retention in (False, True):
            suffix = "_retention_pct" if retention else ""
            metrics = [
                metric + suffix for metric in ("episode_reward", "survival_time", "food_collected")
            ]
            if any(f"{metric}_mean" not in frame for metric in metrics):
                continue
            fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout="constrained")
            for ax, metric in zip(axes, metrics, strict=True):
                for (model, strategy), group in frame.groupby(["model", "strategy"]):
                    label = model + (f" / {strategy}" if strategy != "none" else "")
                    if name in ("comparison", "generalization", "ablation"):
                        labels = group.condition.tolist()
                        x = np.arange(len(labels), dtype=float)
                        ax.set_xticks(x, labels, rotation=20, ha="right")
                    else:
                        group = group.sort_values("level")
                        x = group.level.to_numpy() * 100
                        ax.set_xlabel("Perturbation (%)")
                    y = group[f"{metric}_mean"].to_numpy()
                    sem = group[f"{metric}_sem"].to_numpy()
                    ax.plot(x, y, "o-", label=label, markersize=4)
                    if np.isfinite(sem).any():
                        ax.fill_between(x, y - sem, y + sem, alpha=0.13)
                ax.set_ylabel(metric.replace("_", " "))
            axes[0].legend(fontsize=8)
            fig.suptitle(title + (" — retention" if retention else ""))
            path_out = plot_dir / f"{name}{'_retention' if retention else ''}.png"
            fig.savefig(path_out, dpi=150)
            plt.close(fig)
            written.append(path_out)
    # Each line is one actual training seed. Never interpolate nonexistent episodes.
    manifest_path = output / "experiment_manifest.json"
    if manifest_path.exists():
        checkpoints = json.loads(manifest_path.read_text())["checkpoints"].values()
        episode_files = [Path(path).parent / "episodes.csv" for path in checkpoints]
    else:
        episode_files = list((output / "models").glob("*/episodes.csv"))
    for metric in (
        "episode_reward",
        "survival_time",
        "food_collected",
        "distance_travelled",
        "stationary_fraction",
    ):
        fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
        plotted = False
        synthetic_training = False
        for path in episode_files:
            data = pd.read_csv(path)
            if data.empty or metric not in data:
                continue
            plotted = True
            run_manifest = path.parent / "run.json"
            if run_manifest.exists():
                provenance = json.loads(run_manifest.read_text())["provenance"]
                synthetic_training |= "synthetic" in (
                    provenance["data_kind"],
                    provenance.get("parent_data_kind"),
                )
            label = f"{data.model.iloc[0]} / seed {data.train_seed.iloc[0]}"
            ax.plot(data.timesteps, data[metric].rolling(10, min_periods=1).mean(), label=label)
        if plotted:
            ax.set(
                xlabel="Environment steps",
                ylabel=metric.replace("_", " "),
                title="Training episodes — trailing 10-episode mean"
                + (" — SYNTHETIC SOFTWARE DEMO" if synthetic_training else ""),
            )
            ax.legend(fontsize=7)
            path = plot_dir / f"training_{metric}.png"
            fig.savefig(path, dpi=150)
            written.append(path)
        plt.close(fig)
    if not written:
        logging.getLogger(__name__).warning("No measured results found under %s", output)
    return written
