"""Single installed CLI; scripts are thin, tested entry-point adapters."""

from __future__ import annotations

import argparse
import json
import logging
import sys

from .utils.config import load_config, resolve_path


def parser() -> argparse.ArgumentParser:
    """Build the public CLI and command-specific defaults."""
    root = argparse.ArgumentParser(description="Fruit Fly Connectome Survival AI")
    root.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    commands = root.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="Train a PPO agent")
    train.add_argument("--config", default="configs/baseline.yaml")
    train.add_argument("--model", choices=["baseline", "connectome", "fixed", "random_sparse"])
    train.add_argument("--seed", type=int)
    train.add_argument("--timesteps", type=int)
    train.add_argument("--reuse", action="store_true")
    download = commands.add_parser("download", help="Cache a real measured neuPrint subgraph")
    download.add_argument("--output", default="data/processed/malecns")
    download.add_argument("--size", type=int, default=2000)
    download.add_argument("--seed-ids", type=int, nargs="+")
    download.add_argument("--neuron-type")
    download.add_argument("--force", action="store_true")
    download.add_argument("--allow-synthetic-fallback", action="store_true")
    demo = commands.add_parser("demo", help="Immediate heuristic or saved-checkpoint demo")
    demo.add_argument("--config", default="configs/default.yaml")
    demo.add_argument("--headless", action="store_true")
    demo.add_argument("--steps", type=int, default=1000)
    demo.add_argument("--checkpoint")
    demo.add_argument("--snapshot")
    exp = commands.add_parser("experiments", help="Train and evaluate matched models")
    exp.add_argument("--config", default="configs/experiments.yaml")
    exp.add_argument(
        "--experiment",
        default="all",
        choices=[
            "all",
            "comparison",
            "generalization",
            "sensor_noise",
            "neuron_damage",
            "edge_damage",
            "ablation",
        ],
    )
    plots = commands.add_parser("plots", help="Plot existing measured CSVs")
    plots.add_argument("--config", default="configs/default.yaml")
    plots.add_argument("--output", help="Default: output_dir from the selected config")
    graph = commands.add_parser("graph", help="Inspect and plot a bounded graph sample")
    graph.add_argument("--path", default="data/sample/synthetic")
    graph.add_argument("--output", default="results/plots/connectome.png")
    graph.add_argument("--limit", type=int, default=100)
    evaluate = commands.add_parser("evaluate", help="Evaluate a saved checkpoint")
    evaluate.add_argument("checkpoint")
    evaluate.add_argument("--seeds", nargs="+", type=int, default=[10001, 10002, 10003])
    evaluate.add_argument("--output", help="Default: evaluation_<mode>.csv beside checkpoint")
    evaluate.add_argument("--noise", type=float, default=0.0)
    evaluate.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample actions reproducibly instead of taking argmax",
    )
    return root


def main(argv: list[str] | None = None) -> None:
    """Dispatch a command, logging concise actionable failures without secret payloads."""
    args = parser().parse_args(argv)
    logging.basicConfig(
        level=args.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    logger = logging.getLogger("fly-ai")
    try:
        if args.command == "train":
            from .training.train import train_model

            config = load_config(args.config)
            if args.timesteps is not None:
                config["training"]["timesteps"] = args.timesteps
            logger.info("Checkpoint: %s", train_model(config, args.model, args.seed, args.reuse))
        elif args.command == "download":
            from .connectome.downloader import download_connectome

            graph = download_connectome(
                resolve_path(args.output),
                args.size,
                args.seed_ids,
                args.neuron_type,
                args.force,
                args.allow_synthetic_fallback,
            )
            logger.info("Loaded %d neurons (%s)", graph.n, graph.provenance["data_kind"])
        elif args.command == "demo":
            from .demo import run_demo

            if args.steps < 1:
                raise ValueError("--steps must be positive")
            config = load_config(args.config)
            if args.checkpoint:
                config = load_config(resolve_path(args.checkpoint).parent / "config.yaml")
            info = run_demo(
                config,
                args.headless,
                args.steps,
                resolve_path(args.checkpoint) if args.checkpoint else None,
                resolve_path(args.snapshot) if args.snapshot else None,
            )
            logger.info("Demo finished: %s", json.dumps(info))
        elif args.command == "experiments":
            from .experiments.runner import run_experiments

            logger.info(
                "CSV results: %s", run_experiments(load_config(args.config), args.experiment)
            )
        elif args.command == "plots":
            from .visualization.result_plots import generate_plots

            output = args.output or load_config(args.config)["output_dir"]
            logger.info("Created %d plots", len(generate_plots(resolve_path(output))))
        elif args.command == "graph":
            from .connectome.graph import load_connectome
            from .connectome.statistics import graph_statistics
            from .visualization.connectome_plot import plot_connectome

            graph = load_connectome(resolve_path(args.path))
            stats = graph_statistics(graph)
            logger.info("Graph: %s", {k: v for k, v in stats.items() if not isinstance(v, dict)})
            plot_connectome(graph, resolve_path(args.output), args.limit)
        elif args.command == "evaluate":
            from stable_baselines3 import PPO

            from .training.diagnostics import learning_health, log_learning_health
            from .training.evaluation import evaluate
            from .utils.seeding import seed_everything

            path = resolve_path(args.checkpoint)
            config = load_config(path.parent / "config.yaml")
            seed_everything(config["seed"], config["torch_threads"])
            result = evaluate(
                PPO.load(path, device="cpu"),
                config,
                args.seeds,
                args.noise,
                deterministic=not args.stochastic,
            )
            log_learning_health(learning_health(result), str(path.parent.name))
            manifest = json.loads((path.parent / "run.json").read_text(encoding="utf-8"))
            result["model"] = manifest["model"]
            result["train_seed"] = manifest["train_seed"]
            result["data_kind"] = manifest["provenance"]["data_kind"]
            result["checkpoint_fingerprint"] = manifest["fingerprint"]
            result["noise_level"] = args.noise
            mode = "stochastic" if args.stochastic else "deterministic"
            output = (
                resolve_path(args.output) if args.output else path.parent / f"evaluation_{mode}.csv"
            )
            output.parent.mkdir(parents=True, exist_ok=True)
            result.to_csv(output, index=False)
            logger.info("Evaluation: %s", output)
    except (ValueError, FileNotFoundError, RuntimeError, KeyError) as error:
        logger.error("%s", error)
        sys.exit(1)
