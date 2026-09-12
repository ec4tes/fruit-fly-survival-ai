"""Train a connectome-constrained PPO agent."""

import sys

from fly_connectome.cli import main

if __name__ == "__main__":
    main(["train", "--config", "configs/connectome.yaml", "--model", "connectome", *sys.argv[1:]])
