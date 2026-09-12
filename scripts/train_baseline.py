"""Train the dense PPO baseline."""

import sys

from fly_connectome.cli import main

if __name__ == "__main__":
    main(["train", "--config", "configs/baseline.yaml", "--model", "baseline", *sys.argv[1:]])
