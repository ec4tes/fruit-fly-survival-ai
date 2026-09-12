"""Run the immediate heuristic demo."""

import sys

from fly_connectome.cli import main

if __name__ == "__main__":
    main(["demo", *sys.argv[1:]])
