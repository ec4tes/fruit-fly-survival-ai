"""Run the paired-seed experiment suite."""

import sys

from fly_connectome.cli import main

if __name__ == "__main__":
    main(["experiments", *sys.argv[1:]])
