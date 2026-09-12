"""Download and cache a real neuPrint subgraph."""

import sys

from fly_connectome.cli import main

if __name__ == "__main__":
    main(["download", *sys.argv[1:]])
