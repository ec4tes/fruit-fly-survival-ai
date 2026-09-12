"""Generate figures from existing result CSVs."""

import sys

from fly_connectome.cli import main

if __name__ == "__main__":
    main(["plots", *sys.argv[1:]])
