"""Environment, graph and experiment figures."""

import os
from pathlib import Path

# Keep Matplotlib's writable cache inside generated, git-ignored project artifacts.
_cache = Path(__file__).resolve().parents[3] / "results" / ".matplotlib"
_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_cache))
