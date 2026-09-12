"""Smoke-check every documented script entry point without network calls."""

import subprocess
import sys

import pytest

from fly_connectome.cli import parser
from fly_connectome.utils.config import ROOT


@pytest.mark.parametrize(
    "script",
    [
        "run_demo.py",
        "train_baseline.py",
        "train_connectome.py",
        "download_connectome.py",
        "run_experiments.py",
        "generate_plots.py",
    ],
)
def test_script_help(script):
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script), "--help"],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=ROOT,
    )
    assert completed.returncode == 0, completed.stderr
    assert "usage:" in completed.stdout


def test_module_parser():
    args = parser().parse_args(["train", "--model", "baseline", "--timesteps", "64"])
    assert args.timesteps == 64
