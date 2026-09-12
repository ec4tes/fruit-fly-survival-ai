"""Explicit RNG and CPU thread control."""

import random

import numpy as np
import torch


def seed_everything(seed: int, threads: int = 2) -> None:
    """Seed Python, NumPy and Torch; Gymnasium is seeded separately on reset."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(device: str) -> str:
    """Select CUDA only if available; reject an unavailable explicit request."""
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable. Use device: cpu or auto.")
    return device
