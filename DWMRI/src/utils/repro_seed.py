"""
Centralized RNG and cuDNN settings for training and evaluation.

Fast mode (default): cuDNN benchmark on for throughput; GPU runs are not
bit-for-bit reproducible.

Reproducible mode: deterministic algorithms where supported; still not a full
guarantee on all GPU ops, but improves repeatability for debugging and small
smoke tests.
"""

from __future__ import annotations

import logging
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set Python, NumPy, and PyTorch CPU/CUDA RNG seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_cudnn(*, fast: bool = True) -> None:
    """
    Args:
        fast: If True, enable cuDNN benchmark (default training behavior).
              If False, disable benchmark and enable deterministic algorithms.
    """
    torch.backends.cudnn.enabled = True
    if fast:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        logging.info("cuDNN: benchmark=True (fast, not fully reproducible on GPU)")
    else:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)
        logging.info("cuDNN: benchmark=False, deterministic=True (reproducible-ish mode)")


def log_runtime_env(device: str) -> None:
    logging.info(
        "Runtime: Python=%s, torch=%s, CUDA=%s, cudnn=%s, device=%s",
        __import__("sys").version.split()[0],
        torch.__version__,
        torch.version.cuda,
        torch.backends.cudnn.version(),
        device,
    )


def make_dataloader_generator(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(int(seed))
    return g
