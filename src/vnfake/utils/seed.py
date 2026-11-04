"""Reproducibility helpers.

Fix RNG seeds across Python, NumPy, and PyTorch. Call :func:`set_seed` at the
start of every entrypoint that consumes randomness.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch

from vnfake.utils.logging import get_logger

logger = get_logger(__name__)


def set_seed(seed: int) -> None:
    """Set RNG seeds for Python, NumPy, and PyTorch.

    Args:
        seed: Seed value. Read from ``cfg.seed`` in entrypoints.

    Note:
        Enabling ``cudnn.deterministic`` may reduce throughput slightly but
        is required for trustworthy evaluation.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.info("Seeded Python/NumPy/PyTorch (CUDA). seed=%d", seed)
    else:
        logger.info("Seeded Python/NumPy/PyTorch (CPU). seed=%d", seed)
