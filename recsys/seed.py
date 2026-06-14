"""Reproducibility — one place to seed every RNG the project might touch."""
from __future__ import annotations

import os
import random

import numpy as np

from .config import SEED


def seed_everything(seed: int = SEED) -> int:
    """Seed Python, NumPy, and (if present) PyTorch. Returns the seed used."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:  # torch is optional for the eval/demo path
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    return seed
