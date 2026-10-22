"""Reproducibility: seed Python, NumPy, PyTorch."""

import random
import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Set seeds for python, numpy, torch (CPU + all GPUs)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
