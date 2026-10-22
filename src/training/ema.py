"""Exponential Moving Average (EMA) for model weights."""

import copy
import torch
import torch.nn as nn


class ModelEMA:
    """Maintains an exponential moving average of model parameters and copies buffers."""

    def __init__(self, model: nn.Module, decay: float):
        self.d = float(decay)
        self.m = copy.deepcopy(model).eval()
        # Detach EMA parameters
        for p in self.m.parameters():
            p.requires_grad_(False)

    def update(self, model: nn.Module):
        """Update EMA parameters with current model weights."""
        with torch.no_grad():
            for e, p in zip(self.m.parameters(), model.parameters()):
                e.data.mul_(self.d).add_(p.data, alpha=1.0 - self.d)
            for be, b in zip(self.m.buffers(), model.buffers()):
                be.copy_(b)
