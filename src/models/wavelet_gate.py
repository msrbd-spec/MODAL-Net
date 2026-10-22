"""Wavelet-gated residual stem (optional high-frequency attention)."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WaveletGate(nn.Module):
    """Zero-initialised wavelet-gated residual stem.

    Extracts high-frequency 2D Haar wavelet energy, computes spatial attention,
    and gates high-frequency spatial residuals. With gamma=0 (initial state),
    forward(x) strictly equals identity x.
    """

    def __init__(self, hidden: int = 8):
        super().__init__()
        self.att = nn.Sequential(
            nn.Conv2d(1, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, 1, kernel_size=3, padding=1),
        )
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expects even spatial dimensions
        a = x[..., 0::2, 0::2]
        b = x[..., 0::2, 1::2]
        c = x[..., 1::2, 0::2]
        d = x[..., 1::2, 1::2]

        lh = (a + b - c - d) / 4.0
        hl = (a - b + c - d) / 4.0
        hh = (a - b - c + d) / 4.0
        e = ((lh ** 2 + hl ** 2 + hh ** 2).sum(dim=1, keepdim=True)).sqrt()

        g = F.interpolate(
            torch.sigmoid(self.att(e)),
            size=x.shape[-2:],
            mode='bilinear',
            align_corners=False,
        )
        low_res = F.avg_pool2d(x, kernel_size=2)
        low_up = F.interpolate(low_res, scale_factor=2, mode='nearest')
        return x + self.gamma * g * (x - low_up)
