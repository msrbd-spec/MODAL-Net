"""MixStyle: domain generalisation feature statistics mixing."""

import torch
import torch.nn as nn


class MixStyle(nn.Module):
    """MixStyle (Zhou et al. 2021) with cross-domain pairing.

    When training, mixes feature mean and standard deviation across halves
    of the batch [domain A | domain B].
    In eval mode or when random draw > p, returns x unchanged (identity).
    """

    def __init__(self, p: float = 0.5, alpha: float = 0.1, eps: float = 1e-6):
        super().__init__()
        self.p = float(p)
        self.alpha = float(alpha)
        self.eps = float(eps)
        self.beta = torch.distributions.Beta(self.alpha, self.alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or torch.rand(1).item() > self.p:
            return x

        B = x.size(0)
        # Compute spatial mean and std per channel
        mu = x.mean(dim=(2, 3), keepdim=True).detach()
        sig = (x.var(dim=(2, 3), keepdim=True, unbiased=False) + self.eps).sqrt().detach()
        xn = (x - mu) / sig

        lam = self.beta.sample((B, 1, 1, 1)).to(x.device)
        h = B // 2
        if B % 2 == 0:
            perm = torch.cat([torch.arange(h, B), torch.arange(0, h)]).to(x.device)
        else:
            perm = torch.randperm(B, device=x.device)

        mu_perm = mu[perm]
        sig_perm = sig[perm]

        mixed_sig = lam * sig + (1.0 - lam) * sig_perm
        mixed_mu = lam * mu + (1.0 - lam) * mu_perm
        return xn * mixed_sig + mixed_mu
