"""Loss functions: soft cross-entropy, squared EMD, CORAL, ordinal Gaussian smoothing."""

import numpy as np
import torch
import torch.nn.functional as F


def ordinal_gaussian_np(y: int, K: int = 5, sigma: float = 0.7) -> np.ndarray:
    """Ordinal Gaussian distribution around scalar label y (NumPy)."""
    k = np.arange(K, dtype=np.float32)
    g = np.exp(-((k - float(y)) ** 2) / (2.0 * sigma ** 2))
    return g / g.sum()


def base_target(y: int, cfg: dict) -> np.ndarray:
    """Compute target probability vector for training sample y.

    base_target = (1 - eps) * onehot + eps * ordinal_gaussian(y, sigma)
    When eps == 0.0, returns exact one-hot vector.
    """
    K = int(cfg.get('num_classes', 5))
    eps = float(cfg.get('ordinal_eps', 0.0))
    sigma = float(cfg.get('ordinal_sigma', 0.7))

    onehot = np.zeros(K, dtype=np.float32)
    onehot[int(y)] = 1.0

    if eps <= 0.0:
        return onehot

    g = ordinal_gaussian_np(int(y), K=K, sigma=sigma)
    return (1.0 - eps) * onehot + eps * g


def ordinal_gaussian(y: torch.Tensor, K: int = 5, sigma: float = 0.7) -> torch.Tensor:
    """Ordinal Gaussian distribution around integer labels y (PyTorch tensor)."""
    k = torch.arange(K, device=y.device, dtype=torch.float32).unsqueeze(0)
    y_f = y.unsqueeze(1).float()
    g = torch.exp(-((k - y_f) ** 2) / (2.0 * sigma ** 2))
    return g / g.sum(dim=1, keepdim=True)


def soft_ce(z: torch.Tensor, q: torch.Tensor, w: torch.Tensor = None) -> torch.Tensor:
    """Soft cross-entropy loss between logits z (B, K) and targets q (B, K).

    Optional sample weights w (B,).
    """
    l = -(q * F.log_softmax(z, dim=1)).sum(dim=1)
    if w is not None:
        return (l * w).sum() / (w.sum() + 1e-8)
    return l.mean()


def emd2(z: torch.Tensor, q: torch.Tensor, w: torch.Tensor = None) -> torch.Tensor:
    """Squared Earth Mover's Distance loss over ordinal cumulative probabilities.

    l = sum_k (CDF_p[k] - CDF_q[k])^2 / K
    """
    p = F.softmax(z, dim=1)
    cdf_p = p.cumsum(dim=1)
    cdf_q = q.cumsum(dim=1)
    l = ((cdf_p - cdf_q) ** 2).sum(dim=1) / float(z.size(1))
    if w is not None:
        return (l * w).sum() / (w.sum() + 1e-8)
    return l.mean()


def coral_loss(z: torch.Tensor, y: torch.Tensor, K: int = 5) -> torch.Tensor:
    """CORAL (Consistent Rank Logits) ordinal loss.

    z: (B, K - 1) binary ordinal logits
    y: (B,) class indices 0..K-1
    """
    threshold_indices = torch.arange(K - 1, device=y.device).unsqueeze(0)
    levels = (y.unsqueeze(1) > threshold_indices).float()
    return F.binary_cross_entropy_with_logits(z, levels)


def total_loss(z: torch.Tensor, q: torch.Tensor, y: torch.Tensor, cfg: dict, w: torch.Tensor = None) -> torch.Tensor:
    """Compute overall loss according to cfg configuration."""
    if cfg.get('head_type') == 'coral':
        return coral_loss(z, y, K=cfg.get('num_classes', 5))

    loss_ce = soft_ce(z, q, w)
    if cfg.get('loss') == 'ce_emd':
        lambda_emd = float(cfg.get('lambda_emd', 1.0))
        return loss_ce + lambda_emd * emd2(z, q, w)

    return loss_ce
