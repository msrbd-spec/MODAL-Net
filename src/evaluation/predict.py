"""Inference and test-time augmentation (TTA) routines."""

import torch
import torch.nn.functional as F
import numpy as np

from src.models.gradenet import probs_from_logits


def apply_view(x: torch.Tensor, r: int, h: int, v: int) -> torch.Tensor:
    """Apply 90-degree rotations and horizontal/vertical flips to (B, C, H, W) tensor."""
    if r != 0:
        x = torch.rot90(x, r, (2, 3))
    if h:
        x = x.flip(3)
    if v:
        x = x.flip(2)
    return x


TTA_CONFIGS = {
    'none': [(0, 0, 0)],
    'flip': [(0, h, v) for h in (0, 1) for v in (0, 1)],
    'flip_rot': [(r, h, 0) for r in range(4) for h in (0, 1)],
}


def rescale(x: torch.Tensor, s: float) -> torch.Tensor:
    """Rescale tensor by factor s, then interpolate back to original spatial dimensions."""
    if abs(s - 1.0) < 1e-5:
        return x
    orig_size = x.shape[-2:]
    scaled = F.interpolate(x, scale_factor=s, mode='bilinear', align_corners=False)
    return F.interpolate(scaled, size=orig_size, mode='bilinear', align_corners=False)


@torch.no_grad()
def predict_probs(models, loader, cfg: dict, tta: str = None, scales: list = None, device: str = None):
    """Compute ensemble probabilities over models, TTA views, and multi-scale rescalings.

    Args:
        models: List of nn.Module (e.g. 5 fold models).
        loader: DataLoader returning dict with 'x', 'y', 'id'.
        cfg: Config dict.
        tta: 'none', 'flip', or 'flip_rot' (defaults to cfg['tta']).
        scales: List of scale floats (defaults to cfg['tta_scales']).
        device: Device to use for inference.

    Returns:
        all_ids: List of image id strings.
        all_y: (N,) int array of ground-truth labels.
        all_P: (N, 5) float array of ensemble-averaged probabilities.
        per_model_P: List of (N, 5) float arrays (one per model).
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if tta is None:
        tta = cfg.get('tta', 'none')
    views = TTA_CONFIGS.get(tta, [(0, 0, 0)])

    if scales is None:
        scales = cfg.get('tta_scales', [1.0])
    if not isinstance(scales, (list, tuple)):
        scales = [float(scales)]

    head_type = cfg.get('head_type', 'softmax')

    for m in models:
        m.eval().to(device)

    all_ids = []
    all_y = []
    per_model_probs = [[] for _ in models]

    for batch in loader:
        x_raw = batch['x'].to(device)
        y = batch['y'].cpu().numpy()
        b_ids = batch['id']

        all_ids.extend([str(i) for i in b_ids])
        all_y.extend(list(y))

        # Evaluate each model on this batch
        for m_idx, m in enumerate(models):
            accum_p = torch.zeros(
                (x_raw.size(0), int(cfg.get('num_classes', 5))),
                device=device,
                dtype=torch.float32,
            )
            count = 0
            for s in scales:
                x_scaled = rescale(x_raw, float(s))
                for r, h, v in views:
                    x_aug = apply_view(x_scaled, r, h, v)
                    z = m(x_aug)
                    p = probs_from_logits(z, head_type)
                    accum_p += p
                    count += 1
            accum_p /= max(count, 1)
            per_model_probs[m_idx].append(accum_p.cpu().numpy())

    all_y = np.array(all_y, dtype=int)
    stacked_per_model = [np.concatenate(chunks, axis=0) for chunks in per_model_probs]
    ensemble_P = np.mean(stacked_per_model, axis=0)

    return all_ids, all_y, ensemble_P, stacked_per_model
