"""Test T9: MixStyle properties, amplitude mixing, and config validation guard."""

import numpy as np
import torch
import pytest

from src.models.mixstyle import MixStyle
from src.data.amplitude_mix import amplitude_mix
from src.utils.config import validate_cfg


def test_mixstyle_properties():
    """MixStyle: shape preserved, identity in eval mode, halves swapped in train mode."""
    ms = MixStyle(p=1.0, alpha=0.1)

    # 1. Identity in eval mode
    ms.eval()
    x = torch.randn(4, 8, 16, 16)
    out_eval = ms(x)
    assert torch.allclose(out_eval, x, atol=1e-7)

    # 2. Shape preserved in train mode
    ms.train()
    out_train = ms(x)
    assert out_train.shape == x.shape

    # 3. Check halves swapped when alpha is chosen to give distinct mixture
    # Create two halves with distinctly different mean levels
    x_distinct = torch.zeros(4, 4, 8, 8)
    x_distinct[:2] = 10.0   # first half: domain A
    x_distinct[2:] = -10.0  # second half: domain B
    out_mixed = ms(x_distinct)
    # The statistics should be blended between domain A and domain B
    assert not torch.allclose(out_mixed[:2], x_distinct[:2])
    assert not torch.allclose(out_mixed[2:], x_distinct[2:])


def test_amplitude_mix_black_border_and_dtype():
    """amplitude_mix keeps the black border and returns uint8."""
    # Create a source with a black border
    src = np.zeros((128, 128, 3), dtype=np.uint8)
    # Center disc with non-zero intensity
    src[32:96, 32:96] = 180

    # Target image with different texture
    trg = np.full((128, 128, 3), 100, dtype=np.uint8)

    mixed = amplitude_mix(src, trg, beta=0.1, lam=0.5)

    assert mixed.dtype == np.uint8
    assert mixed.shape == src.shape

    # Black border (outside center region) must remain strictly 0
    assert (mixed[0:20, :, :] == 0).all()
    assert (mixed[108:, :, :] == 0).all()
    assert (mixed[:, 0:20, :] == 0).all()
    assert (mixed[:, 108:, :] == 0).all()

    # Center must remain non-zero
    assert mixed[64, 64, 0] > 0


def test_amp_mix_legacy_preproc_rejected():
    """amp_mix with preproc=legacy must raise AssertionError via validate_cfg."""
    invalid_cfg = {
        'amp_mix': True,
        'preproc': 'legacy',
        'backbone': 'maxvit_large_tf_512',
        'head_type': 'softmax',
        'uda_style_pool': 'none',
        'tta': 'none',
        'recalibrate': 'none',
    }
    with pytest.raises(AssertionError, match='amp_mix needs preproc != legacy'):
        validate_cfg(invalid_cfg)
