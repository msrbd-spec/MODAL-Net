"""Test T8: WaveletGate identity, emd2 distance properties, and CORAL probability sum."""

import torch
import torch.nn.functional as F
import pytest

from src.models.wavelet_gate import WaveletGate
from src.losses.losses import emd2
from src.models.gradenet import probs_from_logits


def test_wavelet_gate_gamma_zero_identity():
    """WaveletGate with gamma=0 must strictly preserve identity."""
    gate = WaveletGate(hidden=8)
    assert gate.gamma.item() == 0.0

    x = torch.randn(2, 3, 64, 64)
    out = gate(x)
    assert torch.allclose(out, x, atol=1e-7)


def test_emd2_zero_and_ordinal_monotonicity():
    """emd2 should be 0 when p == q, and strictly larger for farther grades."""
    # Target q is class 0: [1, 0, 0, 0, 0]
    q = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0]])

    # Perfect prediction: high logit on class 0 -> softmax ≈ q
    z_perfect = torch.tensor([[100.0, -100.0, -100.0, -100.0, -100.0]])
    loss_perfect = emd2(z_perfect, q).item()
    assert pytest.approx(loss_perfect, abs=1e-5) == 0.0

    # Prediction on grade 1 (adjacent)
    z_grade1 = torch.tensor([[-100.0, 100.0, -100.0, -100.0, -100.0]])
    loss_grade1 = emd2(z_grade1, q).item()

    # Prediction on grade 4 (distant)
    z_grade4 = torch.tensor([[-100.0, -100.0, -100.0, -100.0, 100.0]])
    loss_grade4 = emd2(z_grade4, q).item()

    assert loss_grade1 > loss_perfect
    assert loss_grade4 > loss_grade1


def test_coral_probabilities_sum_to_one():
    """CORAL probabilities derived from arbitrary logits must sum to 1.0."""
    torch.manual_seed(42)
    B = 10
    # Arbitrary random logits for 4 cutpoints (num_classes - 1)
    z_coral = torch.randn(B, 4)

    probs = probs_from_logits(z_coral, head_type='coral')
    assert probs.shape == (B, 5)
    assert (probs >= 0.0).all()

    row_sums = probs.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones(B), atol=1e-5)
