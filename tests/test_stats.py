"""Test T4: Statistical tests (paired bootstrap, Holm correction, McNemar symmetry)."""

import numpy as np
import pytest
from src.evaluation.stats import paired_bootstrap, holm, mcnemar_exact


def test_bootstrap_deterministic():
    """Paired bootstrap test must be completely deterministic for a given seed."""
    rng = np.random.default_rng(123)
    N = 100
    y = rng.integers(0, 5, size=N)
    # System A probabilities vs System B probabilities
    Pa = rng.uniform(size=(N, 5))
    Pa /= Pa.sum(axis=1, keepdims=True)
    Pb = rng.uniform(size=(N, 5))
    Pb /= Pb.sum(axis=1, keepdims=True)

    def simple_acc(y_sub, P_sub):
        return (P_sub.argmax(axis=1) == y_sub).mean()

    delta1, ci1, p1 = paired_bootstrap(y, Pa, Pb, simple_acc, B=500, seed=42)
    delta2, ci2, p2 = paired_bootstrap(y, Pa, Pb, simple_acc, B=500, seed=42)

    assert delta1 == delta2
    assert ci1 == ci2
    assert p1 == p2


def test_holm_monotone():
    """Holm-Bonferroni correction must produce monotone non-decreasing adjusted p-values."""
    raw_p = [0.005, 0.012, 0.035, 0.040, 0.150]
    adj_p = holm(raw_p)

    # Check monotonicity in order of ascending raw p-values
    for i in range(len(adj_p) - 1):
        assert adj_p[i] <= adj_p[i + 1]

    # Adjusted p-values must not exceed 1.0
    assert (adj_p <= 1.0).all()
    assert (adj_p >= 0.0).all()

    # Known step-down computation:
    # m = 3, raw = [0.01, 0.04, 0.05]
    # k=1: 3 * 0.01 = 0.03
    # k=2: max(0.03, 2 * 0.04) = 0.08
    # k=3: max(0.08, 1 * 0.05) = 0.08
    expected = [0.03, 0.08, 0.08]
    np.testing.assert_allclose(holm([0.01, 0.04, 0.05]), expected, rtol=1e-5)


def test_mcnemar_symmetry():
    """McNemar exact test must be symmetric when exchanging models A and B."""
    y = np.array([0, 1, 2, 3, 4] * 20)
    # pa has 15 discordant errors where pa wrong, pb right
    pa = y.copy()
    pb = y.copy()
    pa[:15] = (pa[:15] + 1) % 5
    pb[15:20] = (pb[15:20] + 1) % 5

    p_ab = mcnemar_exact(y, pa, pb)
    p_ba = mcnemar_exact(y, pb, pa)
    assert pytest.approx(p_ab, abs=1e-7) == p_ba

    # Edge case: identical predictions gives p-value = 1.0
    assert mcnemar_exact(y, pa, pa) == 1.0
