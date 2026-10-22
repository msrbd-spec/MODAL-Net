"""Test T3: Evaluation metrics (QWK, ECE, binary sens@spec) and 2-decimal formatting."""

import numpy as np
import pytest
from src.evaluation.metrics import (
    metrics5,
    metrics_bin,
    compute_oof_thresholds,
    ece,
    nll,
    brier,
)
from src.utils.io_utils import fmt2


def test_qwk_known_cases():
    """QWK should be 1.0 for perfect predictions and lower for distant errors."""
    y = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
    # Perfect probabilities (one-hot)
    P_perfect = np.eye(5)[y]
    m_perfect = metrics5(y, P_perfect)
    assert pytest.approx(m_perfect['qwk'], abs=1e-5) == 1.0
    assert pytest.approx(m_perfect['acc'], abs=1e-5) == 100.0

    # Mild errors (1 off) vs Severe errors (3 off)
    pred_mild = np.clip(y + 1, 0, 4)
    P_mild = np.eye(5)[pred_mild]
    m_mild = metrics5(y, P_mild)

    pred_severe = np.clip(y + 3, 0, 4)
    P_severe = np.eye(5)[pred_severe]
    m_severe = metrics5(y, P_severe)

    assert m_mild['qwk'] > m_severe['qwk']


def test_ece_known_cases():
    """ECE should be 0.0 for perfectly calibrated predictions and 1.0 for completely wrong confident predictions."""
    y = np.array([0] * 50 + [1] * 50)
    # 100% confidence, 100% correct
    P_perfect = np.zeros((100, 5))
    P_perfect[np.arange(100), y] = 1.0
    assert pytest.approx(ece(P_perfect, y), abs=1e-5) == 0.0

    # 100% confidence on class 2, but true is 0 or 1 -> 0% correct
    P_wrong = np.zeros((100, 5))
    P_wrong[:, 2] = 1.0
    assert pytest.approx(ece(P_wrong, y), abs=1e-5) == 1.0


def test_binary_metrics_and_thresholds():
    """Test referable DR binary metrics and specificity thresholding."""
    rng = np.random.default_rng(42)
    N = 200
    y = rng.integers(0, 5, size=N)
    y_bin = (y >= 2).astype(int)

    # Simulated probabilities: higher score for referable classes
    scores = rng.beta(1.5, 3.0, size=N)
    scores[y_bin == 1] = rng.beta(3.0, 1.5, size=np.sum(y_bin == 1))

    P = np.zeros((N, 5))
    # Put (1 - scores) into class 0, and scores into class 2
    P[:, 0] = 1.0 - scores
    P[:, 2] = scores

    thr = compute_oof_thresholds(y, P)
    assert 'thr_spec90' in thr and 'thr_spec95' in thr
    assert 0.0 <= thr['thr_spec90'] <= 1.0
    assert 0.0 <= thr['thr_spec95'] <= 1.0
    assert thr['thr_spec95'] >= thr['thr_spec90']

    # Verify specificity on this set meets or exceeds target
    pred_90 = (scores >= thr['thr_spec90']).astype(int)
    neg_mask = (y_bin == 0)
    spec_achieved = (pred_90[neg_mask] == 0).mean()
    assert spec_achieved >= 0.90

    # Test metrics_bin output
    m_bin = metrics_bin(y, P, thresholds=thr)
    assert 'bin_acc' in m_bin
    assert 'bin_auroc' in m_bin
    assert 'sens_at_spec90' in m_bin
    assert not np.isnan(m_bin['sens_at_spec90'])
    assert not np.isnan(m_bin['sens_at_spec95'])


def test_formatting_2_decimals():
    """All numbers must format to exactly 2 decimals via fmt2."""
    assert fmt2(85.191) == '85.19'
    assert fmt2(85.199) == '85.20'
    assert fmt2(0.9) == '0.90'
    assert fmt2(0.0) == '0.00'
    assert fmt2(12) == '12.00'
    assert fmt2(None) == ''
    assert fmt2(np.nan) == ''
    assert fmt2(float('nan')) == ''
