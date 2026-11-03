"""Unit tests for WP6: Inference, TTA, Conformal Prediction, and Selective Classification."""

import os
import tempfile
import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from src.evaluation.predict import apply_view, rescale, TTA_CONFIGS
from src.evaluation.conformal import (
    interval_scores,
    compute_qhat,
    interval_set,
    predict_intervals,
    selective_prediction_metrics,
    evaluate_conformal_predictions,
)
from src.evaluation.calibration import fit_devcut, apply_devcut
from src.evaluation.experiment import (
    run_inference_experiment,
    run_ensemble_experiment,
    run_calibrate_conformal,
)
from src.utils.config import ckpt_owner


def test_t10_conformal_exchangeable_coverage():
    """Test T10: Empirical coverage >= 1 - alpha - 0.02 on synthetic exchangeable data."""
    rng = np.random.default_rng(42)
    N = 3000
    K = 5
    alpha = 0.10

    # Generate synthetic Dirichlet probability distributions
    raw_alphas = np.array([2.0, 1.5, 1.0, 0.8, 0.5])
    P = rng.dirichlet(raw_alphas, size=N)

    # Sample true labels y according to the distribution P_i
    y = np.array([rng.choice(K, p=P[i]) for i in range(N)])

    # Split into calibration and test sets (exchangeable split)
    n_cal = 1500
    P_cal, y_cal = P[:n_cal], y[:n_cal]
    P_test, y_test = P[n_cal:], y[n_cal:]

    # Compute conformal quantile on calibration set
    scores_cal = interval_scores(P_cal, y_cal)
    qhat = compute_qhat(scores_cal, alpha=alpha)

    # Predict intervals on test set
    intervals = predict_intervals(P_test, qhat)
    lo = intervals[:, 0]
    hi = intervals[:, 1]

    # Check empirical coverage
    covered = (y_test >= lo) & (y_test <= hi)
    emp_coverage = float(np.mean(covered))

    target_coverage = 1.0 - alpha
    # T10 spec: empirical coverage >= 1 - alpha - 0.02
    assert emp_coverage >= target_coverage - 0.02, (
        f"Empirical coverage {emp_coverage:.4f} is below target {target_coverage - 0.02:.4f}"
    )


def test_conformal_interval_properties():
    """Verify mathematical properties of ordinal-interval conformal sets."""
    # 1. Monotonicity: larger score means more mass needed
    P = np.array([
        [0.8, 0.1, 0.05, 0.03, 0.02],
        [0.05, 0.1, 0.7, 0.1, 0.05],
    ])
    y_easy = np.array([0, 2])
    y_hard = np.array([4, 0])

    scores_easy = interval_scores(P, y_easy)
    scores_hard = interval_scores(P, y_hard)

    assert np.isclose(scores_easy[0], 0.8)
    assert np.isclose(scores_easy[1], 0.7)
    assert np.isclose(scores_hard[0], 1.0)
    assert np.isclose(scores_hard[1], 0.95)

    # 2. Interval set includes argmax always
    for p in P:
        lo, hi = interval_set(p, qhat=0.5)
        argmax = int(np.argmax(p))
        assert lo <= argmax <= hi
        assert lo <= hi

    # 3. Referable abstention logic
    # Sample with interval [1, 2] crosses grade 1|2 boundary -> must abstain
    y_test = np.array([1, 3])
    P_test = np.array([
        [0.1, 0.45, 0.45, 0.0, 0.0],  # interval will cover 1 and 2
        [0.0, 0.0, 0.1, 0.8, 0.1],     # interval will cover [3, 3] or [2, 3]
    ])
    qhat = 0.85
    eval_res = evaluate_conformal_predictions(P_test, y_test, qhat, alpha=0.10)
    assert float(eval_res['abstention_referable']) == 50.0  # 1 out of 2 samples


def test_selective_prediction():
    """Verify selective classification metrics and risk-coverage relationship."""
    rng = np.random.default_rng(123)
    N = 200
    y = rng.choice(5, size=N)
    # Generate probabilities: high confidence for correct, low confidence for incorrect
    P = np.zeros((N, 5))
    for i in range(N):
        if i < 150:  # 75% correct
            P[i, y[i]] = rng.uniform(0.7, 0.99)
            rem = (1.0 - P[i, y[i]]) / 4
            for c in range(5):
                if c != y[i]:
                    P[i, c] = rem
        else:  # incorrect
            wrong = (y[i] + 1) % 5
            P[i, wrong] = rng.uniform(0.3, 0.5)
            rem = (1.0 - P[i, wrong]) / 4
            for c in range(5):
                if c != wrong:
                    P[i, c] = rem

    metrics = selective_prediction_metrics(P, y)
    assert 0.0 <= float(metrics['aurc']) <= 1.0
    # Higher confidence threshold (lower coverage) should have higher accuracy
    assert float(metrics['acc_at_cov70']) >= float(metrics['acc_at_cov90']) - 1.0


def test_predict_tta_and_rescale():
    """Verify apply_view rotations/flips and multi-scale interpolation."""
    B, C, H, W = 2, 3, 32, 32
    x = torch.randn(B, C, H, W)

    # 1. apply_view
    for r in range(4):
        for h in (0, 1):
            for v in (0, 1):
                out = apply_view(x, r, h, v)
                assert out.shape == (B, C, H, W)

    # Identity check
    assert torch.allclose(apply_view(x, 0, 0, 0), x)

    # 2. rescale
    # Scale factor 1.0 is exact identity
    assert torch.allclose(rescale(x, 1.0), x)

    # Scale factors 0.9 and 1.1 preserve shape exactly
    for s in (0.9, 1.1):
        x_scaled = rescale(x, s)
        assert x_scaled.shape == (B, C, H, W)


def test_ckpt_owner_resolution():
    """Verify ckpt_owner traces back through ckpt_from and base chains."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create owner folder with dummy fold0.pth
        owner_dir = os.path.join('checkpoints', 'owner_test_tag')
        os.makedirs(owner_dir, exist_ok=True)
        with open(os.path.join(owner_dir, 'fold0.pth'), 'w') as f:
            f.write('dummy')

        # Create intermediate tag pointing to owner
        inter_dir = os.path.join('checkpoints', 'inter_test_tag')
        os.makedirs(inter_dir, exist_ok=True)
        with open(os.path.join(inter_dir, 'config_snapshot.yaml'), 'w') as f:
            yaml.safe_dump({'ckpt_from': 'owner_test_tag'}, f)

        # Query ckpt_owner with child config pointing to inter
        child_cfg = {'ckpt_from': 'inter_test_tag'}
        resolved = ckpt_owner(child_cfg)
        assert resolved == 'owner_test_tag'

        # Cleanup
        import shutil
        shutil.rmtree(owner_dir)
        shutil.rmtree(inter_dir)


# The calibrate_conformal / ensemble / inference workflows are exercised end to end on a synthetic
# project with every evaluation set in tests/test_pipeline.py.
