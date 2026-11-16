"""Conformal fix of 2026-11-16: OOF calibration on the temperature-scaled scale, and float tolerance.

Before the fix the threshold was calibrated on the raw (over-confident) OOF probabilities but applied to the
temperature-scaled y_prob of the evaluation sets, and the greedy set growth compared floats without tolerance.
For the main model this gave qhat = 1 (up to rounding) and every prediction set contained all five grades.
"""
import json
import os

import numpy as np
import pandas as pd

from src.evaluation import experiment as ex
from src.evaluation.conformal import compute_qhat, interval_scores, interval_set
from src.evaluation.metrics import apply_temperature


def test_interval_set_float_tolerance():
    p = np.array([0.1, 0.2, 0.7])
    s = float(interval_scores(p[None], np.array([1]))[0])        # 0.7 + 0.2 = 0.8999999999999999
    assert s < 0.9
    assert interval_set(p, 0.9) == (1, 2)                          # not widened to all three grades
    assert interval_set(p, 0.95) == (0, 2)


def test_qhat_capped_at_one():
    assert compute_qhat(np.array([0.5, 1.0000000000000002, 1.0000000000000002]), alpha=0.1) <= 1.0


def _softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def _synthetic(rng, n, sharp=4.0):
    """Ordinal data with label noise and over-confident raw probabilities."""
    y = rng.integers(0, 5, n)
    score = y + rng.normal(0, 0.9, n)
    z = -sharp * (np.arange(5)[None, :] - score[:, None]) ** 2
    return y, _softmax(z)


def test_calibrate_conformal_uses_temperature(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rng = np.random.default_rng(0)
    base, T = 'base_tag', 2.5
    bdir = tmp_path / 'results' / 'phase3b_joint' / base
    bdir.mkdir(parents=True)
    (tmp_path / 'checkpoints' / base).mkdir(parents=True)
    with open(tmp_path / 'checkpoints' / base / 'thresholds.json', 'w') as f:
        json.dump({'temperature': T}, f)
    y_oof, P_oof = _synthetic(rng, 2000)
    np.savez(bdir / 'oof.npz', ids=np.arange(2000).astype(str), y=y_oof, P=P_oof)
    y_t, P_raw = _synthetic(rng, 1500)
    np.savez(bdir / 'predictions_aptos_test.npz', ids=np.arange(1500).astype(str), y_true=y_t,
             y_pred=P_raw.argmax(1), y_prob=apply_temperature(P_raw, T), y_prob_raw=P_raw)
    monkeypatch.setattr(ex, 'eval_set_names', lambda cfg: ['aptos_test'])

    ex.run_calibrate_conformal({'tag': 'conf_tag', 'phase': 'phase8_inference', 'base': base,
                                'conformal_alpha': 0.10})
    r = pd.read_csv(tmp_path / 'results' / 'phase8_inference' / 'conf_tag' / 'conformal.csv').iloc[0]

    qhat_raw = compute_qhat(interval_scores(P_oof, y_oof), alpha=0.10)
    assert abs(float(r['temperature']) - T) < 1e-9
    assert float(r['qhat']) < qhat_raw                       # calibrated on the tempered scale
    assert float(r['mean_width']) < 5.0                      # sets are no longer all five grades
    assert float(r['empirical_coverage']) >= 88.0            # exchangeable data: about 90 % coverage
