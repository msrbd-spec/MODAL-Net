"""Ordinal-interval split conformal prediction and selective classification.

Implements:
- S7j: ordinal-interval split conformal prediction on pooled OOF.
- Interval scores and conformal quantile qhat.
- Greedy interval set expansion around argmax.
- Conformal metrics: empirical coverage, mean width, singleton share/accuracy,
  referable abstention (boundary 1|2 crossing).
- Selective prediction: risk-coverage curves, AURC, accuracy at 70/80/90% coverage.
- Diagnostic figure generation (Fig 17).
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.utils.io_utils import fmt2


def interval_scores(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute minimal contiguous-interval probability mass around argmax containing true label.

    Args:
        P: (N, K) predicted probability matrix.
        y: (N,) integer true labels in range 0..K-1.

    Returns:
        out: (N,) conformal non-conformity scores (mass required to cover y).
    """
    out = np.zeros(len(y), dtype=np.float64)
    K = P.shape[1]
    for i in range(len(y)):
        lo = hi = int(P[i].argmax())
        m = float(P[i, lo])
        while not (lo <= y[i] <= hi):
            l = float(P[i, lo - 1]) if lo > 0 else -1.0
            r = float(P[i, hi + 1]) if hi < K - 1 else -1.0
            if l >= r:
                lo -= 1
                m += float(P[i, lo])
            else:
                hi += 1
                m += float(P[i, hi])
        out[i] = m
    return out


def compute_qhat(scores_cal: np.ndarray, alpha: float = 0.10) -> float:
    """Compute the finite-sample conformal quantile threshold qhat.

    Level is ceil((n + 1) * (1 - alpha)) / n, with method='higher'.
    """
    n = len(scores_cal)
    if n == 0:
        return 1.0
    level = min(1.0, float(np.ceil((n + 1) * (1.0 - alpha)) / n))
    qhat = float(np.quantile(scores_cal, level, method='higher'))
    return qhat


def interval_set(p: np.ndarray, qhat: float) -> tuple:
    """Grow contiguous interval greedily around argmax until cumulative mass >= qhat.

    Args:
        p: (K,) probability distribution.
        qhat: conformal mass threshold.

    Returns:
        (lo, hi): tuple of lower and upper class indices (inclusive).
    """
    K = len(p)
    lo = hi = int(np.argmax(p))
    m = float(p[lo])
    while m < qhat and (lo > 0 or hi < K - 1):
        l = float(p[lo - 1]) if lo > 0 else -1.0
        r = float(p[hi + 1]) if hi < K - 1 else -1.0
        if l >= r:
            lo -= 1
            m += float(p[lo])
        else:
            hi += 1
            m += float(p[hi])
    return int(lo), int(hi)


def predict_intervals(P: np.ndarray, qhat: float) -> np.ndarray:
    """Predict conformal intervals for all samples in probability matrix P.

    Returns:
        (N, 2) array of [lo, hi] for each row.
    """
    n = len(P)
    intervals = np.empty((n, 2), dtype=int)
    for i in range(n):
        lo, hi = interval_set(P[i], qhat)
        intervals[i, 0] = lo
        intervals[i, 1] = hi
    return intervals


def selective_prediction_metrics(P: np.ndarray, y: np.ndarray) -> dict:
    """Compute risk-coverage profile, AURC, and accuracy at fixed coverages.

    Args:
        P: (N, K) predicted probabilities.
        y: (N,) ground-truth labels.

    Returns:
        Dictionary containing AURC and accuracy at 70%, 80%, 90% coverage.
    """
    n = len(y)
    if n == 0:
        return {
            'aurc': 0.0,
            'acc_at_cov70': 100.0,
            'acc_at_cov80': 100.0,
            'acc_at_cov90': 100.0,
        }

    preds = P.argmax(axis=1)
    conf = P.max(axis=1)
    correct = (preds == y).astype(float)

    # Sort in descending order of confidence
    order = np.argsort(-conf)
    sorted_correct = correct[order]

    # Evaluate at coverage grid from 1/n to 1.0
    coverages = np.linspace(1.0 / n, 1.0, min(n, 200))
    risks = []
    accs = []
    for cov in coverages:
        k = max(1, int(np.round(cov * n)))
        acc_k = float(sorted_correct[:k].mean())
        accs.append(acc_k * 100.0)
        risks.append(1.0 - acc_k)

    # AURC (trapezoidal integration of risk across coverage)
    try:
        from scipy.integrate import trapezoid
        aurc = float(trapezoid(risks, coverages))
    except ImportError:
        aurc = float(np.trapz(risks, coverages))

    # Interpolate accuracy at specific coverage levels: 70%, 80%, 90%
    def get_acc_at_cov(target_cov):
        idx = int(np.clip(int(np.round(target_cov * n)), 1, n))
        return float(sorted_correct[:idx].mean() * 100.0)

    return {
        'aurc': aurc,
        'acc_at_cov70': get_acc_at_cov(0.70),
        'acc_at_cov80': get_acc_at_cov(0.80),
        'acc_at_cov90': get_acc_at_cov(0.90),
    }


def evaluate_conformal_predictions(P: np.ndarray, y: np.ndarray, qhat: float, alpha: float = 0.10) -> dict:
    """Evaluate split conformal prediction and selective classification metrics.

    Args:
        P: (N, K) predicted probabilities.
        y: (N,) ground-truth labels.
        qhat: calibration cutoff.
        alpha: nominal error rate (default 0.10 => 90% target coverage).

    Returns:
        Dictionary of formatted metrics.
    """
    n = len(y)
    intervals = predict_intervals(P, qhat)
    lo = intervals[:, 0]
    hi = intervals[:, 1]

    # Coverage
    covered = (y >= lo) & (y <= hi)
    coverage = float(np.mean(covered))

    # Interval width (number of classes in interval: hi - lo + 1)
    widths = hi - lo + 1
    mean_width = float(np.mean(widths))

    # Singletons
    singletons = (lo == hi)
    singleton_share = float(np.mean(singletons)) * 100.0
    singleton_acc = float(np.mean(y[singletons] == lo[singletons]) * 100.0) if singletons.any() else 0.0

    # Abstention for referable decisions:
    # Referable DR is grade >= 2. Non-referable is 0 or 1.
    # An interval crossing the boundary (lo <= 1 and hi >= 2) abstains.
    crossing_boundary = (lo <= 1) & (hi >= 2)
    abstention_referable = float(np.mean(crossing_boundary)) * 100.0

    # Selective prediction metrics
    sel_metrics = selective_prediction_metrics(P, y)

    return {
        'target_coverage': fmt2((1.0 - alpha) * 100.0),
        'qhat': fmt2(qhat),
        'empirical_coverage': fmt2(coverage * 100.0),
        'mean_width': fmt2(mean_width),
        'singleton_share': fmt2(singleton_share),
        'singleton_acc': fmt2(singleton_acc),
        'abstention_referable': fmt2(abstention_referable),
        'aurc': fmt2(sel_metrics['aurc']),
        'acc_at_cov70': fmt2(sel_metrics['acc_at_cov70']),
        'acc_at_cov80': fmt2(sel_metrics['acc_at_cov80']),
        'acc_at_cov90': fmt2(sel_metrics['acc_at_cov90']),
    }


def plot_conformal_diagnostics(
    intervals: np.ndarray,
    y: np.ndarray,
    qhat: float,
    alpha: float,
    out_dir: str,
    prefix: str = 'conformal'
):
    """Generate Fig 17 diagnostic plots: coverage vs target and interval width distribution."""
    os.makedirs(out_dir, exist_ok=True)
    lo, hi = intervals[:, 0], intervals[:, 1]
    widths = hi - lo + 1
    covered = (y >= lo) & (y <= hi)
    emp_cov = float(np.mean(covered) * 100.0)
    target_cov = (1.0 - alpha) * 100.0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.0), tight_layout=True)

    # 1. Target vs Empirical coverage
    bars = ax1.bar(['Target', 'Empirical'], [target_cov, emp_cov], color=['#999999', '#0072B2'], width=0.5)
    ax1.set_ylabel('Coverage (%)')
    ax1.set_ylim(0, 105)
    for bar in bars:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., h + 2, f'{h:.2f}%', ha='center', va='bottom', fontsize=8)

    # 2. Interval width distribution
    width_counts = np.bincount(widths, minlength=6)[1:6]
    width_shares = (width_counts / len(widths)) * 100.0
    ax2.bar(range(1, 6), width_shares, color='#0072B2', width=0.6)
    ax2.set_xlabel('Interval Width (Classes)')
    ax2.set_ylabel('Share (%)')
    ax2.set_xticks(range(1, 6))

    from src.evaluation.figures import save_fig, set_bold_style
    set_bold_style()
    save_fig(fig, os.path.join(out_dir, prefix))
