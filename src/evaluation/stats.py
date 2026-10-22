"""Statistical tests: paired bootstrap, Holm correction, McNemar exact."""

import numpy as np
import scipy.stats


def paired_bootstrap(y, Pa, Pb, fn, B=10000, seed=0):
    """Paired bootstrap test for metric difference fn(y, Pa) - fn(y, Pb).

    Args:
        y:  (N,) true labels.
        Pa: (N, ...) predictions from system A (probabilities or classes).
        Pb: (N, ...) predictions from system B.
        fn: callable(y_subset, P_subset) -> scalar metric.
        B:  number of bootstrap resamples.
        seed: random seed for reproducibility.

    Returns:
        delta: point estimate fn(y, Pa) - fn(y, Pb).
        ci:    [2.5th, 97.5th] percentile of bootstrap distribution.
        p:     two-sided p-value.
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    d = np.empty(B)
    for b in range(B):
        i = rng.integers(0, n, n)
        d[b] = fn(y[i], Pa[i]) - fn(y[i], Pb[i])
    delta = fn(y, Pa) - fn(y, Pb)
    ci = np.percentile(d, [2.5, 97.5]).tolist()
    p = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
    return delta, ci, p


def holm(p_values):
    """Holm-Bonferroni step-down correction.

    Args:
        p_values: 1-D array of raw p-values.

    Returns:
        adj: array of adjusted p-values (same length), monotone non-decreasing
             in the order of the original indices when sorted by raw p.
    """
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    o = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(o):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj


def mcnemar_exact(y, pa, pb):
    """Exact McNemar test (two-sided, using binomial test).

    Args:
        y:  (N,) true labels.
        pa: (N,) predictions from system A (class labels).
        pb: (N,) predictions from system B (class labels).

    Returns:
        p-value (float).  Returns 1.0 if no discordant pairs.
    """
    a_correct = (pa == y)
    b_correct = (pb == y)
    n01 = int((~a_correct & b_correct).sum())   # A wrong, B right
    n10 = int((a_correct & ~b_correct).sum())    # A right, B wrong
    total = n01 + n10
    if total == 0:
        return 1.0
    return scipy.stats.binomtest(min(n01, n10), total, 0.5).pvalue
