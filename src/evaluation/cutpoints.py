"""Ordinal cut-point analysis: is external under-grading a cut-point (threshold) problem?

Every model outputs class probabilities P (N, 5). Its expected grade E = sum_k k * P_k is a single
severity score; for the rank-consistent coral_src head E is a monotone function of the learned latent
score, so cut-points on E and cut-points on the latent score partition images identically.

  fit_cutpoints(E, y)   4 ordered cut-points that maximise QWK on labelled images (coordinate search)
  apply_cutpoints(E, c) grades 0..4
  devcal(...)           cut-points fitted on a dataset's DEV split, applied to its TEST split
  fewshot_curve(...)    the same with only n labelled DEV images (n = 10, 25, 50, 100; repeated draws)

If refitting four numbers on a site's own DEV images recovers most of the external QWK / mild-recall
gap while the referable AUROC (a ranking metric, unchanged by cut-points) is already high, the
under-grading is a cut-point shift rather than a feature failure.
"""

import numpy as np

K_DEFAULT = 5
DEFAULT_CUTS = np.array([0.5, 1.5, 2.5, 3.5])


def expected_grade(P):
    P = np.asarray(P, dtype=float)
    return P @ np.arange(P.shape[1], dtype=float)


def qwk_from_pred(y, pred, K=K_DEFAULT):
    """Quadratic weighted kappa from integer labels (identical to sklearn's quadratic kappa)."""
    y = np.asarray(y, dtype=int)
    pred = np.asarray(pred, dtype=int)
    n = len(y)
    if n == 0:
        return float('nan')
    cm = np.bincount(y * K + pred, minlength=K * K).reshape(K, K).astype(float)
    W = (np.arange(K)[:, None] - np.arange(K)[None, :]) ** 2.0
    expected = np.outer(cm.sum(1), cm.sum(0)) / n
    den = (W * expected).sum()
    if den == 0:
        return 1.0 if (W * cm).sum() == 0 else 0.0
    return float(1.0 - (W * cm).sum() / den)


def apply_cutpoints(E, cuts):
    return np.digitize(np.asarray(E, dtype=float), np.asarray(cuts, dtype=float)).astype(int)


def fit_cutpoints(E, y, K=K_DEFAULT, n_grid=200, max_iter=10):
    """Ordered cut-points c_1 < ... < c_{K-1} maximising QWK of apply_cutpoints(E, c) against y.

    Coordinate search over a grid of candidate values (quantiles of E plus the default half-integer
    cut-points), starting from the default cut-points 0.5, 1.5, 2.5, 3.5. Deterministic.
    """
    E = np.asarray(E, dtype=float)
    y = np.asarray(y, dtype=int)
    cuts = DEFAULT_CUTS[:K - 1].copy()
    if len(np.unique(y)) < 2 or len(y) < 2:
        return cuts
    grid = np.unique(np.concatenate([np.quantile(E, np.linspace(0, 1, n_grid)), DEFAULT_CUTS[:K - 1],
                                     [E.min() - 1e-6, E.max() + 1e-6]]))
    best = qwk_from_pred(y, apply_cutpoints(E, cuts), K)
    for _ in range(max_iter):
        improved = False
        for j in range(K - 1):
            lo = cuts[j - 1] if j > 0 else -np.inf
            hi = cuts[j + 1] if j < K - 2 else np.inf
            cand = grid[(grid > lo) & (grid < hi)]
            for v in cand:
                trial = cuts.copy()
                trial[j] = v
                q = qwk_from_pred(y, apply_cutpoints(E, trial), K)
                if q > best + 1e-12:
                    best, cuts, improved = q, trial, True
        if not improved:
            break
    return cuts


def grade_metrics(y, pred, K=K_DEFAULT):
    """Grade-level metrics for hard predictions (used for cut-point recalibrated predictions)."""
    y = np.asarray(y, dtype=int)
    pred = np.asarray(pred, dtype=int)
    n = max(1, len(y))
    cm = np.bincount(y * K + pred, minlength=K * K).reshape(K, K).astype(float)
    tp = np.diag(cm)
    rec = np.divide(tp, cm.sum(1), out=np.zeros(K), where=cm.sum(1) > 0)
    prec = np.divide(tp, cm.sum(0), out=np.zeros(K), where=cm.sum(0) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(K), where=(prec + rec) > 0)
    yb, pb = (y >= 2).astype(int), (pred >= 2).astype(int)
    out = {
        'acc': 100.0 * tp.sum() / n,
        'qwk': qwk_from_pred(y, pred, K),
        'f1_macro': float(f1.mean()),
        'mild_recall': float(rec[1]),
        'pred_nodr_share': 100.0 * float((pred == 0).mean()) if len(pred) else float('nan'),
        'bin_sens': float(pb[yb == 1].mean()) if (yb == 1).any() else float('nan'),
        'bin_spec': float(1 - pb[yb == 0].mean()) if (yb == 0).any() else float('nan'),
    }
    for c in range(K):
        out[f'rec_c{c}'] = float(rec[c])
    return out


def devcal(P_dev, y_dev, P_test, y_test, K=K_DEFAULT):
    """Cut-points fitted on DEV, applied to TEST. Returns (cuts, metrics on TEST)."""
    cuts = fit_cutpoints(expected_grade(P_dev), y_dev, K)
    pred = apply_cutpoints(expected_grade(P_test), cuts)
    return cuts, grade_metrics(y_test, pred, K)


def fewshot_curve(P_dev, y_dev, P_test, y_test, ns=(10, 25, 50, 100), draws=20, seed=0, K=K_DEFAULT):
    """Few-shot cut-point refit: n random labelled DEV images -> 4 cut-points -> metrics on TEST.

    Rows: n=0 'argmax' (no target labels, the model's own prediction), n=0 'default_cuts'
    (E with cut-points 0.5..3.5), every n in ns (mean and SD over `draws` random draws), and
    n=len(DEV) 'all' (full DEV refit = devcal).
    """
    P_dev = np.asarray(P_dev, dtype=float)
    P_test = np.asarray(P_test, dtype=float)
    y_dev = np.asarray(y_dev, dtype=int)
    y_test = np.asarray(y_test, dtype=int)
    E_dev, E_test = expected_grade(P_dev), expected_grade(P_test)
    rows = []

    def _row(n, kind, mets_list):
        r = {'n': int(n), 'kind': kind, 'draws': len(mets_list)}
        for key in ('qwk', 'acc', 'mild_recall', 'pred_nodr_share', 'f1_macro', 'bin_sens', 'bin_spec'):
            vals = np.array([m[key] for m in mets_list], dtype=float)
            r[f'{key}_mean'] = float(np.nanmean(vals))
            r[f'{key}_sd'] = float(np.nanstd(vals)) if len(vals) > 1 else 0.0
        return r

    rows.append(_row(0, 'argmax', [grade_metrics(y_test, P_test.argmax(1), K)]))
    rows.append(_row(0, 'default_cuts', [grade_metrics(y_test, apply_cutpoints(E_test, DEFAULT_CUTS[:K - 1]), K)]))
    rng = np.random.default_rng(seed)
    for n in ns:
        if n >= len(y_dev):
            continue
        mets = []
        for _ in range(int(draws)):
            idx = rng.choice(len(y_dev), size=int(n), replace=False)
            cuts = fit_cutpoints(E_dev[idx], y_dev[idx], K)
            mets.append(grade_metrics(y_test, apply_cutpoints(E_test, cuts), K))
        rows.append(_row(n, 'fewshot', mets))
    cuts = fit_cutpoints(E_dev, y_dev, K)
    rows.append(_row(len(y_dev), 'all', [grade_metrics(y_test, apply_cutpoints(E_test, cuts), K)]))
    return rows
