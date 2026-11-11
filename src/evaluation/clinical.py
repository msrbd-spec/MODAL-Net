"""Clinically oriented metrics, operating points and cluster bootstrap for 5-grade DR predictions.

  full_metrics(y, pred, score, pref)   grade_metrics + MAE, adjacent accuracy, per-class recall, confusion
                                       matrix, referrals / missed referable per 1000, ordinal c-index and
                                       AUROCs of the severity score E, referable AUROC of P(grade >= 2)
  ppv_npv(sens, spec, prevalence)      predictive values at an assumed referable prevalence
  threshold_for_specificity(...)       operating point chosen on labelled DEV images
  binary_at_threshold(...)             sensitivity / specificity of score >= threshold
  sens_at_spec_oracle(...)             best sensitivity at a specificity on the SAME labelled set (reporting
                                       only: an oracle operating point)
  cluster_bootstrap_indices / _weights resample whole groups (patients / exams) with replacement
  batch_metrics_from_weights(...)      qwk, mild_recall, bin_sens, bin_spec, f1_macro, acc for many bootstrap
                                       resamples at once (resample = per-image multiplicity weights)

Referable = grade >= 2 (moderate NPDR or worse), as everywhere in the project.
"""

import numpy as np
from scipy.stats import rankdata

from src.evaluation.cutpoints import grade_metrics

K_DEFAULT = 5
PREVALENCES = (0.05, 0.10, 0.20, 0.30)


def _auroc(ybin, s):
    """Mann-Whitney AUROC with mid-ranks (ties count 1/2); NaN when one class is absent."""
    ybin = np.asarray(ybin, dtype=bool)
    s = np.asarray(s, dtype=float)
    n1, n0 = int(ybin.sum()), int((~ybin).sum())
    if n1 == 0 or n0 == 0:
        return float('nan')
    r = rankdata(s)
    return float((r[ybin].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def ordinal_c_index(y, score, K=K_DEFAULT):
    """Harrell-style concordance of a scalar score with ordinal grades.

    Over all image pairs with different grades: share with the higher-graded image having the higher
    score, ties in the score counting 1/2. Computed exactly as the pair-count weighted mean of the pairwise
    Mann-Whitney AUROCs of every grade pair (a < b). NaN when fewer than two grades are present.
    """
    y = np.asarray(y, dtype=int)
    s = np.asarray(score, dtype=float)
    num, den = 0.0, 0.0
    for a in range(K):
        for b in range(a + 1, K):
            ma, mb = y == a, y == b
            na, nb = int(ma.sum()), int(mb.sum())
            if na == 0 or nb == 0:
                continue
            m = ma | mb
            num += _auroc(mb[m], s[m]) * na * nb
            den += na * nb
    return float(num / den) if den > 0 else float('nan')


def full_metrics(y, pred, score=None, pref=None, K=K_DEFAULT):
    """All grade-level metrics of hard predictions `pred` against labels `y`.

    Keys: everything in cutpoints.grade_metrics (acc %, qwk, f1_macro, mild_recall, pred_nodr_share %,
    bin_sens, bin_spec, rec_c0..), plus n, mae, adj_acc (% within one grade), recall_0..recall_{K-1},
    cm_i_j (count of true i predicted j), referrals_per_1000 (pred >= 2), missed_referable_per_1000
    (true >= 2 but pred < 2), false_referrals_per_1000 (true < 2, pred >= 2), referable_prevalence.
    With score (E): c_index, auroc_ovr_k (grade k vs rest scored by -|E - k|; for k = 0 and K-1 this is
    exactly -E and E) and auroc_ge_k (grade >= k vs < k, scored by E, k = 1..K-1).
    With pref (P(grade >= 2)): auroc_referable.
    """
    y = np.asarray(y, dtype=int)
    pred = np.asarray(pred, dtype=int)
    n = len(y)
    out = dict(grade_metrics(y, pred, K))
    out['n'] = n
    nn = max(n, 1)
    out['mae'] = float(np.abs(pred - y).mean()) if n else float('nan')
    out['adj_acc'] = 100.0 * float((np.abs(pred - y) <= 1).sum()) / nn
    cm = np.bincount(y * K + pred, minlength=K * K).reshape(K, K)
    for c in range(K):
        out[f'recall_{c}'] = out[f'rec_c{c}']
    for i in range(K):
        for j in range(K):
            out[f'cm_{i}_{j}'] = int(cm[i, j])
    yb, pb = y >= 2, pred >= 2
    out['referable_prevalence'] = float(yb.mean()) if n else float('nan')
    out['referrals_per_1000'] = 1000.0 * float(pb.sum()) / nn
    out['missed_referable_per_1000'] = 1000.0 * float((yb & ~pb).sum()) / nn
    out['false_referrals_per_1000'] = 1000.0 * float((~yb & pb).sum()) / nn
    if score is not None:
        s = np.asarray(score, dtype=float)
        out['c_index'] = ordinal_c_index(y, s, K)
        for k in range(K):
            out[f'auroc_ovr_{k}'] = _auroc(y == k, -np.abs(s - k))
        for k in range(1, K):
            out[f'auroc_ge_{k}'] = _auroc(y >= k, s)
    if pref is not None:
        out['auroc_referable'] = _auroc(yb, np.asarray(pref, dtype=float))
    return out


def ppv_npv(sens, spec, prevalence):
    """(PPV, NPV) of a test with sensitivity `sens` and specificity `spec` at referable prevalence p.

    PPV = sens p / (sens p + (1 - spec)(1 - p)),  NPV = spec (1 - p) / (spec (1 - p) + (1 - sens) p).
    NaN where a denominator is 0 or an input is NaN (no silent substitution). Works element-wise.
    """
    sens, spec, p = (np.asarray(v, dtype=float) for v in (sens, spec, prevalence))
    with np.errstate(divide='ignore', invalid='ignore'):
        a, b = sens * p, (1 - spec) * (1 - p)
        c, d = spec * (1 - p), (1 - sens) * p
        ppv = np.where((a + b) > 0, a / (a + b), np.nan)
        npv = np.where((c + d) > 0, c / (c + d), np.nan)
    if ppv.ndim == 0:
        return float(ppv), float(npv)
    return ppv, npv


def threshold_for_specificity(score_dev, ybin_dev, target_spec):
    """Threshold t (positive = score >= t) reaching specificity >= target_spec on labelled DEV images.

    t is placed just above the ceil(target_spec * n_neg)-th smallest negative score: midway to the next
    larger DEV score (any label), or nextafter() when none is larger. Among thresholds that reach the target
    on DEV this is (up to the midpoint placement) the most sensitive one. NaN when DEV has no negatives.
    This is a LABELLED baseline: it uses every DEV label.
    """
    s = np.asarray(score_dev, dtype=float)
    yb = np.asarray(ybin_dev, dtype=bool)
    neg = np.sort(s[~yb])
    if len(neg) == 0:
        return float('nan')
    k = int(np.ceil(float(target_spec) * len(neg) - 1e-9))
    if k <= 0:
        return float(neg[0])
    v = neg[min(k, len(neg)) - 1]
    above = s[s > v]
    return float((v + above.min()) / 2.0) if len(above) else float(np.nextafter(v, np.inf))


def binary_at_threshold(score, ybin, threshold):
    """{'sens', 'spec', 'referral_share'} of score >= threshold (NaN for an absent class, and all NaN for a
    NaN threshold, e.g. threshold_for_specificity on a DEV split without negatives)."""
    s = np.asarray(score, dtype=float)
    yb = np.asarray(ybin, dtype=bool)
    if threshold is None or not np.isfinite(float(threshold)):
        return {'sens': float('nan'), 'spec': float('nan'), 'referral_share': float('nan')}
    pos = s >= threshold
    return {'sens': float(pos[yb].mean()) if yb.any() else float('nan'),
            'spec': float(1 - pos[~yb].mean()) if (~yb).any() else float('nan'),
            'referral_share': float(pos.mean()) if len(pos) else float('nan')}


def roc_points(score, ybin):
    """(fpr, tpr) of every threshold (score >= t), from (0, 0) to (1, 1)."""
    s = np.asarray(score, dtype=float)
    yb = np.asarray(ybin, dtype=bool)
    t = np.unique(s)[::-1]
    P, N = max(int(yb.sum()), 1), max(int((~yb).sum()), 1)
    tpr = np.array([0.0] + [float((s[yb] >= v).sum()) / P for v in t])
    fpr = np.array([0.0] + [float((s[~yb] >= v).sum()) / N for v in t])
    return fpr, tpr


def sens_at_spec_oracle(score, ybin, target_spec, return_spec=False):
    """Largest sensitivity with specificity >= target over thresholds chosen on the SAME labelled images.
    Oracle (uses the evaluation labels for the threshold): reporting only, never a method.

    With return_spec=True returns (sens, spec) where spec is the specificity ACHIEVED at that operating
    point (>= target; the largest one among the points with that sensitivity), so that PPV / NPV are
    computed at the real operating point rather than at the target."""
    fpr, tpr = roc_points(score, ybin)
    ok = (1 - fpr) >= target_spec - 1e-12
    if not ok.any():
        return (float('nan'), float('nan')) if return_spec else float('nan')
    sens = float(tpr[ok].max())
    if not return_spec:
        return sens
    best = ok & (tpr >= sens - 1e-12)
    return sens, float((1 - fpr[best]).max())


# ---------------------------------------------------------------------------
# Cluster bootstrap
# ---------------------------------------------------------------------------

def _group_codes(groups):
    _, codes = np.unique(np.asarray([str(g) for g in groups]), return_inverse=True)
    return codes


def cluster_bootstrap_weights(groups, B, rng):
    """(B, N) integer multiplicities: each resample draws G groups (G = number of distinct groups) with
    replacement; an image counts as many times as its group was drawn. Image-level bootstrap when every
    group is unique. Consumes rng exactly like cluster_bootstrap_indices."""
    codes = _group_codes(groups)
    G = int(codes.max()) + 1 if len(codes) else 0
    draws = rng.integers(0, G, size=(int(B), G))
    counts = np.stack([np.bincount(d, minlength=G) for d in draws]) if B else np.zeros((0, G), int)
    return counts[:, codes]


def cluster_bootstrap_indices(groups, B, rng):
    """List of B index arrays; resample b holds every image of each drawn group (with repetition)."""
    W = cluster_bootstrap_weights(groups, B, rng)
    return [np.repeat(np.arange(W.shape[1]), w) for w in W]


def batch_metrics_from_weights(y, pred, W, K=K_DEFAULT):
    """Metrics of `pred` vs `y` for every resample (row) of the multiplicity matrix W (B, N) at once.

    Identical definitions to cutpoints.grade_metrics evaluated on the resampled images. Returns a dict of
    (B,) arrays: qwk, acc, f1_macro, mild_recall, bin_sens, bin_spec.
    """
    y = np.asarray(y, dtype=int)
    pred = np.asarray(pred, dtype=int)
    onehot = np.zeros((len(y), K * K))
    onehot[np.arange(len(y)), y * K + pred] = 1.0
    cm = (np.asarray(W, dtype=float) @ onehot).reshape(-1, K, K)
    n = cm.sum((1, 2))
    row, col = cm.sum(2), cm.sum(1)
    Wq = (np.arange(K)[:, None] - np.arange(K)[None, :]) ** 2.0
    num = (cm * Wq).sum((1, 2))
    with np.errstate(divide='ignore', invalid='ignore'):
        exp = row[:, :, None] * col[:, None, :] / np.where(n > 0, n, 1)[:, None, None]
        den = (exp * Wq).sum((1, 2))
        qwk = np.where(den > 0, 1 - num / np.where(den > 0, den, 1), np.where(num == 0, 1.0, 0.0))
        qwk = np.where(n > 0, qwk, np.nan)
        tp = np.diagonal(cm, axis1=1, axis2=2)
        rec = np.where(row > 0, tp / np.where(row > 0, row, 1), 0.0)
        prec = np.where(col > 0, tp / np.where(col > 0, col, 1), 0.0)
        f1 = np.where(prec + rec > 0, 2 * prec * rec / np.where(prec + rec > 0, prec + rec, 1), 0.0)
        pos = cm[:, 2:, :].sum((1, 2))
        neg = cm[:, :2, :].sum((1, 2))
        bin_sens = np.where(pos > 0, cm[:, 2:, 2:].sum((1, 2)) / np.where(pos > 0, pos, 1), np.nan)
        bin_spec = np.where(neg > 0, cm[:, :2, :2].sum((1, 2)) / np.where(neg > 0, neg, 1), np.nan)
        acc = 100.0 * tp.sum(1) / np.where(n > 0, n, 1)
    return {'qwk': qwk, 'acc': acc, 'f1_macro': f1.mean(1), 'mild_recall': rec[:, 1],
            'bin_sens': bin_sens, 'bin_spec': bin_spec}
