"""Evaluation metrics: 5-class, binary (referable), calibration.

All functions take y (N,) int and P (N, 5) float probability arrays.
Numbers are returned as raw floats; use io_utils.fmt2 for 2-decimal formatting.
"""

import numpy as np
from sklearn.metrics import (
    cohen_kappa_score,
    precision_score, recall_score, f1_score,
    precision_recall_fscore_support,
    roc_auc_score, average_precision_score,
    roc_curve, confusion_matrix,
)


# ── 5-class metrics ──────────────────────────────────────────────────────

def metrics5(y, P):
    """Compute 5-class DR grading metrics.

    Args:
        y: (N,) true labels 0..4.
        P: (N, 5) predicted probabilities.

    Returns:
        dict with keys: acc, qwk, prec_macro, rec_macro, f1_macro,
        adj_acc, mae, mild_recall, pred_nodr_share,
        f1_c0..c4, rec_c0..c4, prec_c0..c4, support_c0..c4.
    """
    pred = P.argmax(1)
    n = len(y)
    acc = 100.0 * (pred == y).sum() / n
    qwk = cohen_kappa_score(y, pred, weights='quadratic')
    prec_macro = precision_score(y, pred, average='macro',
                                 labels=range(5), zero_division=0)
    rec_macro = recall_score(y, pred, average='macro',
                             labels=range(5), zero_division=0)
    f1_macro = f1_score(y, pred, average='macro',
                        labels=range(5), zero_division=0)

    # Per-class P/R/F1/support
    prec_pc, rec_pc, f1_pc, sup_pc = precision_recall_fscore_support(
        y, pred, labels=range(5), zero_division=0)

    adj_acc = 100.0 * (np.abs(pred - y) <= 1).sum() / n
    mae = float(np.abs(pred - y).mean())

    # Mild recall (class 1)
    mask1 = (y == 1)
    mild_recall = float((pred[mask1] == 1).mean()) if mask1.any() else 0.0

    # Predicted-No-DR share
    pred_nodr_share = 100.0 * (pred == 0).sum() / n

    out = dict(
        acc=acc, qwk=qwk,
        prec_macro=prec_macro, rec_macro=rec_macro, f1_macro=f1_macro,
        adj_acc=adj_acc, mae=mae,
        mild_recall=mild_recall, pred_nodr_share=pred_nodr_share,
    )
    for c in range(5):
        out[f'f1_c{c}'] = float(f1_pc[c])
        out[f'rec_c{c}'] = float(rec_pc[c])
        out[f'prec_c{c}'] = float(prec_pc[c])
        out[f'support_c{c}'] = int(sup_pc[c])
    return out


def confusion(y, P):
    """Return 5x5 confusion matrix (rows=true, cols=predicted)."""
    return confusion_matrix(y, P.argmax(1), labels=range(5))


# ── Binary (referable) metrics ───────────────────────────────────────────

def metrics_bin(y, P, thresholds=None):
    """Compute referable-DR binary metrics.

    Referable = grades 2, 3, 4 vs. 0, 1.
    Score = P[:, 2] + P[:, 3] + P[:, 4].

    Args:
        y: (N,) true labels 0..4.
        P: (N, 5) predicted probabilities.
        thresholds: dict with keys 'thr_spec90', 'thr_spec95' (from OOF fit).

    Returns:
        dict with: bin_acc (%), bin_prec, bin_sens, bin_spec, bin_f1,
        bin_auroc, bin_auprc, sens_at_spec90, sens_at_spec95.
    """
    y_bin = (y >= 2).astype(int)
    s = P[:, 2:].sum(1)
    pred = (s >= 0.5).astype(int)

    acc = 100.0 * (pred == y_bin).mean()
    prec = precision_score(y_bin, pred, zero_division=0)
    sens = recall_score(y_bin, pred, zero_division=0)
    # Specificity = recall of the negative class
    spec = recall_score(1 - y_bin, 1 - pred, zero_division=0)
    f1 = f1_score(y_bin, pred, zero_division=0)

    if len(np.unique(y_bin)) > 1:
        auroc = roc_auc_score(y_bin, s)
        auprc = average_precision_score(y_bin, s)
    else:
        auroc = float('nan')
        auprc = float('nan')

    out = dict(bin_acc=acc, bin_prec=prec, bin_sens=sens, bin_spec=spec,
               bin_f1=f1, bin_auroc=auroc, bin_auprc=auprc,
               sens_at_spec90=float('nan'), sens_at_spec95=float('nan'))

    if thresholds:
        for key, col in [('thr_spec90', 'sens_at_spec90'),
                         ('thr_spec95', 'sens_at_spec95')]:
            if key in thresholds:
                thr = thresholds[key]
                pred_t = (s >= thr).astype(int)
                out[col] = recall_score(y_bin, pred_t, zero_division=0)

    return out


def compute_oof_thresholds(y, P):
    """Fit referable thresholds on pooled OOF probabilities.

    Returns dict: thr_spec90, thr_spec95 (threshold on referable score
    s = P[:, 2:].sum(1) that achieves >= 90% / 95% specificity on OOF).
    """
    y_bin = (y >= 2).astype(int)
    s = P[:, 2:].sum(1)
    if len(np.unique(y_bin)) < 2:
        return {'thr_spec90': 0.5, 'thr_spec95': 0.5}
    fpr, tpr, thresholds = roc_curve(y_bin, s)
    out = {}
    for spec_target, key in [(0.90, 'thr_spec90'), (0.95, 'thr_spec95')]:
        # specificity = 1 - fpr; want fpr <= 1 - spec_target
        max_fpr = 1.0 - spec_target
        idx = np.where(fpr <= max_fpr)[0]
        if len(idx) > 0:
            out[key] = float(thresholds[idx[-1]])
        else:
            out[key] = float(thresholds[0])
    return out


# ── Calibration ──────────────────────────────────────────────────────────

def ece(P, y, bins=15):
    """Expected Calibration Error.

    ECE = sum_b |acc_b - conf_b| * frac_b
    """
    confidences = P.max(1)
    predictions = P.argmax(1)
    correct = (predictions == y).astype(float)
    bin_edges = np.linspace(0, 1, bins + 1)
    ece_val = 0.0
    n = len(y)
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        acc_b = correct[mask].mean()
        conf_b = confidences[mask].mean()
        frac_b = mask.sum() / n
        ece_val += abs(acc_b - conf_b) * frac_b
    return float(ece_val)


def nll(P, y):
    """Negative log-likelihood (mean cross-entropy)."""
    eps = 1e-15
    probs = np.clip(P[np.arange(len(y)), y], eps, 1.0)
    return float(-np.log(probs).mean())


def brier(P, y, K=5):
    """Multi-class Brier score."""
    onehot = np.eye(K)[y]
    return float(((P - onehot) ** 2).sum(1).mean())


def fit_temperature(log_probs, y):
    """Fit temperature T on OOF log-probabilities by minimising NLL.

    Args:
        log_probs: (N, K) log-probabilities (pre-softmax or log_softmax).
        y: (N,) true labels.

    Returns:
        Optimal temperature T (float > 0).
    """
    from scipy.optimize import minimize_scalar

    def _nll_at_T(T):
        scaled = log_probs / max(T, 1e-4)
        # Stable softmax
        shifted = scaled - scaled.max(1, keepdims=True)
        exp_s = np.exp(shifted)
        log_sum = np.log(exp_s.sum(1))
        ll = shifted[np.arange(len(y)), y] - log_sum
        return -ll.mean()

    res = minimize_scalar(_nll_at_T, bounds=(0.1, 10.0), method='bounded')
    return float(res.x)


def apply_temperature(P, T):
    """Apply temperature scaling to probability matrix.

    Args:
        P: (N, K) probabilities.
        T: temperature scalar.

    Returns:
        (N, K) re-calibrated probabilities.
    """
    log_p = np.log(np.clip(P, 1e-15, 1.0))
    scaled = log_p / max(T, 1e-4)
    shifted = scaled - scaled.max(1, keepdims=True)
    exp_s = np.exp(shifted)
    return exp_s / exp_s.sum(1, keepdims=True)
