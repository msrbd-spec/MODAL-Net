"""Few-label last-layer fine-tuning anchored to the original head (CPU, on saved features; no GPU).

Baseline for the few-shot cut-point recalibration of tools/adaptation_curve.py: with n labelled DEV images of
a new site, refit the final linear layer of every fold model instead of 4 cut-points.

Input: results/<phase>/<tag>/features/ written by `python main.py --mode extract_features --tag <tag>`
(CE / softmax heads only): <d>_{dev,test}.npz (ids, y_true, feats (n_folds, N, D)) and head.npz (W0, b0,
OOF temperature T, held-out datasets).

For every tag, held-out dataset d, n in --ns and draw (the EXACT draws of adaptation_curve: DEV images sorted
by id, adaptation_curve._draw_indices(n_dev, n, draws, d, seed); n >= n_dev is skipped as there), per fold k:

    min_{W, b}  mean_i CE(y_i, softmax((f_ik W^T + b) / T))  +  lam * (||W - W0_k||^2 + ||b - b0_k||^2)

over the n drawn labelled DEV images i (L-BFGS-B, scipy, analytic gradient, float64, start = W0_k, b0_k;
the problem is strictly convex, so the optimum is unique and deterministic). TEST prediction exactly as
evaluate.py forms the normal prediction: P = apply_temperature(mean_k softmax(f_k W_k^T + b_k), T), argmax
(T is the run's OOF temperature; T = 1 when the run has temperature_scaling off). With lam -> infinity this is
the run's own argmax prediction.

lam is chosen per tag, target dataset d and n by leave-one-site-out over the OTHER held-out datasets o != d
(the same idea as adaptation_curve's lambda selection): fit on n drawn DEV images of o
(_draw_indices(n_dev_o, n, max(5, draws // 2), o + '_inner', seed), sizes with n >= n_dev_o - 5 skipped),
score QWK on the rest of o's DEV split, average over draws and over the other sites, take the best lam of
the fixed grid LAMBDAS (ties -> larger lam, i.e. closer to the original head). A run with a single held-out
site has no other site: lam = DEFAULT_LAM (fixed a priori, recorded with chosen_from='').
The target's labels beyond its n drawn DEV images, and every TEST label, are never used for fitting or for
lam: the fitting functions receive DEV arrays only; TEST labels enter grade_metrics after prediction.

Outputs:
  --out (default results/master/recal/fewshot_head_draws.csv): one row per (tag, dataset, n, draw) with columns
      tag, dataset, method, info, n, draw, n_test, qwk, acc, f1_macro, mild_recall, bin_sens, bin_spec,
      pred_nodr_share       (method = 'fewshot_head_ft', info = 'n_labels'; metrics = cutpoints.grade_metrics)
  fewshot_head_lambda.csv next to it: tag, dataset, n, lam, chosen_from, inner_qwk_<lam> for every grid value

Usage:
  python tools/fewshot_head.py --tags p3b_joint,p10d_ce,p10q_ce [--ns 10,25,50,100] [--draws 20] [--seed 0]
                               [--out results/master/recal/fewshot_head_draws.csv]
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.cutpoints import grade_metrics, qwk_from_pred  # noqa: E402
from src.evaluation.metrics import apply_temperature  # noqa: E402
from src.utils.config import find_exp  # noqa: E402
from tools.adaptation_curve import _draw_indices, _load  # noqa: E402

METHOD = 'fewshot_head_ft'
INFO = 'n_labels'
COLUMNS = ['tag', 'dataset', 'method', 'info', 'n', 'draw', 'n_test', 'qwk', 'acc', 'f1_macro', 'mild_recall',
           'bin_sens', 'bin_spec', 'pred_nodr_share']
LAMBDAS = (0.01, 0.1, 1.0, 10.0, 100.0)
DEFAULT_LAM = 1.0
DATASETS = ('messidor', 'ddr', 'deepdrid', 'idrid')


# ---------------------------------------------------------------------------
# Head fitting (DEV arrays only)
# ---------------------------------------------------------------------------

def fit_head(F, y, W0, b0, T, lam, maxiter=1000):
    """Anchored multinomial logistic regression of one fold. F (n, D), y (n,). Returns (W (K, D), b (K,))."""
    F = np.asarray(F, dtype=np.float64)
    y = np.asarray(y, dtype=int)
    W0 = np.asarray(W0, dtype=np.float64)
    b0 = np.asarray(b0, dtype=np.float64)
    K, D = W0.shape
    n = len(y)
    Y = np.zeros((n, K))
    Y[np.arange(n), y] = 1.0
    T = float(T)
    lam = float(lam)
    theta0 = np.concatenate([W0.ravel(), b0])

    def fun(theta):
        W = theta[:K * D].reshape(K, D)
        b = theta[K * D:]
        Z = (F @ W.T + b) / T
        Z = Z - Z.max(1, keepdims=True)
        logS = Z - np.log(np.exp(Z).sum(1, keepdims=True))
        dW, db = W - W0, b - b0
        f = -(Y * logS).sum() / n + lam * ((dW ** 2).sum() + (db ** 2).sum())
        G = (np.exp(logS) - Y) / (n * T)
        gW = G.T @ F + 2.0 * lam * dW
        gb = G.sum(0) + 2.0 * lam * db
        return f, np.concatenate([gW.ravel(), gb])

    res = minimize(fun, theta0, jac=True, method='L-BFGS-B',
                   options={'maxiter': int(maxiter), 'gtol': 1e-9, 'ftol': 1e-14})
    th = res.x
    return th[:K * D].reshape(K, D), th[K * D:]


def fit_folds(F_folds, y, W0, b0, T, lam):
    """fit_head for every fold. F_folds (n_folds, n, D). Returns [(W_k, b_k)]."""
    return [fit_head(F_folds[k], y, W0[k], b0[k], T, lam) for k in range(len(W0))]


def fit_temperature_of(head):
    """Temperature inside the fitting loss: the run's OOF T, or 1 when the run has temperature scaling off
    (head['use_T'] False: its saved probabilities are unscaled, so the loss must be too)."""
    return float(head['T']) if head['use_T'] else 1.0


def predict(F_folds, heads, T, use_T):
    """Grades of images with features F_folds (n_folds, N, D) under fold heads [(W_k, b_k)]."""
    P = []
    for k, (W, b) in enumerate(heads):
        Z = np.asarray(F_folds[k], dtype=np.float64) @ W.T + b
        Z = Z - Z.max(1, keepdims=True)
        E = np.exp(Z)
        P.append(E / E.sum(1, keepdims=True))
    P = np.mean(P, axis=0)
    if use_T:
        P = apply_temperature(P, T)
    return P.argmax(1)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def _load_feats(exp_dir, ds, split):
    """(ids, y, feats (n_folds, N, D) float64) of a features file, sorted by id (adaptation_curve order)."""
    p = os.path.join(exp_dir, 'features', f'{ds}_{split}.npz')
    if not os.path.isfile(p):
        return None
    with np.load(p, allow_pickle=False) as f:
        ids = np.array([str(i) for i in f['ids']])
        y = np.asarray(f['y_true'], dtype=int)
        X = np.asarray(f['feats'], dtype=np.float64)
    order = np.argsort(ids, kind='stable')
    return ids[order], y[order], X[:, order]


def load_tag(tag):
    """{'head': {...}, 'data': {ds: {'dev': (ids, y, F), 'test': (ids, y, F)}}} or None without features."""
    exp_dir = find_exp(tag)
    hp = os.path.join(exp_dir, 'features', 'head.npz')
    if not os.path.isfile(hp):
        print(f'fewshot_head: {tag}: no features/head.npz -> skipped (run python main.py --mode extract_features '
              f'--tag {tag}; CE heads only)')
        return None
    with np.load(hp, allow_pickle=False) as f:
        head = {'W': np.asarray(f['W'], np.float64), 'b': np.asarray(f['b'], np.float64), 'T': float(f['T']),
                'use_T': bool(f['use_T']), 'heldout': [str(x) for x in np.atleast_1d(f['heldout'])]}
    data = {}
    for ds in [d for d in DATASETS if d in head['heldout']]:
        dev, test = _load_feats(exp_dir, ds, 'dev'), _load_feats(exp_dir, ds, 'test')
        if dev is None or test is None:
            print(f'fewshot_head: {tag}: no {ds} dev/test features -> skipped')
            continue
        for split, rec in (('dev', dev), ('test', test)):
            pred = _load(exp_dir, ds, split)          # the normal prediction file, ids sorted the same way
            if pred is None:
                print(f'fewshot_head: {tag}: no normal {ds}_{split} predictions to pair with; using feature ids')
            elif not np.array_equal(pred[0], rec[0]):
                raise ValueError(f'fewshot_head: {tag}: {ds}_{split} feature ids differ from the prediction file '
                                 f'ids (draws would not be paired with adaptation_curve)')
        data[ds] = {'dev': dev, 'test': test}
    return {'head': head, 'data': data}


# ---------------------------------------------------------------------------
# Lambda selection (other held-out sites' DEV only) and the per-draw table
# ---------------------------------------------------------------------------

def inner_table(dev_data, head, ns, draws, seed):
    """Q[(ds, n, lam)] = mean QWK on the rest of ds's DEV split after fitting on n drawn DEV images of ds.
    dev_data: {ds: (ids, y, F)} DEV splits only."""
    Q = {}
    for ds, (_, y, F) in dev_data.items():
        for n in ns:
            if n >= len(y) - 5:
                continue
            idx_list = _draw_indices(len(y), n, draws, ds + '_inner', seed)
            for lam in LAMBDAS:
                vals = []
                for idx in idx_list:
                    rest = np.setdiff1d(np.arange(len(y)), idx)
                    heads = fit_folds(F[:, idx], y[idx], head['W'], head['b'], fit_temperature_of(head), lam)
                    vals.append(qwk_from_pred(y[rest], predict(F[:, rest], heads, head['T'], head['use_T'])))
                Q[(ds, n, lam)] = float(np.mean(vals))
    return Q


def pick_lambda(Q, others, n):
    """Best grid lam for size n, averaged over the other sites (ties -> larger lam). None if no score."""
    scores = {}
    for lam in LAMBDAS:
        v = [Q[(o, n, lam)] for o in others if (o, n, lam) in Q]
        if v:
            scores[lam] = float(np.mean(v))
    if not scores:
        return None, scores
    return max(scores, key=lambda k: (scores[k], k)), scores


def analyse_tag(tag, ns, draws, seed, keep_fits=False, loaded=None):
    """Rows of the per-draw table, rows of the lambda table and (optionally) the fitted heads per
    (dataset, n, draw). TEST labels are only used by grade_metrics after the prediction."""
    loaded = loaded if loaded is not None else load_tag(tag)
    if not loaded or not loaded['data']:
        return [], [], {}
    head, data = loaded['head'], loaded['data']
    dev_only = {ds: d['dev'] for ds, d in data.items()}
    Q = inner_table(dev_only, head, ns, max(5, draws // 2), seed) if len(dev_only) > 1 else {}
    rows, lam_rows, fits = [], [], {}
    for ds, d in data.items():
        _, y_dev, F_dev = d['dev']
        _, y_test, F_test = d['test']
        others = [o for o in data if o != ds]
        for n in ns:
            if n >= len(y_dev):
                continue
            lam, scores = pick_lambda(Q, others, n)
            chosen_from = ','.join(o for o in others if any((o, n, l) in Q for l in LAMBDAS))
            if lam is None:
                lam, chosen_from = DEFAULT_LAM, ''
            lam_rows.append({'tag': tag, 'dataset': ds, 'n': int(n), 'lam': lam, 'chosen_from': chosen_from,
                             **{f'inner_qwk_{l:g}': scores.get(l, np.nan) for l in LAMBDAS}})
            for j, idx in enumerate(_draw_indices(len(y_dev), n, draws, ds, seed)):
                heads = fit_folds(F_dev[:, idx], y_dev[idx], head['W'], head['b'], fit_temperature_of(head), lam)
                pred = predict(F_test, heads, head['T'], head['use_T'])
                if keep_fits:
                    fits[(ds, int(n), j)] = heads
                m = grade_metrics(y_test, pred)
                rows.append({'tag': tag, 'dataset': ds, 'method': METHOD, 'info': INFO, 'n': int(n), 'draw': j,
                             'n_test': len(y_test), **{c: m[c] for c in COLUMNS[7:]}})
    return rows, lam_rows, fits


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tags', required=True, help='comma-separated CE-head tags with extracted features')
    ap.add_argument('--ns', default='10,25,50,100')
    ap.add_argument('--draws', type=int, default=20)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=os.path.join('results', 'master', 'recal', 'fewshot_head_draws.csv'))
    a = ap.parse_args()
    tags = [t.strip() for t in a.tags.split(',') if t.strip()]
    ns = [int(x) for x in a.ns.split(',') if x.strip()]
    rows, lam_rows = [], []
    for tag in tags:
        r, lr, _ = analyse_tag(tag, ns, a.draws, a.seed)
        rows += r
        lam_rows += lr
    if not rows:
        print('fewshot_head: no features found for any tag')
        sys.exit(1)
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    df = pd.DataFrame(rows, columns=COLUMNS)
    df.to_csv(a.out, index=False)
    lam_path = os.path.join(os.path.dirname(a.out) or '.', 'fewshot_head_lambda.csv')
    pd.DataFrame(lam_rows).to_csv(lam_path, index=False)
    show = df.groupby(['tag', 'dataset', 'n'])[['qwk', 'mild_recall', 'bin_sens']].mean().reset_index()
    with pd.option_context('display.width', 200, 'display.max_rows', 500):
        print(show.round(3).to_string(index=False))
    print(f'\nwritten: {a.out} ({len(df)} rows), {lam_path}')


if __name__ == '__main__':
    main()
