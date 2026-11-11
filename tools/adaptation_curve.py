"""Cut-point adaptation to a new site, on saved predictions (CPU, no GPU, no retraining).

Question: when a model meets a site it never saw, how much target-site information is needed to fix its
grading, and does an ordinal shared-severity head need less of it than a plain CE head?

Every model gives a single severity score E = sum_k k P_k (expected grade). Only the 4 cut-points on E are
adapted; the network is untouched. Levels of target information (all fitted on the site's DEV split only;
TEST is used only for reporting):

  L0  argmax          nothing: the model's own prediction
  L0  default_cuts    nothing: cut-points 0.5, 1.5, 2.5, 3.5 on E
  L1  prevalence_oracle  ORACLE: grade shares from the site's DEV labels; upper bound, not label-free.
                      With its unlabelled DEV images: cut-points = quantiles of E that reproduce the grade shares
  L2  fewshot         n labelled DEV images (n in --ns, --draws random draws): QWK-optimal cut-points
  L2  fewshot_shrink  the same, shrunk towards the default cut-points (QWK - lam * ||c - c0||^2). lam is
                      chosen per model and per target site by leave-one-site-out over the OTHER held-out
                      sites' DEV splits (fit on n DEV images, score on the rest of that DEV split); it never
                      sees the target site's labels beyond the n images, and never any TEST image.
  L3  all             every labelled DEV image (upper bound; = devcal); written with n = -1

Only the datasets a run never trained on are used (held-out sites, from the run's train_sources).
Few-shot draws are identical across the models of a group (DEV images sorted by id, same seed), so the
model comparison is paired.

Outputs (results/master/):
  adaptation_curve.csv     tag x dataset x level x n: mean / SD over draws of QWK, accuracy, macro F1,
                           mild recall, referable sensitivity / specificity, predicted-No-DR share
                           (dataset 'MEAN' = mean over the run's held-out datasets)
  adaptation_compare.csv   every model vs the first tag of its group, same level and n: mean paired
                           difference and 95 % interval (few-shot: over draws; other levels: paired TEST
                           bootstrap, each dataset resampled independently). The TEST bootstrap resamples
                           whole patients / exams when splits/external/<ds>_test.csv has a `group` column
                           whose ids match the TEST ids (DeepDRiD: 4 images per patient; Messidor-2: exam
                           pairs; matching exactly as tools/recalibration_suite._test_groups), else images;
                           both models of a pair always see the same resamples. Column bootstrap_unit
                           records the unit ('draws' for few-shot rows; MEAN: the per-dataset units)
  adaptation_lambda.csv    the lam chosen for every (tag, target dataset, n)
  results/figures/fig_adaptation.{png,pdf}   QWK vs n per group (mean over held-out datasets;
                           fig_adaptation_<out_dir name> when --out_dir is not results/master)

Usage:
  python tools/adaptation_curve.py --groups "p10d_ce,p10d_coralsh,p10d_coralsrc;p10q_ce,p10q_coralsh,p10q_coralsrc"
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.cutpoints import (DEFAULT_CUTS, apply_cutpoints, expected_grade, fit_cutpoints,  # noqa: E402
                                      grade_metrics, prevalence_cutpoints)
from src.utils.config import find_exp  # noqa: E402

DATASETS = ('messidor', 'ddr', 'deepdrid', 'idrid')
LEGACY = {('messidor', 'dev'): 'predictions_external_dev.npz', ('messidor', 'test'): 'predictions_external_test.npz'}
METRICS = ('qwk', 'acc', 'f1_macro', 'mild_recall', 'bin_sens', 'bin_spec', 'pred_nodr_share')
LAMBDAS = (0.0, 0.03, 0.1, 0.3, 1.0, 3.0)
K = 5


def _load(exp_dir, ds, split):
    for name in (f'predictions_{ds}_{split}.npz', LEGACY.get((ds, split))):
        if name and os.path.isfile(os.path.join(exp_dir, name)):
            with np.load(os.path.join(exp_dir, name), allow_pickle=True) as f:
                y = np.asarray(f['y_true'] if 'y_true' in f.files else f['y'], dtype=int)
                P = np.asarray(f['y_prob'] if 'y_prob' in f.files else f['P'], dtype=float)
                ids = np.array([str(i) for i in f['ids']])
            order = np.argsort(ids, kind='stable')          # identical order (and draws) across models
            return ids[order], y[order], P[order]
    return None


def _heldout(exp_dir):
    from src.evaluation.tables import _train_sources_of
    summ = pd.read_csv(os.path.join(exp_dir, 'summary.csv')).iloc[0].to_dict()
    src = _train_sources_of(summ, exp_dir)
    return [d for d in DATASETS if d not in src]


def _draw_indices(n_dev, n, draws, ds, seed):
    rng = np.random.default_rng([seed, int(n), sum(map(ord, ds))])
    return [rng.choice(n_dev, size=int(n), replace=False) for _ in range(int(draws))]


def _inner_table(data, ns, draws, seed):
    """Q[(ds, n, lam)] = mean QWK on the rest of ds's DEV split after a shrunk fit on n DEV images."""
    from src.evaluation.cutpoints import qwk_from_pred
    Q = {}
    for ds, d in data.items():
        E, y = expected_grade(d['P_dev']), d['y_dev']
        for n in ns:
            if n >= len(y) - 5:
                continue
            idx_list = _draw_indices(len(y), n, draws, ds + '_inner', seed)
            for lam in LAMBDAS:
                vals = []
                for idx in idx_list:
                    rest = np.setdiff1d(np.arange(len(y)), idx)
                    c = fit_cutpoints(E[idx], y[idx], prior=DEFAULT_CUTS, lam=lam)
                    vals.append(qwk_from_pred(y[rest], apply_cutpoints(E[rest], c)))
                Q[(ds, n, lam)] = float(np.mean(vals))
    return Q


def _pick_lambda(Q, target, others, n):
    scores = {lam: np.mean([Q[(o, n, lam)] for o in others if (o, n, lam) in Q] or [np.nan]) for lam in LAMBDAS}
    scores = {k: v for k, v in scores.items() if not np.isnan(v)}
    return max(scores, key=lambda k: (scores[k], -k)) if scores else 0.0     # ties -> smaller lam


def analyse_tag(tag, ns, draws, seed):
    exp_dir = find_exp(tag)
    data = {}
    for ds in _heldout(exp_dir):
        dev, test = _load(exp_dir, ds, 'dev'), _load(exp_dir, ds, 'test')
        if dev is None or test is None:
            print(f'adaptation_curve: {tag}: no {ds} dev/test predictions -> skipped '
                  f'(run --mode evaluate --tag {tag})')
            continue
        data[ds] = {'ids_dev': dev[0], 'y_dev': dev[1], 'P_dev': dev[2],
                    'ids_test': test[0], 'y_test': test[1], 'P_test': test[2]}
    if not data:
        return [], {}, []
    Q = _inner_table(data, ns, max(5, draws // 2), seed) if len(data) > 1 else {}
    rows, preds, lam_rows = [], {}, []
    for ds, d in data.items():
        E_dev, E_test, y_dev, y_test = expected_grade(d['P_dev']), expected_grade(d['P_test']), d['y_dev'], d['y_test']
        others = [o for o in data if o != ds]

        def add(level, n, pred_list, lam=np.nan):
            mets = [grade_metrics(y_test, p) for p in pred_list]
            r = {'tag': tag, 'dataset': ds, 'level': level, 'n': int(n), 'lam': lam, 'draws': len(pred_list),
                 'n_dev': len(y_dev), 'n_test': len(y_test)}
            for m in METRICS:
                v = np.array([x[m] for x in mets], dtype=float)
                r[f'{m}_mean'] = float(np.nanmean(v))
                r[f'{m}_sd'] = float(np.nanstd(v)) if len(v) > 1 else 0.0
            rows.append(r)
            preds[(ds, level, int(n))] = pred_list

        add('argmax', 0, [d['P_test'].argmax(1)])
        add('default_cuts', 0, [apply_cutpoints(E_test, DEFAULT_CUTS)])
        props = np.bincount(y_dev, minlength=K) / len(y_dev)
        add('prevalence_oracle', 0, [apply_cutpoints(E_test, prevalence_cutpoints(E_dev, props))])
        for n in ns:
            if n >= len(y_dev):
                continue
            idx_list = _draw_indices(len(y_dev), n, draws, ds, seed)
            add('fewshot', n, [apply_cutpoints(E_test, fit_cutpoints(E_dev[i], y_dev[i])) for i in idx_list], 0.0)
            lam = _pick_lambda(Q, ds, others, n) if others else 0.0
            lam_rows.append({'tag': tag, 'dataset': ds, 'n': int(n), 'lam': lam, 'chosen_from': ','.join(others)})
            add('fewshot_shrink', n, [apply_cutpoints(E_test, fit_cutpoints(E_dev[i], y_dev[i], prior=DEFAULT_CUTS,
                                                                             lam=lam)) for i in idx_list], lam)
        add('all', -1, [apply_cutpoints(E_test, fit_cutpoints(E_dev, y_dev))])     # n = -1: every DEV image
        preds[('_y', ds)] = y_test
        preds[('_ids', ds)] = np.concatenate([d['ids_dev'], ['|'], d['ids_test']])
    df = pd.DataFrame(rows)
    mean_rows = []
    for (level, n), g in df.groupby(['level', 'n'], sort=False):
        if g['dataset'].nunique() != len(data):
            continue
        r = {'tag': tag, 'dataset': 'MEAN', 'level': level, 'n': int(n), 'lam': np.nan,
             'draws': int(g['draws'].min()), 'n_dev': int(g['n_dev'].sum()), 'n_test': int(g['n_test'].sum())}
        for c in g.columns:
            if c.endswith('_mean'):
                r[c] = float(g[c].mean())
        mean_rows.append(r)
    return rows + mean_rows, preds, lam_rows


def _metric_values(y, pred_list, metric):
    return np.array([grade_metrics(y, p)[metric] for p in pred_list], dtype=float)


CMP_COLUMNS = ['reference', 'tag', 'dataset', 'level', 'n', 'metric', 'diff_mean', 'diff_lo', 'diff_hi',
               'n_resamples', 'bootstrap_unit']


class _SilentNotes:
    """Stand-in for recalibration_suite.Notes (its _test_groups records why it fell back to images)."""

    def __init__(self):
        self.rows = []

    def add(self, *args, **kwargs):
        self.rows.append(args)


def _stem(x):
    return os.path.splitext(os.path.basename(str(x)))[0]


def _test_groups_local(ds, ids_test):
    """Identical rule to tools/recalibration_suite._test_groups (used only if that cannot be imported):
    split-file id_code matched exactly, else by file stem; every TEST id must be found."""
    p = os.path.join('splits', 'external', f'{ds}_test.csv')
    if not os.path.isfile(p):
        return np.asarray(ids_test), 'image (no split file)'
    d = pd.read_csv(p, dtype={'id_code': str, 'group': str}, keep_default_na=False)
    if 'group' not in d.columns:
        return np.asarray(ids_test), 'image (split file has no group column)'
    g = dict(zip(d['id_code'].astype(str), d['group'].astype(str)))
    g.update({_stem(k): v for k, v in list(g.items()) if _stem(k) not in g})
    out = [g.get(str(i), g.get(_stem(i))) for i in ids_test]
    if any(v is None or v == '' for v in out):
        return np.asarray(ids_test), 'image (ids not matched in split file)'
    out = np.asarray(out)
    if len(np.unique(out)) == len(out):
        return out, 'image'
    return out, f'cluster (group column; {len(np.unique(out))} groups / {len(out)} images)'


def _resolve_test_groups(ds, ids_test):
    """(groups, unit) of the TEST images of ds. Reuses tools/recalibration_suite._test_groups so that both
    tools resample the same units; its local copy is used only if that import / call fails."""
    ids_test = np.asarray(ids_test)
    try:
        from tools.recalibration_suite import _test_groups as suite_groups
        groups, unit = suite_groups(ds, ids_test, '-', _SilentNotes())
        groups = np.asarray(groups)
        if len(groups) != len(ids_test):
            raise ValueError('group vector length mismatch')
    except Exception:                                          # noqa: BLE001 - identical local rule
        groups, unit = _test_groups_local(ds, ids_test)
    return groups, str(unit)


def _test_ids(P, ds):
    """TEST ids of ds from the ('_ids', ds) entry (DEV ids, '|', TEST ids)."""
    ids = np.asarray(P[('_ids', ds)])
    sep = np.flatnonzero(ids == '|')
    return ids[sep[-1] + 1:] if len(sep) else ids


def _boot_indices(groups, unit, n, B, rng):
    """B resamples of TEST image indices. Image level (unit 'image...'): rng.integers(0, n, n) per resample,
    exactly the original image bootstrap. Cluster level: G groups drawn with replacement per resample, every
    image of each drawn group included (a group drawn twice contributes its images twice)."""
    if str(unit).startswith('image'):
        for _ in range(B):
            yield rng.integers(0, n, n)
        return
    _, codes = np.unique(np.asarray(groups).astype(str), return_inverse=True)
    G = int(codes.max()) + 1
    members = [np.flatnonzero(codes == g) for g in range(G)]
    for _ in range(B):
        yield np.concatenate([members[g] for g in rng.integers(0, G, G)])


def _mean_unit(units):
    u = list(dict.fromkeys(units.values()))
    return u[0] if len(u) == 1 else '; '.join(f'{ds}: {v}' for ds, v in units.items())


def compare(ref, tag, P_ref, P_tag, B, seed):
    """Paired differences tag - ref per (dataset, level, n) and their mean over common held-out datasets.

    Deterministic levels: paired TEST bootstrap, clustered by patient / exam when the split file provides
    groups (see _resolve_test_groups); the same resample indices are applied to both models."""
    rows = []
    keys = sorted({k for k in P_tag if k[0] not in ('_y', '_ids') and k in P_ref}, key=str)
    common_ds = sorted({k[0] for k in keys})
    for ds in common_ds:
        if not np.array_equal(P_ref[('_ids', ds)], P_tag[('_ids', ds)]):
            raise ValueError(f'adaptation_curve: {tag} and {ref} have different {ds} DEV/TEST images')
    groups = {}
    for ds in common_ds:
        ids_t = _test_ids(P_ref, ds)
        g, unit = _resolve_test_groups(ds, ids_t)
        if len(g) != len(P_ref[('_y', ds)]):
            g, unit = ids_t, 'image (TEST ids / labels length mismatch)'
        groups[ds] = (g, unit)
    by_level = {}
    for ds, level, n in keys:
        by_level.setdefault((level, n), []).append(ds)
    rng = np.random.default_rng(seed)
    for (level, n), dss in by_level.items():
        for metric in ('qwk', 'mild_recall', 'bin_sens'):
            per_ds, units = {}, {}
            for ds in dss:
                y = P_ref[('_y', ds)]
                a, r = P_tag[(ds, level, n)], P_ref[(ds, level, n)]
                if len(a) > 1:                                     # few-shot: paired over identical draws
                    per_ds[ds] = _metric_values(y, a, metric) - _metric_values(y, r, metric)
                    units[ds] = 'draws'
                else:                                              # deterministic: paired TEST bootstrap
                    g, unit = groups[ds]
                    d = np.empty(B)
                    for b, i in enumerate(_boot_indices(g, unit, len(y), B, rng)):
                        d[b] = grade_metrics(y[i], a[0][i])[metric] - grade_metrics(y[i], r[0][i])[metric]
                    per_ds[ds] = d
                    units[ds] = unit
            for ds, d in per_ds.items():
                rows.append({'reference': ref, 'tag': tag, 'dataset': ds, 'level': level, 'n': n, 'metric': metric,
                             'diff_mean': float(np.nanmean(d)), 'diff_lo': float(np.nanpercentile(d, 2.5)),
                             'diff_hi': float(np.nanpercentile(d, 97.5)), 'n_resamples': len(d),
                             'bootstrap_unit': units[ds]})
            if len(per_ds) == len(common_ds) and len(per_ds) > 1:
                m = min(len(v) for v in per_ds.values())
                d = np.mean([v[:m] for v in per_ds.values()], axis=0)
                rows.append({'reference': ref, 'tag': tag, 'dataset': 'MEAN', 'level': level, 'n': n,
                             'metric': metric, 'diff_mean': float(np.nanmean(d)),
                             'diff_lo': float(np.nanpercentile(d, 2.5)), 'diff_hi': float(np.nanpercentile(d, 97.5)),
                             'n_resamples': m, 'bootstrap_unit': _mean_unit(units)})
    return rows


def plot(curve, groups, out_stem):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(groups), figsize=(5.2 * len(groups), 4.0), squeeze=False)
    for ax, group in zip(axes[0], groups):
        for tag in group:
            g = curve[(curve['tag'] == tag) & (curve['dataset'] == 'MEAN')]
            if g.empty:
                continue
            fs = g[g['level'] == 'fewshot_shrink'].sort_values('n')
            line = ax.plot(fs['n'], fs['qwk_mean'], marker='o', label=f'{tag} (few-shot, shrunk)')[0]
            col = line.get_color()
            for level, mk in (('argmax', 'x'), ('prevalence_oracle', 's')):
                v = g.loc[g['level'] == level, 'qwk_mean']
                if len(v):
                    ax.scatter([5], v.values[:1], marker=mk, color=col, zorder=3)
            v = g.loc[g['level'] == 'all', 'qwk_mean']
            if len(v):
                ax.axhline(v.values[0], color=col, ls=':', lw=1)
        ax.set_xscale('log')
        ax.set_xlabel('labelled target-site DEV images (n)')
        ax.set_ylabel('QWK on held-out TEST (mean over sites)')
        ax.set_title('x = argmax, square = prevalence_oracle (DEV-label shares), dotted = all DEV', fontsize=8)
        ax.legend(fontsize=7)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_stem), exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(f'{out_stem}.{ext}', dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--groups', required=True,
                    help='";"-separated groups of ","-separated tags; the first tag of a group is its reference')
    ap.add_argument('--ns', default='10,25,50,100')
    ap.add_argument('--draws', type=int, default=20)
    ap.add_argument('--B', type=int, default=1000, help='bootstrap resamples for deterministic levels')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out_dir', default=os.path.join('results', 'master'))
    a = ap.parse_args()
    groups = [[t.strip() for t in g.split(',') if t.strip()] for g in a.groups.split(';') if g.strip()]
    ns = [int(x) for x in a.ns.split(',') if x.strip()]
    curve_rows, cmp_rows, lam_rows, preds = [], [], [], {}
    for group in groups:
        for tag in group:
            if tag in preds:
                continue
            r, p, lr = analyse_tag(tag, ns, a.draws, a.seed)
            curve_rows += r
            lam_rows += lr
            preds[tag] = p
        ref = group[0]
        for tag in group[1:]:
            if preds.get(ref) and preds.get(tag):
                cmp_rows += compare(ref, tag, preds[ref], preds[tag], a.B, a.seed)
    if not curve_rows:
        print('adaptation_curve: no held-out dev/test predictions found for any tag')
        sys.exit(1)
    os.makedirs(a.out_dir, exist_ok=True)
    curve = pd.DataFrame(curve_rows)
    curve.round(4).to_csv(os.path.join(a.out_dir, 'adaptation_curve.csv'), index=False)
    pd.DataFrame(cmp_rows, columns=CMP_COLUMNS).round(4).to_csv(
        os.path.join(a.out_dir, 'adaptation_compare.csv'), index=False)
    pd.DataFrame(lam_rows, columns=['tag', 'dataset', 'n', 'lam', 'chosen_from']).to_csv(
        os.path.join(a.out_dir, 'adaptation_lambda.csv'), index=False)
    default_out = os.path.normpath(a.out_dir) == os.path.normpath(os.path.join('results', 'master'))
    fig_stem = os.path.join('results', 'figures', 'fig_adaptation' if default_out else
                            f'fig_adaptation_{os.path.basename(os.path.normpath(a.out_dir))}')
    plot(curve, groups, fig_stem)
    show = curve[curve['dataset'] == 'MEAN'][['tag', 'level', 'n', 'qwk_mean', 'mild_recall_mean', 'bin_sens_mean']]
    with pd.option_context('display.width', 200, 'display.max_rows', 500):
        print(show.round(3).to_string(index=False))
        if cmp_rows:
            c = pd.DataFrame(cmp_rows)
            c = c[(c['dataset'] == 'MEAN') & (c['metric'] == 'qwk')]
            print()
            print(c[['reference', 'tag', 'level', 'n', 'diff_mean', 'diff_lo', 'diff_hi']].round(3).to_string(index=False))
    print(f'\nwritten: {a.out_dir}/adaptation_curve.csv, adaptation_compare.csv, adaptation_lambda.csv, '
          f'{fig_stem}.png/.pdf')


if __name__ == '__main__':
    main()
