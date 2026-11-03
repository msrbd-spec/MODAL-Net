"""Cut-point diagnostic on saved predictions (no GPU, no retraining).

Question: is the external under-grading a cut-point (threshold) problem? For each tag and every
external dataset that has saved dev and test predictions, the 4 ordinal cut-points on the expected
grade E = sum_k k P_k are refitted on the DEV split and applied to the TEST split. Compared with the
model's own argmax prediction:
  * QWK, accuracy, mild recall, predicted-No-DR share on TEST (argmax vs refitted cut-points)
  * paired bootstrap 95 % CI of the QWK gain (TEST images resampled; cut-points fixed from DEV)
  * referable AUROC on TEST (a ranking metric; it is the same for both, shown for context)
If refitting four numbers on DEV recovers a large part of the QWK / mild-recall gap while AUROC is
already high, the model ranks images well and only its cut-points are mis-placed for the new site.

Two estimates are written, for two different uses:
  devcf_*  cross-fitted INSIDE the DEV split (exam/patient groups split in halves 20 times: fit on one half,
           score on the other) -> uses no TEST image; this is the number used to MOTIVATE the method.
  *_devcal on TEST (cut-points from the whole DEV) -> the number REPORTED in the paper.

Works with the files the old code wrote (predictions_external_dev.npz / predictions_external_test.npz
for Messidor-2) and with the new predictions_<dataset>_{dev,test}.npz files.

Usage:
  python tools/cutpoint_diagnostic.py --tags p2b_none,p3b_joint,p4_coral
  python tools/cutpoint_diagnostic.py --tags p2b_none,p3b_joint --out results/master/cutpoint_diagnostic.csv
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.cutpoints import apply_cutpoints, expected_grade, fit_cutpoints, grade_metrics, qwk_from_pred
from src.utils.config import find_exp

DATASETS = ('messidor', 'ddr', 'deepdrid', 'idrid')
LEGACY = {('messidor', 'dev'): 'predictions_external_dev.npz', ('messidor', 'test'): 'predictions_external_test.npz'}


def _load(exp_dir, ds, split):
    for name in (f'predictions_{ds}_{split}.npz', LEGACY.get((ds, split))):
        if name and os.path.isfile(os.path.join(exp_dir, name)):
            with np.load(os.path.join(exp_dir, name), allow_pickle=True) as f:
                y = np.asarray(f['y_true'] if 'y_true' in f.files else f['y'], dtype=int)
                P = np.asarray(f['y_prob'] if 'y_prob' in f.files else f['P'])
                ids = [str(i) for i in f['ids']]
            return y, P, ids
    return None


def diagnose(tag, B=2000, seed=0):
    exp_dir = find_exp(tag)
    rows = []
    groups_of = {}
    for ds in DATASETS:
        p = os.path.join('splits', 'external', f'{ds}_dev.csv')
        if os.path.isfile(p):
            d = pd.read_csv(p, dtype=str, keep_default_na=False)
            groups_of[ds] = dict(zip(d['id_code'], d['group']))
    for ds in DATASETS:
        dev, test = _load(exp_dir, ds, 'dev'), _load(exp_dir, ds, 'test')
        if dev is None or test is None:
            continue
        (yd, Pd, idd), (yt, Pt, _) = dev, test
        # cross-fit inside DEV: grouped random halves, fit on one half, score the other (no TEST image used)
        g = np.array([groups_of.get(ds, {}).get(i, i) for i in idd])
        ug = np.unique(g)
        rng_cf = np.random.default_rng(seed)
        cf_arg, cf_cal = [], []
        for _ in range(20):
            half = set(rng_cf.permutation(ug)[:len(ug) // 2])
            a = np.array([x in half for x in g])
            for fit_m, score_m in ((a, ~a), (~a, a)):
                if fit_m.sum() < 5 or score_m.sum() < 5:
                    continue
                c = fit_cutpoints(expected_grade(Pd[fit_m]), yd[fit_m])
                cf_cal.append(qwk_from_pred(yd[score_m], apply_cutpoints(expected_grade(Pd[score_m]), c)))
                cf_arg.append(qwk_from_pred(yd[score_m], Pd[score_m].argmax(1)))
        cuts = fit_cutpoints(expected_grade(Pd), yd)
        pred_cal = apply_cutpoints(expected_grade(Pt), cuts)
        pred_arg = Pt.argmax(1)
        m_arg, m_cal = grade_metrics(yt, pred_arg), grade_metrics(yt, pred_cal)
        rng = np.random.default_rng(seed)
        d = np.empty(B)
        for b in range(B):
            i = rng.integers(0, len(yt), len(yt))
            d[b] = qwk_from_pred(yt[i], pred_cal[i]) - qwk_from_pred(yt[i], pred_arg[i])
        yb = (yt >= 2).astype(int)
        auroc = roc_auc_score(yb, Pt[:, 2:].sum(1)) if 0 < yb.sum() < len(yb) else np.nan
        cf = np.array(cf_cal) - np.array(cf_arg)
        nan = float('nan')
        rows.append({'tag': tag, 'dataset': ds, 'n_dev': len(yd), 'n_test': len(yt),
                     'devcf_qwk_argmax': float(np.mean(cf_arg)) if cf_arg else nan,
                     'devcf_qwk_refit': float(np.mean(cf_cal)) if cf_cal else nan,
                     'devcf_qwk_gain': float(cf.mean()) if len(cf) else nan,
                     'devcf_gain_lo': float(np.percentile(cf, 2.5)) if len(cf) else nan,
                     'devcf_gain_hi': float(np.percentile(cf, 97.5)) if len(cf) else nan,
                     'cuts': ' '.join(f'{c:.2f}' for c in cuts), 'test_bin_auroc': auroc,
                     'qwk_argmax': m_arg['qwk'], 'qwk_devcal': m_cal['qwk'],
                     'qwk_gain': m_cal['qwk'] - m_arg['qwk'], 'qwk_gain_ci_lo': np.percentile(d, 2.5),
                     'qwk_gain_ci_hi': np.percentile(d, 97.5),
                     'acc_argmax': m_arg['acc'], 'acc_devcal': m_cal['acc'],
                     'mild_recall_argmax': m_arg['mild_recall'], 'mild_recall_devcal': m_cal['mild_recall'],
                     'pred_nodr_argmax': m_arg['pred_nodr_share'], 'pred_nodr_devcal': m_cal['pred_nodr_share'],
                     'true_nodr_share': 100.0 * float((yt == 0).mean())})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tags', required=True, help='comma-separated experiment tags')
    ap.add_argument('--out', default=os.path.join('results', 'master', 'cutpoint_diagnostic.csv'))
    ap.add_argument('--B', type=int, default=2000, help='bootstrap resamples for the QWK-gain CI')
    a = ap.parse_args()
    rows = []
    for t in [x.strip() for x in a.tags.split(',') if x.strip()]:
        r = diagnose(t, B=a.B)
        if not r:
            print(f'{t}: no dev+test prediction pair found (run --mode evaluate --tag {t} for the new datasets)')
        rows += r
    if not rows:
        sys.exit(1)
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    df.round(4).to_csv(a.out, index=False)
    with pd.option_context('display.width', 200, 'display.max_columns', 30):
        print(df[['tag', 'dataset', 'n_dev', 'devcf_qwk_argmax', 'devcf_qwk_refit', 'devcf_qwk_gain', 'devcf_gain_lo',
                  'devcf_gain_hi']].round(3).to_string(index=False))
        print()
        print(df[['tag', 'dataset', 'n_test', 'test_bin_auroc', 'qwk_argmax', 'qwk_devcal', 'qwk_gain',
                  'qwk_gain_ci_lo', 'qwk_gain_ci_hi', 'mild_recall_argmax', 'mild_recall_devcal',
                  'pred_nodr_argmax', 'pred_nodr_devcal', 'true_nodr_share']].round(3).to_string(index=False))
    print(f'\nwritten: {a.out}')


if __name__ == '__main__':
    main()
