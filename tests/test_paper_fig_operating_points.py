import os

import numpy as np
import pandas as pd

from tools import paper_fig_operating_points as pf


def test_roc_auc_matches_rank_auc():
    rng = np.random.default_rng(0)
    yb = rng.integers(0, 2, 400)
    s = rng.random(400) + 0.6 * yb
    fpr, tpr, auc = pf.roc_curve(yb, np.round(s, 2))           # rounding creates ties
    from sklearn.metrics import roc_auc_score
    assert abs(auc - roc_auc_score(yb, np.round(s, 2))) < 1e-9
    assert fpr[0] == 0 and tpr[-1] == 1


def test_figure_written(tmp_path):
    rng = np.random.default_rng(1)
    exp = tmp_path / 'exp'
    exp.mkdir()
    rows = []
    for site in ('messidor', 'ddr'):
        y = rng.integers(0, 5, 200)
        P = rng.dirichlet(np.ones(5), 200)
        name = 'predictions_external_test.npz' if site == 'messidor' else f'predictions_{site}_test.npz'
        np.savez(exp / name, ids=np.arange(200).astype(str), y_true=y, y_prob=P)
        for m, *_ in pf.RULES:
            rows.append({'tag': 't', 'dataset': site, 'method': m, 'bin_sens_mean': 0.7, 'bin_spec_mean': 0.9})
    pd.DataFrame(rows).to_csv(tmp_path / 's.csv', index=False)
    out = tmp_path / 'fig'
    pf.main(['--exp_dir', str(exp), '--tag', 't', '--summary', str(tmp_path / 's.csv'), '--out', str(out)])
    assert os.path.isfile(str(out) + '.png') and os.path.isfile(str(out) + '.pdf')
