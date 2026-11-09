"""Cut-point adaptation analysis (tools/adaptation_curve.py and the new cutpoints.py helpers)."""
import os
import subprocess
import sys

import numpy as np
import pandas as pd

from src.evaluation.cutpoints import (DEFAULT_CUTS, apply_cutpoints, expected_grade, fit_cutpoints,
                                      prevalence_cutpoints, qwk_from_pred)

TOOL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tools', 'adaptation_curve.py')


def _probs_from_score(s, cuts, temp=0.25):
    """Cumulative-logit class probabilities for latent scores s and cut-points cuts (K = 5)."""
    cum = 1.0 / (1.0 + np.exp(-(s[:, None] - np.asarray(cuts)[None, :]) / temp))     # P(y > k)
    P = np.concatenate([1 - cum[:, :1], cum[:, :-1] - cum[:, 1:], cum[:, -1:]], axis=1)
    P = np.clip(P, 1e-6, None)
    return P / P.sum(1, keepdims=True)


def test_fit_cutpoints_lam0_unchanged_and_shrinkage_moves_towards_prior():
    rng = np.random.default_rng(0)
    E = rng.uniform(0, 4, 300)
    y = apply_cutpoints(E, [0.9, 1.8, 2.7, 3.6])
    assert np.array_equal(fit_cutpoints(E, y), fit_cutpoints(E, y, prior=DEFAULT_CUTS, lam=0.0))
    free = fit_cutpoints(E[:15], y[:15])
    shrunk = fit_cutpoints(E[:15], y[:15], prior=DEFAULT_CUTS, lam=100.0)
    assert np.abs(shrunk - DEFAULT_CUTS).sum() <= np.abs(free - DEFAULT_CUTS).sum() + 1e-9
    assert np.all(np.diff(shrunk) > 0)


def test_prevalence_cutpoints_reproduce_grade_shares():
    rng = np.random.default_rng(1)
    E = rng.uniform(0, 4, 2000)
    props = np.array([0.5, 0.2, 0.15, 0.1, 0.05])
    pred = apply_cutpoints(E, prevalence_cutpoints(E, props))
    share = np.bincount(pred, minlength=5) / len(pred)
    assert np.abs(share - props).max() < 0.01
    # empty grade -> still strictly ordered
    c = prevalence_cutpoints(E, [0.6, 0.0, 0.2, 0.1, 0.1])
    assert np.all(np.diff(c) > 0)


def _write_run(phase, tag, sources, data):
    ed = os.path.join('results', phase, tag)
    os.makedirs(ed, exist_ok=True)
    pd.DataFrame([{'phase': phase, 'experiment': tag, 'train_sources': ','.join(sources)}]).to_csv(
        os.path.join(ed, 'summary.csv'), index=False)
    for (ds, split), (ids, y, P) in data.items():
        np.savez(os.path.join(ed, f'predictions_{ds}_{split}.npz'), ids=ids, y_true=y, y_prob=P)


def test_adaptation_tool_end_to_end():
    rng = np.random.default_rng(2)
    true_cuts = {'messidor': [0.8, 1.9, 2.8, 3.7], 'deepdrid': [0.3, 1.2, 2.2, 3.3], 'idrid': [0.6, 1.4, 2.6, 3.5],
                 'ddr': [0.5, 1.5, 2.5, 3.5]}
    runs = {'m_ce': 0.6, 'm_ord': 0.35}                      # score noise: the ordinal model ranks better
    data = {t: {} for t in runs}
    for ds, tc in true_cuts.items():
        for split, n in (('dev', 160), ('test', 240)):
            s = rng.uniform(0, 4, n)
            y = apply_cutpoints(s, tc)
            ids = np.array([f'{ds}_{split}_{i:04d}' for i in range(n)])
            perm = rng.permutation(n)                        # files need not be sorted by id
            for t, noise in runs.items():
                s_hat = s + rng.normal(0, noise, n)
                P = _probs_from_score(s_hat, DEFAULT_CUTS)
                data[t][(ds, split)] = (ids[perm], y[perm], P[perm])
    for t in runs:
        _write_run('phase10_ms_ddr', t, ['aptos', 'eyepacs', 'ddr'], data[t])
    r = subprocess.run([sys.executable, TOOL, '--groups', 'm_ce,m_ord', '--ns', '10,50', '--draws', '6', '--B', '50'],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    curve = pd.read_csv('results/master/adaptation_curve.csv')
    assert set(curve['dataset']) == {'messidor', 'deepdrid', 'idrid', 'MEAN'}       # DDR was a training source
    assert set(curve['level']) == {'argmax', 'default_cuts', 'prevalence', 'fewshot', 'fewshot_shrink', 'all'}
    mean = curve[(curve['tag'] == 'm_ce') & (curve['dataset'] == 'MEAN')]
    q = lambda level: float(mean.loc[mean['level'] == level, 'qwk_mean'].iloc[0])
    # the sites' cut-points are shifted: refitting on DEV must beat the model's own argmax
    assert q('all') > q('argmax')
    cmp = pd.read_csv('results/master/adaptation_compare.csv')
    assert {'MEAN'} <= set(cmp['dataset'])
    assert (cmp['reference'] == 'm_ce').all() and (cmp['tag'] == 'm_ord').all()
    lam = pd.read_csv('results/master/adaptation_lambda.csv')
    assert len(lam) == 2 * 3 * 2                                    # 2 tags x 3 held-out sets x 2 n
    assert os.path.isfile('results/figures/fig_adaptation.png')


def test_adaptation_tool_refuses_mismatched_images():
    rng = np.random.default_rng(3)
    for t, off in (('a_ref', 0), ('a_arm', 1)):
        d = {}
        for ds in ('messidor', 'deepdrid'):
            for split in ('dev', 'test'):
                y = rng.integers(0, 5, 60)
                ids = np.array([f'{ds}_{split}_{i + off}' for i in range(60)])
                d[(ds, split)] = (ids, y, _probs_from_score(y + rng.normal(0, .5, 60), DEFAULT_CUTS))
        _write_run('phase10_ms_ddr', t, ['aptos', 'eyepacs', 'ddr', 'idrid'], d)
    r = subprocess.run([sys.executable, TOOL, '--groups', 'a_ref,a_arm', '--ns', '10', '--draws', '3', '--B', '20'],
                       capture_output=True, text=True)
    assert r.returncode != 0 and 'different' in (r.stdout + r.stderr)
