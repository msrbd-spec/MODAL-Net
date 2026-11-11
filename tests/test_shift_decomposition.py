"""tools/shift_decomposition.py on fabricated runs: prior shift vs annotation (cut-point) shift.

World model (exact, so the expected behaviour is known): an image of grade y has a latent severity
s ~ N(y + shift, SD); the "network" outputs the Bayes posterior of the SOURCE world (shift 0, prior PI_SRC),
so its probabilities are calibrated on the source.
  r_prior : target grades ~ PI_TGT, shift 0          -> pure prior shift (p(x|y) identical to the source)
  r_annot : target grades ~ PI_SRC, shift ANNOT       -> same priors, the site grades the same image lower
                                                         (its cut-points sit ANNOT higher): p(y|x) changes
"""
import os

import numpy as np
import pandas as pd
import pytest

from src.evaluation import label_shift as ls
from tools import shift_decomposition as sd

K = 5
SD = 0.7
ANNOT = 0.6
PI_SRC = np.array([0.50, 0.10, 0.22, 0.08, 0.10])
PI_TGT = np.array([0.20, 0.30, 0.25, 0.15, 0.10])
SITES = ('messidor', 'idrid')                       # held out (train_sources = aptos,ddr,deepdrid)
N_OOF, N_DEV, N_TEST = 3000, 700, 700


def _posterior(s):
    ll = -(s[:, None] - np.arange(K)[None]) ** 2 / (2 * SD * SD) + np.log(PI_SRC)[None]
    ll -= ll.max(1, keepdims=True)
    P = np.exp(ll)
    return P / P.sum(1, keepdims=True)


def _sample(rng, pi, n, shift):
    y = rng.choice(K, size=n, p=pi)
    return y, _posterior(y + shift + rng.normal(0, SD, n))


def _fabricate(seed=0, runs=(('r_prior', PI_TGT, 0.0), ('r_annot', PI_SRC, ANNOT))):
    rng = np.random.default_rng(seed)
    y_oof, P_oof = _sample(rng, PI_SRC, N_OOF, 0.0)
    for tag, pi, shift in runs:
        ed = os.path.join('results', 'phase10_ms_ddr', tag)
        os.makedirs(ed, exist_ok=True)
        pd.DataFrame([{'phase': 'phase10_ms_ddr', 'experiment': tag,
                       'train_sources': 'aptos,ddr,deepdrid'}]).to_csv(os.path.join(ed, 'summary.csv'), index=False)
        perm = rng.permutation(N_OOF)                    # file order != id order
        np.savez(os.path.join(ed, 'oof.npz'), ids=np.array([f'oof{i:05d}' for i in range(N_OOF)])[perm],
                 y_true=y_oof[perm], y_prob=P_oof[perm])
        for ds in SITES:
            for split, n in (('dev', N_DEV), ('test', N_TEST)):
                y, P = _sample(rng, pi, n, shift)
                ids = np.array([f'{ds}_{split}_{i:04d}' for i in range(n)])
                p = rng.permutation(n)
                np.savez(os.path.join(ed, f'predictions_{ds}_{split}.npz'), ids=ids[p], y_true=y[p], y_prob=P[p],
                         y_prob_raw=P[p])           # T = 1 recovered from the (raw, scaled) pairs


def _run(out, tags='r_prior,r_annot', R=8, seed=0):
    sd.main(['--tags', tags, '--R', str(R), '--seed', str(seed), '--out_dir', out,
             '--fig_stem', os.path.join('results', 'figures', 'fig_shift_decomposition')])
    return {f: pd.read_csv(os.path.join(out, f'shift_decomposition{f}.csv'))
            for f in ('', '_contrasts', '_draws', '_fitted', '_notes')}


def _c(con, tag, scen, name, metric='qwk', ds='MEAN', col='mean'):
    r = con[(con['tag'] == tag) & (con['dataset'] == ds) & (con['scenario'] == scen) & (con['contrast'] == name)
            & (con['metric'] == metric)]
    assert len(r) == 1, (tag, scen, name, len(r))
    return float(r[col].iloc[0])


@pytest.fixture
def world():
    _fabricate()
    out = os.path.join('results', 'master', 'recal')
    return _run(out), out


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def test_resampling_counts_split_and_missing_grades():
    y = np.repeat(np.arange(K), [50, 30, 20, 10, 1])
    idx, miss = sd.stratified_resample(y, [0.2, 0.2, 0.2, 0.2, 0.2], 101, np.random.default_rng(0))
    assert len(idx) == 101 and miss == []
    assert sorted(np.bincount(y[idx], minlength=K)) == [20, 20, 20, 20, 21]
    y2 = y[y != 3]
    idx, miss = sd.stratified_resample(y2, [0.5, 0, 0, 0.5, 0], 40, np.random.default_rng(1))
    assert miss == [3] and np.all(y2[idx] == 0) and len(idx) == 40          # absent grade -> renormalised
    S, T = sd.split_source(np.repeat(np.arange(K), [100, 21, 40, 7, 9]), 0)
    assert len(np.intersect1d(S, T)) == 0 and len(S) + len(T) == 177
    a, b, shared = sd._halves(y, np.random.default_rng(2))
    assert shared == [4] and len(np.intersect1d(a, b)) == 1                  # only the single grade-4 image


def test_fit_predict_has_no_test_label_argument_and_is_label_invariant():
    import inspect
    assert 'y_test' not in inspect.signature(sd.fit_predict).parameters
    rng = np.random.default_rng(3)
    ys, Ps = _sample(rng, PI_SRC, 800, 0.0)
    src = sd.fit_source(Ps, ys)
    yd, Pd = _sample(rng, PI_TGT, 300, 0.0)
    yt, Pt = _sample(rng, PI_TGT, 300, 0.0)
    pred, fit = sd.fit_predict(src, Pd, yd, Pt)
    assert set(pred) == set(sd.METHODS) and all(len(v) == 300 for v in pred.values())
    m1 = sd.Scorer(yt).score(pred['argmax'])
    m2 = sd.Scorer(rng.permutation(yt)).score(pred['argmax'])                 # TEST labels change ...
    pred2, fit2 = sd.fit_predict(src, Pd, yd, Pt)
    assert fit == fit2 and all(np.array_equal(pred[m], pred2[m]) for m in pred)   # ... nothing fitted does
    assert m1['qwk'] != m2['qwk']
    m = sd.score_metrics(np.array([0, 0, 2, 2]), np.array([0, 2, 2, 2]))      # grades 1, 3, 4 absent
    assert np.isnan(m['mild_recall']) and abs(m['f1_macro'] - np.mean([2 / 3, 0.8])) < 1e-12


# ---------------------------------------------------------------------------
# simulations
# ---------------------------------------------------------------------------

def _show(con, tag):
    names = ('gain_bayes', 'gain_cuts_over_bayes', 'gain_quantile', 'mlls_recovery')
    for scen in ('A0', 'A', 'B', 'C', 'B-A0', 'C-A', 'A-A0'):
        vals = [f'{n}={_c(con, tag, scen, n):+.4f}' for n in names if scen in sd.SCENARIOS or n != 'mlls_recovery']
        print(f'[{tag}] {scen:5s} ' + ' '.join(vals))


def test_pure_prior_shift_C_behaves_like_A(world):
    (o, out) = world
    con = o['_contrasts']
    print()
    _show(con, 'r_prior')
    gb = {s: _c(con, 'r_prior', s, 'gain_bayes') for s in sd.SCENARIOS}
    gc = {s: _c(con, 'r_prior', s, 'gain_cuts_over_bayes') for s in sd.SCENARIOS}
    # prior correction helps where the shares differ from the source (A, C), not where they equal it (A0, B)
    assert gb['C'] > 0.02 and gb['A'] > 0.02
    assert _c(con, 'r_prior', 'C', 'gain_bayes', col='lo') > 0
    assert _c(con, 'r_prior', 'A-A0', 'gain_bayes') > 0.015
    assert abs(gb['B']) < 0.015 and abs(gb['A0']) < 0.015
    # no annotation shift: once priors are corrected, re-fitting cut-points adds nothing beyond the matched
    # in-distribution reference (C vs A, B vs A0)
    assert abs(gc['C']) < 0.015 and abs(gc['A']) < 0.015
    assert abs(_c(con, 'r_prior', 'C-A', 'gain_cuts_over_bayes')) < 0.015
    assert abs(_c(con, 'r_prior', 'B-A0', 'gain_cuts_over_bayes')) < 0.015
    # C behaves like A (tolerance: A is conditional on one finite pool T, ~0.01 QWK pool effect)
    assert abs(gb['C'] - gb['A']) < 0.03
    assert _c(con, 'r_prior', 'A', 'mlls_recovery') > 0.6 and _c(con, 'r_prior', 'C', 'mlls_recovery') > 0.6


def test_annotation_shift_shows_cut_gain_beyond_prior_correction(world):
    (o, out) = world
    con = o['_contrasts']
    print()
    _show(con, 'r_annot')
    gb = {s: _c(con, 'r_annot', s, 'gain_bayes') for s in sd.SCENARIOS}
    gc = {s: _c(con, 'r_annot', s, 'gain_cuts_over_bayes') for s in sd.SCENARIOS}
    for s in ('B', 'C'):
        assert abs(gb[s]) < 0.015, s                                           # no prior shift to correct
        assert gc[s] > 0.05 and gc[s] > 4 * abs(gb[s]), s                      # cut-points fix the annotation
        assert _c(con, 'r_annot', s, 'gain_cuts_over_bayes', col='lo') > 0.02, s
    assert abs(gc['A']) < 0.015 and abs(gc['A0']) < 0.015                      # APTOS images: no annotation shift
    assert _c(con, 'r_annot', 'B-A0', 'gain_cuts_over_bayes') > 0.05
    assert _c(con, 'r_annot', 'C-A', 'gain_cuts_over_bayes') > 0.05
    assert _c(con, 'r_annot', 'C-A', 'gain_cuts_over_bayes', ds='messidor', col='lo') > 0


# ---------------------------------------------------------------------------
# leakage, schema, determinism
# ---------------------------------------------------------------------------

def _relabel_test_files(rng):
    """Change AND permute every held-out TEST label (shares change); probabilities untouched."""
    for root, _, files in os.walk('results'):
        for f in files:
            if f.startswith('predictions_') and f.endswith('_test.npz'):
                p = os.path.join(root, f)
                with np.load(p, allow_pickle=True) as z:
                    d = {k: z[k] for k in z.files}
                d['y_true'] = rng.permutation((d['y_true'] + 2) % K)
                np.savez(p, **d)


def test_no_test_label_leakage():
    _fabricate()
    o1 = _run(os.path.join('results', 'master', 'run1'), R=3)
    _relabel_test_files(np.random.default_rng(5))
    o2 = _run(os.path.join('results', 'master', 'run2'), R=3)
    pd.testing.assert_frame_equal(o1['_fitted'], o2['_fitted'])                # every fitted quantity identical
    d1, d2 = o1['_draws'], o2['_draws']
    pd.testing.assert_series_equal(d1['n_dev'], d2['n_dev'])
    assert not np.allclose(d1['qwk'], d2['qwk'])                               # metrics did use TEST labels
    assert len(o1['_fitted']) > 0 and set(o1['_fitted']['scenario']) == {'source', 'A0', 'A', 'B', 'C'}


def test_schemas_figure_and_determinism(world):
    (o, out) = world
    s, con, dr = o[''], o['_contrasts'], o['_draws']
    for c in ['tag', 'dataset', 'scenario', 'method', 'info', 'R', 'n_dev', 'n_test', 'caveat'] + \
            [f'{m}_{x}' for m in sd.METRICS for x in ('mean', 'lo', 'hi', 'point')]:
        assert c in s.columns, c
    assert set(s['method']) == set(sd.METHODS) and set(s['scenario']) == set(sd.SCENARIOS)
    assert set(s['dataset']) == set(SITES) | {'MEAN'} and set(s['tag']) == {'r_prior', 'r_annot'}
    assert (s['R'] == 8).all()
    assert s.loc[s['scenario'] == 'C', 'qwk_point'].notna().all()
    assert s.loc[s['scenario'] != 'C', 'qwk_point'].isna().all()
    assert s.loc[s['scenario'] != 'C', 'caveat'].str.contains('semi-synthetic').all()
    assert (s.loc[s['method'].str.contains('oracle'), 'info'] == 'prevalence_oracle').all()
    # scenario sizes and shares: A / B DEV share exactly the site DEV / source S shares (exact counts)
    one = s[(s['tag'] == 'r_prior') & (s['dataset'] == 'messidor')]
    assert (one['n_dev'] == N_DEV).all() and (one['n_test'] == N_TEST).all()
    mean = s[(s['dataset'] == 'MEAN') & (s['tag'] == 'r_prior')]
    assert (mean['n_test'] == 2 * N_TEST).all()
    fit = o['_fitted']
    piS = fit[(fit['tag'] == 'r_prior') & (fit['method'] == 'source') & fit['param'].str.startswith('pi_S_')]
    piS = piS.sort_values('param')['value'].values
    piB = fit[(fit['tag'] == 'r_prior') & (fit['scenario'] == 'B') & (fit['method'] == 'oracle_bayes')
              & (fit['draw'] == 0) & (fit['dataset'] == 'idrid')].sort_values('param')['value'].values
    assert np.abs(piB - piS).max() <= 1.0 / N_DEV + 1e-9
    t = fit[(fit['method'] == 'source') & (fit['param'] == 'T_oof')]['value']
    assert np.allclose(t, 1.0, atol=1e-6)
    for c in ('tag', 'dataset', 'scenario', 'contrast', 'metric', 'mean', 'lo', 'hi', 'point', 'n_valid', 'R'):
        assert c in con.columns, c
    assert {'gain_bayes', 'gain_cuts_over_bayes', 'gain_quantile', 'mlls_recovery'} <= set(con['contrast'])
    assert set(sd.SCENARIOS) | {'B-A0', 'C-A', 'A-A0', 'C-B'} == set(con['scenario'])
    assert set(dr.loc[dr['scenario'] == 'C', 'draw']) == set(range(-1, 8))
    assert not dr.duplicated(['tag', 'dataset', 'scenario', 'draw', 'method']).any()
    for ext in ('png', 'pdf'):
        assert os.path.getsize(os.path.join('results', 'figures', f'fig_shift_decomposition.{ext}')) > 1000
    # determinism: a second run with the same seed reproduces every file exactly; another seed does not
    o2 = _run(os.path.join('results', 'master', 'again'))
    for k in ('', '_contrasts', '_draws', '_fitted'):
        pd.testing.assert_frame_equal(o[k], o2[k])
    o3 = _run(os.path.join('results', 'master', 'seed1'), seed=1)
    assert not o3['_draws']['qwk'].equals(o['_draws']['qwk'])


def test_missing_run_and_partial_sites_are_handled():
    _fabricate(runs=(('r_prior', PI_TGT, 0.0),))
    os.remove(os.path.join('results', 'phase10_ms_ddr', 'r_prior', 'predictions_idrid_test.npz'))
    o = _run(os.path.join('results', 'master', 'partial'), tags='r_prior,nope', R=2)
    assert 'MEAN' not in set(o['']['dataset']) and set(o['']['dataset']) == {'messidor'}
    assert o['_notes']['note'].str.contains('skipped').any()
    assert o['_notes']['tag'].eq('nope').any()
