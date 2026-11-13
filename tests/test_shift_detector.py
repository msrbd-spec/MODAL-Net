"""tools/shift_detector.py: label-free MLLS-BBSE disagreement as a detector of unsafe prior correction.

Uses the exact fabricated world of tests/test_shift_decomposition.py:
  r_prior : pure prior shift (label-shift assumption holds)  -> MLLS and BBSE agree, correction safe
  r_annot : annotation shift (p(y|x) changes)                -> they may disagree, correction can hurt
"""
import os

import numpy as np
import pandas as pd
import pytest

from tests.test_shift_decomposition import _fabricate, _relabel_test_files
from tools import shift_detector as det

FILES = ('draws', 'summary', 'gating', 'notes')


def _run(out, tags='r_prior,r_annot', R=6, B=50, seed=0):
    det.main(['--tags', tags, '--R', str(R), '--B', str(B), '--seed', str(seed), '--out_dir', out,
              '--fig_stem', os.path.join('results', 'figures', 'fig_shift_detector')])
    return {f: pd.read_csv(os.path.join(out, f'shift_detector_{f}.csv')) for f in FILES}


@pytest.fixture
def world():
    _fabricate()
    out = os.path.join('results', 'master', 'recal')
    return _run(out), out


# ---------------------------------------------------------------------------
# unit tests
# ---------------------------------------------------------------------------

def test_auroc_matches_definition():
    assert det.auroc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == 1.0
    assert det.auroc([0.9, 0.8, 0.2, 0.1], [0, 0, 1, 1]) == 0.0
    assert det.auroc([0.5, 0.5, 0.5, 0.5], [0, 1, 0, 1]) == 0.5               # ties count one half
    assert np.isnan(det.auroc([0.1, 0.2], [1, 1]))                            # one class only
    rng = np.random.default_rng(0)
    s, y = rng.normal(size=300), rng.integers(0, 2, 300)
    pairs = np.mean([(a > b) + 0.5 * (a == b) for a in s[y == 1] for b in s[y == 0]])
    assert det.auroc(s, y) == pytest.approx(pairs)


def test_fit_tau_maximises_mean_gated_qwk():
    s = np.array([0.1, 0.2, 0.3, 0.4])
    q_arg = np.array([0.8, 0.8, 0.8, 0.8])
    q_cor = np.array([0.9, 0.9, 0.5, 0.5])                                    # correction helps only below 0.25
    assert det.fit_tau(s, q_arg, q_cor) == pytest.approx(0.2)
    assert det.fit_tau(s, q_arg, q_arg - 0.1) == -np.inf                      # always worse -> never correct
    assert det.fit_tau(s, q_arg, q_arg + 0.1) == pytest.approx(0.4)           # always better -> always correct
    q_nan = np.array([0.9, np.nan, 0.9, 0.5])                                  # one NaN QWK must not break it
    assert det.fit_tau(s, q_arg, q_nan) == pytest.approx(0.3)


def test_unsafe_is_nan_for_nan_harm_and_auroc_drops_it():
    assert det._unsafe(-0.05) == 1.0 and det._unsafe(0.0) == 0.0 and np.isnan(det._unsafe(np.nan))
    assert det.auroc([0.1, 0.9, 0.5], [0, 1, np.nan]) == 1.0


def test_loso_threshold_ignores_the_heldout_site():
    rng = np.random.default_rng(1)
    rows = []
    for ds in ('a', 'b', 'c'):
        for scen in ('A', 'C'):
            for r in ([-1] if scen == 'C' else []) + list(range(5)):
                row = {'tag': 't', 'dataset': ds, 'scenario': scen, 'draw': r, 'qwk_argmax': 0.8}
                row.update({s: float(rng.uniform()) for s in det.SCORES})
                row.update({f'qwk_{m}': float(rng.uniform(0.6, 0.9)) for m in det.CORRECTIONS.values()})
                rows.append(row)
    d1 = pd.DataFrame(rows)
    d2 = d1.copy()
    m = d2['dataset'] == 'a'
    d2.loc[m, [f'qwk_{v}' for v in det.CORRECTIONS.values()]] = 0.0           # wreck the held-out site only
    for s in det.SCORES:
        d2.loc[m, s] = 0.0
    g1, g2 = det.gating(d1, 0, 0), det.gating(d2, 0, 0)
    k = ['tag', 'correction', 'score', 'tau_fit_on', 'dataset']
    t1 = g1[g1['dataset'] == 'a'].set_index(k)['tau']
    t2 = g2[g2['dataset'] == 'a'].set_index(k)['tau']
    pd.testing.assert_series_equal(t1, t2)                                     # tau of site a never sees site a


def test_detector_receives_no_labels():
    import inspect
    assert list(inspect.signature(det.detector_scores).parameters) == ['src', 'P_dev']
    assert 'y' not in ''.join(inspect.signature(det.predict_methods).parameters)


# ---------------------------------------------------------------------------
# end to end on the fabricated world
# ---------------------------------------------------------------------------

def test_disagreement_small_under_prior_shift_large_under_annotation_shift(world):
    o, _ = world
    d = o['draws']
    prior_c = d[(d['tag'] == 'r_prior') & (d['scenario'] == 'C')]['D_mb'].mean()
    annot_c = d[(d['tag'] == 'r_annot') & (d['scenario'] == 'C')]['D_mb'].mean()
    a_rows = d[d['scenario'].isin(['A0', 'A'])]['D_mb'].mean()
    assert annot_c > 2 * prior_c, (annot_c, prior_c)
    assert annot_c > 2 * a_rows, (annot_c, a_rows)
    # MLLS correction is harmless under pure prior shift and harmful under annotation shift
    h_prior = d[(d['tag'] == 'r_prior') & (d['scenario'] == 'C')]['harm_mlls'].mean()
    h_annot = d[(d['tag'] == 'r_annot') & (d['scenario'] == 'C')]['harm_mlls'].mean()
    assert h_prior > -0.01 and h_annot < 0 and h_annot < h_prior - 0.02, (h_prior, h_annot)


def test_primary_auroc_and_gating(world):
    o, _ = world
    s = o['summary']
    pr = s[s['primary'] == 1]
    assert len(pr) == 1
    assert (pr['tag'].iloc[0], pr['stratum'].iloc[0], pr['score'].iloc[0], pr['label'].iloc[0]) == \
        ('ALL', 'C', 'D_mb', 'unsafe_mlls')
    assert pr['auroc'].iloc[0] > 0.8
    assert pr['auroc_lo'].iloc[0] <= pr['auroc'].iloc[0] <= pr['auroc_hi'].iloc[0]
    assert pr['n_clusters'].iloc[0] == 2                                      # clusters = held-out datasets
    assert set(s['stratum']) == {'C', 'site_images', 'all'}
    assert set(s[s['stratum'] == 'all']['label']) == set(det.LABELS)
    g = o['gating']
    gm = g[(g['tag'] == 'ALL') & (g['dataset'] == 'MEAN') & (g['score'] == 'D_mb')
           & (g['correction'] == 'mlls_bcts_quantile') & (g['tau_fit_on'] == 'all_scenarios')]
    assert len(gm) == 1
    assert set(g['tau_fit_on']) == {'all_scenarios', 'C_only'}
    # gating never does materially worse than argmax and beats always-correcting
    assert gm['qwk_gated_C'].iloc[0] >= gm['qwk_argmax_C'].iloc[0] - 0.01
    assert gm['qwk_gated_C'].iloc[0] >= gm['qwk_always_C'].iloc[0] - 1e-9


def test_no_test_label_leakage():
    _fabricate()
    o1 = _run(os.path.join('results', 'master', 'run1'), R=2, B=0)
    _relabel_test_files(np.random.default_rng(5))
    o2 = _run(os.path.join('results', 'master', 'run2'), R=2, B=0)
    d1, d2 = o1['draws'], o2['draws']
    label_free = list(det.SCORES) + [c for c in d1.columns if c.startswith('pi_')] + ['n_dev', 'n_test']
    pd.testing.assert_frame_equal(d1[label_free], d2[label_free])            # detector identical
    assert not np.allclose(d1['qwk_argmax'], d2['qwk_argmax'])                # outcomes did use TEST labels


def test_schemas_figure_and_determinism(world):
    o, out = world
    d = o['draws']
    for c in ['tag', 'dataset', 'scenario', 'draw', 'n_dev', 'n_test', 'tv_true_dev', 'site_images',
              'bbse_cond', 'bbse_degenerate', *det.SCORES, *[f'qwk_{m}' for m in det.QWK_METHODS],
              'harm_mlls', 'harm_bbse', 'harm_vs_oracle_mlls', 'unsafe_mlls', 'unsafe_bbse',
              'unsafe_vs_oracle_mlls']:
        assert c in d.columns, c
    assert set(d['scenario']) == {'A0', 'A', 'B', 'C'}
    assert (d[d['scenario'] == 'C']['draw'] == -1).sum() == 4                 # 2 tags x 2 sites observed
    assert set(o['summary']['tag']) == {'r_prior', 'r_annot', 'ALL'}
    assert set(o['gating']['correction']) == set(det.CORRECTIONS.values())
    assert list(o['gating'].columns) == det.GATING_COLS
    for ext in ('png', 'pdf'):
        assert os.path.isfile(os.path.join('results', 'figures', f'fig_shift_detector.{ext}'))
    o2 = _run(os.path.join('results', 'master', 'again'))
    pd.testing.assert_frame_equal(o['draws'], o2['draws'])
    pd.testing.assert_frame_equal(o['summary'], o2['summary'])


def test_missing_run_is_skipped_with_a_note():
    _fabricate(runs=(('r_prior', np.array([0.2, 0.3, 0.25, 0.15, 0.1]), 0.0),))
    o = _run(os.path.join('results', 'master', 'recal'), tags='r_prior,does_not_exist', R=2, B=0)
    assert set(o['draws']['tag']) == {'r_prior'}
    assert o['notes']['tag'].eq('does_not_exist').any()


def _relabel_dev_files(rng):
    for root, _, files in os.walk('results'):
        for f in files:
            if f.startswith('predictions_') and f.endswith('_dev.npz'):
                p = os.path.join(root, f)
                with np.load(p, allow_pickle=True) as z:
                    d = {k: z[k] for k in z.files}
                d['y_true'] = rng.permutation((d['y_true'] + 1) % 5)
                np.savez(p, **d)


def test_observed_scores_do_not_use_dev_labels():
    _fabricate()
    o1 = _run(os.path.join('results', 'master', 'run1'), R=1, B=0)
    _relabel_dev_files(np.random.default_rng(3))
    o2 = _run(os.path.join('results', 'master', 'run2'), R=1, B=0)
    k = list(det.SCORES) + [c for c in o1['draws'].columns if c.startswith('pi_')]
    a = o1['draws'][o1['draws']['draw'] == -1].reset_index(drop=True)
    b = o2['draws'][o2['draws']['draw'] == -1].reset_index(drop=True)
    pd.testing.assert_frame_equal(a[k], b[k])            # observed split: DEV labels never reach the detector


def test_single_heldout_site_does_not_crash():
    _fabricate(runs=(('r_prior', np.array([0.2, 0.3, 0.25, 0.15, 0.1]), 0.0),))
    from tests import test_shift_decomposition as tsd
    for f in os.listdir(os.path.join('results', 'phase10_ms_ddr', 'r_prior')):
        if f.startswith('predictions_idrid'):
            os.remove(os.path.join('results', 'phase10_ms_ddr', 'r_prior', f))
    assert tsd.SITES[1] == 'idrid'
    o = _run(os.path.join('results', 'master', 'one'), tags='r_prior', R=2, B=10)
    assert set(o['draws']['dataset']) == {'messidor'}
    assert o['gating'].empty and list(o['gating'].columns) == det.GATING_COLS
    assert os.path.isfile(os.path.join('results', 'figures', 'fig_shift_detector.png'))
