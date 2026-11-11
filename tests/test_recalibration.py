"""Recalibration / label-shift suite: src/evaluation/label_shift.py, src/evaluation/clinical.py and
tools/recalibration_suite.py (end to end on fabricated runs, incl. a proof that no fitted quantity
depends on TEST labels)."""
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from scipy.stats import ks_2samp

from src.evaluation import clinical as cl
from src.evaluation import label_shift as ls
from src.evaluation.cutpoints import DEFAULT_CUTS, grade_metrics

TOOL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tools', 'recalibration_suite.py')
K = 5
MU, SD = np.arange(K, dtype=float), 0.7


def _simulate(pi, n, rng, train_prior=None, T=1.0):
    """Pure label shift: fixed class-conditional score densities N(k, SD); classifier posterior computed
    with `train_prior` (miscalibrated w.r.t. pi when they differ) and temperature T."""
    y = rng.choice(K, size=n, p=pi)
    s = rng.normal(MU[y], SD)
    ll = -(s[:, None] - MU[None]) ** 2 / (2 * SD * SD)
    prior = pi if train_prior is None else train_prior
    z = (ll + np.log(prior)) / T
    z -= z.max(1, keepdims=True)
    P = np.exp(z)
    return y, P / P.sum(1, keepdims=True)


PI_SRC = np.array([0.50, 0.10, 0.20, 0.10, 0.10])
PI_TGT = np.array([0.20, 0.30, 0.20, 0.20, 0.10])
TRAIN_PRIOR = np.array([0.30, 0.20, 0.20, 0.15, 0.15])


# ---------------------------------------------------------------------------
# label_shift.py
# ---------------------------------------------------------------------------

def test_mlls_bcts_and_bbse_recover_target_prior_and_correction_helps():
    rng = np.random.default_rng(0)
    ys, Ps = _simulate(PI_SRC, 5000, rng, TRAIN_PRIOR, T=1.5)
    yt, Pt = _simulate(PI_TGT, 5000, rng, TRAIN_PRIOR, T=1.5)
    bcts = ls.fit_bcts(Ps, ys)
    assert bcts['T'] > 0 and abs(bcts['b'].mean()) < 1e-9
    pi_src = ls.class_shares(ys)
    pi_mlls, P_adj = ls.mlls_em(ls.apply_bcts(Pt, bcts), pi_src)
    pi_bbse = ls.bbse(Ps, ys, Pt)
    assert ls.tv_distance(pi_mlls, PI_TGT) < 0.05
    assert ls.tv_distance(pi_bbse, PI_TGT) < 0.05
    assert ls.tv_distance(pi_src, PI_TGT) > 0.25                       # the shift is real
    Pc = ls.prior_correct(ls.apply_bcts(Pt, bcts), pi_src, pi_mlls)
    assert (Pc.argmax(1) == yt).mean() > (ls.apply_bcts(Pt, bcts).argmax(1) == yt).mean() + 0.03
    assert np.allclose(P_adj, Pc)
    for x in (pi_mlls, pi_bbse, Pc):
        assert np.all(np.isfinite(x))


def test_em_leaves_prior_unchanged_without_shift():
    rng = np.random.default_rng(1)
    ys, Ps = _simulate(PI_SRC, 4000, rng)                              # calibrated to PI_SRC
    pi_exact = Ps.mean(0)                                              # exact fixed point
    pi_hat, P_adj = ls.mlls_em(Ps, pi_exact)
    assert np.abs(pi_hat - pi_exact).max() < 1e-8 and np.allclose(P_adj, Ps)
    yt, Pt = _simulate(PI_SRC, 4000, rng)                              # new sample, same prior
    assert ls.tv_distance(ls.mlls_em(Pt, PI_SRC)[0], PI_SRC) < 0.04


def test_zero_mass_classes_never_give_nan():
    P = np.zeros((6, K))
    P[:, 0] = 1.0                                                      # one-hot, empty classes
    y = np.zeros(6, int)
    pi = np.array([1.0, 0, 0, 0, 0])
    for out in (ls.mlls_em(P, pi)[0], ls.bbse(P, y, P), ls.prior_correct(P, pi, [0, 0, 0, 0, 1.0]),
                ls.apply_bcts(P, ls.fit_bcts(P, y)), ls.smoothed_prevalence([]),
                ls.vector_scale_apply(P, ls.vector_scale_fit(P, y))):
        assert np.all(np.isfinite(out))
    assert np.allclose(ls.smoothed_prevalence([0, 0, 1], alpha=1.0), np.array([3, 2, 1, 1, 1]) / 8.0)


def test_quantile_align_maps_onto_source_distribution():
    rng = np.random.default_rng(2)
    E_src = np.clip(rng.normal(1.0, 0.6, 3000), 0, 4)
    E_ref = np.clip(rng.normal(1.8, 1.0, 1500), 0, 4)                  # shifted + wider target (DEV)
    E_new = np.clip(rng.normal(1.8, 1.0, 1500), 0, 4)                  # fresh target images (TEST)
    m = ls.fit_quantile_map(E_ref, E_src)
    assert ks_2samp(E_new, E_src).statistic > 0.3
    assert ks_2samp(m(E_new), E_src).statistic < 0.05
    grid = np.linspace(-1, 5, 200)
    assert np.all(np.diff(m(grid)) >= -1e-12)                          # monotone
    assert np.allclose(ls.quantile_align(E_ref, E_src, E_new), m(E_new))


@pytest.mark.parametrize('delta', [0.05, 0.10, 0.3])
def test_perturb_prevalence_tv_exact_and_in_simplex(delta):
    rng = np.random.default_rng(3)
    for pi in (PI_TGT, np.array([0.7, 0.0, 0.2, 0.1, 0.0])):
        for _ in range(50):
            q = ls.perturb_prevalence(pi, 'tv_random', delta, rng)
            assert abs(ls.tv_distance(q, pi) - delta) < 1e-9
            assert q.min() >= 0 and abs(q.sum() - 1) < 1e-9
        for d in (-delta, delta):
            if 0 <= pi[0] + d <= 1:
                q = ls.perturb_prevalence(pi, 'nodr_shift', d)
                assert abs(ls.tv_distance(q, pi) - abs(d)) < 1e-9 and abs(q[0] - pi[0] - d) < 1e-12
                if q[1:].sum() > 0:
                    assert np.allclose(q[1:] / q[1:].sum(), pi[1:] / pi[1:].sum())
    with pytest.raises(ValueError):
        ls.perturb_prevalence(np.array([0.02, 0.5, 0.48, 0, 0]), 'nodr_shift', -0.05)


def test_fit_cut_offset_recovers_a_common_shift():
    rng = np.random.default_rng(4)
    E = rng.uniform(0, 4, 400)
    y = np.digitize(E, DEFAULT_CUTS + 0.3)
    s = ls.fit_cut_offset(E, y)
    assert abs(s - 0.3) < 0.05
    assert ls.fit_cut_offset(E[:3], np.zeros(3, int)) == 0.0


# ---------------------------------------------------------------------------
# clinical.py
# ---------------------------------------------------------------------------

def test_cluster_bootstrap_resamples_whole_groups():
    groups = np.repeat([f'p{i}' for i in range(30)], 4)
    rng = np.random.default_rng(5)
    for idx in cl.cluster_bootstrap_indices(groups, 20, rng):
        g, c = np.unique(groups[idx], return_counts=True)
        assert len(idx) == len(groups)                                  # 30 groups drawn x 4 images
        assert np.all(c % 4 == 0)
        for gg in g:                                                    # each drawn group: all its images
            members, mc = np.unique(idx[groups[idx] == gg], return_counts=True)
            assert len(members) == 4 and len(set(mc)) == 1
    W = cl.cluster_bootstrap_weights(groups, 10, np.random.default_rng(6))
    assert np.all(W.reshape(10, 30, 4).std(2) == 0)
    W_img = cl.cluster_bootstrap_weights(np.arange(50).astype(str), 200, np.random.default_rng(7))
    assert W_img.sum(1).tolist() == [50] * 200 and W_img.std(0).min() > 0     # image-level when unique


def test_batch_metrics_match_grade_metrics():
    rng = np.random.default_rng(8)
    y = rng.integers(0, 5, 300)
    pred = np.clip(y + rng.integers(-1, 2, 300), 0, 4)
    groups = np.repeat(np.arange(100), 3).astype(str)
    idx = cl.cluster_bootstrap_indices(groups, 5, np.random.default_rng(9))
    W = cl.cluster_bootstrap_weights(groups, 5, np.random.default_rng(9))
    bm = cl.batch_metrics_from_weights(y, pred, W)
    for b, i in enumerate(idx):
        gm = grade_metrics(y[i], pred[i])
        for k in ('qwk', 'acc', 'f1_macro', 'mild_recall', 'bin_sens', 'bin_spec'):
            assert abs(bm[k][b] - gm[k]) < 1e-9, k


def test_full_metrics_c_index_and_referral_arithmetic():
    rng = np.random.default_rng(10)
    y = rng.integers(0, 5, 2000)
    assert cl.full_metrics(y, y, score=y + rng.uniform(0, 0.9, 2000))['c_index'] == 1.0
    assert abs(cl.full_metrics(y, y, score=rng.uniform(0, 4, 2000))['c_index'] - 0.5) < 0.03
    assert abs(cl.ordinal_c_index(y, np.zeros(2000)) - 0.5) < 1e-12            # all ties -> 1/2
    y = np.array([0, 0, 0, 1, 2, 2, 3, 4, 4, 1])
    p = np.array([0, 2, 0, 1, 1, 2, 3, 1, 4, 0])
    m = cl.full_metrics(y, p, score=p.astype(float), pref=(p >= 2).astype(float))
    assert m['referrals_per_1000'] == 1000 * 4 / 10                     # pred >= 2: idx 1, 5, 6, 8
    assert m['missed_referable_per_1000'] == 1000 * 2 / 10              # true >= 2 with pred < 2: idx 4, 7
    assert m['false_referrals_per_1000'] == 1000 * 1 / 10
    assert m['cm_0_2'] == 1 and sum(m[f'cm_{i}_{j}'] for i in range(5) for j in range(5)) == 10
    assert abs(m['mae'] - np.abs(y - p).mean()) < 1e-12
    assert m['recall_4'] == 0.5 and 'auroc_referable' in m and 'auroc_ovr_2' in m
    ppv, npv = cl.ppv_npv(0.9, 0.95, 0.1)
    assert abs(ppv - 0.09 / (0.09 + 0.045)) < 1e-12 and abs(npv - 0.855 / (0.855 + 0.01)) < 1e-12


def test_threshold_for_specificity_reaches_target_on_dev():
    rng = np.random.default_rng(11)
    yb = rng.random(500) < 0.3
    s = rng.normal(yb * 1.5, 1.0)
    for spec in (0.9, 0.95, 0.99):
        t = cl.threshold_for_specificity(s, yb, spec)
        r = cl.binary_at_threshold(s, yb, t)
        assert r['spec'] >= spec
        eps_t = np.sort(s[~yb])[int(np.ceil(spec * (~yb).sum())) - 1]  # one negative lower -> below target
        assert cl.binary_at_threshold(s, yb, eps_t)['spec'] < spec


def test_other_sites_prior_never_uses_target():
    shares = {'a': np.array([1.0, 0, 0, 0, 0]), 'b': np.array([0.5, 0.5, 0, 0, 0]),
              'c': np.array([0.5, 0, 0.5, 0, 0])}
    from tools.recalibration_suite import other_sites_prior
    q = other_sites_prior(shares, 'a')
    assert np.allclose(q, [0.5, 0.25, 0.25, 0, 0])
    shares['a'] = np.array([0, 0, 0, 0, 1.0])                          # changing the target changes nothing
    assert np.allclose(other_sites_prior(shares, 'a'), q)


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------

SITES = {'messidor': (0.7, [0.62, 0.08, 0.2, 0.06, 0.04]), 'deepdrid': (-0.2, [0.45, 0.15, 0.2, 0.12, 0.08]),
         'idrid': (0.4, [0.35, 0.05, 0.35, 0.15, 0.10]), 'ddr': (0.0, [0.5, 0.05, 0.35, 0.03, 0.07])}


def _probs(s, temp=0.35):
    cum = 1.0 / (1.0 + np.exp(-(s[:, None] - DEFAULT_CUTS[None, :]) / temp))
    P = np.concatenate([1 - cum[:, :1], cum[:, :-1] - cum[:, 1:], cum[:, -1:]], axis=1)
    P = np.clip(P, 1e-6, None)
    return P / P.sum(1, keepdims=True)


def _fabricate(rng):
    """2 tags trained on aptos+eyepacs+ddr (held out: messidor, deepdrid, idrid) + split CSVs + leakage."""
    os.makedirs(os.path.join('splits', 'external'), exist_ok=True)
    runs = {'r_ce': ('phase10_ms_ddr', 0.45), 'r_ord': ('phase4_joint', 0.3)}
    labels = {}
    for ds, (shift, pi) in SITES.items():
        for split, n in (('dev', 120), ('test', 160)):
            y = rng.choice(K, size=n, p=pi)
            lat = y + rng.uniform(-0.45, 0.45, n)
            ids = np.array([f'{ds}_{split}_{i:04d}' for i in range(n)])
            grp = [f'{ds}_pt{i // 4}' for i in range(n)] if ds == 'deepdrid' else list(ids)
            pd.DataFrame({'id_code': ids, 'diagnosis': y, 'path': '', 'group': grp}).to_csv(
                os.path.join('splits', 'external', f'{ds}_{split}.csv'), index=False)
            labels[(ds, split)] = (ids, y, lat - shift)              # site under-/over-grades by `shift`
    y_oof = rng.choice(K, size=400, p=[0.49, 0.1, 0.27, 0.05, 0.09])
    y_at = rng.choice(K, size=120, p=[0.49, 0.1, 0.27, 0.05, 0.09])
    for tag, (phase, noise) in runs.items():
        ed = os.path.join('results', phase, tag)
        os.makedirs(ed, exist_ok=True)
        pd.DataFrame([{'phase': phase, 'experiment': tag, 'train_sources': 'aptos,eyepacs,ddr'}]).to_csv(
            os.path.join(ed, 'summary.csv'), index=False)
        T = 1.3 if tag == 'r_ce' else 1.0
        from src.evaluation.metrics import apply_temperature
        for (ds, split), (ids, y, lat) in labels.items():
            raw = _probs(lat + rng.normal(0, noise, len(y)))
            perm = rng.permutation(len(y))
            kw = {'y_prob_raw': raw[perm]} if tag == 'r_ce' else {}
            np.savez(os.path.join(ed, f'predictions_{ds}_{split}.npz'), ids=ids[perm], y_true=y[perm],
                     y_prob=apply_temperature(raw, T)[perm], **kw)
        P_oof = _probs(y_oof + rng.uniform(-0.45, 0.45, 400) + rng.normal(0, noise, 400))
        np.savez(os.path.join(ed, 'oof.npz'), ids=np.array([f'oof{i}' for i in range(400)]), y_true=y_oof,
                 y_prob=P_oof)
        P_at = apply_temperature(_probs(y_at + rng.normal(0, noise + 0.3, 120)), T)
        np.savez(os.path.join(ed, 'predictions_aptos_test.npz'), ids=np.array([f'at{i:03d}' for i in range(120)]),
                 y_true=y_at, y_prob=P_at)
    # test-time adaptation predictions for one tag / one site (written by another tool on the server)
    ids, y, lat = labels[('idrid', 'test')]
    td = os.path.join('results', 'phase10_ms_ddr', 'r_ce', 'target_adapt', 'bn')
    os.makedirs(td, exist_ok=True)
    np.savez(os.path.join(td, 'predictions_idrid_test.npz'), ids=ids, y_true=y, y_prob=_probs(lat + 0.3))
    os.makedirs(os.path.join('results', 'master'), exist_ok=True)
    pd.DataFrame([{'tag': 'r_ce', 'set': 'aptos_test', 'eval_id': f'at{i:03d}', 'train_dataset': 'APTOS',
                   'train_id': f'tr{i}', 'hamming_distance': 2} for i in (3, 7, 11)] +
                 [{'tag': 'r_ce', 'set': 'aptos_test', 'eval_id': 'at019', 'train_dataset': 'EyePACS',
                   'train_id': 'ep1', 'hamming_distance': 3}] +
                 [{'tag': 'r_ce', 'set': 'messidor_dev', 'eval_id': 'messidor_dev_0001', 'train_dataset': 'APTOS',
                   'train_id': 'trx', 'hamming_distance': 1}]).to_csv(
        os.path.join('results', 'master', 'leakage_pairs.csv'), index=False)
    return labels


def _run(out_dir):
    r = subprocess.run([sys.executable, TOOL, '--tags', 'r_ce,r_ord', '--ns', '10,25', '--draws', '3', '--B', '40',
                        '--misspec_draws', '4', '--out_dir', out_dir], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout


def _relabel_test_files(rng):
    """Permute AND change every held-out TEST label (shares change too); images / probabilities untouched."""
    for root, _, files in os.walk('results'):
        for f in files:
            if f.startswith('predictions_') and f.endswith('_test.npz') and 'aptos' not in f:
                p = os.path.join(root, f)
                with np.load(p, allow_pickle=True) as z:
                    d = {k: z[k] for k in z.files}
                d['y_true'] = rng.permutation((d['y_true'] + 2) % K)
                np.savez(p, **d)


def test_recalibration_suite_end_to_end_and_no_test_label_leakage():
    rng = np.random.default_rng(12)
    labels = _fabricate(rng)
    out = os.path.join('results', 'master', 'recal')
    os.makedirs(out, exist_ok=True)
    # last-layer fine-tuning rows written by tools/fewshot_head.py (same schema) are merged
    pd.DataFrame([{'tag': 'r_ce', 'dataset': ds, 'method': 'fewshot_head', 'info': 'n_labels', 'n': 10, 'draw': d,
                   'n_test': 160, 'qwk': 0.7, 'acc': 60.0, 'f1_macro': 0.5, 'mild_recall': 0.3, 'bin_sens': 0.8,
                   'bin_spec': 0.9, 'pred_nodr_share': 50.0}
                  for ds in ('messidor', 'deepdrid', 'idrid') for d in range(3)]).to_csv(
        os.path.join(out, 'fewshot_head_draws.csv'), index=False)
    stdout = _run(out)
    assert 'MEAN external QWK' in stdout
    for f in ('recal_draws.csv', 'recal_summary.csv', 'recal_compare.csv', 'prevalence_estimates.csv',
              'binary_operating_points.csv', 'ordinal_discrimination.csv', 'indist_control.csv',
              'aptos_dup_sensitivity.csv', 'recal_fitted.csv', 'recal_notes.csv'):
        assert os.path.isfile(os.path.join(out, f)), f
    for f in ('fig_recal_levels', 'fig_recal_misspec', 'fig_recal_operating_points'):
        for ext in ('png', 'pdf'):
            assert os.path.isfile(os.path.join('results', 'figures', f'{f}.{ext}'))

    from tools.recalibration_suite import AGGREGATES, DRAW_COLS, INFO_LEVELS, SRC_JOINT, SRC_TEST
    draws = pd.read_csv(os.path.join(out, 'recal_draws.csv'))
    assert list(draws.columns[:len(DRAW_COLS)]) == DRAW_COLS
    summ = pd.read_csv(os.path.join(out, 'recal_summary.csv'))
    for df in (draws, summ):
        assert set(df['info']) <= set(INFO_LEVELS)
        assert set(df.loc[~df['dataset'].isin(AGGREGATES), 'dataset']) == {'messidor', 'deepdrid', 'idrid'}  # DDR
        oracle = df['info'] == 'prevalence_oracle'
        assert df.loc[oracle, 'method'].str.contains('oracle').all()
        assert (df.loc[df['method'].str.contains('oracle'), 'info'] == 'prevalence_oracle').all()
    assert not draws.duplicated(['tag', 'dataset', 'method', 'info', 'n', 'draw']).any()
    methods = set(summ['method'])
    for m in ('argmax', 'default_cuts', 'source_cuts', 'upa', 'mlls_bcts_bayes', 'mlls_bcts_quantile',
              'em_raw_bayes', 'em_raw_quantile', 'bbse_bayes', 'bbse_quantile', 'oracle_quantile', 'oracle_bayes',
              'other_sites_quantile', 'misspecified_quantile_nodr+0.05', 'misspecified_quantile_tv0.10',
              'fewshot_plain', 'fewshot_shrink', 'fewshot_prev_quantile', 'fewshot_offset', 'fewshot_temp',
              'fewshot_vector', 'all_dev_cuts', 'all_dev_temp', 'bn_adapt_argmax', 'fewshot_head',
              'mlls_bcts_bagged_quantile', 'mlls_bcts_shrunk_quantile', 'bbse_bagged_quantile', 'bbse_shrunk_quantile'):
        assert m in methods, m
    info_of = summ.drop_duplicates('method').set_index('method')['info']
    for m in ('mlls_bcts_bagged_quantile', 'mlls_bcts_shrunk_quantile', 'bbse_bagged_quantile', 'bbse_shrunk_quantile'):
        assert info_of[m] == 'unlabelled_target'
        assert set(summ.loc[summ['method'] == m, 'n']) == {0} and np.isfinite(summ.loc[summ['method'] == m, 'qwk_mean']).all()
    # both tags hold out the same sites -> MEAN_COMMON == MEAN, `sites` names them
    mc = summ[summ['dataset'] == 'MEAN_COMMON'].set_index(['tag', 'method', 'n'])
    mn = summ[summ['dataset'] == 'MEAN'].set_index(['tag', 'method', 'n'])
    assert set(mc['sites']) == {'messidor,deepdrid,idrid'} and set(mn['sites']) == {'messidor,deepdrid,idrid'}
    assert np.allclose(mc.loc[mn.index, 'qwk_mean'], mn['qwk_mean'], equal_nan=True)
    assert info_of['upa'] == 'unlabelled_target' and info_of['source_cuts'] == 'none'
    assert info_of['all_dev_cuts'] == 'all_dev_labels' and info_of['fewshot_shrink'] == 'n_labels'
    assert set(summ.loc[summ['method'] == 'all_dev_cuts', 'n']) == {-1}
    mean = summ[(summ['dataset'] == 'MEAN') & (summ['tag'] == 'r_ce')].set_index(['method', 'n'])
    assert mean.loc[('fewshot_plain', 25), 'draws'] == 3 and np.isfinite(mean.loc[('argmax', 0), 'qwk_lo'])
    # the sites' grading is shifted: oracle prevalence / all-DEV cut-points beat argmax on average
    assert mean.loc[('all_dev_cuts', -1), 'qwk_mean'] > mean.loc[('argmax', 0), 'qwk_mean']
    # few-shot draws are those of tools/adaptation_curve (same seed convention)
    from tools.adaptation_curve import _draw_indices
    assert len(_draw_indices(120, 10, 3, 'messidor', 0)) == 3
    # other_sites prior = mean of the OTHER sites' DEV shares (never the target's)
    fit = pd.read_csv(os.path.join(out, 'recal_fitted.csv'))
    sh = {ds: np.bincount(labels[(ds, 'dev')][1], minlength=K) / 120.0 for ds in ('messidor', 'deepdrid', 'idrid')}
    got = fit[(fit['tag'] == 'r_ce') & (fit['dataset'] == 'idrid') & (fit['method'] == 'other_sites')]
    got = got.sort_values('param')['value'].values
    assert np.allclose(got, (sh['messidor'] + sh['deepdrid']) / 2, atol=1e-9)
    # temperature: recovered from the (raw, scaled) file pairs for r_ce, refit for r_ord
    t = fit[(fit['method'] == 'source') & (fit['param'] == 'T_oof')].set_index('tag')['value']
    assert abs(t['r_ce'] - 1.3) < 1e-6

    cmp = pd.read_csv(os.path.join(out, 'recal_compare.csv'))
    assert {'argmax', 'mlls_bcts_bayes', 'fewshot_plain'} <= set(cmp['reference'])
    assert 'MEAN' in set(cmp['dataset'])
    units = cmp[cmp['n'] <= 0].groupby('dataset')['bootstrap_unit'].first()
    assert units['deepdrid'].startswith('cluster') and units['messidor'] == 'image'
    assert (cmp.loc[(cmp['reference'] == 'fewshot_plain'), 'bootstrap_unit'].str.contains('draw')).all()
    assert len(cmp[(cmp['method'] == 'fewshot_head') & (cmp['reference'] == 'fewshot_plain')])
    # few-label rows: the primary interval is the joint draws x TEST-bootstrap one; draws-only kept aside
    fl = cmp[(cmp['info'] == 'n_labels') & (cmp['method'] != 'fewshot_head')]
    assert len(fl) and (fl['interval_source'] == SRC_JOINT).all() and (fl['n_resamples'] == 40).all()
    assert np.allclose(fl['diff_lo'], fl['diff_lo_joint']) and np.allclose(fl['diff_hi'], fl['diff_hi_joint'])
    assert fl['diff_lo_draws'].notna().all() and {'MEAN', 'MEAN_COMMON'} <= set(fl['dataset'])
    fh = cmp[cmp['method'] == 'fewshot_head']
    assert fh['interval_source'].str.startswith('draws only').all() and fh['diff_lo_joint'].isna().all()
    assert np.allclose(fh['diff_lo'], fh['diff_lo_draws'])
    assert (cmp.loc[cmp['method'] == 'mlls_bcts_shrunk_quantile', 'interval_source'] == SRC_TEST).all()
    prev = pd.read_csv(os.path.join(out, 'prevalence_estimates.csv'))
    assert {'mlls_bcts', 'em_raw', 'bbse', 'other_sites', 'source_prior', 'dev_labels_oracle',
            'mlls_bcts_shrunk', 'bbse_shrunk'} <= set(prev['estimator'])
    pb = prev[prev['estimator'].isin(['mlls_bcts', 'bbse'])]
    for k in range(K):
        assert (pb[f'pi_lo_{k}'] <= pb[f'pi_hi_{k}'] + 1e-12).all()
    assert pb['shrink_weight'].between(0, 1).all() and (pb['prior_boot_B'] == 200).all()
    units = pb.groupby('dataset')['prior_boot_unit'].first()
    assert units['deepdrid'].startswith('cluster') and units['messidor'] == 'image'      # DEV groups used
    pbt = fit[fit['method'].isin(['mlls_bcts_prior_boot', 'bbse_prior_boot']) & (fit['param'] == 'shrink_w')]
    assert len(pbt) == 2 * 2 * 3 and pbt['value'].between(0, 1).all()
    # the shrunk prior is pi_src + w (pi_hat - pi_src)
    for (tg, ds), g in prev[prev['estimator'].isin(['mlls_bcts', 'mlls_bcts_shrunk', 'source_prior'])].groupby(
            ['tag', 'dataset']):
        g = g.set_index('estimator')
        pc = [f'pi_hat_{k}' for k in range(K)]
        w = g.loc['mlls_bcts_shrunk', 'shrink_weight']
        assert np.allclose(g.loc['mlls_bcts_shrunk', pc].astype(float),
                           g.loc['source_prior', pc].astype(float)
                           + w * (g.loc['mlls_bcts', pc].astype(float) - g.loc['source_prior', pc].astype(float)),
                           atol=2e-6)
    binop = pd.read_csv(os.path.join(out, 'binary_operating_points.csv'))
    dt = binop[binop['method'] == 'dev_threshold_pref']
    assert set(np.round(dt['target_spec'], 2)) == {0.90, 0.95, 0.97, 0.99} and 'ppv_at_0.10' in binop.columns
    assert (binop.loc[binop['method'] == 'test_oracle_roc', 'info'] == 'test_labels_oracle_reporting_only').all()
    orc = binop[(binop['method'] == 'test_oracle_roc') & (binop['dataset'] != 'MEAN')]
    assert (orc['spec'] >= orc['target_spec'] - 1e-12).all()        # achieved spec, PPV computed at it
    assert np.allclose(orc['ppv_at_0.10'], cl.ppv_npv(orc['sens'].values, orc['spec'].values, 0.10)[0])
    # fewshot_shrink records where its lambda came from: the OTHER held-out sites only
    sh = draws[draws['method'] == 'fewshot_shrink']
    assert len(sh) and all(ds not in str(f).split(',') and len(str(f).split(',')) == 2
                           for ds, f in zip(sh['dataset'], sh['lam_from']))
    od = pd.read_csv(os.path.join(out, 'ordinal_discrimination.csv'))
    assert {'c_index', 'auroc_referable', 'auroc_ovr_0'} <= set(od.columns) and 'MEAN' in set(od['dataset'])
    ind = pd.read_csv(os.path.join(out, 'indist_control.csv'))
    assert np.isfinite(ind['net_external_gain']).all() and set(ind['tag']) == {'r_ce', 'r_ord'}
    dup = pd.read_csv(os.path.join(out, 'aptos_dup_sensitivity.csv'))
    # r_ce has its own leakage rows -> every training source is excluded (3 APTOS + 1 EyePACS duplicate)
    ex = dup[(dup['tag'] == 'r_ce') & (dup['subset'] == 'excl_dups_of_training_pool')]
    assert len(ex) and (ex['n_excluded'] == 4).all() and (ex['n'] == 116).all()
    # r_ord has none -> falls back to the APTOS-train duplicates of other tags (same APTOS pool)
    exo = dup[(dup['tag'] == 'r_ord') & (dup['subset'] == 'excl_dups_of_aptos_train')]
    assert len(exo) and (exo['n_excluded'] == 3).all()
    # estimator-bias control on APTOS test
    eb = ind[ind['method'].isin(['mlls_bcts_quantile', 'mlls_bcts_bayes', 'bbse_quantile', 'bbse_bayes'])]
    assert len(eb) == 8 and eb['prior_tv_error_vs_aptos_test'].between(0, 1).all()

    # ---- leakage proof: change every TEST label, refit -> every fitted quantity identical ----------------
    keep = {f: pd.read_csv(os.path.join(out, f)) for f in ('recal_fitted.csv', 'prevalence_estimates.csv',
                                                           'binary_operating_points.csv', 'recal_summary.csv')}
    _relabel_test_files(np.random.default_rng(13))
    out2 = os.path.join('results', 'master', 'recal_relabel')
    _run(out2)
    f1, f2 = keep['recal_fitted.csv'], pd.read_csv(os.path.join(out2, 'recal_fitted.csv'))
    pd.testing.assert_frame_equal(f1, f2)
    p1, p2 = keep['prevalence_estimates.csv'], pd.read_csv(os.path.join(out2, 'prevalence_estimates.csv'))
    pi_cols = [c for c in p1.columns if c.startswith(('pi_hat_', 'dev_share_', 'pi_lo_', 'pi_hi_', 'shrink_weight'))]
    assert any(c.startswith('pi_lo_') for c in pi_cols) and 'shrink_weight' in pi_cols
    pd.testing.assert_frame_equal(p1[pi_cols], p2[pi_cols])
    assert not np.allclose(p1['test_share_0'], p2['test_share_0'])     # TEST labels really changed
    b1, b2 = (b[b['method'] == 'dev_threshold_pref'].reset_index(drop=True) for b in
              (keep['binary_operating_points.csv'], pd.read_csv(os.path.join(out2, 'binary_operating_points.csv'))))
    pd.testing.assert_frame_equal(b1[['tag', 'dataset', 'target_spec', 'threshold', 'dev_sens', 'dev_spec']],
                                  b2[['tag', 'dataset', 'target_spec', 'threshold', 'dev_sens', 'dev_spec']])
    s1, s2 = keep['recal_summary.csv'], pd.read_csv(os.path.join(out2, 'recal_summary.csv'))
    s1 = s1[s1['method'] != 'fewshot_head'].reset_index(drop=True)
    assert len(s1) == len(s2)
    assert not np.allclose(s1['qwk_mean'].fillna(0), s2['qwk_mean'].fillna(0))    # metrics did use TEST labels


# ---------------------------------------------------------------------------
# audit regressions
# ---------------------------------------------------------------------------

def test_nan_threshold_gives_nan_operating_point():
    """Regression: a NaN DEV threshold (DEV without negatives) was scored as sens 0 / spec 1."""
    r = cl.binary_at_threshold(np.array([0.1, 0.9]), np.array([False, True]), float('nan'))
    assert all(np.isnan(v) for v in r.values())
    t = cl.threshold_for_specificity(np.array([0.2, 0.8]), np.array([True, True]), 0.9)
    assert np.isnan(t) and np.isnan(cl.binary_at_threshold(np.array([0.2, 0.8]), np.array([True, True]), t)['sens'])


def test_oracle_operating_point_reports_the_achieved_specificity():
    """Regression: the TEST-oracle row reported the TARGET specificity (and PPV/NPV at it), not the
    specificity of the chosen operating point."""
    # 10 negatives (6 at 0.1, 4 at 0.5), 5 positives (2 at 0.5, 3 at 0.9)
    s = np.array([0.1] * 6 + [0.5] * 4 + [0.5] * 2 + [0.9] * 3)
    yb = np.array([False] * 10 + [True] * 5)
    sens, spec = cl.sens_at_spec_oracle(s, yb, 0.65, return_spec=True)
    assert abs(sens - 0.6) < 1e-12 and spec == 1.0         # only threshold 0.9 reaches spec >= 0.65
    assert cl.sens_at_spec_oracle(s, yb, 0.65) == sens
    sens, spec = cl.sens_at_spec_oracle(s, yb, 0.5, return_spec=True)
    assert sens == 1.0 and abs(spec - 0.6) < 1e-12        # threshold 0.5: 6 of 10 negatives below, not 0.5


def test_recalibration_suite_without_bootstrap_B0():
    """Regression: --B 0 crashed in summarise (percentile of an empty bootstrap)."""
    _fabricate(np.random.default_rng(21))
    out = os.path.join('results', 'master', 'recal_b0')
    r = subprocess.run([sys.executable, TOOL, '--tags', 'r_ce', '--ns', '10', '--draws', '2', '--B', '0',
                        '--misspec_draws', '2', '--out_dir', out], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    summ = pd.read_csv(os.path.join(out, 'recal_summary.csv'))
    det = summ[(summ['method'] == 'argmax')]
    assert det['qwk_mean'].notna().all() and ('qwk_lo' not in summ.columns or det['qwk_lo'].isna().all())


def test_shrinkage_lambda_ignores_the_target_site():
    """fewshot_shrink's lambda (tools/adaptation_curve._pick_lambda, used by the suite) depends only on the
    other sites' inner scores: changing every inner score of the target site changes nothing."""
    from tools.adaptation_curve import LAMBDAS, _pick_lambda
    rng = np.random.default_rng(0)
    Q = {(ds, 10, lam): float(rng.random()) for ds in ('a', 'b', 'c') for lam in LAMBDAS}
    lam = _pick_lambda(Q, 'a', ['b', 'c'], 10)
    Q2 = dict(Q)
    for l_ in LAMBDAS:
        Q2[('a', 10, l_)] = 10.0 if l_ == LAMBDAS[-1] else -10.0
    assert _pick_lambda(Q2, 'a', ['b', 'c'], 10) == lam
    best = max(LAMBDAS, key=lambda l_: (np.mean([Q[('b', 10, l_)], Q[('c', 10, l_)]]), -l_))
    assert lam == best


# ---------------------------------------------------------------------------
# reviewer extensions: joint few-label intervals, MEAN_COMMON, uncertainty-aware prevalence
# ---------------------------------------------------------------------------

def _fake_tag(rng, D=5, n_test=150, B=400, flip=0.02):
    """A hand-built TagResult with one site: argmax (deterministic) and two draw-based methods whose draws
    differ only slightly (flip share of images changed per draw), on a noisy TEST set."""
    from tools.recalibration_suite import Scorer, TagResult
    y = rng.integers(0, K, n_test)
    base = np.clip(y + rng.integers(-1, 2, n_test), 0, K - 1)
    W = cl.cluster_bootstrap_weights(np.arange(n_test).astype(str), B, np.random.default_rng(99))
    sc = Scorer(y, W, 'image')
    R = TagResult('t')
    R.heldout, R.sites, R.units, R.scorers = ['a'], ['a'], {'a': 'image'}, {'a': sc}
    am = np.clip(base + (rng.random(n_test) < 0.15), 0, K - 1)
    R.boot[('a', 'argmax')], R.full[('a', 'argmax')] = sc.boot(am), sc.full(am)
    rows = [{'tag': 't', 'dataset': 'a', 'method': 'argmax', 'info': 'none', 'n': 0, 'draw': 0, 'n_test': n_test,
             **{m: sc.grade(am)[m] for m in ('qwk', 'acc', 'f1_macro', 'mild_recall', 'bin_sens', 'bin_spec',
                                             'pred_nodr_share')}}]
    preds = {}
    for method in ('fewshot_plain', 'fewshot_copy'):
        for d in range(D):
            if method == 'fewshot_plain':
                p = base.copy()
                ch = rng.random(n_test) < flip
                p[ch] = rng.integers(0, K, ch.sum())
                preds[d] = p
            p = preds[d]                                           # fewshot_copy == fewshot_plain, draw by draw
            R.dpred.setdefault(('a', method, 10), {})[d] = p.astype(np.int8)
            rows.append({'tag': 't', 'dataset': 'a', 'method': method, 'info': 'n_labels', 'n': 10, 'draw': d,
                         'n_test': n_test, **{m: sc.grade(p)[m] for m in ('qwk', 'acc', 'f1_macro', 'mild_recall',
                                                                          'bin_sens', 'bin_spec', 'pred_nodr_share')}})
    return R, pd.DataFrame(rows), y, W, am, preds


def test_joint_interval_uses_identical_resamples_and_is_wider_than_draws_only():
    from tools.recalibration_suite import SRC_JOINT, compare
    rng = np.random.default_rng(30)
    R, draws, y, W, am, preds = _fake_tag(rng)
    sc = R.scorers['a']
    # resample b is row b of the deterministic methods' W, scored with draw b mod D
    jc = sc.boot_cycled([preds[d] for d in range(5)])
    idx = cl.cluster_bootstrap_indices(np.arange(len(y)).astype(str), W.shape[0], np.random.default_rng(99))
    for b in (0, 1, 7, 123, 399):
        assert abs(jc['qwk'][b] - grade_metrics(y[idx[b]], preds[b % 5][idx[b]])['qwk']) < 1e-9
    assert np.array_equal(sc.boot_cycled([am])['qwk'], R.boot[('a', 'argmax')]['qwk'])      # D = 1: same as boot
    cmp = compare(draws, [R], common=['a'])
    r = cmp[(cmp['method'] == 'fewshot_plain') & (cmp['reference'] == 'argmax') & (cmp['metric'] == 'qwk')
            & (cmp['dataset'] == 'a')].iloc[0]
    expect = jc['qwk'] - R.boot[('a', 'argmax')]['qwk']                       # paired resample by resample
    assert r['interval_source'] == SRC_JOINT and r['n_resamples'] == 400
    assert np.isclose(r['diff_lo_joint'], np.percentile(expect, 2.5))
    assert np.isclose(r['diff_hi_joint'], np.percentile(expect, 97.5))
    assert r['diff_lo'] == r['diff_lo_joint'] and r['diff_hi'] == r['diff_hi_joint']
    assert r['diff_hi_joint'] - r['diff_lo_joint'] >= r['diff_hi_draws'] - r['diff_lo_draws']
    assert r['diff_hi_joint'] - r['diff_lo_joint'] > 2 * (r['diff_hi_draws'] - r['diff_lo_draws'])   # TEST noise
    # vs fewshot_plain: paired on the same draw AND the same resample -> identical predictions give exactly 0
    c = cmp[(cmp['method'] == 'fewshot_copy') & (cmp['reference'] == 'fewshot_plain')]
    assert len(c) and (c['interval_source'] == SRC_JOINT).all()
    assert (c[['diff_lo', 'diff_hi', 'diff_lo_joint', 'diff_hi_joint', 'diff_lo_draws', 'diff_hi_draws']] == 0).all().all()
    # MEAN / MEAN_COMMON over the single site equal the site row
    m = cmp[(cmp['method'] == 'fewshot_plain') & (cmp['reference'] == 'argmax') & (cmp['metric'] == 'qwk')]
    assert set(m['dataset']) == {'a', 'MEAN', 'MEAN_COMMON'}
    assert np.allclose(m['diff_lo_joint'], r['diff_lo_joint']) and set(m['sites']) == {'a'}


def test_joint_interval_falls_back_to_draws_without_bootstrap():
    from tools.recalibration_suite import compare
    rng = np.random.default_rng(31)
    R, draws, *_ = _fake_tag(rng, B=0)
    cmp = compare(draws, [R], common=['a'])
    fl = cmp[cmp['method'] == 'fewshot_plain']
    assert len(fl) and (fl['interval_source'] == 'draws only (--B 0)').all()
    assert np.allclose(fl['diff_lo'], fl['diff_lo_draws']) and fl['diff_lo_joint'].isna().all()


def test_mean_common_uses_exactly_the_intersection_of_held_out_sites():
    _fabricate(np.random.default_rng(32))
    # r_ord now trains on aptos+eyepacs only -> it also holds out DDR; r_ce holds out messidor/deepdrid/idrid
    pd.DataFrame([{'phase': 'phase4_joint', 'experiment': 'r_ord', 'train_sources': 'aptos,eyepacs'}]).to_csv(
        os.path.join('results', 'phase4_joint', 'r_ord', 'summary.csv'), index=False)
    out = os.path.join('results', 'master', 'recal_common')
    r = subprocess.run([sys.executable, TOOL, '--tags', 'r_ce,r_ord', '--ns', '10', '--draws', '2', '--B', '20',
                        '--misspec_draws', '2', '--B_pi', '20', '--out_dir', out], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert 'MEAN_COMMON external QWK' in r.stdout and 'timing:' in r.stdout
    summ = pd.read_csv(os.path.join(out, 'recal_summary.csv'))
    common = ['messidor', 'deepdrid', 'idrid']
    assert set(summ.loc[(summ['tag'] == 'r_ord') & (summ['dataset'] == 'MEAN'), 'sites']) == \
        {'messidor,ddr,deepdrid,idrid'}
    mc = summ[summ['dataset'] == 'MEAN_COMMON']
    assert set(mc['tag']) == {'r_ce', 'r_ord'} and set(mc['sites']) == {','.join(common)}
    for tag in ('r_ce', 'r_ord'):
        s = summ[summ['tag'] == tag]
        for method, n in (('argmax', 0), ('mlls_bcts_shrunk_quantile', 0), ('fewshot_plain', 10)):
            per = s[(s['method'] == method) & (s['n'] == n) & s['dataset'].isin(common)]
            assert len(per) == 3
            got = s[(s['method'] == method) & (s['n'] == n) & (s['dataset'] == 'MEAN_COMMON')]['qwk_mean'].iloc[0]
            assert abs(got - per['qwk_mean'].mean()) < 1e-5
    s = summ[(summ['tag'] == 'r_ord') & (summ['method'] == 'argmax')].set_index('dataset')
    assert abs(s.loc['MEAN', 'qwk_mean'] - s.loc[['messidor', 'ddr', 'deepdrid', 'idrid'], 'qwk_mean'].mean()) < 1e-5
    assert abs(s.loc['MEAN', 'qwk_mean'] - s.loc['MEAN_COMMON', 'qwk_mean']) > 1e-6       # DDR really excluded
    cmp = pd.read_csv(os.path.join(out, 'recal_compare.csv'))
    c = cmp[(cmp['tag'] == 'r_ord') & (cmp['method'] == 'default_cuts') & (cmp['reference'] == 'argmax')
            & (cmp['metric'] == 'qwk')].set_index('dataset')
    assert c.loc['MEAN_COMMON', 'sites'] == ','.join(common)
    assert abs(c.loc['MEAN_COMMON', 'diff_point'] - c.loc[common, 'diff_point'].mean()) < 1e-5
    assert abs(c.loc['MEAN', 'diff_point'] - c.loc[['messidor', 'ddr', 'deepdrid', 'idrid'], 'diff_point'].mean()) < 1e-5
    f = cmp[(cmp['method'] == 'fewshot_plain') & (cmp['dataset'] == 'MEAN_COMMON')]
    assert len(f) and set(f['tag']) == {'r_ce', 'r_ord'} and f['interval_source'].str.startswith('joint').all()
    notes = pd.read_csv(os.path.join(out, 'recal_notes.csv'))
    nt = notes.loc[notes['dataset'] == 'MEAN_COMMON', 'note'].iloc[0]
    assert 'messidor,deepdrid,idrid' in nt and 'r_ord: messidor,ddr,deepdrid,idrid' in nt
    bop = pd.read_csv(os.path.join(out, 'binary_operating_points.csv'))
    assert 'MEAN_COMMON' not in set(bop['dataset']) and 'MEAN' in set(bop['dataset'])


def test_shrinkage_weight_limits_and_range():
    pi_src = np.array([0.5, 0.1, 0.2, 0.1, 0.1])
    pi_hat = np.array([0.3, 0.2, 0.2, 0.2, 0.1])                       # ||shift||^2 = 0.06
    assert ls.shrinkage_weight(pi_hat, pi_src, 1e-8) > 0.9999          # precise estimate of a real shift
    assert ls.shrinkage_weight(pi_hat, pi_src, 0.06) == pytest.approx(0.5)
    tiny = pi_src + np.array([-1e-3, 1e-3, 0, 0, 0])                   # shift tiny relative to the noise
    assert ls.shrinkage_weight(tiny, pi_src, 0.01) < 1e-3
    assert ls.shrinkage_weight(pi_src, pi_src, 0.01) == 0.0 and ls.shrinkage_weight(pi_src, pi_src, 0.0) == 1.0
    rng = np.random.default_rng(33)
    for _ in range(200):
        a, b = rng.dirichlet(np.ones(K)), rng.dirichlet(np.ones(K))
        w = ls.shrinkage_weight(a, b, rng.exponential(0.05))
        assert 0.0 <= w <= 1.0
        p, w2 = ls.shrink_prior(a, b, 0.02)
        assert np.allclose(p, b + w2 * (a - b)) and abs(p.sum() - 1) < 1e-12 and p.min() >= 0


def test_bootstrap_priors_identical_resamples_ci_and_shrinkage_behaviour():
    rng = np.random.default_rng(34)
    ys, Ps = _simulate(PI_SRC, 3000, rng, TRAIN_PRIOR, T=1.5)
    bcts = ls.fit_bcts(Ps, ys)
    # strong shift, many images -> precise estimate, w close to 1, CI covers the truth
    yt, Pt = _simulate(PI_TGT, 2000, rng, TRAIN_PRIOR, T=1.5)
    groups = np.repeat(np.arange(1000), 2)
    idx, unit = ls.bootstrap_indices(len(Pt), 100, np.random.default_rng(1), groups)
    idx2, _ = ls.bootstrap_indices(len(Pt), 100, np.random.default_rng(1), groups)
    assert unit.startswith('cluster') and all(np.array_equal(i, j) for i, j in zip(idx, idx2))   # deterministic
    for i in idx[:5]:                                                  # whole clusters
        g, c = np.unique(groups[i], return_counts=True)
        assert np.all(c % 2 == 0)
    bt = ls.bootstrap_label_shift_priors(Pt, Ps, ys, bcts=bcts, indices=idx)
    pi_src = ls.class_shares(ys)
    for name in ('mlls_bcts', 'bbse'):
        r = bt[name]
        assert r['draws'].shape == (100, K) and np.allclose(r['draws'].sum(1), 1)
        assert np.all(r['lo'] <= r['pi'] + 1e-12) and np.all(r['pi'] <= r['hi'] + 1e-12)
        assert np.all(r['lo'] - 0.02 <= PI_TGT) and np.all(PI_TGT <= r['hi'] + 0.02)
        assert ls.shrinkage_weight(r['pi'], pi_src, r['cov_trace']) > 0.9
    assert np.allclose(bt['mlls_bcts']['pi'], ls.mlls_em(ls.apply_bcts(Pt, bcts), pi_src)[0], atol=1e-12)
    assert np.allclose(bt['bbse']['pi'], ls.bbse(Ps, ys, Pt))
    # both estimators saw the same resamples: re-estimating MLLS on resample b reproduces draw b
    b = 17
    assert np.allclose(bt['mlls_bcts']['draws'][b], ls.mlls_em(ls.apply_bcts(Pt[idx[b]], bcts), pi_src)[0], atol=1e-6)
    assert np.allclose(bt['bbse']['draws'][b], ls.bbse(Ps, ys, Pt[idx[b]]))
    # no shift, few images -> the estimated shift is mostly noise and w is small
    yt0, Pt0 = _simulate(pi_src, 60, np.random.default_rng(2), TRAIN_PRIOR, T=1.5)
    bt0 = ls.bootstrap_label_shift_priors(Pt0, Ps, ys, B=200, rng=np.random.default_rng(3), bcts=bcts)
    assert bt0['unit'] == 'image'
    w0 = ls.shrinkage_weight(bt0['mlls_bcts']['pi'], pi_src, bt0['mlls_bcts']['cov_trace'])
    assert w0 < 0.6
    # bagged cut-points: strictly increasing, equal to the plain ones when every draw is the same prior
    from src.evaluation.cutpoints import prevalence_cutpoints
    E = (np.arange(K)[None] * Pt).sum(1)
    c = ls.bagged_prevalence_cutpoints(E, bt['mlls_bcts']['draws'])
    assert np.all(np.diff(c) > 0)
    assert np.allclose(ls.bagged_prevalence_cutpoints(E, np.tile(PI_TGT, (7, 1))), prevalence_cutpoints(E, PI_TGT))
    cz = ls.bagged_prevalence_cutpoints(E, np.tile([0.5, 0.0, 0.0, 0.25, 0.25], (3, 1)))
    assert np.all(np.diff(cz) > 0)
