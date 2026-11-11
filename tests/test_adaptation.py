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
    assert set(curve['level']) == {'argmax', 'default_cuts', 'prevalence_oracle', 'fewshot', 'fewshot_shrink',
                                   'all'}
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


def _grouped_preds(rng, ds_groups, n_dev=20):
    """Prediction dicts in the analyse_tag layout for two models; TEST images come in whole patients (all
    images of a patient share label and predictions: maximal within-patient correlation)."""
    P_ref, P_tag = {}, {}
    for ds, (n_groups, size) in ds_groups.items():
        y_g = rng.integers(0, 5, n_groups)
        y = np.repeat(y_g, size)
        ids_t = np.array([f'{ds}_p{g:03d}_{k}' for g in range(n_groups) for k in range(size)])
        ids = np.concatenate([[f'{ds}_dev_{i:03d}' for i in range(n_dev)], ['|'], ids_t])
        for P in (P_ref, P_tag):
            P[('_y', ds)], P[('_ids', ds)] = y, ids
            for level in ('argmax', 'all'):
                P[(ds, level, 0)] = [np.repeat(np.clip(y_g + rng.integers(-1, 2, n_groups), 0, 4), size)]
            P[(ds, 'fewshot', 10)] = [np.clip(y + rng.integers(-1, 2, len(y)), 0, 4) for _ in range(3)]
    return P_ref, P_tag


def _write_split(ds, ids_test, groups, ext='.jpg'):
    os.makedirs(os.path.join('splits', 'external'), exist_ok=True)
    pd.DataFrame({'id_code': [f'{i}{ext}' for i in ids_test], 'diagnosis': 0,
                  'path': [f'/data/{ds}/{i}{ext}' for i in ids_test], 'group': groups}).to_csv(
        os.path.join('splits', 'external', f'{ds}_test.csv'), index=False)


def test_compare_cluster_bootstrap_resamples_whole_patients(monkeypatch):
    import tools.adaptation_curve as ac
    rng = np.random.default_rng(7)
    P_ref, P_tag = _grouped_preds(rng, {'deepdrid': (40, 4), 'messidor': (60, 2)})
    ids_dd = ac._test_ids(P_ref, 'deepdrid')
    patient = np.array([i.rsplit('_', 1)[0] for i in ids_dd])
    _write_split('deepdrid', ids_dd, patient)                       # ids match by file stem (.jpg dropped)
    # messidor: no split file -> image-level bootstrap
    seen = []
    real = ac._boot_indices

    def spy(groups, unit, n, B, rng_):
        for i in real(groups, unit, n, B, rng_):
            seen.append((unit, np.asarray(groups), i))
            yield i

    monkeypatch.setattr(ac, '_boot_indices', spy)
    rows = pd.DataFrame(ac.compare('m_ref', 'm_arm', P_ref, P_tag, 40, 0))
    det = rows[rows['level'] != 'fewshot']
    assert set(det.loc[det['dataset'] == 'deepdrid', 'bootstrap_unit']) == \
        {'cluster (group column; 40 groups / 160 images)'}
    assert set(det.loc[det['dataset'] == 'messidor', 'bootstrap_unit']) == {'image (no split file)'}
    assert set(rows.loc[rows['level'] == 'fewshot', 'bootstrap_unit']) == {'draws'}
    mean_units = set(det.loc[det['dataset'] == 'MEAN', 'bootstrap_unit'])
    assert len(mean_units) == 1 and 'deepdrid: cluster' in mean_units.pop()
    cluster = [(g, i) for u, g, i in seen if u.startswith('cluster')]
    assert cluster and len([1 for u, _, _ in seen if u.startswith('image')])
    for g, i in cluster:
        drawn = pd.Series(g[i]).value_counts()
        for grp, cnt in drawn.items():
            members = np.flatnonzero(g == grp)
            assert cnt % len(members) == 0                           # whole patients only
            # every image of a drawn patient appears equally often
            assert len(set(np.bincount(i, minlength=len(g))[members])) == 1
    # within-patient correlation is real here, so the patient-level interval must be wider than the image one
    monkeypatch.setattr(ac, '_boot_indices', real)

    def width(df):
        s = df[(df['dataset'] == 'deepdrid') & (df['level'] == 'argmax') & (df['metric'] == 'qwk')]
        return float(s['diff_hi'].iloc[0] - s['diff_lo'].iloc[0]), s['bootstrap_unit'].iloc[0]

    w_cl, u_cl = width(pd.DataFrame(ac.compare('m_ref', 'm_arm', P_ref, P_tag, 400, 0)))
    os.remove(os.path.join('splits', 'external', 'deepdrid_test.csv'))
    w_im, u_im = width(pd.DataFrame(ac.compare('m_ref', 'm_arm', P_ref, P_tag, 400, 0)))
    assert u_cl.startswith('cluster') and u_im == 'image (no split file)'
    assert w_cl > 1.3 * w_im, (w_cl, w_im)


def test_compare_unmatched_or_unique_groups_fall_back_to_images():
    import tools.adaptation_curve as ac
    rng = np.random.default_rng(8)
    P_ref, P_tag = _grouped_preds(rng, {'deepdrid': (20, 4), 'idrid': (30, 1)})
    ids_dd = ac._test_ids(P_ref, 'deepdrid')
    _write_split('deepdrid', ids_dd[:-1], [i.rsplit('_', 1)[0] for i in ids_dd[:-1]])   # one TEST id missing
    ids_id = ac._test_ids(P_ref, 'idrid')
    _write_split('idrid', ids_id, [f'g{k}' for k in range(len(ids_id))])                  # every group unique
    rows = pd.DataFrame(ac.compare('a', 'b', P_ref, P_tag, 20, 0))
    det = rows[rows['level'] != 'fewshot']
    assert set(det.loc[det['dataset'] == 'deepdrid', 'bootstrap_unit']) == {'image (ids not matched in split file)'}
    assert set(det.loc[det['dataset'] == 'idrid', 'bootstrap_unit']) == {'image'}


def test_adaptation_tool_writes_bootstrap_unit_with_group_file():
    rng = np.random.default_rng(9)
    data = {t: {} for t in ('g_ce', 'g_ord')}
    for ds, n_pat in (('deepdrid', 30), ('messidor', 50)):
        for split in ('dev', 'test'):
            n = 4 * n_pat if ds == 'deepdrid' else n_pat
            s = np.repeat(rng.uniform(0, 4, n // 4), 4) if ds == 'deepdrid' else rng.uniform(0, 4, n)
            y = apply_cutpoints(s, [0.6, 1.6, 2.4, 3.4])
            ids = np.array([f'{ds}_{split}_p{i // 4:03d}_{i % 4}' for i in range(n)])
            for t, noise in (('g_ce', 0.6), ('g_ord', 0.4)):
                data[t][(ds, split)] = (ids, y, _probs_from_score(s + rng.normal(0, noise, n), DEFAULT_CUTS))
            if ds == 'deepdrid' and split == 'test':
                _write_split(ds, ids, [i.rsplit('_', 1)[0] for i in ids], ext='.png')
    for t in data:
        _write_run('phase10_ms_ddr', t, ['aptos', 'eyepacs', 'ddr', 'idrid'], data[t])
    r = subprocess.run([sys.executable, TOOL, '--groups', 'g_ce,g_ord', '--ns', '10', '--draws', '4', '--B', '30'],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    cmp = pd.read_csv('results/master/adaptation_compare.csv')
    assert list(cmp.columns)[-1] == 'bootstrap_unit'
    det = cmp[~cmp['level'].isin(['fewshot', 'fewshot_shrink'])]
    assert set(det.loc[det['dataset'] == 'deepdrid', 'bootstrap_unit']) == \
        {'cluster (group column; 30 groups / 120 images)'}
    assert set(det.loc[det['dataset'] == 'messidor', 'bootstrap_unit']) == {'image (no split file)'}
    assert set(cmp.loc[cmp['level'].isin(['fewshot', 'fewshot_shrink']), 'bootstrap_unit']) == {'draws'}


def test_local_group_rule_matches_recalibration_suite(monkeypatch):
    """The local copy (used only if tools.recalibration_suite cannot be imported) gives the same groups."""
    import tools.adaptation_curve as ac
    ids = np.array([f'dd_p{g}_{k}' for g in range(5) for k in range(4)])
    _write_split('deepdrid', ids, [i.rsplit('_', 1)[0] for i in ids])
    g_suite, u_suite = ac._resolve_test_groups('deepdrid', ids)
    monkeypatch.setitem(sys.modules, 'tools.recalibration_suite', None)       # import now fails
    g_local, u_local = ac._resolve_test_groups('deepdrid', ids)
    assert u_suite == u_local == 'cluster (group column; 5 groups / 20 images)'
    assert np.array_equal(g_suite, g_local)
