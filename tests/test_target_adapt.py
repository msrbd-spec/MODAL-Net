"""Target-site adaptation baselines: --mode target_adapt (bn / tent), --mode extract_features and
tools/fewshot_head.py, on a miniature synthetic project (resnet18 backbone: has BatchNorm)."""

import argparse
import copy
import hashlib
import os
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from tests import synth

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(REPO, 'tools', 'fewshot_head.py')
PHASE = 'phase3a_pretrain'
HELDOUT = ['messidor', 'ddr', 'deepdrid', 'idrid']
NPZ_KEYS = {'ids', 'y_true', 'y_pred', 'y_prob', 'y_prob_raw', 'y_prob_folds', 'y', 'P', 'P_per_model'}


def _cfg(mode, tag=None, phase=None, sets=()):
    from src.utils.config import build_cfg
    ns = argparse.Namespace(mode=mode, tag=tag, phase=phase, inherit=None, base=None, members=None,
                            kind=None, method=None, set=list(sets))
    return build_cfg(ns)


def _hashes(paths):
    out = {}
    for root in paths:
        for dp, dns, fns in os.walk(root):
            dns[:] = [d for d in dns if d not in ('target_adapt', 'features')]
            for fn in fns:
                p = os.path.join(dp, fn)
                out[p] = hashlib.sha256(open(p, 'rb').read()).hexdigest()
    return out


@pytest.fixture(scope='module')
def project(tmp_path_factory):
    """Synthetic project, one tiny trained CE run (p_ref), one fabricated coral_src run (p_csrc), and the
    outputs of target_adapt bn / tent and extract_features on p_ref (with file hashes before / after)."""
    root = tmp_path_factory.mktemp('ta_proj')
    cwd = os.getcwd()
    os.chdir(root)
    try:
        shutil.copytree(os.path.join(REPO, 'configs'), 'configs')
        cfg = synth.build(str(root))
        cfg['dedup_exclude_max'] = -1
        synth.install_config(cfg)
        from src.data.splits import make_splits
        from src.data.external import make_external_splits
        from tools.dedup_audit import run_dedup_audit, run_dedup_external
        from src.evaluation.experiment import run_experiment
        make_splits(cfg)
        run_dedup_audit(cfg)
        run_dedup_external(cfg)
        make_external_splits(_cfg('make_external_splits'))
        run_experiment(_cfg('experiment', 'p_ref', PHASE))

        # fabricated coral_src run: random-init fold checkpoints, snapshot, temperature, results folder
        from src.models.gradenet import GradeNet
        snap = yaml.safe_load(open(os.path.join('checkpoints', 'p_ref', 'config_snapshot.yaml')))
        snap.update({'tag': 'p_csrc', 'head_type': 'coral_src', 'external_sets': ['idrid']})
        os.makedirs(os.path.join('checkpoints', 'p_csrc'))
        os.makedirs(os.path.join('results', PHASE, 'p_csrc'))
        yaml.safe_dump(snap, open(os.path.join('checkpoints', 'p_csrc', 'config_snapshot.yaml'), 'w'))
        shutil.copy(os.path.join('checkpoints', 'p_ref', 'thresholds.json'), os.path.join('checkpoints', 'p_csrc'))
        for k in range(2):
            torch.manual_seed(k)
            torch.save(GradeNet(dict(snap, pretrained=False)).state_dict(),
                       os.path.join('checkpoints', 'p_csrc', f'fold{k}.pth'))

        from src.evaluation.target_adapt import run_extract_features, run_target_adapt
        watched = [os.path.join('checkpoints', 'p_ref'), os.path.join('results', PHASE, 'p_ref')]
        before = _hashes(watched)
        run_target_adapt(_cfg('target_adapt', 'p_ref'))
        run_target_adapt(_cfg('target_adapt', 'p_ref', sets=['target_adapt_method=tent', 'tent_lr=0.01',
                                                             'tent_epochs=2']))
        run_extract_features(_cfg('extract_features', 'p_ref'))
        after = _hashes(watched)
        yield {'root': root, 'before': before, 'after': after}
    finally:
        os.chdir(cwd)


@pytest.fixture
def at_project(project, monkeypatch):
    monkeypatch.chdir(project['root'])
    return project


def _exp(tag='p_ref'):
    return os.path.join('results', PHASE, tag)


def _fold_model_and_loader(tag='p_ref', k=0, sets=(), set_name='messidor_dev'):
    from src.evaluation.target_adapt import _loader, load_fold_model, run_context
    cfg = _cfg('target_adapt', tag, sets=sets)
    ctx = run_context(cfg)
    return cfg, load_fold_model(cfg, ctx['owner'], k), _loader(cfg, set_name)


def _split_state(model):
    params = {k: v.detach().clone() for k, v in model.named_parameters()}
    buffers = {k: v.detach().clone() for k, v in model.named_buffers()}
    return params, buffers


# ---------------------------------------------------------------------------
# (1) bn mode
# ---------------------------------------------------------------------------

def test_bn_changes_running_stats_only_and_is_a_cumulative_average(at_project):
    from src.evaluation.target_adapt import adapt_bn, bn_layers, _loader
    cfg, model, _ = _fold_model_and_loader(sets=['batch_size=64'])
    one_batch = _loader(cfg, 'messidor_dev')                       # every DEV image in one batch
    p0, b0 = _split_state(model)
    bn1 = model.backbone.bn1                                       # resnet18 stem: conv1 -> bn1
    n = adapt_bn(model, one_batch, 'cpu')
    assert n == len(bn_layers(model)) == 20
    p1, b1 = _split_state(model)
    assert all(torch.equal(p0[k], p1[k]) for k in p0), 'bn adaptation must not change any parameter'
    changed = [k for k in b0 if k.endswith('running_mean') and not torch.equal(b0[k], b1[k])]
    assert len(changed) == 20
    with torch.no_grad():
        x = model.backbone.conv1(next(iter(one_batch))['x'])       # input of bn1 for every DEV image
    assert torch.allclose(bn1.running_mean, x.mean((0, 2, 3)), atol=1e-5)
    assert torch.allclose(bn1.running_var, x.var((0, 2, 3), unbiased=True), atol=1e-4)
    assert not model.training and all(not m.training for m in model.modules())
    assert all(m.momentum == 0.1 for m in bn_layers(model)), 'original momentum restored'


def test_bn_mode_outputs(at_project):
    from src.evaluation.metrics import apply_temperature
    out = os.path.join(_exp(), 'target_adapt', 'bn')
    T = float(yaml.safe_load(open(os.path.join('checkpoints', 'p_ref', 'thresholds.json')))['temperature'])
    for d in HELDOUT:
        for split in ('dev', 'test'):
            new = np.load(os.path.join(out, f'predictions_{d}_{split}.npz'))
            old = np.load(os.path.join(_exp(), f'predictions_{d}_{split}.npz'))
            assert set(new.files) == set(old.files) == NPZ_KEYS
            assert list(new['ids']) == list(old['ids']), 'same image order as the normal prediction file'
            assert np.array_equal(new['y_true'], old['y_true'])
            assert new['y_prob_folds'].shape == old['y_prob_folds'].shape == (2, len(old['ids']), 5)
            assert np.allclose(new['y_prob_raw'], new['y_prob_folds'].mean(0))
            assert np.allclose(new['y_prob'], apply_temperature(new['y_prob_raw'], T))
            assert np.array_equal(new['y_pred'], new['y_prob'].argmax(1))
            assert not np.allclose(new['y_prob_raw'], old['y_prob_raw']), 'BN statistics changed the outputs'
    s = pd.read_csv(os.path.join(out, 'summary.csv'))
    assert list(s['set']) == [f'{d}_{sp}' for d in HELDOUT for sp in ('dev', 'test')]
    for c in ('qwk', 'acc', 'f1_macro', 'mild_recall', 'bin_sens', 'bin_spec', 'pred_nodr_share', 'bin_auroc',
              'orig_qwk', 'n_bn_layers', 'n_adapted_params'):
        assert c in s.columns, c
    assert (s['n_bn_layers'] == 20).all() and (s['n_adapted_params'] == 0).all()
    log = pd.read_csv(os.path.join(out, 'adapt_log.csv'))
    assert len(log) == len(HELDOUT) * 2


def test_heldout_only(at_project):
    """A training source is never adapted to: a run trained on DDR has no ddr target-adaptation outputs."""
    from src.evaluation.evaluate import heldout_datasets
    assert heldout_datasets(dict(_cfg('target_adapt', 'p_ref'), extra_sources=['ddr'])) == \
        ['messidor', 'deepdrid', 'idrid']
    assert heldout_datasets(_cfg('target_adapt', 'p_csrc')) == ['idrid']


# ---------------------------------------------------------------------------
# (2) tent
# ---------------------------------------------------------------------------

def _tent_check(tag, sets=(), epochs=3, set_name='messidor_dev'):
    from src.evaluation.target_adapt import (_loader, adapt_bn, adapt_tent, mean_entropy, norm_affine_params)
    cfg, model, stats = _fold_model_and_loader(tag, sets=sets, set_name=set_name)
    head = cfg.get('head_type', 'softmax')
    adapt_bn(model, stats, 'cpu', head)
    p0, _ = _split_state(model)
    e0 = mean_entropy(model, stats, 'cpu', head)
    info = adapt_tent(model, _loader(cfg, set_name, shuffle=True, seed=0, drop_last=True), 'cpu', head,
                      lr=0.01, epochs=epochs, stats_loader=stats)
    p1, _ = _split_state(model)
    e1 = mean_entropy(model, stats, 'cpu', head)
    norm = {k for k, _ in norm_affine_params(model)}
    assert norm and info['n_params'] == sum(p0[k].numel() for k in norm)
    for k in p0:
        if k not in norm:
            assert torch.equal(p0[k], p1[k]), f'{k} is not a normalisation affine parameter but changed'
    assert any(not torch.equal(p0[k], p1[k]) for k in norm)
    assert e1 < e0, (e0, e1)
    assert all(p.requires_grad for p in model.parameters())
    return info


def test_tent_changes_only_norm_affine_and_reduces_entropy(at_project):
    info = _tent_check('p_ref', sets=['batch_size=4'])
    assert info['steps'] == 3 * (10 // 4)                       # 10 DEV images, incomplete batch dropped


def test_tent_ordinal_coral_src_head(at_project):
    _tent_check('p_csrc', sets=['batch_size=8'], epochs=2, set_name='idrid_dev')
    from src.evaluation.target_adapt import run_target_adapt
    out = run_target_adapt(_cfg('target_adapt', 'p_csrc', sets=['target_adapt_method=tent', 'batch_size=8']))
    P = np.load(os.path.join(out, 'predictions_idrid_test.npz'))['y_prob_folds']
    assert P.shape[0] == 2 and np.allclose(P.sum(-1), 1.0, atol=1e-5)
    assert sorted(os.listdir(out)) == sorted(['predictions_idrid_dev.npz', 'predictions_idrid_test.npz',
                                              'summary.csv', 'adapt_log.csv', 'config.yaml'])


def test_tent_mode_outputs(at_project):
    out = os.path.join(_exp(), 'target_adapt', 'tent')
    log = pd.read_csv(os.path.join(out, 'adapt_log.csv'))
    assert (log['n_adapted_params'] > 0).all() and (log['tent_steps'] > 0).all()
    assert (log['dev_entropy_final'] < log['dev_entropy_orig']).all()
    bn = np.load(os.path.join(_exp(), 'target_adapt', 'bn', 'predictions_idrid_test.npz'))['y_prob_raw']
    tn = np.load(os.path.join(out, 'predictions_idrid_test.npz'))['y_prob_raw']
    assert not np.allclose(bn, tn)
    assert yaml.safe_load(open(os.path.join(out, 'config.yaml')))['target_adapt_method'] == 'tent'


# ---------------------------------------------------------------------------
# (3) originals untouched
# ---------------------------------------------------------------------------

def test_checkpoints_and_normal_results_unchanged(at_project):
    before, after = at_project['before'], at_project['after']
    assert any(p.endswith('fold0.pth') for p in before) and any(p.endswith('predictions_ddr_test.npz') for p in before)
    assert before == after


# ---------------------------------------------------------------------------
# (4) extract_features
# ---------------------------------------------------------------------------

def test_features_and_head_reproduce_normal_predictions(at_project):
    from src.evaluation.metrics import apply_temperature
    from src.evaluation.target_adapt import head_probs
    fd = os.path.join(_exp(), 'features')
    h = np.load(os.path.join(fd, 'head.npz'))
    assert h['W'].dtype == np.float32 and h['W'].shape == (2, 5, 512) and h['b'].shape == (2, 5)
    assert list(h['heldout']) == HELDOUT
    for d in HELDOUT:
        for split in ('dev', 'test'):
            f = np.load(os.path.join(fd, f'{d}_{split}.npz'))
            old = np.load(os.path.join(_exp(), f'predictions_{d}_{split}.npz'))
            assert list(f['ids']) == list(old['ids']) and np.array_equal(f['y_true'], old['y_true'])
            assert f['feats'].dtype == np.float16 and f['feats'].shape == (2, len(old['ids']), 512)
            P = head_probs(f['feats'], h['W'], h['b'])
            assert np.abs(P - old['y_prob_folds']).max() < 1e-3
            assert np.abs(apply_temperature(P.mean(0), float(h['T'])) - old['y_prob']).max() < 1e-3
    chk = pd.read_csv(os.path.join(fd, 'reproduction_check.csv'))
    assert chk['checked_vs_saved'].all() and (chk['max_abs_diff_saved_y_prob'] < 1e-5).all()


def test_extract_features_refuses_ordinal_heads(at_project):
    from src.evaluation.target_adapt import run_extract_features
    with pytest.raises(ValueError, match='CE'):
        run_extract_features(_cfg('extract_features', 'p_csrc'))


def test_cli_modes_and_config_keys(at_project):
    from src.utils.config import parse_set
    assert parse_set(['target_adapt_method=tent', 'tent_lr=0.001', 'tent_epochs=2']) == \
        {'target_adapt_method': 'tent', 'tent_lr': 0.001, 'tent_epochs': 2}
    with pytest.raises(AssertionError):
        _cfg('target_adapt', 'p_ref', sets=['target_adapt_method=adabn'])
    cfg = _cfg('target_adapt', 'p_ref')
    assert cfg['backbone'] == 'resnet18' and cfg['tag'] == 'p_ref'        # the run's own snapshot is loaded
    main_src = open(os.path.join(REPO, 'main.py')).read()
    assert "'target_adapt', 'extract_features'" in main_src


# ---------------------------------------------------------------------------
# (5) / (6) tools/fewshot_head.py
# ---------------------------------------------------------------------------

def test_fewshot_head_schema_and_draws(at_project):
    from tools.fewshot_head import COLUMNS
    out = os.path.join('results', 'master', 'recal', 'fewshot_head_draws.csv')
    r = subprocess.run([sys.executable, TOOL, '--tags', 'p_ref', '--ns', '3,5,12', '--draws', '4'],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    df = pd.read_csv(out)
    assert list(df.columns) == COLUMNS == ['tag', 'dataset', 'method', 'info', 'n', 'draw', 'n_test', 'qwk', 'acc',
                                           'f1_macro', 'mild_recall', 'bin_sens', 'bin_spec', 'pred_nodr_share']
    assert set(df['method']) == {'fewshot_head_ft'} and set(df['info']) == {'n_labels'}
    n_dev = {d: len(pd.read_csv(os.path.join('splits', 'external', f'{d}_dev.csv'))) for d in HELDOUT}
    n_test = {d: len(pd.read_csv(os.path.join('splits', 'external', f'{d}_test.csv'))) for d in HELDOUT}
    want = {(d, n, j) for d in HELDOUT for n in (3, 5, 12) if n < n_dev[d] for j in range(4)}
    got = list(zip(df['dataset'], df['n'], df['draw']))
    assert len(got) == len(set(got)) and set(got) == want
    assert all(df['n_test'] == df['dataset'].map(n_test))
    lam = pd.read_csv(os.path.join('results', 'master', 'recal', 'fewshot_head_lambda.csv'))
    assert set(lam['lam']) <= {0.01, 0.1, 1.0, 10.0, 100.0}
    for _, row in lam.iterrows():
        assert row['dataset'] not in str(row['chosen_from']).split(',')


def test_fewshot_head_uses_adaptation_curve_draws_and_never_test_labels(at_project):
    from tools.adaptation_curve import _draw_indices, _load
    from tools.fewshot_head import analyse_tag, fit_folds, load_tag
    loaded = load_tag('p_ref')
    rows, lam_rows, fits = analyse_tag('p_ref', [3, 5], 3, 0, keep_fits=True, loaded=loaded)
    # draws = adaptation_curve draws on the DEV split sorted by id
    ids_dev, y_dev, F_dev = loaded['data']['idrid']['dev']
    assert np.array_equal(ids_dev, _load(_exp(), 'idrid', 'dev')[0])
    idx = _draw_indices(len(y_dev), 5, 3, 'idrid', 0)[1]
    lam = [r['lam'] for r in lam_rows if r['dataset'] == 'idrid' and r['n'] == 5][0]
    h = loaded['head']
    manual = fit_folds(F_dev[:, idx], y_dev[idx], h['W'], h['b'], h['T'], lam)
    assert all(np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]) for a, b in zip(manual, fits[('idrid', 5, 1)]))
    # permuting / replacing everything of TEST leaves every fitted head and every chosen lambda identical
    leak = copy.deepcopy(loaded)
    rng = np.random.default_rng(0)
    for d in leak['data'].values():
        ids, y, F = d['test']
        d['test'] = (ids, rng.permutation(y), F)
    rows2, lam2, fits2 = analyse_tag('p_ref', [3, 5], 3, 0, keep_fits=True, loaded=leak)
    assert fits.keys() == fits2.keys()
    for key in fits:
        for (W1, b1), (W2, b2) in zip(fits[key], fits2[key]):
            assert np.array_equal(W1, W2) and np.array_equal(b1, b2), key
    assert pd.DataFrame(lam_rows).equals(pd.DataFrame(lam2))
    assert [r['draw'] for r in rows] == [r['draw'] for r in rows2]


def test_fewshot_head_lambda_uses_other_sites_only(at_project):
    from tools.fewshot_head import analyse_tag, load_tag
    loaded = load_tag('p_ref')
    _, lam_rows, _ = analyse_tag('p_ref', [3], 3, 0, loaded=loaded)
    # destroy the target site's whole DEV split (labels and features): its own lambda cannot change
    for target in ('ddr', 'idrid'):
        alt = copy.deepcopy(loaded)
        ids, y, F = alt['data'][target]['dev']
        rng = np.random.default_rng(1)
        alt['data'][target]['dev'] = (ids, rng.integers(0, 5, len(y)), rng.normal(size=F.shape))
        _, lam_alt, _ = analyse_tag('p_ref', [3], 3, 0, loaded=alt)
        a = [r for r in lam_rows if r['dataset'] == target]
        b = [r for r in lam_alt if r['dataset'] == target]
        assert len(a) == 1 and pd.DataFrame(a).equals(pd.DataFrame(b)), target
        assert target not in a[0]['chosen_from'].split(',')
    # a single held-out site has no other site: the a-priori default
    single = copy.deepcopy(loaded)
    single['data'] = {'idrid': single['data']['idrid']}
    _, lam_single, _ = analyse_tag('p_ref', [3], 3, 0, loaded=single)
    assert lam_single[0]['lam'] == 1.0 and lam_single[0]['chosen_from'] == ''


# ---------------------------------------------------------------------------
# (7) audit regressions
# ---------------------------------------------------------------------------

def test_adapt_bn_is_image_weighted_with_a_small_last_batch():
    """Regression: momentum=None weighted every batch equally, so a small last batch (here 3 outlying images
    out of 35) dominated the running statistics. They must be the image-weighted statistics."""
    from src.evaluation.target_adapt import adapt_bn
    torch.manual_seed(0)
    x = torch.cat([torch.randn(32, 3, 4, 4), 5.0 + torch.randn(3, 3, 4, 4)])
    loader = [{'x': x[i:i + 16]} for i in range(0, len(x), 16)]                 # batches of 16, 16, 3
    model = torch.nn.Sequential(torch.nn.BatchNorm2d(3))
    adapt_bn(model, loader, 'cpu')
    bn = model[0]
    assert torch.allclose(bn.running_mean, x.mean((0, 2, 3)), atol=1e-5)
    within = sum(len(b['x']) * b['x'].transpose(0, 1).reshape(3, -1).var(1, unbiased=True) for b in loader)
    assert torch.allclose(bn.running_var, within / len(x), atol=1e-5)
    assert bn.momentum == 0.1 and not bn.training


def test_bn_stats_loader_is_shuffled_complete_and_deterministic(at_project):
    """Regression: the BatchNorm statistics pass used the split-file order (consecutive rows = one patient /
    exam), which under-estimates the variance; it must see every DEV image once in a fixed random order."""
    from src.evaluation.target_adapt import stats_loader_for
    cfg = _cfg('target_adapt', 'p_ref', sets=['batch_size=3'])
    order = [[str(i) for b in stats_loader_for(cfg, 'messidor_dev') for i in b['id']] for _ in range(2)]
    file_order = [str(i) for i in pd.read_csv(os.path.join('splits', 'external', 'messidor_dev.csv'))['id_code']]
    assert order[0] == order[1], 'deterministic'
    assert sorted(order[0]) == sorted(file_order) and len(order[0]) == len(file_order), 'every image once'
    assert order[0] != file_order, 'not the split-file order'


def test_fewshot_head_fit_uses_T_1_when_temperature_scaling_is_off():
    """Regression: the head was fitted at the OOF temperature even for runs with temperature_scaling off."""
    from tools.fewshot_head import analyse_tag, fit_folds, fit_temperature_of
    from tools.adaptation_curve import _draw_indices
    rng = np.random.default_rng(0)
    W0, b0 = rng.normal(size=(2, 5, 6)), rng.normal(size=(2, 5))
    dev = (np.array([f'd{i:02d}' for i in range(30)]), rng.integers(0, 5, 30), rng.normal(size=(2, 30, 6)))
    test = (np.array([f't{i:02d}' for i in range(20)]), rng.integers(0, 5, 20), rng.normal(size=(2, 20, 6)))
    for use_T, T_fit in ((True, 2.5), (False, 1.0)):
        head = {'W': W0, 'b': b0, 'T': 2.5, 'use_T': use_T, 'heldout': ['idrid']}
        assert fit_temperature_of(head) == T_fit
        _, lam_rows, fits = analyse_tag('x', [10], 2, 0, keep_fits=True,
                                        loaded={'head': head, 'data': {'idrid': {'dev': dev, 'test': test}}})
        idx = _draw_indices(30, 10, 2, 'idrid', 0)[0]
        want = fit_folds(dev[2][:, idx], dev[1][idx], W0, b0, T_fit, lam_rows[0]['lam'])
        other = fit_folds(dev[2][:, idx], dev[1][idx], W0, b0, 1.0 if use_T else 2.5, lam_rows[0]['lam'])
        assert all(np.allclose(a[0], b[0]) for a, b in zip(want, fits[('idrid', 10, 0)]))
        assert not all(np.allclose(a[0], b[0]) for a, b in zip(other, fits[('idrid', 10, 0)]))


def test_fewshot_head_and_recalibration_suite_share_the_draw_schema():
    from tools.fewshot_head import COLUMNS
    from tools.recalibration_suite import DRAW_COLS, INFO_LEVELS
    from tools.fewshot_head import INFO
    assert COLUMNS == DRAW_COLS and INFO in INFO_LEVELS
