"""End-to-end pipeline on a miniature synthetic project with all seven datasets.

Covers: make_splits -> duplicate audit -> make_external_splits -> experiment (softmax, legacy joint,
coral_src multi-source, seed replicate) -> every metric on every set -> evaluate (re-use + new sets)
-> inference / ensemble / conformal -> evaluate_all -> phase_select (both endpoints) -> master_table
-> seed_summary -> report (tables + figures) -> tools (cut-point diagnostic, leakage check) -> XAI.
"""

import argparse
import json
import os
import shutil

import numpy as np
import pandas as pd
import pytest

from tests import synth

SETS = ['aptos_test', 'messidor_dev', 'messidor_test', 'ddr_dev', 'ddr_test', 'deepdrid_dev', 'deepdrid_test',
        'idrid_dev', 'idrid_test']


def _cfg(mode, tag=None, phase=None, inherit=None, base=None, members=None, sets=()):
    from src.utils.config import build_cfg
    ns = argparse.Namespace(mode=mode, tag=tag, phase=phase, inherit=inherit, base=base, members=members,
                            kind=None, method=None, set=list(sets))
    return build_cfg(ns)


@pytest.fixture(scope='module')
def project(tmp_path_factory):
    """Build the synthetic project once and run every training mode (module scope keeps the suite fast)."""
    root = tmp_path_factory.mktemp('proj')
    cwd = os.getcwd()
    os.chdir(root)
    try:
        repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        shutil.copytree(os.path.join(repo, 'configs'), 'configs')
        cfg = synth.build(str(root))
        cfg['dedup_exclude_max'] = -1          # synthetic discs look alike; exclusion logic is tested separately
        synth.install_config(cfg)

        from src.data.splits import make_splits
        from src.data.external import make_external_splits
        from tools.dedup_audit import run_dedup_audit, run_dedup_external
        from src.evaluation.experiment import (run_experiment, run_inference_experiment, run_ensemble_experiment,
                                               run_calibrate_conformal)
        make_splits(cfg)
        run_dedup_audit(cfg)
        run_dedup_external(cfg)
        make_external_splits(_cfg('make_external_splits'))

        run_experiment(_cfg('experiment', 'p_ref', 'phase3a_pretrain', sets=['description=reference']))
        run_experiment(_cfg('experiment', 'p_ref_s1', 'phase3c_seeds', inherit='p_ref', sets=['seed=1']))
        run_experiment(_cfg('experiment', 'p_joint', 'phase3b_joint', inherit='p_ref', sets=['joint_eyepacs=true']))
        run_experiment(_cfg('experiment', 'p_src', 'phase3b_joint', inherit='p_ref',
                            sets=['head_type=coral_src', 'extra_sources=[ddr,deepdrid]']))
        run_inference_experiment(_cfg('inference_experiment', 'p_tta', 'phase8_inference', base='p_ref',
                                      sets=['tta=flip']))
        run_ensemble_experiment(_cfg('ensemble_experiment', 'p_ens', 'phase5_architecture', members='p_ref,p_joint'))
        run_calibrate_conformal(_cfg('calibrate_conformal', 'p_conf', 'phase8_inference', base='p_ref'))
        yield root
    finally:
        os.chdir(cwd)


@pytest.fixture
def at_project(project, monkeypatch):
    monkeypatch.chdir(project)
    return project


def test_every_set_has_every_metric(at_project):
    from src.evaluation.evaluate import SUMMARY_METRICS
    for tag, phase in (('p_ref', 'phase3a_pretrain'), ('p_joint', 'phase3b_joint'), ('p_src', 'phase3b_joint'),
                       ('p_tta', 'phase8_inference'), ('p_ens', 'phase5_architecture')):
        d = os.path.join('results', phase, tag)
        long = pd.read_csv(os.path.join(d, 'metrics_long.csv'))
        assert set(long['set']) == set(SETS) | {'oof'}, tag
        for s in SETS + ['oof']:
            got = set(long[(long['set'] == s) & (long['task'] != 'devcal')]['metric'])
            assert set(SUMMARY_METRICS) <= got, (tag, s, set(SUMMARY_METRICS) - got)
        summ = pd.read_csv(os.path.join(d, 'summary.csv')).iloc[0]
        for s in SETS:
            for m in ('acc', 'qwk', 'prec_macro', 'rec_macro', 'f1_macro', 'bin_acc', 'bin_prec', 'bin_sens',
                      'bin_spec', 'bin_f1', 'bin_auroc'):
                assert f'{s}_{m}' in summ.index, (tag, s, m)
            assert os.path.isfile(os.path.join(d, f'predictions_{s}.npz'))
            assert os.path.isfile(os.path.join(d, f'confusion_{s}.csv'))
        for legacy in ('predictions_test.npz', 'predictions_external_dev.npz', 'predictions_external_test.npz',
                       'confusion_test.csv', 'confusion_external.csv', 'oof.npz', 'per_class.csv', 'binary.csv',
                       'calibration.csv', 'cutpoints.csv', 'fewshot.csv', 'per_fold.csv', 'config.yaml'):
            assert os.path.isfile(os.path.join(d, legacy)), (tag, legacy)
        # legacy columns still mean what they meant: test = APTOS test, ext = Messidor-2 TEST
        assert summ['test_qwk'] == summ['aptos_test_qwk']
        assert summ['ext_qwk'] == summ['messidor_test_qwk']
        assert pd.read_csv(os.path.join(d, 'cutpoints.csv'))['dataset'].tolist() == ['messidor', 'ddr', 'deepdrid',
                                                                                     'idrid']


def test_multisource_coral_src_outputs(at_project):
    d = os.path.join('results', 'phase3b_joint', 'p_src')
    summ = pd.read_csv(os.path.join(d, 'summary.csv')).iloc[0]
    assert summ['train_sources'] == 'aptos,ddr,deepdrid'
    assert summ['heldout_sets'] == 'messidor,idrid'
    assert summ['ext_mean_test_sets'] == 'messidor_test,idrid_test'
    long = pd.read_csv(os.path.join(d, 'metrics_long.csv'))
    flags = long.drop_duplicates('set').set_index('set')['in_source']
    assert bool(flags['ddr_test']) and bool(flags['deepdrid_dev']) and not bool(flags['messidor_test'])
    thr = json.load(open(os.path.join(d, 'source_thresholds_fold0.json')))
    assert set(thr) == {'shared', 'aptos', 'ddr', 'deepdrid'}
    for v in thr.values():
        assert all(a < b for a, b in zip(v, v[1:])), 'cut-points must be ordered'
    P = np.load(os.path.join(d, 'predictions_messidor_test.npz'))['y_prob']
    assert np.allclose(P.sum(1), 1.0, atol=1e-5)


def test_evaluate_reuses_and_fills_missing_sets(at_project):
    from src.evaluation.evaluate import run_evaluate
    d = os.path.join('results', 'phase3a_pretrain', 'p_ref')
    s = pd.read_csv(os.path.join(d, 'summary.csv'))
    s['decision'] = 'REFERENCE'
    s.to_csv(os.path.join(d, 'summary.csv'), index=False)
    before = np.load(os.path.join(d, 'predictions_messidor_test.npz'))['y_prob']
    os.remove(os.path.join(d, 'predictions_ddr_test.npz'))
    run_evaluate(_cfg('evaluate', 'p_ref'))
    assert os.path.isfile(os.path.join(d, 'predictions_ddr_test.npz'))
    after = np.load(os.path.join(d, 'predictions_messidor_test.npz'))['y_prob']
    assert np.allclose(before, after), 're-evaluation must reproduce the stored predictions'
    assert pd.read_csv(os.path.join(d, 'summary.csv')).iloc[0]['decision'] == 'REFERENCE'


def test_legacy_prediction_files_are_read_exactly(at_project, tmp_path):
    from src.evaluation.evaluate import load_saved_set
    from src.evaluation.metrics import apply_temperature
    rng = np.random.default_rng(0)
    folds = rng.dirichlet(np.ones(5), size=(2, 6))
    raw = folds.mean(0)
    scaled = apply_temperature(raw, 1.7)
    np.savez(tmp_path / 'predictions_external_test.npz', ids=np.array([f'i{k}' for k in range(6)]),
             y_true=np.arange(6) % 5, y_prob=scaled, y_prob_folds=folds, y=np.arange(6) % 5, P=scaled)
    rec = load_saved_set(str(tmp_path), 'messidor_test')
    assert np.allclose(rec['P_raw'], raw), 'legacy files hold scaled probabilities; raw = mean over folds'


def test_evaluate_all_and_selection(at_project):
    from src.evaluation.evaluate import run_evaluate_all
    from src.evaluation.tables import phase_select, master_table, seed_summary
    rows = run_evaluate_all(_cfg('evaluate_all'))
    status = {r['tag']: r['status'] for r in rows}
    assert all(v == 'ok' for v in status.values()), status
    assert set(status) >= {'p_ref', 'p_ref_s1', 'p_joint', 'p_src', 'p_tta', 'p_ens', 'p_conf'}

    win = phase_select('phase3b_joint', 'p_ref', cfg=_cfg('phase_select', sets=['select_endpoint=amended']))
    assert win in ('p_ref', 'p_joint', 'p_src')
    pt = pd.read_csv(os.path.join('results', 'phase3b_joint', 'phase_table.csv'))
    for col in ('decision_original', 'decision_amended', 'd_dev_mean_qwk', 'd_dev_mean_qwk_ci_lo', 'oof_equivalent',
                'dev_sets_used', 'p_holm'):
        assert col in pt.columns
    # p_src trains on DDR and DeepDRiD, so its amended endpoint may only use the held-out DEV sets
    assert pt.set_index('experiment').loc['p_src', 'dev_sets_used'] == 'messidor,idrid'
    winner_before = open(os.path.join('results', 'phase3b_joint', 'WINNER.txt')).read()
    phase_select('phase3b_joint', 'p_ref', cfg=_cfg('phase_select', sets=['select_endpoint=original',
                                                                         'select_report_only=true']))
    assert os.path.isfile(os.path.join('results', 'phase3b_joint', 'phase_table_original.csv'))
    assert open(os.path.join('results', 'phase3b_joint', 'WINNER.txt')).read() == winner_before

    master_table()
    m = pd.read_csv(os.path.join('results', 'master', 'master_table.csv'))
    assert {'p_ref', 'p_ref_s1', 'p_src', 'p_ens'} <= set(m['experiment'])
    assert os.path.isfile(os.path.join('results', 'master', 'master_long.csv'))

    seed_summary(reference='p_ref')
    sv = pd.read_csv(os.path.join('results', 'master', 'seed_variance.csv'))
    g = sv[(sv['group'] == 'p_ref') & (sv['metric'] == 'oof_qwk')].iloc[0]
    assert g['n_seeds'] == 2 and set(g['members'].split(',')) == {'p_ref', 'p_ref_s1'}
    assert 'p_tta' not in ','.join(sv['members']) and 'p_ens' not in ','.join(sv['members'])


def test_report_uses_real_data_only(at_project):
    from src.evaluation.tables import build_paper_tables, master_table
    from src.evaluation.figures import build_paper_figures
    cfg = _cfg('report', sets=['report_final=p_src', 'report_baseline=p_ref'])
    master_table()
    build_paper_tables(cfg=cfg)
    done, skipped = build_paper_figures(cfg=cfg)
    t3 = pd.read_csv(os.path.join('results', 'tables', 'table03_main_5class.csv'))
    assert set(t3['set']) == set(SETS) | {'oof'}
    assert {'Accuracy', 'QWK', 'Precision', 'Recall', 'F1'} <= set(t3.columns)
    t4 = pd.read_csv(os.path.join('results', 'tables', 'table04_main_binary.csv'))
    assert {'Accuracy', 'Precision', 'Sensitivity', 'Specificity', 'F1', 'AUROC'} <= set(t4.columns)
    for f in os.listdir(os.path.join('results', 'tables')):
        txt = open(os.path.join('results', 'tables', f)).read()
        assert 'TODO' not in txt and 'MODAL-Net (Final)' not in txt
    assert 'fig03_confusion_matrices' in done and 'fig10_fewshot_cutpoints' in done
    assert 'fig11_source_cutpoints' in done, skipped


def test_tools_on_saved_predictions(at_project):
    from tools.cutpoint_diagnostic import diagnose
    from tools.leakage_check import check
    from src.data.external import load_audit_pairs
    rows = diagnose('p_ref', B=50)
    assert [r['dataset'] for r in rows] == ['messidor', 'ddr', 'deepdrid', 'idrid']
    out, _ = check('p_joint', 4, load_audit_pairs())
    assert {r['set'] for r in out} == set(SETS)


def test_xai_grid_and_expert_mask_localisation(at_project):
    from src.xai.grid import run_xai_experiment
    from src.xai.localization import run_xai_localization
    run_xai_experiment(_cfg('xai', 'p_ref', sets=['xai_split=ddr_test', 'xai_top_k=1', 'xai_methods=[gradcam]']))
    assert os.path.isfile(os.path.join('results', 'figures', 'xai_grid.png'))
    res = run_xai_localization(_cfg('xai_localization', 'p_src', sets=['xai_loc_dilate=2', 'xai_loc_method=eigencam']))
    assert set(res['dataset']) == {'ddr', 'idrid'}
    assert {'energy_in_mask', 'enrichment', 'pointing_game'} <= set(res['metric'])
