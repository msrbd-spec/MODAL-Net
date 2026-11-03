"""External datasets (frames, xlsx, frozen splits, duplicate exclusion), --set whitelist, cut-points,
soft-label hard stop."""

import argparse
import os

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import cohen_kappa_score

from tests import synth


def test_set_whitelist_rejects_typos_and_missing_equals():
    from src.utils.config import parse_set
    assert parse_set(['seed=1', 'extra_sources=[ddr,deepdrid]', 'pretrain_path=null']) == \
        {'seed': 1, 'extra_sources': ['ddr', 'deepdrid'], 'pretrain_path': None}
    with pytest.raises(ValueError, match='aptos_repeat'):
        parse_set(['aptos_repeat=6'])
    with pytest.raises(ValueError, match='Did you mean'):
        parse_set(['eyepacs_maxx=5000'])
    with pytest.raises(ValueError, match="has no '='"):
        parse_set(['seed'])


def test_validate_cfg_new_options():
    from src.utils.config import validate_cfg
    import yaml
    base = yaml.safe_load(open('configs/config.yaml'))
    validate_cfg({**base, 'head_type': 'coral_src', 'extra_sources': ['ddr']})
    for bad in ({'head_type': 'coral_src', 'soft_labels': 'aasl'}, {'extra_sources': ['messidor']},
                {'joint_eyepacs': True, 'extra_sources': ['ddr']}, {'external_sets': ['aptos']},
                {'select_endpoint': 'best'}, {'head_type': 'mlp'}):
        with pytest.raises(AssertionError):
            validate_cfg({**base, **bad})


def test_xlsx_reader_roundtrip(tmp_path):
    from src.data.external import read_xlsx_first_sheet
    p = str(tmp_path / 'a.xlsx')
    synth.write_xlsx(p, ['image_id', 'DR_Levels'], [['1_l1', 3], ['1_r1', 0], ['2_l1', 4]])
    d = read_xlsx_first_sheet(p)
    assert list(d.columns) == ['image_id', 'DR_Levels']
    assert d['image_id'].tolist() == ['1_l1', '1_r1', '2_l1'] and d['DR_Levels'].tolist() == [3, 0, 4]


@pytest.fixture
def mini(tmp_path):
    cfg = synth.build(str(tmp_path))
    from src.data.splits import make_splits
    make_splits(cfg)
    return cfg


def test_raw_frames_follow_official_splits(mini):
    from src.data.external import raw_frame
    ddr = raw_frame(mini, 'ddr', 'test')
    assert len(ddr) == 13 and ddr['diagnosis'].max() == 4, 'grade 5 (ungradable) must be dropped'
    dd_tr = raw_frame(mini, 'deepdrid', 'train')
    assert len(dd_tr) == 16 and dd_tr['path'].map(os.path.isfile).all()
    # the row whose label sits in the "other" eye column is still labelled
    assert dd_tr.set_index('id_code').loc['1_l2', 'diagnosis'] == 1
    dd_te = raw_frame(mini, 'deepdrid', 'test')
    assert len(dd_te) == 16 and set(dd_te['group']).isdisjoint(set(dd_tr['group']))
    idr_dev, idr_te = raw_frame(mini, 'idrid', 'dev'), raw_frame(mini, 'idrid', 'test')
    assert idr_dev['id_code'].str.startswith('tr_').all() and idr_te['id_code'].str.startswith('te_').all()
    assert set(idr_dev['id_code']).isdisjoint(set(idr_te['id_code']))
    with pytest.raises(ValueError):
        raw_frame(mini, 'idrid', 'train')


def test_make_external_splits_freezes_and_excludes_duplicates(mini):
    from src.data.external import make_external_splits, external_frame, eval_set_names, check_eval_inputs
    audit = os.path.join('results', 'phase0_protocol', 'audit')
    os.makedirs(audit, exist_ok=True)
    pd.DataFrame([
        {'dataset_a': 'DDR', 'id_a': 'test001', 'dataset_b': 'EyePACS', 'id_b': '3_left', 'hamming_distance': 2},
        {'dataset_a': 'APTOS', 'id_a': 'tr000', 'dataset_b': 'Messidor', 'id_b': '20051020_00003_0100_PP',
         'hamming_distance': 3},
        {'dataset_a': 'IDRiD', 'id_a': 'te_IDRiD_002', 'dataset_b': 'DDR', 'id_b': 'test004', 'hamming_distance': 1},
        {'dataset_a': 'DDR', 'id_a': 'test002', 'dataset_b': 'EyePACS', 'id_b': '4_left', 'hamming_distance': 9},
    ]).to_csv(os.path.join(audit, 'duplicates_external.csv'), index=False)
    mini['dedup_exclude_max'] = 4
    with pytest.raises(FileNotFoundError, match='make_external_splits'):
        external_frame(mini, 'ddr', 'test')
    manifest = make_external_splits(mini)
    ddr = external_frame(mini, 'ddr', 'test')
    assert 'test001' not in set(ddr['id_code']), 'duplicate of a training (EyePACS) image must be excluded'
    assert 'test002' in set(ddr['id_code']), 'distance above dedup_exclude_max must be kept'
    mess = external_frame(mini, 'messidor', 'test')
    mess_dev = external_frame(mini, 'messidor', 'dev')
    assert '20051020_00003_0100_PP' in set(mess['id_code']) | set(mess_dev['id_code']), \
        'Messidor-2 DEV/TEST are never filtered (finished runs were scored on them)'
    assert len(mess) + len(mess_dev) == 40
    assert 'te_IDRiD_002' in set(external_frame(mini, 'idrid', 'test')['id_code']), \
        'a duplicate between two evaluation sets is not leakage'
    ex = pd.read_csv(os.path.join('splits', 'external', 'exclusions.csv'))
    assert set(ex['id_code']) == {'test001'}
    assert manifest['splits']['ddr_test']['n'] == manifest['splits']['ddr_test']['n_raw'] - 1
    assert eval_set_names(mini)[0] == 'aptos_test' and len(eval_set_names(mini)) == 9
    check_eval_inputs(mini)
    summ = pd.read_csv(os.path.join('results', 'phase0_protocol', 'splits_summary.csv'))
    assert 'ddr_test' in set(summ['split']) and 'aptos_test' in set(summ['split'])


def test_missing_dataset_fails_loudly(mini):
    from src.data.external import make_external_splits
    mini['paths']['idrid']['root'] = '/nonexistent/idrid'
    with pytest.raises(FileNotFoundError, match='idrid'):
        make_external_splits(mini)
    make_external_splits({**mini, 'external_sets': ['messidor', 'ddr', 'deepdrid']})


def test_qwk_matches_sklearn_and_cutpoints_recover_shift():
    from src.evaluation.cutpoints import qwk_from_pred, fit_cutpoints, apply_cutpoints, fewshot_curve, devcal
    rng = np.random.default_rng(0)
    y = rng.integers(0, 5, 400)
    p = np.clip(y + rng.integers(-1, 2, 400), 0, 4)
    assert abs(qwk_from_pred(y, p) - cohen_kappa_score(y, p, weights='quadratic')) < 1e-9
    # a well-ranking model whose scores are shifted down by one grade (systematic under-grading)
    score = y + rng.normal(0, 0.35, 400) - 1.0
    base = qwk_from_pred(y, apply_cutpoints(score, [0.5, 1.5, 2.5, 3.5]))
    cuts = fit_cutpoints(score, y)
    assert all(a < b for a, b in zip(cuts, cuts[1:]))
    assert qwk_from_pred(y, apply_cutpoints(score, cuts)) > base + 0.05
    P = np.zeros((400, 5))
    P[np.arange(400), np.clip(np.round(score), 0, 4).astype(int)] = 1.0
    rows = fewshot_curve(P[:200], y[:200], P[200:], y[200:], ns=(10, 50), draws=4)
    kinds = [r['kind'] for r in rows]
    assert kinds == ['argmax', 'default_cuts', 'fewshot', 'fewshot', 'all']
    assert rows[-1]['qwk_mean'] > rows[0]['qwk_mean']
    cuts2, met = devcal(P[:200], y[:200], P[200:], y[200:])
    assert met['qwk'] == pytest.approx(rows[-1]['qwk_mean'])


def test_aasl_without_soft_label_files_stops(mini):
    from src.training.trainer import train_fold
    import logging
    cfg = {**mini, 'soft_labels': 'aasl', 'soft_label_run': 'sl_missing', 'tag': 't', 'phase': 'p'}
    with pytest.raises(FileNotFoundError, match='outer0.npz'):
        train_fold(cfg, 0, 't', logging.getLogger('t'))
    with pytest.raises(ValueError, match='soft_label_run'):
        train_fold({**cfg, 'soft_label_run': None}, 0, 't', logging.getLogger('t'))
    os.makedirs('softlabels/sl_part', exist_ok=True)
    np.savez('softlabels/sl_part/outer0.npz', ids=np.array(['tr000']), Q=np.eye(5)[:1])
    with pytest.raises(ValueError, match='no soft label'):
        train_fold({**cfg, 'soft_label_run': 'sl_part'}, 0, 't', logging.getLogger('t'))


def test_ordinal_head_with_mixup_stops(mini):
    from src.training.trainer import train_fold
    import logging
    with pytest.raises(ValueError, match='mix_prob=0'):
        train_fold({**mini, 'head_type': 'coral', 'mix_prob': 0.5, 'tag': 't', 'phase': 'p'}, 0, 't',
                   logging.getLogger('t'))


def test_zoo_rows_keep_full1744_numbers_apart(tmp_path):
    from src.utils.io_utils import import_zoo, normalize_zoo_row
    csv = tmp_path / 'zoo.csv'
    pd.DataFrame([{'Model': 'MAXVIT_L', 'Accuracy': '85.19 ± 0.95', 'QWK': '0.90 ± 0.01', 'Ext_Acc': '65.21 ± 0.85',
                   'Ext_QWK': '0.59 ± 0.03', 'Ext_Bin_Acc': '85.17 ± 1.09'}]).to_csv(csv, index=False)
    import_zoo(str(csv), 'phase1_zoo')
    row = pd.read_csv('results/phase1_zoo/MAXVIT_L/summary.csv').iloc[0]
    assert pd.isna(row['ext_qwk']) and row['ext_full1744_qwk'] == 0.59
    legacy = normalize_zoo_row({'ext_set': 'full1744', 'ext_qwk': 0.59, 'ext_acc': 65.21})
    assert 'ext_qwk' not in legacy and legacy['ext_full1744_qwk'] == 0.59


def test_snapshot_paths_are_merged_not_replaced(tmp_path):
    """A run snapshot written before the new datasets keeps the default paths of DDR / DeepDRiD / IDRiD."""
    import yaml
    from src.utils.config import build_cfg
    os.makedirs('checkpoints/old_run', exist_ok=True)
    old_paths = {'aptos': {'train_images': 'x'}, 'messidor': {'images': 'm', 'csv': 'm.csv'}}
    yaml.safe_dump({'paths': old_paths, 'seed': 3, 'set_overrides': '{"seed": 3}', 'inherit': 'p0'},
                   open('checkpoints/old_run/config_snapshot.yaml', 'w'))
    cfg = build_cfg(argparse.Namespace(mode='evaluate', tag='old_run', phase=None, inherit=None, base=None,
                                       members=None, kind=None, method=None, set=[]))
    assert cfg['paths']['aptos']['train_images'] == 'x' and 'ddr' in cfg['paths'] and 'idrid' in cfg['paths']
    assert cfg['paths']['aptos']['test_csv'].endswith('test.csv')
