"""Test T7: Master table and phase table compilation, schemas, and zoo merge."""

import os
import shutil
import numpy as np
import pandas as pd
import pytest
import yaml

from src.evaluation.tables import phase_select, master_table, MASTER_TABLE_COLS
from src.utils.io_utils import SUMMARY_COLUMNS, fmt2


@pytest.fixture
def synthetic_phase_env():
    """Create a temporary synthetic results directory with reference and candidate experiments."""
    phase = "phase2a_recipe"
    phase_dir = os.path.join("results", phase)
    os.makedirs(phase_dir, exist_ok=True)

    rng = np.random.default_rng(42)
    N_oof = 100
    N_dev = 50

    y_oof = rng.integers(0, 5, size=N_oof)
    y_dev = rng.integers(0, 5, size=N_dev)

    # Helper to generate synthetic experiment folder
    def _create_exp(tag, n_errors=30, winner_eligible=True):
        ed = os.path.join(phase_dir, tag)
        os.makedirs(ed, exist_ok=True)

        pred_oof = y_oof.copy()
        pred_oof[:n_errors] = (pred_oof[:n_errors] + 1) % 5
        P_oof = np.eye(5)[pred_oof] * 0.9 + 0.02
        P_oof /= P_oof.sum(axis=1, keepdims=True)

        pred_dev = y_dev.copy()
        pred_dev[:n_errors // 2] = (pred_dev[:n_errors // 2] + 1) % 5
        P_dev = np.eye(5)[pred_dev] * 0.9 + 0.02
        P_dev /= P_dev.sum(axis=1, keepdims=True)

        np.savez(os.path.join(ed, 'oof.npz'), ids=np.arange(N_oof), y=y_oof, P=P_oof)
        np.savez(os.path.join(ed, 'predictions_external_dev.npz'), ids=np.arange(N_dev), y=y_dev, P=P_dev)

        sum_dict = {c: '' for c in SUMMARY_COLUMNS}
        sum_dict.update({
            'phase': phase,
            'experiment': tag,
            'description': f'Synthetic {tag}',
            'set_overrides': '{}',
            'inherit': 'None',
            'protocol': '5fold_ensemble',
            'ext_set': 'messidor_test',
            'n_params_M': '25.00',
            'gpu_hours': '1.20',
            'oof_qwk': fmt2(1.0 - (n_errors / N_oof)),
            'test_acc': fmt2(100.0 - (n_errors / N_oof * 100)),
            'test_qwk': fmt2(1.0 - (n_errors / N_oof)),
            'extdev_qwk': fmt2(1.0 - (n_errors / 2 / N_dev)),
            'extdev_bin_auroc': fmt2(0.88),
            'ext_acc': fmt2(75.0),
            'ext_qwk': fmt2(0.68),
            'winner_eligible': str(winner_eligible).lower(),
            'decision': '',
            'is_winner': '',
        })
        pd.DataFrame([sum_dict]).to_csv(os.path.join(ed, 'summary.csv'), index=False)
        with open(os.path.join(ed, 'config.yaml'), 'w') as f_c:
            yaml.safe_dump({'tag': tag, 'phase': phase, 'winner_eligible': winner_eligible}, f_c)

    _create_exp('p2a_plain', n_errors=35)
    _create_exp('p2a_reg', n_errors=5)
    _create_exp('p2a_mix', n_errors=45)

    yield phase

    # Teardown
    shutil.rmtree(phase_dir, ignore_errors=True)


def test_phase_select_and_master_tables(synthetic_phase_env):
    """T7: Test phase selection, WINNER.txt, and master table compilation with zoo merge."""
    phase = synthetic_phase_env

    # 1. Run phase_select
    winner_tag = phase_select(phase, reference='p2a_plain')
    assert winner_tag == 'p2a_reg'

    # Check phase_table.csv exists and has delta columns
    pt_path = os.path.join('results', phase, 'phase_table.csv')
    assert os.path.isfile(pt_path)
    pt_df = pd.read_csv(pt_path)
    assert 'd_oof_qwk' in pt_df.columns
    assert 'ci_lo' in pt_df.columns
    assert 'ci_hi' in pt_df.columns
    assert 'p_holm' in pt_df.columns
    assert 'decision' in pt_df.columns
    assert 'winner' in pt_df.columns

    # Verify reference decision
    ref_row = pt_df[pt_df['experiment'] == 'p2a_plain'].iloc[0]
    assert ref_row['decision'] == 'REFERENCE'

    # Verify winning experiment decision
    winner_row = pt_df[pt_df['experiment'] == 'p2a_reg'].iloc[0]
    assert winner_row['decision'] in ['ADOPT', 'TENTATIVE']
    assert winner_row['winner'] == True

    # Check WINNER.txt
    winner_txt = os.path.join('results', phase, 'WINNER.txt')
    assert os.path.isfile(winner_txt)
    lines = open(winner_txt).read().strip().split('\n')
    assert lines[0] == 'p2a_reg'
    assert len(lines) >= 2  # tag on line 1, reason on line 2

    # Check summary.csv updated
    sum_reg = pd.read_csv(os.path.join('results', phase, 'p2a_reg', 'summary.csv')).iloc[0]
    assert sum_reg['decision'] == winner_row['decision']
    assert str(sum_reg['is_winner']).lower() == 'true'

    # 2. Run master_table
    master_table()

    master_csv = os.path.join('results', 'master', 'master_table.csv')
    assert os.path.isfile(master_csv)
    m_df = pd.read_csv(master_csv)

    # Check column schema
    for col in MASTER_TABLE_COLS:
        assert col in m_df.columns, f"Missing master table column: {col}"

    # Check master_full.csv
    full_csv = os.path.join('results', 'master', 'master_full.csv')
    assert os.path.isfile(full_csv)
    full_df = pd.read_csv(full_csv)
    assert len(full_df.columns) >= len(SUMMARY_COLUMNS)

    # Check zoo rows merged with protocol column
    zoo_rows = m_df[m_df['phase'] == 'phase1_zoo']
    if not zoo_rows.empty:
        assert (zoo_rows['protocol'] == 'single_fold_mean').all()
        assert (zoo_rows['ext_set'] == 'full1744').all()

    # Check LaTeX booktabs table
    tex_path = os.path.join('results', 'master', 'master_table.tex')
    assert os.path.isfile(tex_path)
    tex_content = open(tex_path).read()
    assert '\\toprule' in tex_content
    assert '\\bottomrule' in tex_content
