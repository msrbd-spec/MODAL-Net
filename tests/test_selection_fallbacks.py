"""phase_select fallbacks and phase-specific rules, under both selection endpoints.

original endpoint (pre-registered): Messidor DEV is the only external DEV set it needs.
amended endpoint (default): mean QWK over the held-out DEV sets of Messidor, DDR, DeepDRiD and IDRiD;
phase_select refuses to run when one of these DEV prediction files is missing (evaluate_all first).

All files are written inside the temporary project of conftest.py (cwd = tmp folder).
"""
import os

import numpy as np
import pandas as pd
import pytest

from src.evaluation.tables import phase_select

N_OOF, N_DEV = 300, 200
DEV_SETS = ('messidor', 'ddr', 'deepdrid', 'idrid')


@pytest.fixture
def env():
    rng = np.random.default_rng(7)
    y_oof = rng.integers(0, 5, size=N_OOF)
    y_dev = {ds: rng.integers(0, 5, size=N_DEV) for ds in DEV_SETS}
    return y_oof, y_dev


def _probs(y, n_err):
    """One-hot-like probabilities; the first n_err images are graded one step too high (mod 5)."""
    pred = np.asarray(y).copy()
    pred[:n_err] = (pred[:n_err] + 1) % 5
    P = np.eye(5)[pred] * 0.9 + 0.02
    return P / P.sum(axis=1, keepdims=True)


def _make_exp(phase, tag, y_oof, oof_err, y_dev, dev_err, dev_sets=('messidor',), legacy_messidor=True):
    """Synthetic finished experiment: oof.npz, DEV predictions of `dev_sets`, summary.csv."""
    ed = os.path.join('results', phase, tag)
    os.makedirs(ed, exist_ok=True)
    np.savez(os.path.join(ed, 'oof.npz'), ids=np.arange(N_OOF), y=y_oof, P=_probs(y_oof, oof_err))
    for ds in dev_sets:
        name = ('predictions_external_dev.npz' if (ds == 'messidor' and legacy_messidor)
                else f'predictions_{ds}_dev.npz')
        ids = np.array([f'{ds}_{i}' for i in range(N_DEV)])
        np.savez(os.path.join(ed, name), ids=ids, y=y_dev[ds], P=_probs(y_dev[ds], dev_err))
    pd.DataFrame([{'phase': phase, 'experiment': tag, 'winner_eligible': 'true', 'train_sources': 'aptos',
                   'decision': '', 'is_winner': ''}]).to_csv(os.path.join(ed, 'summary.csv'), index=False)
    return ed


def _winner_file(phase):
    with open(os.path.join('results', phase, 'WINNER.txt')) as fh:
        return fh.read().splitlines()


def _cross_phase_setup(y_oof, y_dev, dev_sets):
    _make_exp('phase2a_recipe', 'p2a_ref', y_oof, 4, y_dev, 4, dev_sets)
    with open(os.path.join('results', 'phase2a_recipe', 'WINNER.txt'), 'w') as fh:
        fh.write('p2a_ref\nselected earlier\n')
    # Both arms are clearly worse than the reference everywhere -> both REJECT; no REFERENCE row here.
    _make_exp('phaseX_bad', 'pX_a', y_oof, 90, y_dev, 70, dev_sets)
    _make_exp('phaseX_bad', 'pX_b', y_oof, 110, y_dev, 80, dev_sets)


def test_all_rejected_with_cross_phase_reference_carries_reference(env):
    """original endpoint: no in-phase winner + reference from another phase -> reference carried forward."""
    y_oof, y_dev = env
    _cross_phase_setup(y_oof, y_dev, ('messidor',))

    winner = phase_select('phaseX_bad', reference='@phase2a_recipe', cfg={'select_endpoint': 'original'})

    assert winner == 'p2a_ref'
    lines = _winner_file('phaseX_bad')
    assert lines[0] == 'p2a_ref'
    assert 'carried forward' in lines[1]
    table = pd.read_csv(os.path.join('results', 'phaseX_bad', 'phase_table.csv'))
    assert set(table['decision']) == {'REJECT'}
    # the reference's own phase is untouched
    assert _winner_file('phase2a_recipe')[0] == 'p2a_ref'


def test_all_rejected_with_cross_phase_reference_carries_reference_amended(env):
    """amended endpoint, all four DEV sets present: same carry-forward behaviour."""
    y_oof, y_dev = env
    _cross_phase_setup(y_oof, y_dev, DEV_SETS)

    winner = phase_select('phaseX_bad', reference='@phase2a_recipe')

    assert winner == 'p2a_ref'
    table = pd.read_csv(os.path.join('results', 'phaseX_bad', 'phase_table.csv'))
    assert set(table['decision_amended']) == {'REJECT'}
    assert set(table['dev_sets_used']) == {','.join(DEV_SETS)}


def test_phase2b_joint_rule_adopts_external_gain(env):
    """original endpoint, phase2b: flat OOF QWK + significant Messidor-DEV gain (incl. Mild recall) -> ADOPT."""
    y_oof, y_dev = env
    phase = 'phase2b_preprocess'
    # Reference and arm are identical internally but the arm is far better on Messidor DEV.
    _make_exp(phase, 'p2b_base', y_oof, 40, y_dev, 70)
    _make_exp(phase, 'p2b_pad', y_oof, 40, y_dev, 0)

    winner = phase_select(phase, reference='p2b_base', cfg={'select_endpoint': 'original'})

    table = pd.read_csv(os.path.join('results', phase, 'phase_table.csv')).set_index('experiment')
    assert table.loc['p2b_pad', 'decision_original'] == 'ADOPT'
    assert table.loc['p2b_base', 'decision'] == 'REFERENCE'
    assert winner in ('p2b_pad', 'p2b_base')       # original ranking = OOF QWK (a tie here)
    assert _winner_file(phase)[0] == winner


def test_amended_refuses_missing_dev_predictions(env):
    """amended endpoint: a missing held-out DEV file is an error, never a silent smaller DEV set."""
    y_oof, y_dev = env
    phase = 'phase2b_preprocess'
    _make_exp(phase, 'p2b_base', y_oof, 40, y_dev, 70)          # Messidor DEV only (pre-evaluate_all state)
    _make_exp(phase, 'p2b_pad', y_oof, 40, y_dev, 0)

    with pytest.raises(FileNotFoundError, match='DEV predictions missing'):
        phase_select(phase, reference='p2b_base')
    # restricting the DEV sets explicitly is allowed
    assert phase_select(phase, reference='p2b_base', cfg={'select_dev_sets': ['messidor']}) == 'p2b_pad'


def test_amended_adopts_heldout_gain_and_reports_both_decisions(env):
    """amended endpoint, all four DEV sets: an arm better on every held-out site and flat on OOF -> ADOPT/winner."""
    y_oof, y_dev = env
    phase = 'phase4j_label_loss'
    _make_exp(phase, 'p4j_ref', y_oof, 40, y_dev, 70, DEV_SETS, legacy_messidor=False)
    _make_exp(phase, 'p4j_arm', y_oof, 40, y_dev, 10, DEV_SETS, legacy_messidor=False)

    winner = phase_select(phase, reference='p4j_ref')

    assert winner == 'p4j_arm'
    table = pd.read_csv(os.path.join('results', phase, 'phase_table.csv')).set_index('experiment')
    assert table.loc['p4j_arm', 'decision_amended'] == 'ADOPT'
    assert table.loc['p4j_arm', 'decision_original'] == 'REJECT'      # OOF identical -> no OOF gain
    assert float(table.loc['p4j_arm', 'd_dev_mean_qwk_ci_lo']) > 0
