"""Paper tables / figures of the recalibration analyses (--mode report): built only from the CSVs written by
tools/recalibration_suite.py and tools/shift_decomposition.py, skipped (and listed) when those are absent;
the ordinal source cut-point figure reads report_ordinal; tool figures are registered, never regenerated."""

import json
import os

import numpy as np
import pandas as pd

from src.evaluation.figures import build_paper_figures
from src.evaluation.tables import build_paper_tables

RECAL = os.path.join('results', 'master', 'recal')
TAGS = 'p3b_joint,p10d_ce,p10q_ce'
LEVEL_COLS = ['Tag', 'Dataset', 'Information', 'Method', 'n',
              'QWK', 'dQWK_vs_argmax_95CI', 'Macro_F1', 'dMacro_F1_vs_argmax_95CI',
              'Mild_recall', 'dMild_recall_vs_argmax_95CI', 'Referable_sens', 'dReferable_sens_vs_argmax_95CI',
              'Referable_spec', 'dReferable_spec_vs_argmax_95CI', 'Difference_interval']
NEW_TABLES = ('table_recal_levels', 'table_recal_clinical', 'table_shift_decomposition', 'table_indist_control',
              'table_aptos_dup')

# (method, info, n, draws); oracle rows deliberately in the middle of the file
METHODS = [('argmax', 'none', 0, 1), ('default_cuts', 'none', 0, 1), ('source_cuts', 'none', 0, 1),
           ('oracle_quantile', 'prevalence_oracle', 0, 1), ('oracle_bayes', 'prevalence_oracle', 0, 1),
           ('upa', 'unlabelled_target', 0, 1), ('mlls_bcts_bayes', 'unlabelled_target', 0, 1),
           ('other_sites_quantile', 'prevalence_other_sites', 0, 1),
           ('mlls_prev_estimated', 'prevalence_estimated', 0, 1),
           ('misspecified_quantile_nodr+0.05', 'prevalence_misspecified', 0, 1),
           ('misspecified_quantile_nodr+0.10', 'prevalence_misspecified', 0, 1),
           ('misspecified_quantile_nodr-0.10', 'prevalence_misspecified', 0, 1),
           ('misspecified_quantile_tv0.05', 'prevalence_misspecified', 0, 20),
           ('misspecified_quantile_tv0.10', 'prevalence_misspecified', 0, 20),
           ('fewshot_plain', 'n_labels', 10, 20), ('fewshot_plain', 'n_labels', 25, 20),
           ('fewshot_shrink', 'n_labels', 25, 20), ('all_dev_cuts', 'all_dev_labels', -1, 1),
           ('all_dev_temp', 'all_dev_labels', -1, 1)]
ORDER = ['argmax', 'default_cuts', 'source_cuts', 'upa', 'mlls_bcts_bayes', 'other_sites_quantile',
         'misspecified_quantile_nodr+0.10', 'misspecified_quantile_nodr-0.10', 'misspecified_quantile_tv0.10',
         'fewshot_plain', 'fewshot_shrink', 'all_dev_cuts', 'all_dev_temp', 'oracle_quantile', 'oracle_bayes']


def _write_recal_inputs():
    os.makedirs(RECAL, exist_ok=True)
    rng = np.random.default_rng(0)
    summ, cmp_rows = [], []
    for tag, dss in (('p3b_joint', ('messidor', 'MEAN', 'MEAN_COMMON')), ('p10d_ce', ('MEAN',)),
                     ('zz_other', ('MEAN',))):
        for ds in dss:
            for k, (m, info, n, draws) in enumerate(METHODS):
                r = {'tag': tag, 'dataset': ds, 'method': m, 'info': info, 'n': n, 'draws': draws, 'n_test': 900,
                     'sites': 'messidor,deepdrid,idrid'}                 # extra column of the newer suite
                for met in ('qwk', 'acc', 'f1_macro', 'mild_recall', 'bin_sens', 'bin_spec', 'pred_nodr_share'):
                    r[f'{met}_mean'] = 0.5 + 0.01 * k + rng.uniform(0, 0.004)
                    r[f'{met}_sd'] = 0.01
                summ.append(r)
                for ref in ('argmax', 'mlls_bcts_bayes'):
                    if m == ref:
                        continue
                    for met in ('qwk', 'mild_recall', 'bin_sens', 'f1_macro'):
                        cmp_rows.append({'reference': ref, 'tag': tag, 'dataset': ds, 'method': m, 'info': info,
                                         'n': n, 'metric': met, 'diff_point': 0.01 * k + (0.5 if ref != 'argmax'
                                                                                           else 0.0),
                                         'diff_mean': 0.01 * k, 'diff_lo': 0.01 * k - 0.04,
                                         'diff_hi': 0.01 * k + 0.04, 'p_two_sided': 0.2, 'n_resamples': 1000,
                                         'bootstrap_unit': 'mean of independent per-dataset resamples',
                                         'joint_lo': -9.0, 'joint_hi': 9.0})          # extra columns tolerated
    pd.DataFrame(summ).to_csv(os.path.join(RECAL, 'recal_summary.csv'), index=False)
    pd.DataFrame(cmp_rows).to_csv(os.path.join(RECAL, 'recal_compare.csv'), index=False)

    bo = []
    for tag in ('p3b_joint', 'p10d_ce', 'p10q_ce'):
        for ds in ('messidor', 'MEAN'):
            for j, spec in enumerate((0.90, 0.95, 0.97, 0.99)):
                for method, info in (('dev_threshold_pref', 'all_dev_labels'),
                                     ('test_oracle_roc', 'test_labels_oracle_reporting_only')):
                    r = {'tag': tag, 'dataset': ds, 'method': method, 'info': info, 'n': -1, 'target_spec': spec,
                         'threshold': np.nan, 'dev_sens': 0.9, 'dev_spec': spec, 'sens': 0.95 - 0.05 * j,
                         'spec': spec - 0.01 if method == 'dev_threshold_pref' else spec + 0.005,
                         'test_referable_prevalence': 0.25, 'referrals_per_1000': 300.0 - 40 * j,
                         'missed_referable_per_1000': 12.5 + 5 * j}
                    for p in (0.05, 0.10, 0.20, 0.30):
                        r[f'ppv_at_{p:.2f}'], r[f'npv_at_{p:.2f}'] = 0.3 + p, 0.99 - p / 10
                    bo.append(r)
        bo.append({'tag': tag, 'dataset': 'MEAN', 'method': 'argmax', 'info': 'none', 'n': 0,
                   'target_spec': np.nan, 'sens': 0.7, 'spec': 0.9})             # implied 5-class row
    pd.DataFrame(bo).to_csv(os.path.join(RECAL, 'binary_operating_points.csv'), index=False)

    con = []
    for tag in ('p3b_joint', 'p10d_ce'):
        for scen in ('A0', 'A', 'B', 'C', 'B-A0', 'C-A'):
            for name in ('gain_bayes', 'gain_cuts_over_bayes', 'gain_quantile', 'gain_cuts_over_quantile',
                         'gain_mlls_bayes', 'mlls_recovery', 'vs_argmax:default_cuts'):
                for metric in ('qwk', 'mild_recall'):
                    con.append({'tag': tag, 'dataset': 'MEAN', 'scenario': scen, 'contrast': name, 'metric': metric,
                                'mean': 0.123, 'lo': 0.051, 'hi': 0.199, 'point': 0.12, 'n_valid': 50, 'R': 50,
                                'note': ''})
                    con.append({'tag': tag, 'dataset': 'messidor', 'scenario': scen, 'contrast': name,
                                'metric': metric, 'mean': 9.0, 'lo': 8.0, 'hi': 10.0, 'point': 9.0, 'n_valid': 50,
                                'R': 50, 'note': ''})
    pd.DataFrame(con).to_csv(os.path.join(RECAL, 'shift_decomposition_contrasts.csv'), index=False)

    pd.DataFrame([{'tag': t, 'method': m, 'internal_definition': 'x', 'aptos_test_n': 366,
                   'prior_tv_error_vs_aptos_test': 0.031, 'aptos_test_qwk_argmax': 0.912,
                   'aptos_test_qwk_method': 0.905, 'internal_gain': -0.007, 'external_gain': 0.081,
                   'net_external_gain': 0.088, 'external_sites': 'messidor,deepdrid', 'note': ''}
                  for t in ('p10d_ce', 'p3b_joint', 'zz_other') for m in ('default_cuts', 'mlls_bcts_bayes')]
                 ).to_csv(os.path.join(RECAL, 'indist_control.csv'), index=False)
    pd.DataFrame([{'tag': t, 'subset': s, 'method': m, 'n_excluded': 12, 'pairs_from': 'leakage_pairs.csv rows',
                   'n': 366 - (12 if s != 'all' else 0), 'qwk': 0.9123, 'acc': 81.234, 'f1_macro': 0.701,
                   'mild_recall': 0.55, 'bin_sens': 0.93, 'bin_spec': 0.96, 'note': ''}
                  for t in ('p3b_joint',) for s in ('all', 'excl_dups_of_training_pool')
                  for m in ('argmax', 'default_cuts')]).to_csv(os.path.join(RECAL, 'aptos_dup_sensitivity.csv'),
                                                               index=False)


def test_recal_tables_structure_and_oracle_rows_last():
    _write_recal_inputs()
    build_paper_tables(master_df=pd.DataFrame(), out_dir='results/tables', cfg={'report_recal_tags': TAGS})
    for name in NEW_TABLES:
        assert os.path.isfile(f'results/tables/{name}.csv'), name
        tex = open(f'results/tables/{name}.tex').read()
        assert '\\toprule' in tex and '\\bottomrule' in tex
    t = pd.read_csv('results/tables/table_recal_levels.csv', dtype=str, keep_default_na=False)
    assert list(t.columns) == LEVEL_COLS
    # p3b_joint MEAN, p3b_joint MEAN_COMMON, p10d_ce MEAN; never a per-site row, never a tag outside the list
    blocks = list(dict.fromkeys(zip(t['Tag'], t['Dataset'])))
    assert blocks == [('p3b_joint', 'MEAN'), ('p3b_joint', 'MEAN_COMMON'), ('p10d_ce', 'MEAN')]
    for (tag, ds), g in t.groupby(['Tag', 'Dataset'], sort=False):
        methods = [m.replace(' (oracle, upper bound)', '') for m in g['Method']]
        assert methods == ORDER, methods                                  # levels grouped, oracle rows last
        oracle = g['Method'].str.endswith('(oracle, upper bound)')
        assert list(oracle) == [False] * (len(ORDER) - 2) + [True, True]
        assert g.loc[oracle, 'Information'].str.endswith('(oracle, upper bound)').all()
        assert not g.loc[~oracle, 'Information'].str.contains('oracle').any()
        assert set(g.loc[g['Method'] == 'fewshot_plain', 'n']) == {'25'}           # only n = 25
        assert set(g.loc[g['Method'] == 'all_dev_cuts', 'n']) == {'all'}
        arg = g[g['Method'] == 'argmax'].iloc[0]
        assert arg['dQWK_vs_argmax_95CI'] == 'reference' and arg['Difference_interval'] == 'reference'
        upa = g[g['Method'] == 'upa'].iloc[0]
        k = [m for m, *_ in METHODS].index('upa')
        assert upa['dQWK_vs_argmax_95CI'] == f'{0.01 * k:+.2f} [{0.01 * k - 0.04:.2f}, {0.01 * k + 0.04:.2f}]'
        assert upa['dReferable_spec_vs_argmax_95CI'] == ''          # no bin_spec difference in recal_compare
        assert upa['Difference_interval'] == 'mean of independent per-dataset resamples'
        for col in ('QWK', 'Macro_F1', 'Mild_recall', 'Referable_sens', 'Referable_spec'):
            assert g[col].str.fullmatch(r'\d+\.\d{2}').all(), col             # 2 decimals
    clin = pd.read_csv('results/tables/table_recal_clinical.csv', dtype=str, keep_default_na=False)
    assert list(clin.columns) == ['Tag', 'Target_spec_DEV', 'Referable_sens_TEST', 'Achieved_spec_TEST',
                                  'Referable_prevalence_TEST', 'Referrals_per_1000', 'Missed_referable_per_1000',
                                  'PPV_at_10', 'NPV_at_10', 'PPV_at_20', 'NPV_at_20']
    assert list(zip(clin['Tag'], clin['Target_spec_DEV'])) == [
        (t, s) for t in ('p3b_joint', 'p10d_ce', 'p10q_ce') for s in ('0.90', '0.95', '0.97', '0.99')]
    first = clin.iloc[0]
    assert (first['Referable_sens_TEST'], first['Achieved_spec_TEST'], first['Missed_referable_per_1000'],
            first['PPV_at_10'], first['NPV_at_20']) == ('0.95', '0.89', '12.50', '0.40', '0.97')
    sd = pd.read_csv('results/tables/table_shift_decomposition.csv', dtype=str, keep_default_na=False)
    assert list(sd.columns) == ['Tag', 'Scenario', 'Draws', 'gain_bayes', 'gain_cuts_over_bayes', 'gain_quantile',
                                'mlls_recovery']
    assert list(zip(sd['Tag'], sd['Scenario'])) == [(t, s) for t in ('p3b_joint', 'p10d_ce')
                                                    for s in ('A0', 'A', 'B', 'C')]
    assert (sd['gain_bayes'] == '0.12 [0.05, 0.20]').all() and (sd['Draws'] == '50').all()
    ind = pd.read_csv('results/tables/table_indist_control.csv', dtype=str, keep_default_na=False)
    assert list(ind['Tag']) == ['p3b_joint', 'p3b_joint', 'p10d_ce', 'p10d_ce']       # report order, no zz_other
    assert {'Internal_gain', 'External_gain', 'Net_external_gain', 'Prior_TV_error'} <= set(ind.columns)
    assert ind['Internal_gain'].iloc[0] == '-0.01' and ind['APTOS_test_n'].iloc[0] == '366'
    dup = pd.read_csv('results/tables/table_aptos_dup.csv', dtype=str, keep_default_na=False)
    assert len(dup) == 4 and dup['QWK'].iloc[0] == '0.91' and set(dup['n']) == {'366', '354'}
    readme = open('results/tables/TABLES_README.txt').read()
    written = readme.split('Skipped:')[0]
    for name in NEW_TABLES:
        assert name in written


def test_recal_tables_skipped_and_listed_when_inputs_absent():
    build_paper_tables(master_df=pd.DataFrame(), out_dir='results/tables', cfg={})
    assert not [f for f in os.listdir('results/tables') if f.startswith('table')]
    skipped = open('results/tables/TABLES_README.txt').read().split('Skipped:')[1]
    for name in NEW_TABLES:
        assert name in skipped, name
    assert 'tools/shift_decomposition.py' in skipped and 'tools/recalibration_suite.py' in skipped
    assert 'p3b_joint,p10d_ce,p10q_ce' in skipped                         # default report_recal_tags


def test_recal_tables_skipped_when_no_row_for_headline_tags():
    _write_recal_inputs()
    build_paper_tables(master_df=pd.DataFrame(), out_dir='results/tables', cfg={'report_recal_tags': ['nope']})
    assert not [f for f in os.listdir('results/tables') if f.startswith('table')]
    skipped = open('results/tables/TABLES_README.txt').read().split('Skipped:')[1]
    for name in NEW_TABLES:
        assert f'{name}:' in skipped and 'no row for report_recal_tags=nope' in skipped


def _write_run(phase, tag, thresholds=None):
    d = os.path.join('results', phase, tag)
    os.makedirs(d, exist_ok=True)
    pd.DataFrame([{'experiment': tag}]).to_csv(os.path.join(d, 'summary.csv'), index=False)
    for k, th in enumerate(thresholds or []):
        with open(os.path.join(d, f'source_thresholds_fold{k}.json'), 'w') as f:
            json.dump(th, f)


def test_ordinal_figure_uses_report_ordinal_and_tool_figures_are_registered():
    _write_run('phase3b_joint', 'p_ce')                                   # headline CE run: no thresholds
    _write_run('phase4j_label_loss', 'p_ord', [{'aptos': [0.5, 1.5, 2.5, 3.5], 'ddr': [0.6, 1.4, 2.4, 3.6]},
                                               {'aptos': [0.4, 1.6, 2.6, 3.4], 'ddr': [0.7, 1.5, 2.3, 3.5]}])
    os.makedirs('results/figures', exist_ok=True)
    for name in ('fig_recal_levels.png', 'fig_recal_levels.pdf', 'fig_recal_extra.png', 'fig_adaptation.png',
                 'fig_adaptation_recal.png'):
        open(os.path.join('results/figures', name), 'wb').write(b'not regenerated')
    done, skipped = build_paper_figures(out_dir='results/figures',
                                        cfg={'report_final': 'p_ce', 'report_ordinal': 'p_ord'})
    assert 'fig11_source_cutpoints' in done, skipped
    assert os.path.isfile('results/figures/fig11_source_cutpoints.png')
    assert os.path.isfile('results/figures/fig11_source_cutpoints.pdf')
    readme = open('results/figures/FIGURES_README.txt').read()
    assert 'fig11_source_cutpoints: drawn from p_ord (report_ordinal)' in readme
    tools = readme.split('Produced by tools')[1]
    assert 'fig_recal_levels: produced by tools/recalibration_suite.py (present; png+pdf)' in tools
    assert 'fig_recal_extra: produced by tools/recalibration_suite.py (present; png)' in tools
    assert 'fig_recal_misspec: produced by tools/recalibration_suite.py (missing' in tools
    assert 'fig_recal_operating_points: produced by tools/recalibration_suite.py (missing' in tools
    assert 'fig_shift_decomposition: produced by tools/shift_decomposition.py (missing' in tools
    assert 'fig_adaptation: produced by tools/adaptation_curve.py (present' in tools
    assert 'fig_adaptation_recal: produced by tools/adaptation_curve.py (present' in tools
    assert open('results/figures/fig_recal_levels.png', 'rb').read() == b'not regenerated'

    # report_final (the CE headline model) alone is not used while the report_ordinal run exists
    done, skipped = build_paper_figures(out_dir='results/figures',
                                        cfg={'report_final': 'p_ord', 'report_ordinal': 'p_ce'})
    assert 'fig11_source_cutpoints' not in done
    assert any('fig11_source_cutpoints' in s and 'report_ordinal p_ce is not a coral_src run' in s for s in skipped)

    # report_ordinal run absent from this project -> report_final, and the README says so
    done, _ = build_paper_figures(out_dir='results/figures', cfg={'report_final': 'p_ord', 'report_ordinal': 'nope'})
    assert 'fig11_source_cutpoints' in done
    assert "report_ordinal run 'nope' not found: report_final 'p_ord' used" in \
        open('results/figures/FIGURES_README.txt').read()


def test_report_keys_are_valid_set_keys():
    from src.utils.config import parse_set, validate_cfg
    ov = parse_set(['report_ordinal=p_ord', 'report_recal_tags=a,b'])
    assert ov == {'report_ordinal': 'p_ord', 'report_recal_tags': 'a,b'}
    validate_cfg(ov)
    validate_cfg(parse_set(['report_recal_tags=[a,b]']))
    import pytest
    with pytest.raises(AssertionError):
        validate_cfg({'report_recal_tags': 3})
