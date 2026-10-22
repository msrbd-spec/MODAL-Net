"""I/O utilities: log_run, import_zoo, formatting helpers."""

import os
import json
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Number formatting (2 decimal places everywhere)
# ---------------------------------------------------------------------------

def fmt2(v):
    """Format a value to 2 decimal places. Returns '' for None/NaN."""
    if v is None:
        return ''
    try:
        if np.isnan(v):
            return ''
    except (TypeError, ValueError):
        pass
    return f'{v:.2f}'


# ---------------------------------------------------------------------------
# log_run: legacy run registry (results/all_runs.npz)
# ---------------------------------------------------------------------------

def log_run(tag, kind, metrics):
    """Append a run entry to results/all_runs.npz.

    Args:
        tag: Experiment tag.
        kind: e.g. 'experiment', 'inference_experiment', 'ensemble_experiment'.
        metrics: dict of metric name -> value.
    """
    path = os.path.join('results', 'all_runs.npz')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    runs = {}
    if os.path.isfile(path):
        with np.load(path, allow_pickle=True) as f:
            if 'runs' in f:
                runs = f['runs'].item()
    runs[tag] = {'kind': kind, **metrics}
    np.savez(path, runs=np.array(runs))


# ---------------------------------------------------------------------------
# import_zoo: read baseline_result.csv -> per-model summary.csv + zoo_table
# ---------------------------------------------------------------------------

# Mapping: baseline_result.csv column -> (summary.csv mean column, std column)
_ZOO_COL_MAP = {
    'Best_Val_Acc':   ('cv_val_acc',      'cv_val_acc_std'),
    'Final_Train_Acc':('train_acc',        'train_acc_std'),
    'Accuracy':       ('test_acc',         'test_acc_std'),
    'QWK':            ('test_qwk',         'test_qwk_std'),
    'Precision':      ('test_prec_macro',  'test_prec_macro_std'),
    'Recall':         ('test_rec_macro',   'test_rec_macro_std'),
    'F1_Score':       ('test_f1_macro',    'test_f1_macro_std'),
    'Bin_Acc':        ('test_bin_acc',     'test_bin_acc_std'),
    'Bin_Prec':       ('test_bin_prec',    'test_bin_prec_std'),
    'Bin_Rec':        ('test_bin_sens',    'test_bin_sens_std'),
    'Bin_F1':         ('test_bin_f1',      'test_bin_f1_std'),
    'Ext_Acc':        ('ext_acc',          'ext_acc_std'),
    'Ext_QWK':        ('ext_qwk',         'ext_qwk_std'),
    'Ext_Bin_Acc':    ('ext_bin_acc',      'ext_bin_acc_std'),
}


def _parse_mean_std(s):
    """Parse '85.19 ± 0.95' -> (85.19, 0.95). Also handles plain numbers."""
    s = str(s).strip()
    if '±' in s:
        parts = s.split('±')
        return float(parts[0].strip()), float(parts[1].strip())
    if '+-' in s:
        parts = s.split('+-')
        return float(parts[0].strip()), float(parts[1].strip())
    try:
        return float(s), 0.0
    except ValueError:
        return None, None


# The full summary.csv column set (S5 line 82-88).  Empty columns are filled
# with '' for zoo rows; this list is the canonical ordering.
SUMMARY_COLUMNS = [
    'phase', 'experiment', 'description', 'set_overrides', 'inherit',
    'protocol', 'ext_set', 'n_params_M', 'gpu_hours',
    # CV / OOF
    'cv_val_acc', 'cv_val_qwk', 'cv_val_bin_auroc',
    'oof_acc', 'oof_qwk', 'oof_bin_auroc',
    # Test 5-class
    'test_acc', 'test_qwk', 'test_prec_macro', 'test_rec_macro',
    'test_f1_macro', 'test_adj_acc', 'test_mae',
    'test_f1_c0', 'test_f1_c1', 'test_f1_c2', 'test_f1_c3', 'test_f1_c4',
    'test_rec_c0', 'test_rec_c1', 'test_rec_c2', 'test_rec_c3', 'test_rec_c4',
    'test_acc_single_mean', 'test_acc_single_std',
    'test_qwk_single_mean', 'test_qwk_single_std',
    # Test binary
    'test_bin_acc', 'test_bin_prec', 'test_bin_sens', 'test_bin_spec',
    'test_bin_f1', 'test_bin_auroc', 'test_bin_auprc', 'test_sens_at_spec90',
    # External dev
    'extdev_acc', 'extdev_qwk', 'extdev_bin_auroc', 'extdev_mild_recall',
    # External (test)
    'ext_acc', 'ext_qwk', 'ext_prec_macro', 'ext_rec_macro',
    'ext_f1_macro', 'ext_adj_acc', 'ext_mild_recall', 'ext_pred_nodr_share',
    'ext_bin_acc', 'ext_bin_prec', 'ext_bin_sens', 'ext_bin_spec',
    'ext_bin_f1', 'ext_bin_auroc', 'ext_bin_auprc', 'ext_sens_at_spec90',
    # Calibration
    'ece_test', 'ece_ext',
    # Diagnostics
    'train_val_gap',
    # Decision
    'decision', 'is_winner', 'notes',
]


def import_zoo(csv_path, phase):
    """Import baseline_result.csv into per-model summary.csv files and zoo_table.csv.

    Args:
        csv_path: Path to the baseline results CSV.
        phase: Phase folder name (e.g. 'phase1_zoo').
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f'Zoo CSV not found: {csv_path}')

    raw = pd.read_csv(csv_path)
    all_rows = []

    for _, zrow in raw.iterrows():
        model = str(zrow['Model']).strip()
        row = {c: '' for c in SUMMARY_COLUMNS}
        row['phase'] = phase
        row['experiment'] = model
        row['description'] = f'Zoo baseline {model}'
        row['protocol'] = 'single_fold_mean'
        row['ext_set'] = 'full1744'

        # Parse mean ± std columns
        for src_col, (mean_col, std_col) in _ZOO_COL_MAP.items():
            if src_col in zrow.index:
                m, s = _parse_mean_std(zrow[src_col])
                if m is not None:
                    row[mean_col] = fmt2(m)
                    if std_col not in SUMMARY_COLUMNS:
                        # Ensure std column exists
                        pass
                    row[std_col] = fmt2(s)

        # Compute train_val_gap if both values exist
        try:
            ta = float(row.get('train_acc', ''))
            va = float(row.get('cv_val_acc', ''))
            row['train_val_gap'] = fmt2(ta - va)
        except (ValueError, TypeError):
            pass

        # Write per-model summary.csv
        model_dir = os.path.join('results', phase, model)
        os.makedirs(model_dir, exist_ok=True)
        # Build the column list for this row (summary + extra std cols + train_acc)
        extra_cols = ['train_acc', 'train_acc_std']
        std_cols = [v[1] for v in _ZOO_COL_MAP.values()]
        all_cols = []
        for c in SUMMARY_COLUMNS:
            all_cols.append(c)
        for c in extra_cols + std_cols:
            if c not in all_cols:
                all_cols.append(c)

        model_df = pd.DataFrame([{c: row.get(c, '') for c in all_cols}])
        model_df.to_csv(os.path.join(model_dir, 'summary.csv'), index=False)

        all_rows.append({c: row.get(c, '') for c in all_cols})

    # Write combined zoo_table.csv
    zoo_df = pd.DataFrame(all_rows)
    zoo_dir = os.path.join('results', phase)
    os.makedirs(zoo_dir, exist_ok=True)
    zoo_df.to_csv(os.path.join(zoo_dir, 'zoo_table.csv'), index=False)
    print(f'import_zoo: wrote {len(all_rows)} models to results/{phase}/')
