"""Evaluation core shared by every mode: predictions on every evaluation set, metrics, artefacts.

Evaluation sets of an experiment (src.data.external.eval_set_names):
    oof                     pooled APTOS out-of-fold predictions (internal validation, 3,296 images)
    aptos_test              APTOS held-out test (366 images, never filtered)
    <dataset>_dev / _test   for every dataset in cfg['external_sets'] (messidor, ddr, deepdrid, idrid)

Per experiment folder results/<phase>/<tag>/ this module writes:
    predictions_<set>.npz    ids, y_true, y_pred, y_prob (temperature-scaled), y_prob_raw, y_prob_folds
    predictions_test.npz, predictions_external_dev.npz, predictions_external_test.npz   legacy aliases
                             (APTOS test, Messidor-2 DEV, Messidor-2 TEST) read by older tools
    oof.npz                  pooled OOF (raw probabilities)
    metrics_long.csv         one row per (set, metric) with value, 95 % bootstrap CI, n, in_source flag
    summary.csv              one row: legacy columns + <set>_<metric> for every set + external means
    per_class.csv            set x class: precision, recall, F1, support
    binary.csv               one row per set: referable-DR metrics
    calibration.csv          one row per set: ECE, NLL, Brier, temperature
    confusion_<set>.csv      5 x 5 confusion matrix (+ legacy confusion_test.csv / confusion_external.csv)
    cutpoints.csv            external datasets: cut-points refitted on DEV and the TEST metrics they give
    fewshot.csv              external datasets: few-shot cut-point refit curve on TEST
    per_fold.csv, config.yaml, figures/
"""

import os
import json
import time

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.stats import rankdata
from torch.utils.data import DataLoader

from src.data.datasets import DRDataset
from src.data.external import (DISPLAY_NAMES, eval_frame, eval_set_names, split_set_name, check_eval_inputs)
from src.evaluation.metrics import metrics5, metrics_bin, ece, nll, brier, apply_temperature
from src.evaluation.calibration import fit_devcut, apply_devcut
from src.evaluation.cutpoints import devcal, fewshot_curve
from src.evaluation.predict import predict_probs
from src.utils.config import train_sources
from src.utils.io_utils import SUMMARY_COLUMNS, fmt2, log_run

CLASS_NAMES = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']

# Metrics written for every set (5-class, binary/referable, calibration).
METRICS_5 = ['acc', 'qwk', 'prec_macro', 'rec_macro', 'f1_macro', 'adj_acc', 'mae', 'mild_recall',
             'pred_nodr_share'] + [f'{m}_c{c}' for m in ('prec', 'rec', 'f1') for c in range(5)]
METRICS_BIN = ['bin_acc', 'bin_prec', 'bin_sens', 'bin_spec', 'bin_f1', 'bin_auroc', 'bin_auprc',
               'sens_at_spec90', 'sens_at_spec95']
METRICS_CAL = ['ece', 'nll', 'brier']
SUMMARY_METRICS = METRICS_5 + METRICS_BIN + METRICS_CAL + ['n']
DEVCAL_METRICS = ['acc', 'qwk', 'f1_macro', 'mild_recall', 'pred_nodr_share', 'bin_sens', 'bin_spec']
CI_METRICS = ['acc', 'qwk', 'prec_macro', 'rec_macro', 'f1_macro', 'mild_recall', 'bin_acc', 'bin_prec',
              'bin_sens', 'bin_spec', 'bin_f1', 'bin_auroc']
EXT_MEAN_METRICS = ['acc', 'qwk', 'f1_macro', 'mild_recall', 'pred_nodr_share', 'bin_acc', 'bin_sens',
                    'bin_spec', 'bin_auroc']

LEGACY_FILES = {'aptos_test': 'predictions_test.npz', 'messidor_dev': 'predictions_external_dev.npz',
                'messidor_test': 'predictions_external_test.npz'}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def all_metrics(y, P, thresholds=None):
    """5-class + referable + calibration metrics of one set (raw floats)."""
    y = np.asarray(y, dtype=int)
    P = np.asarray(P, dtype=float)
    out = {'n': int(len(y))}
    if len(y) == 0:
        return out
    out.update(metrics5(y, P))
    out.update(metrics_bin(y, P, thresholds=thresholds))
    out['ece'] = ece(P, y)
    out['nll'] = nll(P, y)
    out['brier'] = brier(P, y)
    return out


def _fast_ci_metrics(y, pred, s):
    """CI_METRICS for one bootstrap resample (numpy only)."""
    K = 5
    n = len(y)
    cm = np.bincount(y * K + pred, minlength=K * K).reshape(K, K).astype(float)
    tp = np.diag(cm)
    rows, cols = cm.sum(1), cm.sum(0)
    rec = np.divide(tp, rows, out=np.zeros(K), where=rows > 0)
    prec = np.divide(tp, cols, out=np.zeros(K), where=cols > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(K), where=(prec + rec) > 0)
    W = (np.arange(K)[:, None] - np.arange(K)[None, :]) ** 2.0
    den = (W * np.outer(rows, cols) / n).sum()
    qwk = 1.0 - (W * cm).sum() / den if den > 0 else float('nan')
    yb, pb = (y >= 2).astype(int), (s >= 0.5).astype(int)
    tpb = float(((yb == 1) & (pb == 1)).sum())
    fpb = float(((yb == 0) & (pb == 1)).sum())
    fnb = float(((yb == 1) & (pb == 0)).sum())
    tnb = float(((yb == 0) & (pb == 0)).sum())
    sens = tpb / (tpb + fnb) if tpb + fnb > 0 else float('nan')
    spec = tnb / (tnb + fpb) if tnb + fpb > 0 else float('nan')
    bprec = tpb / (tpb + fpb) if tpb + fpb > 0 else 0.0
    bf1 = 2 * bprec * sens / (bprec + sens) if (bprec + sens) > 0 and not np.isnan(sens) else 0.0
    npos, nneg = int(yb.sum()), int((1 - yb).sum())
    if npos > 0 and nneg > 0:
        r = rankdata(s)
        auroc = (r[yb == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg)
    else:
        auroc = float('nan')
    present = rows > 0
    return {
        'acc': 100.0 * tp.sum() / n, 'qwk': qwk,
        'prec_macro': float(prec.mean()), 'rec_macro': float(rec.mean()), 'f1_macro': float(f1.mean()),
        'mild_recall': float(rec[1]) if present[1] else float('nan'),
        'bin_acc': 100.0 * (tpb + tnb) / n, 'bin_prec': bprec, 'bin_sens': sens, 'bin_spec': spec,
        'bin_f1': bf1, 'bin_auroc': auroc,
    }


def bootstrap_cis(y, P, B=1000, seed=0):
    """Percentile 95 % CIs over images for CI_METRICS. Returns {metric: (lo, hi)}."""
    y = np.asarray(y, dtype=int)
    P = np.asarray(P, dtype=float)
    if B <= 0 or len(y) < 2:
        return {}
    pred = P.argmax(1)
    s = P[:, 2:].sum(1)
    rng = np.random.default_rng(seed)
    vals = {m: [] for m in CI_METRICS}
    for _ in range(int(B)):
        i = rng.integers(0, len(y), len(y))
        r = _fast_ci_metrics(y[i], pred[i], s[i])
        for m in CI_METRICS:
            vals[m].append(r[m])
    out = {}
    for m, v in vals.items():
        v = np.asarray(v, dtype=float)
        v = v[~np.isnan(v)]
        if len(v):
            out[m] = (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    return out


# ---------------------------------------------------------------------------
# Prediction sets
# ---------------------------------------------------------------------------

def make_loader(cfg, set_name):
    df, dirs, domain = eval_frame(cfg, set_name)
    ds = DRDataset(df, dirs, cfg, train=False, domain=domain)
    return DataLoader(ds, batch_size=int(cfg.get('batch_size', 16)), shuffle=False,
                      num_workers=int(cfg.get('num_workers', 4)))


def predict_set(models, cfg, set_name, device):
    """Ensemble prediction of one evaluation set. Returns dict ids, y, P_raw, P_folds."""
    ids, y, P, P_single = predict_probs(models, make_loader(cfg, set_name), cfg, tta=cfg.get('tta'),
                                        scales=cfg.get('tta_scales'), device=device)
    return {'ids': [str(i) for i in ids], 'y': np.asarray(y, dtype=int), 'P_raw': np.asarray(P),
            'P_folds': np.asarray(P_single)}


def load_saved_set(exp_dir, set_name):
    """Raw ensemble probabilities of a set saved earlier, or None.

    New files carry y_prob_raw. Legacy files (written before this module) carry the temperature-
    scaled ensemble in y_prob/P and the raw per-fold probabilities in y_prob_folds; the raw ensemble
    is then recovered exactly as the mean over folds (that is how it was computed).
    """
    cand = [os.path.join(exp_dir, f'predictions_{set_name}.npz')]
    if set_name in LEGACY_FILES:
        cand.append(os.path.join(exp_dir, LEGACY_FILES[set_name]))
    for p in cand:
        if not os.path.isfile(p):
            continue
        with np.load(p, allow_pickle=True) as f:
            keys = set(f.files)
            ids = [str(i) for i in f['ids']]
            y = np.asarray(f['y_true'] if 'y_true' in keys else f['y'], dtype=int)
            folds = np.asarray(f['y_prob_folds']) if 'y_prob_folds' in keys else np.empty((0,))
            if 'y_prob_raw' in keys:
                raw = np.asarray(f['y_prob_raw'])
            elif folds.ndim == 3 and folds.shape[0] > 0 and folds.shape[1] == len(ids):
                raw = folds.mean(axis=0)
            else:
                continue
        return {'ids': ids, 'y': y, 'P_raw': raw, 'P_folds': folds, 'source_file': os.path.basename(p)}
    return None


def restrict_to_frame(rec, frame):
    """Keep only the rows of a saved set whose ids are in the current frozen split (same order as the
    split file). Returns None when the saved set does not cover the split (inference needed)."""
    want = [str(i) for i in frame['id_code']]
    pos = {i: k for k, i in enumerate(rec['ids'])}
    if any(i not in pos for i in want):
        return None
    idx = np.array([pos[i] for i in want], dtype=int)
    y_frame = frame['diagnosis'].to_numpy(dtype=int)
    if not np.array_equal(rec['y'][idx], y_frame):
        return None
    folds = rec['P_folds']
    if folds.ndim == 3 and folds.shape[1] == len(rec['ids']):
        folds = folds[:, idx]
    return {'ids': want, 'y': y_frame, 'P_raw': rec['P_raw'][idx], 'P_folds': folds}


def collect_sets(cfg, exp_dir, model_loader, logger, overwrite=False):
    """Predictions for every evaluation set: reuse saved arrays that cover the frozen split,
    otherwise run inference (models are loaded lazily, once)."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sets, models = {}, None
    for name in eval_set_names(cfg):
        frame = eval_frame(cfg, name)[0]
        rec = None if overwrite else load_saved_set(exp_dir, name)
        rec = restrict_to_frame(rec, frame) if rec is not None else None
        if rec is None:
            if models is None:
                models = model_loader()
            logger.info(f'  inference on {name} ({len(frame)} images)')
            rec = predict_set(models, cfg, name, device)
        else:
            logger.info(f'  {name}: reusing saved predictions ({len(rec["ids"])} images)')
        sets[name] = rec
    return sets


# ---------------------------------------------------------------------------
# Artefacts
# ---------------------------------------------------------------------------

def _save_npz(path, ids, y, P_scaled, P_raw, P_folds):
    np.savez(path, ids=np.array(ids), y_true=y, y_pred=P_scaled.argmax(1), y_prob=P_scaled, y_prob_raw=P_raw,
             y_prob_folds=P_folds, y=y, P=P_scaled, P_per_model=P_folds)


def _in_source(cfg):
    src = set(train_sources(cfg))
    return {name: split_set_name(name)[0] in src for name in eval_set_names(cfg) if name != 'aptos_test'}


def heldout_datasets(cfg):
    """External datasets of this experiment that are NOT training sources (true external sets)."""
    src = set(train_sources(cfg))
    return [d for d in (cfg.get('external_sets') or []) if d not in src]


def write_artifacts(cfg, tag, phase, exp_dir, ckpt_dir, t0, n_params_M, fold_metric_records, oof,
                    thresholds, fitted_T, sets, protocol, logger, winner_eligible=True, notes='',
                    kind='experiment', keep_decision=None):
    """Write every evaluation artefact of an experiment (see module docstring). Returns the summary row."""
    os.makedirs(exp_dir, exist_ok=True)
    use_T = bool(cfg.get('temperature_scaling', True)) and fitted_T is not None and fitted_T > 0
    scaled = {n: (apply_temperature(r['P_raw'], fitted_T) if use_T else r['P_raw']) for n, r in sets.items()}

    # Optional legacy devcut recalibration: fitted on each dataset's DEV split, applied to its TEST split
    if cfg.get('recalibrate') == 'devcut':
        winner_eligible = False
        for d in (cfg.get('external_sets') or []):
            if f'{d}_dev' in sets and f'{d}_test' in sets:
                a_opt, b_opt = fit_devcut(sets[f'{d}_dev']['y'], scaled[f'{d}_dev'])
                scaled[f'{d}_test'] = apply_devcut(scaled[f'{d}_test'], a_opt, b_opt)

    # Prediction files
    for n, r in sets.items():
        _save_npz(os.path.join(exp_dir, f'predictions_{n}.npz'), r['ids'], r['y'], scaled[n], r['P_raw'], r['P_folds'])
        if n in LEGACY_FILES:
            legacy = os.path.join(exp_dir, LEGACY_FILES[n])
            if os.path.isfile(legacy):
                with np.load(legacy, allow_pickle=True) as f_old:
                    old_ids = [str(i) for i in f_old['ids']]
                keep = legacy.replace('.npz', '.pre_v2.npz')
                if old_ids != list(r['ids']) and not os.path.isfile(keep):
                    os.replace(legacy, keep)   # never lose predictions written on a different image set
            _save_npz(legacy, r['ids'], r['y'], scaled[n], r['P_raw'], r['P_folds'])
    np.savez(os.path.join(exp_dir, 'oof.npz'), ids=np.array(oof['ids']), y_true=oof['y'],
             y_pred=oof['P'].argmax(1), y_prob=oof['P'], y=oof['y'], P=oof['P'])

    # Metrics (OOF uses raw probabilities, exactly as the OOF thresholds/temperature were fitted)
    probs = {'oof': (oof['y'], oof['P'])}
    probs.update({n: (sets[n]['y'], scaled[n]) for n in sets})
    met = {n: all_metrics(y, P, thresholds) for n, (y, P) in probs.items()}
    B = int(cfg.get('bootstrap_ci', 1000))
    cis = {n: bootstrap_cis(y, P, B=B, seed=0) for n, (y, P) in probs.items()}
    in_src = _in_source(cfg)

    # Cut-point recalibration and few-shot curve on every external dataset
    cut_rows, few_rows, devcal_met = [], [], {}
    for d in (cfg.get('external_sets') or []):
        dv, ts = f'{d}_dev', f'{d}_test'
        if dv not in sets or ts not in sets:
            continue
        cuts, dm = devcal(scaled[dv], sets[dv]['y'], scaled[ts], sets[ts]['y'])
        devcal_met[ts] = dm
        cut_rows.append({'dataset': d, 'in_source': in_src.get(ts, False), 'n_dev': len(sets[dv]['y']),
                         'n_test': len(sets[ts]['y']),
                         **{f'cut{j + 1}': round(float(c), 4) for j, c in enumerate(cuts)},
                         **{f'argmax_{m}': met[ts].get(m) for m in DEVCAL_METRICS},
                         **{f'devcal_{m}': dm[m] for m in DEVCAL_METRICS}})
        for r in fewshot_curve(scaled[dv], sets[dv]['y'], scaled[ts], sets[ts]['y'],
                               ns=cfg.get('fewshot_n', [10, 25, 50, 100]), draws=int(cfg.get('fewshot_draws', 20)),
                               seed=int(cfg.get('seed', 0))):
            few_rows.append({'dataset': d, 'in_source': in_src.get(ts, False), **r})
    pd.DataFrame(cut_rows, columns=None if cut_rows else ['dataset']).round(4).to_csv(
        os.path.join(exp_dir, 'cutpoints.csv'), index=False)
    pd.DataFrame(few_rows, columns=None if few_rows else ['dataset']).round(4).to_csv(
        os.path.join(exp_dir, 'fewshot.csv'), index=False)

    # metrics_long.csv
    long_rows = []
    for n, m in met.items():
        ds, split = ('aptos', 'oof') if n == 'oof' else split_set_name(n)
        for key in SUMMARY_METRICS:
            if key not in m:
                continue
            task = ('binary' if key in METRICS_BIN else 'calibration' if key in METRICS_CAL
                    else 'count' if key == 'n' else '5class')
            lo, hi = cis.get(n, {}).get(key, (np.nan, np.nan))
            long_rows.append({'experiment': tag, 'phase': phase, 'set': n, 'dataset': ds, 'split': split,
                              'in_source': bool(in_src.get(n, ds == 'aptos')), 'task': task, 'metric': key,
                              'value': m[key], 'ci_lo': lo, 'ci_hi': hi, 'n': m['n']})
        if n in devcal_met:
            for key, v in devcal_met[n].items():
                long_rows.append({'experiment': tag, 'phase': phase, 'set': n, 'dataset': ds, 'split': split,
                                  'in_source': bool(in_src.get(n, False)), 'task': 'devcal', 'metric': key,
                                  'value': v, 'ci_lo': np.nan, 'ci_hi': np.nan, 'n': m['n']})
    pd.DataFrame(long_rows).round(4).to_csv(os.path.join(exp_dir, 'metrics_long.csv'), index=False)

    # per_class.csv, binary.csv, calibration.csv, confusion_<set>.csv
    from sklearn.metrics import confusion_matrix
    pc_rows, bin_rows, cal_rows = [], [], []
    for n, m in met.items():
        for c in range(5):
            pc_rows.append({'set': n, 'class': c, 'class_name': CLASS_NAMES[c], 'precision': fmt2(m.get(f'prec_c{c}')),
                            'recall': fmt2(m.get(f'rec_c{c}')), 'f1': fmt2(m.get(f'f1_c{c}')),
                            'support': m.get(f'support_c{c}', '')})
        bin_rows.append({'set': n, **{k: fmt2(m.get(k)) for k in METRICS_BIN}})
        cal_rows.append({'set': n, 'ece': fmt2(m.get('ece')), 'nll': fmt2(m.get('nll')),
                         'brier': fmt2(m.get('brier')),
                         'temperature': fmt2(fitted_T) if (use_T and n != 'oof') else fmt2(1.0)})
        y, P = probs[n]
        cm = pd.DataFrame(confusion_matrix(y, P.argmax(1), labels=range(5)))
        cm.to_csv(os.path.join(exp_dir, f'confusion_{n}.csv'), index=False)
        if n == 'aptos_test':
            cm.to_csv(os.path.join(exp_dir, 'confusion_test.csv'), index=False)
        if n == 'messidor_test':
            cm.to_csv(os.path.join(exp_dir, 'confusion_external.csv'), index=False)
    pd.DataFrame(pc_rows).to_csv(os.path.join(exp_dir, 'per_class.csv'), index=False)
    pd.DataFrame(bin_rows).to_csv(os.path.join(exp_dir, 'binary.csv'), index=False)
    pd.DataFrame(cal_rows).to_csv(os.path.join(exp_dir, 'calibration.csv'), index=False)

    # Single-model statistics on the APTOS test set and per_fold.csv
    single = []
    folds = sets['aptos_test']['P_folds'] if 'aptos_test' in sets else np.empty((0,))
    if folds.ndim == 3 and folds.shape[0] > 0:
        for Pk in folds:
            Pk_s = apply_temperature(Pk, fitted_T) if use_T else Pk
            mk = metrics5(sets['aptos_test']['y'], Pk_s)
            single.append((mk['acc'], mk['qwk']))
    per_fold_rows = []
    n_rows = max(len(fold_metric_records or []), len(single))
    for k in range(n_rows):
        rec = (fold_metric_records or [{}] * n_rows)[k] if k < len(fold_metric_records or []) else {}
        per_fold_rows.append({'fold': k, 'val_acc': fmt2(rec.get('val_acc')), 'val_qwk': fmt2(rec.get('val_qwk')),
                              'val_bin_auroc': fmt2(rec.get('val_bin_auroc')),
                              'test_acc': fmt2(single[k][0]) if k < len(single) else '',
                              'test_qwk': fmt2(single[k][1]) if k < len(single) else ''})
    pd.DataFrame(per_fold_rows).to_csv(os.path.join(exp_dir, 'per_fold.csv'), index=False)

    with open(os.path.join(exp_dir, 'config.yaml'), 'w') as f_cfg:
        yaml.safe_dump(cfg, f_cfg)

    # summary.csv
    elapsed_hours = (time.time() - t0) / 3600.0
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    if fold_metric_records:
        cv_acc = float(np.mean([r.get('val_acc', np.nan) for r in fold_metric_records]))
        cv_qwk = float(np.mean([r.get('val_qwk', np.nan) for r in fold_metric_records]))
        cv_auc = float(np.mean([r.get('val_bin_auroc', np.nan) for r in fold_metric_records]))
    else:
        cv_acc, cv_qwk, cv_auc = met['oof'].get('acc'), met['oof'].get('qwk'), met['oof'].get('bin_auroc')

    row = {c: '' for c in SUMMARY_COLUMNS}
    row.update({
        'phase': phase, 'experiment': tag, 'description': cfg.get('description') or f'Experiment {tag}',
        'set_overrides': cfg.get('set_overrides', ''), 'inherit': str(cfg.get('inherit', '')),
        'protocol': protocol, 'ext_set': 'messidor_test' if 'messidor_test' in sets else '',
        'n_params_M': fmt2(n_params_M), 'gpu_hours': fmt2(elapsed_hours * max(1, n_gpus)),
        'cv_val_acc': fmt2(cv_acc), 'cv_val_qwk': fmt2(cv_qwk), 'cv_val_bin_auroc': fmt2(cv_auc),
        'oof_acc': fmt2(met['oof'].get('acc')), 'oof_qwk': fmt2(met['oof'].get('qwk')),
        'oof_bin_auroc': fmt2(met['oof'].get('bin_auroc')),
        'winner_eligible': 'true' if winner_eligible else 'false', 'notes': notes,
        'train_sources': ','.join(train_sources(cfg)), 'heldout_sets': ','.join(heldout_datasets(cfg)),
        'seed': cfg.get('seed', ''), 'head_type': cfg.get('head_type', 'softmax'),
    })
    if single:
        row['test_acc_single_mean'] = fmt2(np.mean([a for a, _ in single]))
        row['test_acc_single_std'] = fmt2(np.std([a for a, _ in single]))
        row['test_qwk_single_mean'] = fmt2(np.mean([q for _, q in single]))
        row['test_qwk_single_std'] = fmt2(np.std([q for _, q in single]))
    legacy = {'test': 'aptos_test', 'ext': 'messidor_test'}
    for prefix, setn in legacy.items():
        if setn in met:
            m = met[setn]
            for key in ['acc', 'qwk', 'prec_macro', 'rec_macro', 'f1_macro', 'adj_acc', 'mae', 'mild_recall',
                        'pred_nodr_share'] + [f'f1_c{c}' for c in range(5)] + [f'rec_c{c}' for c in range(5)] + \
                    ['bin_acc', 'bin_prec', 'bin_sens', 'bin_spec', 'bin_f1', 'bin_auroc', 'bin_auprc',
                     'sens_at_spec90']:
                col = f'{prefix}_{key}'
                if col in row:
                    row[col] = fmt2(m.get(key))
            row[f'ece_{prefix}'] = fmt2(m.get('ece'))
    if 'messidor_dev' in met:
        m = met['messidor_dev']
        row.update({'extdev_acc': fmt2(m.get('acc')), 'extdev_qwk': fmt2(m.get('qwk')),
                    'extdev_bin_auroc': fmt2(m.get('bin_auroc')), 'extdev_mild_recall': fmt2(m.get('mild_recall'))})
    for n, m in met.items():
        for key in SUMMARY_METRICS:
            row[f'{n}_{key}'] = fmt2(m.get(key)) if key != 'n' else m.get('n', '')
    for n, dm in devcal_met.items():
        for key in DEVCAL_METRICS:
            row[f'{n}_devcal_{key}'] = fmt2(dm.get(key))
    held = heldout_datasets(cfg)
    for split in ('dev', 'test'):
        names = [f'{d}_{split}' for d in held if f'{d}_{split}' in met]
        for key in EXT_MEAN_METRICS:
            vals = [met[n].get(key) for n in names if met[n].get(key) is not None]
            vals = [v for v in vals if not (isinstance(v, float) and np.isnan(v))]
            row[f'ext_mean_{split}_{key}'] = fmt2(float(np.mean(vals))) if vals else ''
        row[f'ext_mean_{split}_sets'] = ','.join(names)
    if keep_decision:
        row.update({k: v for k, v in keep_decision.items() if v is not None})
    pd.DataFrame([row]).to_csv(os.path.join(exp_dir, 'summary.csv'), index=False)

    # Figures
    from src.evaluation.experiment import plot_experiment_figures
    plot_experiment_figures(exp_dir, os.path.join(exp_dir, 'curves'), probs, sets_order=list(sets.keys()))

    log_run(tag, kind, {'oof_qwk': met['oof'].get('qwk'),
                        **{f'{n}_qwk': met[n].get('qwk') for n in sets}})
    msg = ' | '.join(f"{DISPLAY_NAMES.get(n, n)} QWK={fmt2(met[n].get('qwk'))}" for n in ['oof'] + list(sets))
    logger.info(f'Summary for {tag}: {msg}')
    return row


# ---------------------------------------------------------------------------
# evaluate / evaluate_all modes
# ---------------------------------------------------------------------------

def _load_fold_models(cfg, owner, n_folds):
    from src.models.gradenet import GradeNet
    models = []
    for k in range(n_folds):
        pth = os.path.join('checkpoints', owner, f'fold{k}.pth')
        if not os.path.isfile(pth):
            raise FileNotFoundError(f'missing checkpoint {pth}')
        m = GradeNet(dict(cfg, pretrained=False, pretrain_path=None))
        m.load_state_dict(torch.load(pth, map_location='cpu'))
        m.eval()
        models.append(m)
    return models


def _read_fold_records(exp_dir, ckpt_dirs, n_folds):
    recs = []
    for k in range(n_folds):
        for d in ckpt_dirs:
            p = os.path.join(d, f'fold{k}_val_metrics.json')
            if os.path.isfile(p):
                recs.append(json.load(open(p)))
                break
    if len(recs) == n_folds:
        return recs
    pf = os.path.join(exp_dir, 'per_fold.csv')
    if os.path.isfile(pf):
        out = []
        for _, r in pd.read_csv(pf).iterrows():
            if str(r.get('fold', '')).isdigit():
                out.append({k: float(r[k]) for k in ('val_acc', 'val_qwk', 'val_bin_auroc')
                            if k in r and str(r[k]) not in ('', 'nan')})
        return out
    return []


def run_evaluate(cfg):
    """Re-evaluate a finished experiment on every evaluation set without retraining.

    Reuses saved predictions that cover the current frozen splits, runs inference with the saved
    fold checkpoints for everything else (e.g. the new DDR / DeepDRiD / IDRiD sets), and rewrites all
    artefacts of results/<phase>/<tag>/ in place. The phase decision columns are kept.
    """
    from src.utils.config import find_exp, ckpt_owner, resolve
    from src.utils.logger import get_logger
    tag = cfg['tag']
    logger = get_logger(tag, 'evaluate')
    exp_dir = find_exp(tag)
    phase = os.path.basename(os.path.dirname(exp_dir))
    cfg = dict(cfg, phase=phase)
    check_eval_inputs(cfg)
    old = pd.read_csv(os.path.join(exp_dir, 'summary.csv')).iloc[0].to_dict() \
        if os.path.isfile(os.path.join(exp_dir, 'summary.csv')) else {}
    keep = {k: old.get(k) for k in ('decision', 'is_winner', 'description', 'gpu_hours', 'notes', 'protocol',
                                    'winner_eligible', 'set_overrides', 'inherit')}
    keep = {k: ('' if (isinstance(v, float) and np.isnan(v)) else v) for k, v in keep.items()}
    t0 = time.time()

    snap_path = os.path.join('checkpoints', tag, 'config_snapshot.yaml')
    snap = (yaml.safe_load(open(snap_path)) or {}) if os.path.isfile(snap_path) else {}
    # provenance of the run (how it was trained), not of this evaluation call
    for k in ('set_overrides', 'inherit'):
        if snap.get(k) is not None:
            cfg[k] = snap[k]
            keep[k] = snap[k]
        elif keep.get(k) not in (None, ''):
            cfg[k] = keep[k]
    if snap.get('members'):
        # build_cfg resets the CLI metadata keys, so ensemble membership is read from the snapshot
        return run_evaluate_ensemble(dict(cfg, members=snap['members']), exp_dir, phase, logger, keep)

    owner = ckpt_owner(tag)
    n_folds = int(cfg.get('n_folds', 5))
    thr_path = None
    for cand in (os.path.join('checkpoints', tag, 'thresholds.json'), os.path.join('checkpoints', owner, 'thresholds.json')):
        if os.path.isfile(cand):
            thr_path = cand
            break
    if thr_path is None:
        raise FileNotFoundError(f'{tag}: no thresholds.json in checkpoints/{tag} or checkpoints/{owner}')
    thresholds = json.load(open(thr_path))
    fitted_T = float(thresholds.get('temperature', 1.0))
    with np.load(os.path.join(exp_dir, 'oof.npz'), allow_pickle=True) as f:
        oof = {'ids': [str(i) for i in f['ids']], 'y': np.asarray(f['y'], dtype=int),
               'P': np.asarray(f['P'] if 'P' in f.files else f['y_prob'])}
    logger.info(f'=== evaluate {tag} (phase {phase}, checkpoints of {owner}) ===')
    sets = collect_sets(cfg, exp_dir, lambda: _load_fold_models(cfg, owner, n_folds), logger,
                        overwrite=bool(cfg.get('eval_overwrite', False)))
    try:
        from src.models.gradenet import GradeNet
        n_params = sum(p.numel() for p in GradeNet(dict(cfg, pretrained=False, pretrain_path=None)).parameters()) / 1e6
    except Exception:
        n_params = float(old.get('n_params_M') or 0.0)
    recs = _read_fold_records(exp_dir, [os.path.join('checkpoints', tag), os.path.join('checkpoints', owner)], n_folds)
    keep_gpu = keep.pop('gpu_hours', '')
    row = write_artifacts(cfg, tag, phase, exp_dir, os.path.join('checkpoints', tag), t0, n_params, recs, oof,
                          thresholds, fitted_T, sets, keep.get('protocol') or 'ensemble5_tta', logger,
                          winner_eligible=str(keep.get('winner_eligible', 'true')).lower() != 'false',
                          notes=keep.get('notes') or '', kind='evaluate', keep_decision=keep)
    if keep_gpu not in ('', None):
        s = pd.read_csv(os.path.join(exp_dir, 'summary.csv'))
        s['gpu_hours'] = keep_gpu
        s.to_csv(os.path.join(exp_dir, 'summary.csv'), index=False)
    return row


def run_evaluate_ensemble(cfg, exp_dir, phase, logger, keep):
    """Ensemble tags: average the members' raw probabilities on every set (members must be evaluated)."""
    from src.utils.config import find_exp, resolve
    from src.evaluation.metrics import compute_oof_thresholds, fit_temperature
    members = cfg['members']
    if isinstance(members, str):
        members = [resolve(m.strip()) for m in members.split(',') if m.strip()]
    t0 = time.time()
    member_sets, member_oof = [], []
    for m in members:
        m_dir = find_exp(m)
        recs = {}
        for name in eval_set_names(cfg):
            rec = load_saved_set(m_dir, name)
            rec = restrict_to_frame(rec, eval_frame(cfg, name)[0]) if rec is not None else None
            if rec is None:
                raise FileNotFoundError(f'ensemble member {m} has no predictions for {name}: run '
                                        f'--mode evaluate --tag {m} first')
            recs[name] = rec
        member_sets.append(recs)
        with np.load(os.path.join(m_dir, 'oof.npz'), allow_pickle=True) as f:
            member_oof.append({'ids': [str(i) for i in f['ids']], 'y': np.asarray(f['y'], dtype=int),
                               'P': np.asarray(f['P'] if 'P' in f.files else f['y_prob'])})
    for o in member_oof[1:]:
        assert o['ids'] == member_oof[0]['ids'], 'ensemble members have different OOF orderings'
    oof = {'ids': member_oof[0]['ids'], 'y': member_oof[0]['y'], 'P': np.mean([o['P'] for o in member_oof], axis=0)}
    sets = {}
    for name in eval_set_names(cfg):
        raws = [ms[name]['P_raw'] for ms in member_sets]
        sets[name] = {'ids': member_sets[0][name]['ids'], 'y': member_sets[0][name]['y'],
                      'P_raw': np.mean(raws, axis=0), 'P_folds': np.asarray(raws)}
    thresholds = compute_oof_thresholds(oof['y'], oof['P'])
    fitted_T = fit_temperature(np.log(np.clip(oof['P'], 1e-15, 1.0)), oof['y'])
    thresholds['temperature'] = float(fitted_T)
    ckpt_dir = os.path.join('checkpoints', cfg['tag'])
    os.makedirs(ckpt_dir, exist_ok=True)
    with open(os.path.join(ckpt_dir, 'thresholds.json'), 'w') as f_t:
        json.dump(thresholds, f_t, indent=2)
    return write_artifacts(cfg, cfg['tag'], phase, exp_dir, ckpt_dir, t0, 0.0, [], oof, thresholds, fitted_T, sets,
                           'ensemble_of_models', logger, winner_eligible=False,
                           notes=f"Ensemble of {','.join(members)}", kind='ensemble_experiment', keep_decision=keep)


def _all_experiment_dirs():
    import glob
    out = []
    for d in sorted(glob.glob(os.path.join('results', 'phase*', '*'))):
        if os.path.isdir(d) and os.path.isfile(os.path.join(d, 'summary.csv')):
            out.append(d)
    return out


def classify_tag(tag):
    """'zoo' | 'conformal' | 'ensemble' | 'model' | 'unknown' from the tag's config snapshot."""
    snap = os.path.join('checkpoints', tag, 'config_snapshot.yaml')
    if not os.path.isfile(snap):
        return 'unknown'
    data = yaml.safe_load(open(snap)) or {}
    if data.get('kind') == 'calibrate_conformal':
        return 'conformal'
    if data.get('members'):
        return 'ensemble'
    if data.get('conformal_alpha') is not None and data.get('ckpt_from') and \
            not os.path.isfile(os.path.join('checkpoints', tag, 'thresholds.json')):
        return 'conformal'
    return 'model'


def run_evaluate_all(cfg):
    """Evaluate every finished experiment on every evaluation set (models first, then ensembles and
    conformal tags). Every --set value of this call (e.g. external_sets, batch_size, num_workers,
    eval_overwrite) is applied to each tag. Writes results/master/evaluate_all_report.csv (rows of earlier
    calls for other tags are kept, so parallel calls on disjoint tag lists can share it); a failing tag
    is reported, never hidden."""
    import argparse
    import json as _json
    from src.utils.config import build_cfg, validate_cfg
    from src.evaluation.experiment import run_calibrate_conformal
    overrides = _json.loads(cfg.get('set_overrides') or '{}')
    overrides.pop('evaluate_only', None)
    rows = []
    order = {'model': 0, 'ensemble': 1, 'conformal': 2, 'unknown': 3, 'zoo': 4}
    items = []
    for d in _all_experiment_dirs():
        tag, phase = os.path.basename(d), os.path.basename(os.path.dirname(d))
        if phase == 'phase0_protocol':
            continue          # smoke-test leftovers live here; protocol artefacts are not experiments
        kind = 'zoo' if phase == 'phase1_zoo' else classify_tag(tag)
        items.append((order[kind], phase, tag, kind))
    only = list(cfg.get('evaluate_only') or [])
    unknown = sorted(set(only) - {t for _, _, t, _ in items})
    if unknown:
        raise ValueError(f'evaluate_only names tags without a results folder: {unknown}')
    for _, phase, tag, kind in sorted(items):
        if only and tag not in only:
            continue
        if kind == 'zoo':
            rows.append({'tag': tag, 'phase': phase, 'kind': kind, 'status': 'skipped',
                         'detail': 'imported zoo row (no checkpoints in this codebase); see table01 note'})
            continue
        if kind == 'unknown':
            rows.append({'tag': tag, 'phase': phase, 'kind': kind, 'status': 'skipped',
                         'detail': f'checkpoints/{tag}/config_snapshot.yaml missing (not a run of this codebase)'})
            print(f'evaluate_all: {tag}: skipped (no config snapshot)')
            continue
        args = argparse.Namespace(mode='evaluate', tag=tag, phase=phase, inherit=None, base=None, members=None,
                                  kind=None, method=None, set=[])
        t0 = time.time()
        try:
            sub = build_cfg(args)
            sub.update(overrides)
            validate_cfg(sub)
            if kind == 'conformal':
                sub['base'] = sub.get('base') or yaml.safe_load(open(os.path.join('checkpoints', tag,
                                                                                    'config_snapshot.yaml'))).get('base')
                run_calibrate_conformal(sub)
            else:
                run_evaluate(sub)
            rows.append({'tag': tag, 'phase': phase, 'kind': kind, 'status': 'ok',
                         'detail': f'{(time.time() - t0) / 60:.1f} min'})
            print(f'evaluate_all: {tag}: ok')
        except Exception as e:
            rows.append({'tag': tag, 'phase': phase, 'kind': kind, 'status': 'failed',
                         'detail': f'{type(e).__name__}: {e}'})
            print(f'evaluate_all: {tag}: FAILED ({type(e).__name__}: {e})')
    os.makedirs(os.path.join('results', 'master'), exist_ok=True)
    out = os.path.join('results', 'master', 'evaluate_all_report.csv')
    new = pd.DataFrame(rows, columns=['tag', 'phase', 'kind', 'status', 'detail'])
    if os.path.isfile(out):
        try:
            old = pd.read_csv(out)
            new = pd.concat([old[~old['tag'].isin(new['tag'])], new], ignore_index=True)
        except pd.errors.EmptyDataError:
            pass
    tmp = out + f'.{os.getpid()}.tmp'
    new.to_csv(tmp, index=False)
    os.replace(tmp, out)
    n_fail = sum(r['status'] == 'failed' for r in rows)
    print(f'evaluate_all: {len(rows)} tags, {n_fail} failed -> {out}')
    return rows
