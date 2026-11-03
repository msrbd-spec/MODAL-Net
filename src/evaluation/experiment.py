"""Experiment execution: 5-fold training, inference-only variants, ensembles, conformal calibration.

All evaluation and artefact writing is delegated to src.evaluation.evaluate, so every mode reports
the same metrics on the same sets: pooled APTOS OOF, APTOS test, and the dev/test split of every
dataset in cfg['external_sets'] (Messidor-2, DDR, DeepDRiD, IDRiD by default).
"""

import os
import time
import json
import shutil

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import yaml
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix

from src.data.external import DISPLAY_NAMES, check_eval_inputs, eval_set_names, eval_frame
from src.models.gradenet import GradeNet
from src.training.trainer import train_fold
from src.evaluation.metrics import compute_oof_thresholds, ece, fit_temperature
from src.evaluation.evaluate import (collect_sets, write_artifacts, load_saved_set, restrict_to_frame,
                                     run_evaluate_ensemble)
from src.evaluation.conformal import (interval_scores, compute_qhat, predict_intervals,
                                      evaluate_conformal_predictions, plot_conformal_diagnostics)
from src.utils.config import resolve, find_exp, ckpt_owner
from src.utils.io_utils import log_run
from src.utils.logger import get_logger
from src.evaluation.figures import set_bold_style, save_fig

# S12 Style tokens
COLOR_TRAIN = '#1F77B4'
COLOR_VAL = '#D62728'
CLASS_COLORS = ['#0072B2', '#E69F00', '#009E73', '#D55E00', '#CC79A7']
SET_COLORS = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00', '#56B4E9', '#000000']
CLASS_NAMES = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']


def plot_experiment_figures(exp_dir: str, curves_dir: str, probs: dict, sets_order=None):
    """Per-experiment figures (S12 style, all text bold): training curves, a confusion matrix per set,
    referable ROC / PR over every test set, and a reliability diagram per test set.

    probs: {set_name: (y, P)} including 'oof'.
    """
    set_bold_style()
    fig_dir = os.path.join(exp_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    # 1. Training and validation curves (train blue, val red)
    curve_files = sorted(f for f in os.listdir(curves_dir) if f.endswith('.csv')) if os.path.isdir(curves_dir) else []
    curve_dfs = [pd.read_csv(os.path.join(curves_dir, f)) for f in curve_files if f.startswith('fold')]
    curve_dfs = [d for d in curve_dfs if {'epoch', 'train_loss', 'val_acc'} <= set(d.columns) and len(d)]
    if curve_dfs:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.0), tight_layout=True)
        max_ep = max(len(d) for d in curve_dfs)
        grids = {k: np.full((len(curve_dfs), max_ep), np.nan) for k in ('train_loss', 'val_acc', 'train_acc')}
        for idx, df in enumerate(curve_dfs):
            for k in grids:
                if k in df.columns:
                    grids[k][idx, :len(df)] = df[k]
            ax1.plot(df['epoch'] + 1, df['train_loss'], color=COLOR_TRAIN, alpha=0.3)
            ax2.plot(df['epoch'] + 1, df['val_acc'], color=COLOR_VAL, alpha=0.3)
        ep = np.arange(1, max_ep + 1)
        ax1.plot(ep, np.nanmean(grids['train_loss'], axis=0), color=COLOR_TRAIN, linewidth=2.0, label='Train loss')
        if not np.all(np.isnan(grids['train_acc'])):
            ax2.plot(ep, np.nanmean(grids['train_acc'], axis=0), color=COLOR_TRAIN, linewidth=2.0, label='Train acc')
        ax2.plot(ep, np.nanmean(grids['val_acc'], axis=0), color=COLOR_VAL, linewidth=2.0, label='Val acc')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend(loc='best', frameon=False)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend(loc='best', frameon=False)
        save_fig(fig, os.path.join(fig_dir, 'curves'))

    names = [n for n in (sets_order or list(probs)) if n in probs and n != 'oof']

    # 2. Confusion matrices
    for n in names:
        y, P = probs[n]
        if len(y) == 0:
            continue
        cm = confusion_matrix(y, P.argmax(axis=1), labels=range(5))
        fig, ax = plt.subplots(figsize=(3.5, 3.2), tight_layout=True)
        ax.imshow(cm, cmap='Blues', interpolation='nearest')
        for i in range(5):
            for j in range(5):
                color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
                ax.text(j, i, str(cm[i, j]), ha='center', va='center', color=color, fontsize=8, fontweight='bold')
        ax.set_xticks(range(5))
        ax.set_yticks(range(5))
        ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='right', fontsize=7.5)
        ax.set_yticklabels(CLASS_NAMES, fontsize=7.5)
        ax.set_xlabel('Predicted Grade')
        ax.set_ylabel('True Grade')
        ax.set_title(DISPLAY_NAMES.get(n, n))
        save_fig(fig, os.path.join(fig_dir, f'cm_{n}'))
        if n == 'aptos_test':
            shutil.copy2(os.path.join(fig_dir, f'cm_{n}.png'), os.path.join(fig_dir, 'cm_test.png'))
            shutil.copy2(os.path.join(fig_dir, f'cm_{n}.pdf'), os.path.join(fig_dir, 'cm_test.pdf'))
        if n == 'messidor_test':
            shutil.copy2(os.path.join(fig_dir, f'cm_{n}.png'), os.path.join(fig_dir, 'cm_external.png'))
            shutil.copy2(os.path.join(fig_dir, f'cm_{n}.pdf'), os.path.join(fig_dir, 'cm_external.pdf'))

    # 3. Referable ROC and PR over every test set
    test_names = [n for n in names if n.endswith('_test')]
    usable = [n for n in test_names if len(np.unique((probs[n][0] >= 2).astype(int))) > 1]
    if usable:
        fig_roc, ax_roc = plt.subplots(figsize=(3.5, 3.2), tight_layout=True)
        fig_pr, ax_pr = plt.subplots(figsize=(3.5, 3.2), tight_layout=True)
        for i, n in enumerate(usable):
            y, P = probs[n]
            yb, s = (y >= 2).astype(int), P[:, 2:].sum(axis=1)
            fpr, tpr, _ = roc_curve(yb, s)
            prec, rec, _ = precision_recall_curve(yb, s)
            c = SET_COLORS[i % len(SET_COLORS)]
            ax_roc.plot(fpr, tpr, color=c, label=DISPLAY_NAMES.get(n, n))
            ax_pr.plot(rec, prec, color=c, label=DISPLAY_NAMES.get(n, n))
        ax_roc.plot([0, 1], [0, 1], ':', color='grey')
        ax_roc.set_xlabel('False Positive Rate')
        ax_roc.set_ylabel('True Positive Rate')
        ax_roc.legend(loc='lower right', frameon=False, fontsize=6.5)
        ax_pr.set_xlabel('Recall')
        ax_pr.set_ylabel('Precision')
        ax_pr.legend(loc='lower left', fontsize=6.5, frameon=False)
        save_fig(fig_roc, os.path.join(fig_dir, 'roc'))
        save_fig(fig_pr, os.path.join(fig_dir, 'pr'))

    # 4. Reliability diagrams
    for n in test_names:
        y, P = probs[n]
        if len(y) == 0:
            continue
        fig, ax = plt.subplots(figsize=(3.5, 3.2), tight_layout=True)
        conf, correct = P.max(axis=1), (P.argmax(axis=1) == y).astype(float)
        bins = np.linspace(0, 1, 11)
        accs = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (conf > lo) & (conf <= hi)
            accs.append(correct[m].mean() if m.sum() > 0 else 0.0)
        centers = (bins[:-1] + bins[1:]) / 2.0
        ax.bar(centers, accs, width=0.08, color=COLOR_TRAIN, alpha=0.8, edgecolor='black', label='Outputs')
        ax.plot([0, 1], [0, 1], '--', color='grey', label='Ideal')
        ax.set_xlabel('Confidence')
        ax.set_ylabel('Accuracy')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.0])
        ax.set_title(DISPLAY_NAMES.get(n, n))
        ax.legend(title=f'ECE: {ece(P, y, bins=10):.2f}', loc='upper left', fontsize=7, frameon=False)
        save_fig(fig, os.path.join(fig_dir, f'reliability_{n}'))
        if n == 'aptos_test':
            shutil.copy2(os.path.join(fig_dir, f'reliability_{n}.png'), os.path.join(fig_dir, 'reliability.png'))
            shutil.copy2(os.path.join(fig_dir, f'reliability_{n}.pdf'), os.path.join(fig_dir, 'reliability.pdf'))


def _load_models(cfg, owner_tag, n_folds):
    models = []
    for k in range(n_folds):
        m = GradeNet(cfg)
        pth = os.path.join('checkpoints', owner_tag, f'fold{k}.pth')
        if not os.path.isfile(pth):
            raise FileNotFoundError(f'Required checkpoint {pth} not found')
        m.load_state_dict(torch.load(pth, map_location='cpu'))
        m.eval()
        models.append(m)
    return models


def run_experiment(cfg: dict):
    """Run full 5-fold cross-validation experiment, evaluation on every set, and artefact generation."""
    t0 = time.time()
    tag = cfg['tag']
    phase = cfg.get('phase', 'default_phase')
    logger = get_logger(tag, 'experiment')
    logger.info(f"=== Starting Experiment: tag={tag}, phase={phase} ===")

    # Fail before any GPU hour is spent if an evaluation split or an extra-source split is missing
    check_eval_inputs(cfg)

    ckpt_dir = os.path.join('checkpoints', tag)
    exp_dir = os.path.join('results', phase, tag)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(exp_dir, exist_ok=True)
    with open(os.path.join(ckpt_dir, 'config_snapshot.yaml'), 'w') as f_snap:
        yaml.safe_dump(cfg, f_snap)

    n_folds = int(cfg.get('n_folds', 5))
    fold_metric_records = [train_fold(cfg, k, tag, logger) for k in range(n_folds)]

    # Merge OOF predictions
    oof_ids, oof_y, oof_p = [], [], []
    for k in range(n_folds):
        f_oof_path = os.path.join(exp_dir, f'oof_fold{k}.npz')
        if not os.path.isfile(f_oof_path):
            raise FileNotFoundError(f"Missing OOF file for fold {k}: {f_oof_path}")
        with np.load(f_oof_path, allow_pickle=True) as f_oof:
            oof_ids.extend([str(i) for i in f_oof['ids']])
            oof_y.append(f_oof['y'])
            oof_p.append(f_oof['P'])
    oof = {'ids': oof_ids, 'y': np.concatenate(oof_y).astype(int), 'P': np.concatenate(oof_p)}

    thresholds = compute_oof_thresholds(oof['y'], oof['P'])
    fitted_T = fit_temperature(np.log(np.clip(oof['P'], 1e-15, 1.0)), oof['y'])
    thresholds['temperature'] = float(fitted_T)
    with open(os.path.join(ckpt_dir, 'thresholds.json'), 'w') as f_thr:
        json.dump(thresholds, f_thr, indent=2)
    logger.info(f"OOF thresholds: thr_spec90={thresholds['thr_spec90']:.4f}, "
                f"thr_spec95={thresholds['thr_spec95']:.4f}, temperature={fitted_T:.4f}")

    holder = {}

    def _loader():
        holder['models'] = _load_models(cfg, tag, n_folds)
        return holder['models']

    sets = collect_sets(cfg, exp_dir, _loader, logger, overwrite=True)
    n_params_M = sum(p.numel() for p in holder['models'][0].parameters()) / 1e6 if holder.get('models') else 0.0
    return write_artifacts(cfg, tag, phase, exp_dir, ckpt_dir, t0, n_params_M, fold_metric_records, oof,
                           thresholds, fitted_T, sets, 'ensemble5_tta', logger,
                           winner_eligible=bool(cfg.get('winner_eligible', True)), kind='experiment')


def run_inference_experiment(cfg: dict):
    """Inference-only experiment on the owner's fold checkpoints (TTA, temperature, recalibration)."""
    t0 = time.time()
    tag = cfg['tag']
    phase = cfg.get('phase', 'phase8_inference')
    logger = get_logger(tag, 'inference_experiment')
    logger.info(f"=== Starting Inference Experiment: tag={tag}, phase={phase} ===")
    check_eval_inputs(cfg)

    base = resolve(cfg.get('base') or cfg.get('inherit'))
    if not base:
        raise ValueError("inference_experiment requires --base or --inherit")
    owner = ckpt_owner(base)
    logger.info(f"Base tag: {base} -> Owner checkpoint tag: {owner}")

    n_folds = int(cfg.get('n_folds', 5))
    ckpt_dir = os.path.join('checkpoints', tag)
    exp_dir = os.path.join('results', phase, tag)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(exp_dir, exist_ok=True)
    for k in range(n_folds):
        pth = os.path.join('checkpoints', owner, f'fold{k}.pth')
        if not os.path.isfile(pth):
            raise FileNotFoundError(f"Required checkpoint {pth} not found for fold {k} of owner {owner}. "
                                    f"Ensure base experiment {base} has completed training.")

    snap_cfg = dict(cfg, ckpt_from=owner, base=base)
    with open(os.path.join(ckpt_dir, 'config_snapshot.yaml'), 'w') as f_snap:
        yaml.safe_dump(snap_cfg, f_snap)

    owner_exp_dir = find_exp(owner)
    with np.load(os.path.join(owner_exp_dir, 'oof.npz'), allow_pickle=True) as f:
        oof = {'ids': [str(i) for i in f['ids']], 'y': np.asarray(f['y'], dtype=int),
               'P': np.asarray(f['P'] if 'P' in f.files else f['y_prob'])}
    thr_path = os.path.join('checkpoints', owner, 'thresholds.json')
    if os.path.isfile(thr_path):
        thresholds = json.load(open(thr_path))
    else:
        thresholds = compute_oof_thresholds(oof['y'], oof['P'])
        thresholds['temperature'] = float(fit_temperature(np.log(np.clip(oof['P'], 1e-15, 1.0)), oof['y']))
    fitted_T = float(thresholds.get('temperature', 1.0))
    with open(os.path.join(ckpt_dir, 'thresholds.json'), 'w') as f_t:
        json.dump(thresholds, f_t, indent=2)

    owner_curves = os.path.join(owner_exp_dir, 'curves')
    if os.path.isdir(owner_curves):
        shutil.copytree(owner_curves, os.path.join(exp_dir, 'curves'), dirs_exist_ok=True)

    holder = {}

    def _loader():
        holder['models'] = _load_models(cfg, owner, n_folds)
        return holder['models']

    sets = collect_sets(cfg, exp_dir, _loader, logger, overwrite=True)
    n_params_M = sum(p.numel() for p in holder['models'][0].parameters()) / 1e6 if holder.get('models') else 0.0

    from src.evaluation.evaluate import _read_fold_records
    fold_metric_records = _read_fold_records(owner_exp_dir, [os.path.join('checkpoints', owner)], n_folds)
    winner_eligible = bool(cfg.get('winner_eligible', True)) and cfg.get('recalibrate') != 'devcut'
    cfg = dict(cfg, ckpt_from=owner, base=base)
    return write_artifacts(cfg, tag, phase, exp_dir, ckpt_dir, t0, n_params_M, fold_metric_records, oof,
                           thresholds, fitted_T, sets, 'ensemble5_tta', logger, winner_eligible=winner_eligible,
                           notes=f'Inference from {base}', kind='inference_experiment')


def run_ensemble_experiment(cfg: dict):
    """Ensemble of finished experiments: average of the members' raw probabilities on every set."""
    tag = cfg['tag']
    phase = cfg.get('phase', 'phase5_architecture')
    logger = get_logger(tag, 'ensemble_experiment')
    logger.info(f"=== Starting Ensemble Experiment: tag={tag}, phase={phase} ===")
    members_str = cfg.get('members')
    if not members_str:
        raise ValueError("ensemble_experiment requires --members with comma-separated tags")
    member_tags = [resolve(m.strip()) for m in str(members_str).split(',') if m.strip()]
    if len(member_tags) < 2:
        raise ValueError(f"ensemble_experiment requires at least 2 members, got: {member_tags}")
    check_eval_inputs(cfg)
    ckpt_dir = os.path.join('checkpoints', tag)
    exp_dir = os.path.join('results', phase, tag)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(exp_dir, exist_ok=True)
    snap_cfg = dict(cfg, winner_eligible=False, members=member_tags)
    with open(os.path.join(ckpt_dir, 'config_snapshot.yaml'), 'w') as f_snap:
        yaml.safe_dump(snap_cfg, f_snap)
    return run_evaluate_ensemble(dict(cfg, members=member_tags, winner_eligible=False), exp_dir, phase, logger,
                                 keep=None)


def run_calibrate_conformal(cfg: dict):
    """Split-conformal ordinal intervals calibrated on the pooled OOF, evaluated on every set of the base."""
    tag = cfg['tag']
    phase = cfg.get('phase') or 'phase8_inference'
    logger = get_logger(tag, 'calibrate_conformal')
    logger.info(f"=== Starting Calibrate Conformal: tag={tag}, phase={phase} ===")

    base = resolve(cfg.get('base') or cfg.get('inherit'))
    if not base or base == tag:
        snap = os.path.join('checkpoints', tag, 'config_snapshot.yaml')
        base = (yaml.safe_load(open(snap)) or {}).get('base') if os.path.isfile(snap) else None
    if not base:
        raise ValueError("calibrate_conformal requires --base or --inherit pointing to a completed experiment")
    base_dir = find_exp(base)
    try:
        exp_dir = find_exp(tag)
    except FileNotFoundError:
        exp_dir = os.path.join('results', phase, tag)
    phase = os.path.basename(os.path.dirname(exp_dir))
    ckpt_dir = os.path.join('checkpoints', tag)
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    logger.info(f"Calibrating from base experiment: {base} (at {base_dir})")

    oof_path = os.path.join(base_dir, 'oof.npz')
    if not os.path.isfile(oof_path):
        raise FileNotFoundError(f"Missing oof.npz in base directory: {oof_path}")
    with np.load(oof_path, allow_pickle=True) as f:
        oof_y = np.asarray(f['y'], dtype=int)
        oof_p = np.asarray(f['P'] if 'P' in f.files else f['y_prob'])
    shutil.copy2(oof_path, os.path.join(exp_dir, 'oof.npz'))

    alpha = float(cfg.get('conformal_alpha', 0.10))
    qhat = compute_qhat(interval_scores(oof_p, oof_y), alpha=alpha)
    logger.info(f"Computed conformal cutoff qhat={qhat:.4f} for alpha={alpha:.2f} (n_cal={len(oof_y)})")

    rows = []
    fig_dir = os.path.join(exp_dir, 'figures')
    for name in eval_set_names(cfg):
        p = os.path.join(base_dir, f'predictions_{name}.npz')
        if not os.path.isfile(p):
            raise FileNotFoundError(f'{p} missing: run --mode evaluate --tag {base} first')
        with np.load(p, allow_pickle=True) as f:
            y = np.asarray(f['y_true'], dtype=int)
            P = np.asarray(f['y_prob'])
        shutil.copy2(p, os.path.join(exp_dir, f'predictions_{name}.npz'))
        res = evaluate_conformal_predictions(P, y, qhat, alpha=alpha)
        rows.append({'dataset': name, **res})
        plot_conformal_diagnostics(predict_intervals(P, qhat), y, qhat, alpha, fig_dir, prefix=f'conformal_{name}')
    for legacy in ('predictions_test.npz', 'predictions_external_dev.npz', 'predictions_external_test.npz'):
        src = os.path.join(base_dir, legacy)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(exp_dir, legacy))
    pd.DataFrame(rows).to_csv(os.path.join(exp_dir, 'conformal.csv'), index=False)
    logger.info(f"Conformal metrics saved to {os.path.join(exp_dir, 'conformal.csv')}")

    base_curves = os.path.join(base_dir, 'curves')
    if os.path.isdir(base_curves):
        shutil.copytree(base_curves, os.path.join(exp_dir, 'curves'), dirs_exist_ok=True)
    for fname in ('metrics_long.csv', 'per_class.csv', 'binary.csv', 'calibration.csv', 'cutpoints.csv',
                  'fewshot.csv'):
        src = os.path.join(base_dir, fname)
        if os.path.isfile(src):
            d = pd.read_csv(src)
            if 'experiment' in d.columns:
                d['experiment'] = tag
                d['phase'] = phase
            d.to_csv(os.path.join(exp_dir, fname), index=False)
    base_summary_path = os.path.join(base_dir, 'summary.csv')
    if os.path.isfile(base_summary_path):
        sum_df = pd.read_csv(base_summary_path)
        old = os.path.join(exp_dir, 'summary.csv')
        decision = pd.read_csv(old).iloc[0].get('decision', '') if os.path.isfile(old) else ''
        sum_df['experiment'] = tag
        sum_df['phase'] = phase
        sum_df['description'] = cfg.get('description') or f'Conformal calibration on {base}'
        sum_df['winner_eligible'] = 'false'
        sum_df['decision'] = '' if pd.isna(decision) else decision
        sum_df['notes'] = f'conformal qhat={qhat:.4f}, alpha={alpha:.2f}'
        sum_df.to_csv(os.path.join(exp_dir, 'summary.csv'), index=False)

    snap_cfg = dict(cfg, ckpt_from=ckpt_owner(base), base=base, conformal_alpha=alpha, kind='calibrate_conformal')
    with open(os.path.join(ckpt_dir, 'config_snapshot.yaml'), 'w') as f_snap:
        yaml.safe_dump(snap_cfg, f_snap)
    with open(os.path.join(exp_dir, 'config.yaml'), 'w') as f_cfg:
        yaml.safe_dump(dict(cfg, kind='calibrate_conformal', base=base, ckpt_from=ckpt_owner(base)), f_cfg)
    log_run(tag, 'calibrate_conformal', {'qhat': qhat, 'alpha': alpha,
                                         **{f'{r["dataset"]}_coverage': r['empirical_coverage'] for r in rows}})
    by = {r['dataset']: r for r in rows}
    return by.get('aptos_test'), by.get('messidor_test')
