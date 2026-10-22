"""One-shot experiment execution, evaluation, metrics compilation, and figure generation."""

import os
import time
import json
import shutil
import yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
import torch
from torch.utils.data import DataLoader

from src.data.splits import (
    fold_frames,
    test_frame,
    messidor_frame,
    aptos_frames,
)
from src.data.datasets import DRDataset
from src.models.gradenet import GradeNet
from src.training.trainer import train_fold
from src.evaluation.predict import predict_probs
from src.evaluation.metrics import (
    metrics5,
    metrics_bin,
    compute_oof_thresholds,
    ece,
    nll,
    brier,
    fit_temperature,
    apply_temperature,
)
from src.evaluation.calibration import fit_devcut, apply_devcut
from src.evaluation.conformal import (
    interval_scores,
    compute_qhat,
    predict_intervals,
    evaluate_conformal_predictions,
    plot_conformal_diagnostics,
)
from src.utils.config import resolve, find_exp, ckpt_owner
from src.utils.io_utils import SUMMARY_COLUMNS, fmt2, log_run
from src.utils.logger import get_logger

# S12 Style tokens
COLOR_TRAIN = '#1F77B4'
COLOR_VAL = '#D62728'
CLASS_COLORS = ['#0072B2', '#E69F00', '#009E73', '#D55E00', '#CC79A7']
CLASS_NAMES = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']


from src.evaluation.figures import set_bold_style, save_fig


def _setup_plt():
    set_bold_style()


def plot_experiment_figures(exp_dir: str, curves_dir: str, y_test, P_test, y_ext, P_ext, n_folds: int):
    """Generate per-experiment figures according to Part 2 S12 styling."""
    set_bold_style()
    fig_dir = os.path.join(exp_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    # 1. Training and Validation Curves
    curve_files = [os.path.join(curves_dir, f'fold{k}.csv') for k in range(n_folds)]
    curve_dfs = [pd.read_csv(p) for p in curve_files if os.path.isfile(p)]
    if curve_dfs:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.0), tight_layout=True)
        for i, df in enumerate(curve_dfs):
            alpha = 0.3 if len(curve_dfs) > 1 else 0.8
            ax1.plot(df['epoch'] + 1, df['train_loss'], color=COLOR_TRAIN, alpha=alpha)
            ax2.plot(df['epoch'] + 1, df['val_acc'], color=COLOR_VAL, alpha=alpha)
            if 'train_acc' in df.columns:
                ax2.plot(df['epoch'] + 1, df['train_acc'], color=COLOR_TRAIN, alpha=alpha, linestyle='--')

        # Add mean curves if multiple folds
        if len(curve_dfs) > 1:
            max_ep = max(len(d) for d in curve_dfs)
            loss_grid = np.full((len(curve_dfs), max_ep), np.nan)
            vacc_grid = np.full((len(curve_dfs), max_ep), np.nan)
            tacc_grid = np.full((len(curve_dfs), max_ep), np.nan)
            for idx, df in enumerate(curve_dfs):
                loss_grid[idx, :len(df)] = df['train_loss']
                vacc_grid[idx, :len(df)] = df['val_acc']
                if 'train_acc' in df.columns:
                    tacc_grid[idx, :len(df)] = df['train_acc']

            ep_axis = np.arange(1, max_ep + 1)
            ax1.plot(ep_axis, np.nanmean(loss_grid, axis=0), color=COLOR_TRAIN, linewidth=2.0, label='Train Loss')
            ax2.plot(ep_axis, np.nanmean(tacc_grid, axis=0), color=COLOR_TRAIN, linestyle='--', linewidth=1.8, label='Train Acc')
            ax2.plot(ep_axis, np.nanmean(vacc_grid, axis=0), color=COLOR_VAL, linewidth=2.0, label='Val Acc')

        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend(loc='best', frameon=False)

        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend(loc='best', frameon=False)

        save_fig(fig, os.path.join(fig_dir, 'curves'))

    # 2. Confusion Matrices (Internal Test and External Messidor Test)
    def _plot_cm(y_true, y_prob, out_name):
        if len(y_true) == 0 or len(y_prob) == 0:
            return
        preds = y_prob.argmax(axis=1)
        cm = confusion_matrix(y_true, preds, labels=range(5))
        fig, ax = plt.subplots(figsize=(3.5, 3.2), tight_layout=True)
        im = ax.imshow(cm, cmap='Blues', interpolation='nearest')
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
        save_fig(fig, os.path.join(fig_dir, out_name))

    _plot_cm(y_test, P_test, 'cm_test')
    _plot_cm(y_ext, P_ext, 'cm_external')

    # 3. Referable DR ROC and PR curves
    y_test_bin = (y_test >= 2).astype(int) if len(y_test) > 0 else []
    s_test = P_test[:, 2:].sum(axis=1) if len(P_test) > 0 else []
    y_ext_bin = (y_ext >= 2).astype(int) if len(y_ext) > 0 else []
    s_ext = P_ext[:, 2:].sum(axis=1) if len(P_ext) > 0 else []

    if len(y_test_bin) > 0 and len(np.unique(y_test_bin)) > 1:
        # ROC
        fig_roc, ax_roc = plt.subplots(figsize=(3.5, 3.2), tight_layout=True)
        fpr_t, tpr_t, _ = roc_curve(y_test_bin, s_test)
        ax_roc.plot(fpr_t, tpr_t, color='#0072B2', label='Test')
        if len(y_ext_bin) > 0 and len(np.unique(y_ext_bin)) > 1:
            fpr_e, tpr_e, _ = roc_curve(y_ext_bin, s_ext)
            ax_roc.plot(fpr_e, tpr_e, color='#D55E00', linestyle='--', label='Messidor')
        ax_roc.plot([0, 1], [0, 1], ':', color='grey')
        ax_roc.set_xlabel('False Positive Rate')
        ax_roc.set_ylabel('True Positive Rate')
        ax_roc.legend(loc='lower right', frameon=False)
        save_fig(fig_roc, os.path.join(fig_dir, 'roc'))

        # PR
        fig_pr, ax_pr = plt.subplots(figsize=(3.5, 3.2), tight_layout=True)
        prec_t, rec_t, _ = precision_recall_curve(y_test_bin, s_test)
        ax_pr.plot(rec_t, prec_t, color='#0072B2', label='Test')
        if len(y_ext_bin) > 0 and len(np.unique(y_ext_bin)) > 1:
            prec_e, rec_e, _ = precision_recall_curve(y_ext_bin, s_ext)
            ax_pr.plot(rec_e, prec_e, color='#D55E00', linestyle='--', label='Messidor')
        ax_pr.set_xlabel('Recall')
        ax_pr.set_ylabel('Precision')
        ax_pr.legend(loc='lower left', fontsize=6.5, frameon=False)
        save_fig(fig_pr, os.path.join(fig_dir, 'pr'))

    # 4. Reliability Diagram
    if len(y_test) > 0 and len(P_test) > 0:
        fig, ax = plt.subplots(figsize=(3.5, 3.2), tight_layout=True)
        conf = P_test.max(axis=1)
        pred = P_test.argmax(axis=1)
        correct = (pred == y_test).astype(float)
        bins = np.linspace(0, 1, 11)

        accs = []
        confs = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (conf > lo) & (conf <= hi)
            if m.sum() > 0:
                accs.append(correct[m].mean())
                confs.append(conf[m].mean())
            else:
                accs.append(0.0)
                confs.append((lo + hi) / 2.0)

        centers = (bins[:-1] + bins[1:]) / 2.0
        ax.bar(centers, accs, width=0.08, color=COLOR_TRAIN, alpha=0.8, edgecolor='black', label='Outputs')
        ax.plot([0, 1], [0, 1], '--', color='grey', label='Ideal')
        ax.set_xlabel('Confidence')
        ax.set_ylabel('Accuracy')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.0])
        ece_val = ece(P_test, y_test, bins=10)
        ax.legend(title=f'ECE: {ece_val:.2f}', loc='upper left', fontsize=7, frameon=False)
        save_fig(fig, os.path.join(fig_dir, 'reliability'))


def _evaluate_and_save_artifacts(
    cfg: dict,
    tag: str,
    phase: str,
    exp_dir: str,
    ckpt_dir: str,
    t0: float,
    n_params_M: float,
    fold_metric_records: list,
    oof_ids: list,
    oof_y: np.ndarray,
    oof_p: np.ndarray,
    thresholds: dict,
    fitted_T: float,
    t_ids: list,
    t_y: np.ndarray,
    t_p: np.ndarray,
    t_p_single: list,
    dev_ids: list,
    dev_y: np.ndarray,
    dev_p: np.ndarray,
    dev_p_single: list,
    mtest_ids: list,
    mtest_y: np.ndarray,
    mtest_p: np.ndarray,
    mtest_p_single: list,
    protocol: str,
    logger,
    winner_eligible: bool = True,
    notes: str = '',
    kind: str = 'experiment',
):
    """Common evaluation, metric calculation, and output artifact persistence."""
    # Temperature scaling
    if cfg.get('temperature_scaling', True) and fitted_T is not None and fitted_T > 0:
        t_p_scaled = apply_temperature(t_p, fitted_T)
        dev_p_scaled = apply_temperature(dev_p, fitted_T)
        mtest_p_scaled = apply_temperature(mtest_p, fitted_T)
    else:
        t_p_scaled = t_p
        dev_p_scaled = dev_p
        mtest_p_scaled = mtest_p

    # Devcut recalibration
    if cfg.get('recalibrate') == 'devcut':
        a_opt, b_opt = fit_devcut(dev_y, dev_p_scaled)
        mtest_p_scaled = apply_devcut(mtest_p_scaled, a_opt, b_opt)
        winner_eligible = False

    # Save prediction files with full S5 keys (ids, y_true, y_pred, y_prob, y_prob_folds)
    folds_arr = np.array(t_p_single) if t_p_single is not None and len(t_p_single) > 0 else np.empty((0, len(t_y), 5))
    dev_folds_arr = np.array(dev_p_single) if dev_p_single is not None and len(dev_p_single) > 0 else np.empty((0, len(dev_y), 5))
    mtest_folds_arr = np.array(mtest_p_single) if mtest_p_single is not None and len(mtest_p_single) > 0 else np.empty((0, len(mtest_y), 5))

    np.savez(
        os.path.join(exp_dir, 'predictions_test.npz'),
        ids=np.array(t_ids),
        y_true=t_y,
        y_pred=t_p_scaled.argmax(axis=1),
        y_prob=t_p_scaled,
        y_prob_folds=folds_arr,
        # Backward compatibility aliases
        y=t_y,
        P=t_p_scaled,
        P_per_model=folds_arr,
    )

    np.savez(
        os.path.join(exp_dir, 'predictions_external_dev.npz'),
        ids=np.array(dev_ids),
        y_true=dev_y,
        y_pred=dev_p_scaled.argmax(axis=1),
        y_prob=dev_p_scaled,
        y_prob_folds=dev_folds_arr,
        y=dev_y,
        P=dev_p_scaled,
    )

    np.savez(
        os.path.join(exp_dir, 'predictions_external_test.npz'),
        ids=np.array(mtest_ids),
        y_true=mtest_y,
        y_pred=mtest_p_scaled.argmax(axis=1),
        y_prob=mtest_p_scaled,
        y_prob_folds=mtest_folds_arr,
        y=mtest_y,
        P=mtest_p_scaled,
    )

    np.savez(
        os.path.join(exp_dir, 'oof.npz'),
        ids=np.array(oof_ids),
        y_true=oof_y,
        y_pred=oof_p.argmax(axis=1),
        y_prob=oof_p,
        y=oof_y,
        P=oof_p,
    )

    # Compute metrics
    test_m5 = metrics5(t_y, t_p_scaled)
    test_mbin = metrics_bin(t_y, t_p_scaled, thresholds=thresholds)
    test_ece = ece(t_p_scaled, t_y)

    dev_m5 = metrics5(dev_y, dev_p_scaled)
    dev_mbin = metrics_bin(dev_y, dev_p_scaled, thresholds=thresholds)

    mtest_m5 = metrics5(mtest_y, mtest_p_scaled)
    mtest_mbin = metrics_bin(mtest_y, mtest_p_scaled, thresholds=thresholds)
    mtest_ece = ece(mtest_p_scaled, mtest_y)

    oof_m5 = metrics5(oof_y, oof_p)
    oof_mbin = metrics_bin(oof_y, oof_p, thresholds=thresholds)

    # Single-model test statistics (across folds or models)
    single_test_accs = []
    single_test_qwks = []
    if t_p_single is not None and len(t_p_single) > 0:
        for sm_p in t_p_single:
            sm_res = metrics5(t_y, sm_p)
            single_test_accs.append(sm_res['acc'])
            single_test_qwks.append(sm_res['qwk'])
        test_acc_single_mean = float(np.mean(single_test_accs))
        test_acc_single_std = float(np.std(single_test_accs))
        test_qwk_single_mean = float(np.mean(single_test_qwks))
        test_qwk_single_std = float(np.std(single_test_qwks))
    else:
        test_acc_single_mean = test_m5['acc']
        test_acc_single_std = 0.0
        test_qwk_single_mean = test_m5['qwk']
        test_qwk_single_std = 0.0

    # per_fold.csv
    per_fold_rows = []
    if fold_metric_records:
        for k_idx, rec in enumerate(fold_metric_records):
            per_fold_rows.append({
                'fold': k_idx,
                'val_acc': fmt2(rec.get('val_acc')),
                'val_qwk': fmt2(rec.get('val_qwk')),
                'val_bin_auroc': fmt2(rec.get('val_bin_auroc')),
                'test_acc': fmt2(single_test_accs[k_idx]) if k_idx < len(single_test_accs) else '',
                'test_qwk': fmt2(single_test_qwks[k_idx]) if k_idx < len(single_test_qwks) else '',
            })
    elif t_p_single is not None and len(t_p_single) > 0:
        for m_idx in range(len(t_p_single)):
            per_fold_rows.append({
                'fold': m_idx,
                'val_acc': '',
                'val_qwk': '',
                'val_bin_auroc': '',
                'test_acc': fmt2(single_test_accs[m_idx]),
                'test_qwk': fmt2(single_test_qwks[m_idx]),
            })
    pd.DataFrame(per_fold_rows).to_csv(os.path.join(exp_dir, 'per_fold.csv'), index=False)

    # per_class.csv
    per_class_rows = []
    for c in range(5):
        per_class_rows.append({
            'class': c,
            'class_name': CLASS_NAMES[c],
            'test_f1': fmt2(test_m5.get(f'f1_c{c}')),
            'test_rec': fmt2(test_m5.get(f'rec_c{c}')),
            'test_prec': fmt2(test_m5.get(f'prec_c{c}')),
            'ext_f1': fmt2(mtest_m5.get(f'f1_c{c}')),
            'ext_rec': fmt2(mtest_m5.get(f'rec_c{c}')),
            'ext_prec': fmt2(mtest_m5.get(f'prec_c{c}')),
        })
    pd.DataFrame(per_class_rows).to_csv(os.path.join(exp_dir, 'per_class.csv'), index=False)

    # binary.csv
    binary_row = {
        'test_bin_acc': fmt2(test_mbin['bin_acc']),
        'test_bin_prec': fmt2(test_mbin['bin_prec']),
        'test_bin_sens': fmt2(test_mbin['bin_sens']),
        'test_bin_spec': fmt2(test_mbin['bin_spec']),
        'test_bin_f1': fmt2(test_mbin['bin_f1']),
        'test_bin_auroc': fmt2(test_mbin['bin_auroc']),
        'test_bin_auprc': fmt2(test_mbin['bin_auprc']),
        'test_sens_at_spec90': fmt2(test_mbin['sens_at_spec90']),
        'ext_bin_acc': fmt2(mtest_mbin['bin_acc']),
        'ext_bin_prec': fmt2(mtest_mbin['bin_prec']),
        'ext_bin_sens': fmt2(mtest_mbin['bin_sens']),
        'ext_bin_spec': fmt2(mtest_mbin['bin_spec']),
        'ext_bin_f1': fmt2(mtest_mbin['bin_f1']),
        'ext_bin_auroc': fmt2(mtest_mbin['bin_auroc']),
        'ext_bin_auprc': fmt2(mtest_mbin['bin_auprc']),
        'ext_sens_at_spec90': fmt2(mtest_mbin['sens_at_spec90']),
    }
    pd.DataFrame([binary_row]).to_csv(os.path.join(exp_dir, 'binary.csv'), index=False)

    # confusion_test.csv and confusion_external.csv
    pd.DataFrame(confusion_matrix(t_y, t_p_scaled.argmax(axis=1), labels=range(5))).to_csv(
        os.path.join(exp_dir, 'confusion_test.csv'), index=False
    )
    pd.DataFrame(confusion_matrix(mtest_y, mtest_p_scaled.argmax(axis=1), labels=range(5))).to_csv(
        os.path.join(exp_dir, 'confusion_external.csv'), index=False
    )

    # calibration.csv
    cal_row = {
        'ece_test': fmt2(test_ece),
        'nll_test': fmt2(nll(t_p_scaled, t_y)),
        'brier_test': fmt2(brier(t_p_scaled, t_y)),
        'ece_ext': fmt2(mtest_ece),
        'nll_ext': fmt2(nll(mtest_p_scaled, mtest_y)),
        'brier_ext': fmt2(brier(mtest_p_scaled, mtest_y)),
        'temperature': fmt2(fitted_T),
    }
    pd.DataFrame([cal_row]).to_csv(os.path.join(exp_dir, 'calibration.csv'), index=False)

    # config.yaml in the experiment folder
    with open(os.path.join(exp_dir, 'config.yaml'), 'w') as f_cfg:
        yaml.safe_dump(cfg, f_cfg)

    # Wall-clock and GPU calculation
    elapsed_hours = (time.time() - t0) / 3600.0
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    gpu_hours = elapsed_hours * max(1, n_gpus)

    # CV val stats across folds
    if fold_metric_records:
        cv_val_acc = float(np.mean([r['val_acc'] for r in fold_metric_records]))
        cv_val_qwk = float(np.mean([r['val_qwk'] for r in fold_metric_records]))
        cv_val_auroc = float(np.mean([r['val_bin_auroc'] for r in fold_metric_records]))
    else:
        cv_val_acc = oof_m5['acc']
        cv_val_qwk = oof_m5['qwk']
        cv_val_auroc = oof_mbin['bin_auroc']

    # summary.csv
    sum_row = {c: '' for c in SUMMARY_COLUMNS}
    sum_row.update({
        'phase': phase,
        'experiment': tag,
        'description': cfg.get('description', f'Experiment {tag}'),
        'set_overrides': cfg.get('set_overrides', ''),
        'inherit': str(cfg.get('inherit', '')),
        'protocol': protocol,
        'ext_set': 'messidor_test',
        'n_params_M': fmt2(n_params_M),
        'gpu_hours': fmt2(gpu_hours),
        # CV / OOF
        'cv_val_acc': fmt2(cv_val_acc),
        'cv_val_qwk': fmt2(cv_val_qwk),
        'cv_val_bin_auroc': fmt2(cv_val_auroc),
        'oof_acc': fmt2(oof_m5['acc']),
        'oof_qwk': fmt2(oof_m5['qwk']),
        'oof_bin_auroc': fmt2(oof_mbin['bin_auroc']),
        # Test 5-class
        'test_acc': fmt2(test_m5['acc']),
        'test_qwk': fmt2(test_m5['qwk']),
        'test_prec_macro': fmt2(test_m5['prec_macro']),
        'test_rec_macro': fmt2(test_m5['rec_macro']),
        'test_f1_macro': fmt2(test_m5['f1_macro']),
        'test_adj_acc': fmt2(test_m5['adj_acc']),
        'test_mae': fmt2(test_m5['mae']),
        'test_f1_c0': fmt2(test_m5['f1_c0']),
        'test_f1_c1': fmt2(test_m5['f1_c1']),
        'test_f1_c2': fmt2(test_m5['f1_c2']),
        'test_f1_c3': fmt2(test_m5['f1_c3']),
        'test_f1_c4': fmt2(test_m5['f1_c4']),
        'test_rec_c0': fmt2(test_m5['rec_c0']),
        'test_rec_c1': fmt2(test_m5['rec_c1']),
        'test_rec_c2': fmt2(test_m5['rec_c2']),
        'test_rec_c3': fmt2(test_m5['rec_c3']),
        'test_rec_c4': fmt2(test_m5['rec_c4']),
        'test_acc_single_mean': fmt2(test_acc_single_mean),
        'test_acc_single_std': fmt2(test_acc_single_std),
        'test_qwk_single_mean': fmt2(test_qwk_single_mean),
        'test_qwk_single_std': fmt2(test_qwk_single_std),
        # Test binary
        'test_bin_acc': fmt2(test_mbin['bin_acc']),
        'test_bin_prec': fmt2(test_mbin['bin_prec']),
        'test_bin_sens': fmt2(test_mbin['bin_sens']),
        'test_bin_spec': fmt2(test_mbin['bin_spec']),
        'test_bin_f1': fmt2(test_mbin['bin_f1']),
        'test_bin_auroc': fmt2(test_mbin['bin_auroc']),
        'test_bin_auprc': fmt2(test_mbin['bin_auprc']),
        'test_sens_at_spec90': fmt2(test_mbin['sens_at_spec90']),
        # External dev
        'extdev_acc': fmt2(dev_m5['acc']),
        'extdev_qwk': fmt2(dev_m5['qwk']),
        'extdev_bin_auroc': fmt2(dev_mbin['bin_auroc']),
        'extdev_mild_recall': fmt2(dev_m5['mild_recall']),
        # External test
        'ext_acc': fmt2(mtest_m5['acc']),
        'ext_qwk': fmt2(mtest_m5['qwk']),
        'ext_prec_macro': fmt2(mtest_m5['prec_macro']),
        'ext_rec_macro': fmt2(mtest_m5['rec_macro']),
        'ext_f1_macro': fmt2(mtest_m5['f1_macro']),
        'ext_adj_acc': fmt2(mtest_m5['adj_acc']),
        'ext_mild_recall': fmt2(mtest_m5['mild_recall']),
        'ext_pred_nodr_share': fmt2(mtest_m5['pred_nodr_share']),
        'ext_bin_acc': fmt2(mtest_mbin['bin_acc']),
        'ext_bin_prec': fmt2(mtest_mbin['bin_prec']),
        'ext_bin_sens': fmt2(mtest_mbin['bin_sens']),
        'ext_bin_spec': fmt2(mtest_mbin['bin_spec']),
        'ext_bin_f1': fmt2(mtest_mbin['bin_f1']),
        'ext_bin_auroc': fmt2(mtest_mbin['bin_auroc']),
        'ext_bin_auprc': fmt2(mtest_mbin['bin_auprc']),
        'ext_sens_at_spec90': fmt2(mtest_mbin['sens_at_spec90']),
        # Calibration
        'ece_test': fmt2(test_ece),
        'ece_ext': fmt2(mtest_ece),
        # Decision metadata
        'decision': '',
        'is_winner': '',
        'winner_eligible': 'true' if winner_eligible else 'false',
        'notes': notes,
    })

    summary_df = pd.DataFrame([sum_row])
    summary_df.to_csv(os.path.join(exp_dir, 'summary.csv'), index=False)

    # Figures
    plot_experiment_figures(
        exp_dir,
        os.path.join(exp_dir, 'curves'),
        t_y,
        t_p_scaled,
        mtest_y,
        mtest_p_scaled,
        len(fold_metric_records) if fold_metric_records else 5,
    )

    # Legacy registry results/all_runs.npz
    log_run(tag, kind, {
        'test_qwk': test_m5['qwk'],
        'test_acc': test_m5['acc'],
        'ext_qwk': mtest_m5['qwk'],
        'oof_qwk': oof_m5['qwk'],
    })

    logger.info(
        f"Summary for {tag}: OOF QWK={sum_row['oof_qwk']} | Test QWK={sum_row['test_qwk']} | Ext QWK={sum_row['ext_qwk']}"
    )
    return sum_row


def run_experiment(cfg: dict):
    """Run full 5-fold cross-validation experiment, evaluation, and artefact generation."""
    t0 = time.time()
    tag = cfg['tag']
    phase = cfg.get('phase', 'default_phase')
    logger = get_logger(tag, 'experiment')

    logger.info(f"=== Starting Experiment: tag={tag}, phase={phase} ===")

    ckpt_dir = os.path.join('checkpoints', tag)
    exp_dir = os.path.join('results', phase, tag)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(exp_dir, exist_ok=True)

    # Snapshot current configuration
    with open(os.path.join(ckpt_dir, 'config_snapshot.yaml'), 'w') as f_snap:
        yaml.safe_dump(cfg, f_snap)

    n_folds = int(cfg.get('n_folds', 5))
    fold_metric_records = []

    # 1. Train folds (resume-aware)
    for k in range(n_folds):
        f_metrics = train_fold(cfg, k, tag, logger)
        fold_metric_records.append(f_metrics)

    # 2. Merge OOF predictions
    oof_ids = []
    oof_y = []
    oof_p = []
    for k in range(n_folds):
        f_oof_path = os.path.join(exp_dir, f'oof_fold{k}.npz')
        if not os.path.isfile(f_oof_path):
            raise FileNotFoundError(f"Missing OOF file for fold {k}: {f_oof_path}")
        with np.load(f_oof_path, allow_pickle=True) as f_oof:
            oof_ids.extend([str(i) for i in f_oof['ids']])
            oof_y.append(f_oof['y'])
            oof_p.append(f_oof['P'])

    oof_y = np.concatenate(oof_y, axis=0)
    oof_p = np.concatenate(oof_p, axis=0)

    # Compute OOF thresholds and optimal temperature
    thresholds = compute_oof_thresholds(oof_y, oof_p)
    fitted_T = fit_temperature(np.log(np.clip(oof_p, 1e-15, 1.0)), oof_y)
    thresholds['temperature'] = float(fitted_T)
    with open(os.path.join(ckpt_dir, 'thresholds.json'), 'w') as f_thr:
        json.dump(thresholds, f_thr, indent=2)

    logger.info(
        f"OOF fitted thresholds: thr_spec90={thresholds['thr_spec90']:.4f}, "
        f"thr_spec95={thresholds['thr_spec95']:.4f}, temperature={fitted_T:.4f}"
    )

    # 3. Load 5 fold models for ensemble inference
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    models = []
    for k in range(n_folds):
        m = GradeNet(cfg)
        pth = os.path.join(ckpt_dir, f'fold{k}.pth')
        m.load_state_dict(torch.load(pth, map_location='cpu'))
        m.eval()
        models.append(m)

    total_params = sum(p.numel() for p in models[0].parameters())
    n_params_M = total_params / 1e6

    # 4. Predict on APTOS Test Set
    test_df = test_frame(cfg)
    aptos_paths = cfg['paths']['aptos']
    test_ds = DRDataset(test_df, [aptos_paths['test_images']], cfg, train=False, domain=0)
    test_loader = DataLoader(
        test_ds, batch_size=int(cfg.get('batch_size', 16)), shuffle=False, num_workers=int(cfg.get('num_workers', 4))
    )
    t_ids, t_y, t_p, t_p_single = predict_probs(
        models, test_loader, cfg, tta=cfg.get('tta'), scales=cfg.get('tta_scales'), device=device
    )

    # 5. Predict on Messidor-2 DEV and TEST
    mess_img_dir = cfg['paths']['messidor']['images']

    dev_df = messidor_frame(cfg, 'dev')
    dev_ds = DRDataset(dev_df, [mess_img_dir], cfg, train=False, domain=2)
    dev_loader = DataLoader(
        dev_ds, batch_size=int(cfg.get('batch_size', 16)), shuffle=False, num_workers=int(cfg.get('num_workers', 4))
    )
    dev_ids, dev_y, dev_p, dev_p_single = predict_probs(
        models, dev_loader, cfg, tta=cfg.get('tta'), scales=cfg.get('tta_scales'), device=device
    )

    mtest_df = messidor_frame(cfg, 'test')
    mtest_ds = DRDataset(mtest_df, [mess_img_dir], cfg, train=False, domain=2)
    mtest_loader = DataLoader(
        mtest_ds, batch_size=int(cfg.get('batch_size', 16)), shuffle=False, num_workers=int(cfg.get('num_workers', 4))
    )
    mtest_ids, mtest_y, mtest_p, mtest_p_single = predict_probs(
        models, mtest_loader, cfg, tta=cfg.get('tta'), scales=cfg.get('tta_scales'), device=device
    )

    return _evaluate_and_save_artifacts(
        cfg=cfg,
        tag=tag,
        phase=phase,
        exp_dir=exp_dir,
        ckpt_dir=ckpt_dir,
        t0=t0,
        n_params_M=n_params_M,
        fold_metric_records=fold_metric_records,
        oof_ids=oof_ids,
        oof_y=oof_y,
        oof_p=oof_p,
        thresholds=thresholds,
        fitted_T=fitted_T,
        t_ids=t_ids,
        t_y=t_y,
        t_p=t_p,
        t_p_single=t_p_single,
        dev_ids=dev_ids,
        dev_y=dev_y,
        dev_p=dev_p,
        dev_p_single=dev_p_single,
        mtest_ids=mtest_ids,
        mtest_y=mtest_y,
        mtest_p=mtest_p,
        mtest_p_single=mtest_p_single,
        protocol='ensemble5_tta',
        logger=logger,
        winner_eligible=bool(cfg.get('winner_eligible', True)),
        kind='experiment',
    )


def run_inference_experiment(cfg: dict):
    """Run inference-only experiment without training (TTA, temperature scaling, recalibration)."""
    t0 = time.time()
    tag = cfg['tag']
    phase = cfg.get('phase', 'phase8_inference')
    logger = get_logger(tag, 'inference_experiment')

    logger.info(f"=== Starting Inference Experiment: tag={tag}, phase={phase} ===")

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

    # Check that owner has all fold checkpoints
    owner_ckpt_dir = os.path.join('checkpoints', owner)
    for k in range(n_folds):
        pth = os.path.join(owner_ckpt_dir, f'fold{k}.pth')
        if not os.path.isfile(pth):
            raise FileNotFoundError(
                f"Required checkpoint {pth} not found for fold {k} of owner {owner}. "
                f"Ensure base experiment {base} has completed training."
            )

    # Snapshot current configuration with ckpt_from
    snap_cfg = dict(cfg)
    snap_cfg['ckpt_from'] = owner
    snap_cfg['base'] = base
    with open(os.path.join(ckpt_dir, 'config_snapshot.yaml'), 'w') as f_snap:
        yaml.safe_dump(snap_cfg, f_snap)

    # Copy oof.npz and thresholds.json and curves from owner
    owner_exp_dir = find_exp(owner)
    owner_oof = os.path.join(owner_exp_dir, 'oof.npz')
    if os.path.isfile(owner_oof):
        shutil.copy2(owner_oof, os.path.join(exp_dir, 'oof.npz'))
    oof_data = np.load(os.path.join(exp_dir, 'oof.npz'), allow_pickle=True)
    oof_ids = [str(i) for i in oof_data['ids']]
    oof_y = oof_data['y']
    oof_p = oof_data['P'] if 'P' in oof_data else oof_data['y_prob']

    thr_path = os.path.join(owner_ckpt_dir, 'thresholds.json')
    if os.path.isfile(thr_path):
        with open(thr_path) as f_t:
            thresholds = json.load(f_t)
    else:
        thresholds = compute_oof_thresholds(oof_y, oof_p)
        thresholds['temperature'] = float(fit_temperature(np.log(np.clip(oof_p, 1e-15, 1.0)), oof_y))

    fitted_T = thresholds.get('temperature', 1.0)
    with open(os.path.join(ckpt_dir, 'thresholds.json'), 'w') as f_t:
        json.dump(thresholds, f_t, indent=2)

    owner_curves = os.path.join(owner_exp_dir, 'curves')
    if os.path.isdir(owner_curves):
        shutil.copytree(owner_curves, os.path.join(exp_dir, 'curves'), dirs_exist_ok=True)

    # Load 5 fold models
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    models = []
    for k in range(n_folds):
        m = GradeNet(cfg)
        pth = os.path.join(owner_ckpt_dir, f'fold{k}.pth')
        m.load_state_dict(torch.load(pth, map_location='cpu'))
        m.eval()
        models.append(m)

    total_params = sum(p.numel() for p in models[0].parameters())
    n_params_M = total_params / 1e6

    # Predict APTOS Test Set
    test_df = test_frame(cfg)
    aptos_paths = cfg['paths']['aptos']
    test_ds = DRDataset(test_df, [aptos_paths['test_images']], cfg, train=False, domain=0)
    test_loader = DataLoader(
        test_ds, batch_size=int(cfg.get('batch_size', 16)), shuffle=False, num_workers=int(cfg.get('num_workers', 4))
    )
    t_ids, t_y, t_p, t_p_single = predict_probs(
        models, test_loader, cfg, tta=cfg.get('tta'), scales=cfg.get('tta_scales'), device=device
    )

    # Predict Messidor-2 DEV and TEST
    mess_img_dir = cfg['paths']['messidor']['images']
    dev_df = messidor_frame(cfg, 'dev')
    dev_ds = DRDataset(dev_df, [mess_img_dir], cfg, train=False, domain=2)
    dev_loader = DataLoader(
        dev_ds, batch_size=int(cfg.get('batch_size', 16)), shuffle=False, num_workers=int(cfg.get('num_workers', 4))
    )
    dev_ids, dev_y, dev_p, dev_p_single = predict_probs(
        models, dev_loader, cfg, tta=cfg.get('tta'), scales=cfg.get('tta_scales'), device=device
    )

    mtest_df = messidor_frame(cfg, 'test')
    mtest_ds = DRDataset(mtest_df, [mess_img_dir], cfg, train=False, domain=2)
    mtest_loader = DataLoader(
        mtest_ds, batch_size=int(cfg.get('batch_size', 16)), shuffle=False, num_workers=int(cfg.get('num_workers', 4))
    )
    mtest_ids, mtest_y, mtest_p, mtest_p_single = predict_probs(
        models, mtest_loader, cfg, tta=cfg.get('tta'), scales=cfg.get('tta_scales'), device=device
    )

    # Read owner per_fold metrics if available
    fold_metric_records = []
    owner_per_fold = os.path.join(owner_exp_dir, 'per_fold.csv')
    if os.path.isfile(owner_per_fold):
        p_df = pd.read_csv(owner_per_fold)
        for _, r in p_df.iterrows():
            if str(r.get('fold', '')).isdigit():
                fold_metric_records.append({
                    'val_acc': float(r['val_acc']) if 'val_acc' in r and str(r['val_acc']) != '' else 0.0,
                    'val_qwk': float(r['val_qwk']) if 'val_qwk' in r and str(r['val_qwk']) != '' else 0.0,
                    'val_bin_auroc': float(r['val_bin_auroc']) if 'val_bin_auroc' in r and str(r['val_bin_auroc']) != '' else 0.0,
                })

    winner_eligible = bool(cfg.get('winner_eligible', True))
    if cfg.get('recalibrate') == 'devcut':
        winner_eligible = False

    return _evaluate_and_save_artifacts(
        cfg=cfg,
        tag=tag,
        phase=phase,
        exp_dir=exp_dir,
        ckpt_dir=ckpt_dir,
        t0=t0,
        n_params_M=n_params_M,
        fold_metric_records=fold_metric_records,
        oof_ids=oof_ids,
        oof_y=oof_y,
        oof_p=oof_p,
        thresholds=thresholds,
        fitted_T=fitted_T,
        t_ids=t_ids,
        t_y=t_y,
        t_p=t_p,
        t_p_single=t_p_single,
        dev_ids=dev_ids,
        dev_y=dev_y,
        dev_p=dev_p,
        dev_p_single=dev_p_single,
        mtest_ids=mtest_ids,
        mtest_y=mtest_y,
        mtest_p=mtest_p,
        mtest_p_single=mtest_p_single,
        protocol='ensemble5_tta',
        logger=logger,
        winner_eligible=winner_eligible,
        notes=f'Inference from {base}',
        kind='inference_experiment',
    )


def run_ensemble_experiment(cfg: dict):
    """Run ensemble of multiple models by averaging saved probabilities."""
    t0 = time.time()
    tag = cfg['tag']
    phase = cfg.get('phase', 'phase5_architecture')
    logger = get_logger(tag, 'ensemble_experiment')

    logger.info(f"=== Starting Ensemble Experiment: tag={tag}, phase={phase} ===")

    members_str = cfg.get('members')
    if not members_str:
        raise ValueError("ensemble_experiment requires --members with comma-separated tags")

    cfg = dict(cfg)
    cfg.setdefault('temperature_scaling', False)

    member_tags = [resolve(m.strip()) for m in str(members_str).split(',') if m.strip()]
    if len(member_tags) < 2:
        raise ValueError(f"ensemble_experiment requires at least 2 members, got: {member_tags}")

    logger.info(f"Ensemble members: {member_tags}")

    ckpt_dir = os.path.join('checkpoints', tag)
    exp_dir = os.path.join('results', phase, tag)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(exp_dir, exist_ok=True)

    # Snapshot current configuration with winner_eligible=false
    snap_cfg = dict(cfg)
    snap_cfg['winner_eligible'] = False
    snap_cfg['members'] = member_tags
    with open(os.path.join(ckpt_dir, 'config_snapshot.yaml'), 'w') as f_snap:
        yaml.safe_dump(snap_cfg, f_snap)

    # Load predictions from all members
    member_tests = []
    member_devs = []
    member_mtests = []
    member_oofs = []

    t_ids, t_y = None, None
    dev_ids, dev_y = None, None
    mtest_ids, mtest_y = None, None
    oof_ids, oof_y = None, None

    for m_tag in member_tags:
        m_dir = find_exp(m_tag)

        # Test
        t_f = np.load(os.path.join(m_dir, 'predictions_test.npz'), allow_pickle=True)
        t_ids = [str(i) for i in t_f['ids']]
        t_y = t_f['y_true'] if 'y_true' in t_f else t_f['y']
        member_tests.append(t_f['y_prob'] if 'y_prob' in t_f else t_f['P'])

        # Dev
        d_f = np.load(os.path.join(m_dir, 'predictions_external_dev.npz'), allow_pickle=True)
        dev_ids = [str(i) for i in d_f['ids']]
        dev_y = d_f['y_true'] if 'y_true' in d_f else d_f['y']
        member_devs.append(d_f['y_prob'] if 'y_prob' in d_f else d_f['P'])

        # Ext Test
        mt_f = np.load(os.path.join(m_dir, 'predictions_external_test.npz'), allow_pickle=True)
        mtest_ids = [str(i) for i in mt_f['ids']]
        mtest_y = mt_f['y_true'] if 'y_true' in mt_f else mt_f['y']
        member_mtests.append(mt_f['y_prob'] if 'y_prob' in mt_f else mt_f['P'])

        # OOF if available
        oof_path = os.path.join(m_dir, 'oof.npz')
        if os.path.isfile(oof_path):
            o_f = np.load(oof_path, allow_pickle=True)
            oof_ids = [str(i) for i in o_f['ids']]
            oof_y = o_f['y']
            member_oofs.append(o_f['P'] if 'P' in o_f else o_f['y_prob'])

    # Average probabilities
    t_p = np.mean(member_tests, axis=0)
    dev_p = np.mean(member_devs, axis=0)
    mtest_p = np.mean(member_mtests, axis=0)

    if member_oofs and len(member_oofs) == len(member_tags):
        oof_p = np.mean(member_oofs, axis=0)
    elif member_oofs:
        oof_p = member_oofs[0]
    else:
        oof_p = dev_p
        oof_y = dev_y
        oof_ids = dev_ids

    thresholds = compute_oof_thresholds(oof_y, oof_p)
    fitted_T = fit_temperature(np.log(np.clip(oof_p, 1e-15, 1.0)), oof_y)
    thresholds['temperature'] = float(fitted_T)
    with open(os.path.join(ckpt_dir, 'thresholds.json'), 'w') as f_t:
        json.dump(thresholds, f_t, indent=2)

    return _evaluate_and_save_artifacts(
        cfg=cfg,
        tag=tag,
        phase=phase,
        exp_dir=exp_dir,
        ckpt_dir=ckpt_dir,
        t0=t0,
        n_params_M=0.0,
        fold_metric_records=[],
        oof_ids=oof_ids,
        oof_y=oof_y,
        oof_p=oof_p,
        thresholds=thresholds,
        fitted_T=fitted_T,
        t_ids=t_ids,
        t_y=t_y,
        t_p=t_p,
        t_p_single=member_tests,
        dev_ids=dev_ids,
        dev_y=dev_y,
        dev_p=dev_p,
        dev_p_single=member_devs,
        mtest_ids=mtest_ids,
        mtest_y=mtest_y,
        mtest_p=mtest_p,
        mtest_p_single=member_mtests,
        protocol='ensemble_of_models',
        logger=logger,
        winner_eligible=False,
        notes=f"Ensemble of {','.join(member_tags)}",
        kind='ensemble_experiment',
    )


def run_calibrate_conformal(cfg: dict):
    """Calibrate split conformal prediction and selective classification metrics."""
    tag = cfg['tag']
    phase = cfg.get('phase', 'phase8_inference')
    logger = get_logger(tag, 'calibrate_conformal')
    logger.info(f"=== Starting Calibrate Conformal: tag={tag}, phase={phase} ===")

    exp_dir = os.path.join('results', phase, tag)
    ckpt_dir = os.path.join('checkpoints', tag)
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    base = resolve(cfg.get('base') or cfg.get('inherit') or cfg.get('tag'))
    if not base:
        raise ValueError("calibrate_conformal requires --base or --inherit pointing to a completed experiment")

    base_dir = find_exp(base)
    logger.info(f"Calibrating from base experiment: {base} (at {base_dir})")

    # Load base OOF
    oof_path = os.path.join(base_dir, 'oof.npz')
    if not os.path.isfile(oof_path):
        raise FileNotFoundError(f"Missing oof.npz in base directory: {oof_path}")
    oof_data = np.load(oof_path, allow_pickle=True)
    oof_y = oof_data['y']
    oof_p = oof_data['P'] if 'P' in oof_data else oof_data['y_prob']
    shutil.copy2(oof_path, os.path.join(exp_dir, 'oof.npz'))

    # Load base Test predictions
    test_path = os.path.join(base_dir, 'predictions_test.npz')
    if not os.path.isfile(test_path):
        raise FileNotFoundError(f"Missing predictions_test.npz in base directory: {test_path}")
    test_data = np.load(test_path, allow_pickle=True)
    t_y = test_data['y_true'] if 'y_true' in test_data else test_data['y']
    t_p = test_data['y_prob'] if 'y_prob' in test_data else test_data['P']
    shutil.copy2(test_path, os.path.join(exp_dir, 'predictions_test.npz'))

    # Load base External Test predictions
    ext_path = os.path.join(base_dir, 'predictions_external_test.npz')
    if not os.path.isfile(ext_path):
        raise FileNotFoundError(f"Missing predictions_external_test.npz in base directory: {ext_path}")
    ext_data = np.load(ext_path, allow_pickle=True)
    mtest_y = ext_data['y_true'] if 'y_true' in ext_data else ext_data['y']
    mtest_p = ext_data['y_prob'] if 'y_prob' in ext_data else ext_data['P']
    shutil.copy2(ext_path, os.path.join(exp_dir, 'predictions_external_test.npz'))

    # Copy dev predictions if available
    dev_path = os.path.join(base_dir, 'predictions_external_dev.npz')
    if os.path.isfile(dev_path):
        shutil.copy2(dev_path, os.path.join(exp_dir, 'predictions_external_dev.npz'))

    # Copy curves if present
    base_curves = os.path.join(base_dir, 'curves')
    if os.path.isdir(base_curves):
        shutil.copytree(base_curves, os.path.join(exp_dir, 'curves'), dirs_exist_ok=True)

    # Copy base summary and update metadata
    base_summary_path = os.path.join(base_dir, 'summary.csv')
    if os.path.isfile(base_summary_path):
        sum_df = pd.read_csv(base_summary_path)
        sum_df['experiment'] = tag
        sum_df['phase'] = phase
        sum_df['description'] = cfg.get('description', f'Conformal calibration on {base}')
        sum_df.to_csv(os.path.join(exp_dir, 'summary.csv'), index=False)

    # Conformal calibration on OOF
    alpha = float(cfg.get('conformal_alpha', 0.10))
    oof_scores = interval_scores(oof_p, oof_y)
    qhat = compute_qhat(oof_scores, alpha=alpha)
    logger.info(f"Computed conformal cutoff qhat={qhat:.4f} for alpha={alpha:.2f} (n_cal={len(oof_y)})")

    # Evaluate conformal on Test and External Test
    conf_test = evaluate_conformal_predictions(t_p, t_y, qhat, alpha=alpha)
    conf_ext = evaluate_conformal_predictions(mtest_p, mtest_y, qhat, alpha=alpha)

    # Save conformal.csv
    conformal_df = pd.DataFrame([
        {'dataset': 'test', **conf_test},
        {'dataset': 'external_test', **conf_ext},
    ])
    conformal_df.to_csv(os.path.join(exp_dir, 'conformal.csv'), index=False)
    logger.info(f"Conformal metrics saved to {os.path.join(exp_dir, 'conformal.csv')}")

    # Generate diagnostic figures (Fig 17)
    fig_dir = os.path.join(exp_dir, 'figures')
    t_intervals = predict_intervals(t_p, qhat)
    plot_conformal_diagnostics(t_intervals, t_y, qhat, alpha, fig_dir, prefix='conformal_test')
    m_intervals = predict_intervals(mtest_p, qhat)
    plot_conformal_diagnostics(m_intervals, mtest_y, qhat, alpha, fig_dir, prefix='conformal_external')

    # Save snapshots
    snap_cfg = dict(cfg)
    snap_cfg['ckpt_from'] = ckpt_owner(base)
    snap_cfg['base'] = base
    snap_cfg['conformal_alpha'] = alpha
    with open(os.path.join(ckpt_dir, 'config_snapshot.yaml'), 'w') as f_snap:
        yaml.safe_dump(snap_cfg, f_snap)
    with open(os.path.join(exp_dir, 'config.yaml'), 'w') as f_cfg:
        yaml.safe_dump(cfg, f_cfg)

    log_run(tag, 'calibrate_conformal', {
        'qhat': qhat,
        'alpha': alpha,
        'test_coverage': conf_test['empirical_coverage'],
        'ext_coverage': conf_ext['empirical_coverage'],
    })
    return conf_test, conf_ext
