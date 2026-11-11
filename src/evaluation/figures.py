"""Paper figures, drawn only from real results (S12 styling: all text bold, 300 dpi PNG + PDF).

Every figure reads saved artefacts (predictions_<set>.npz, curves/, metrics_long.csv, cutpoints.csv,
fewshot.csv, source_thresholds_fold*.json, phase tables, splits_summary.csv). A figure whose inputs
do not exist is skipped and listed with the reason in results/figures/FIGURES_README.txt; nothing is
simulated. The methods schematic is drawn by the authors, not generated.
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix

# S12 Color tokens
COLOR_TRAIN = '#1F77B4'
COLOR_VAL = '#D62728'
CLASS_COLORS = ['#0072B2', '#E69F00', '#009E73', '#D55E00', '#CC79A7']
CLASS_NAMES = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']
DOMAIN_COLORS = {'aptos': '#332288', 'eyepacs': '#117733', 'messidor': '#CC6677', 'ddr': '#88CCEE',
                 'deepdrid': '#DDCC77', 'idrid': '#AA4499'}
SET_COLORS = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00', '#56B4E9', '#000000']
NEUTRAL = '#7F7F7F'
ADOPT = '#0072B2'
REJECT_HATCH = '///'


from matplotlib.font_manager import weight_dict, FontProperties, findfont

_FONT_REPORTED = False


def weight_value(w) -> int:
    """Map string weights (e.g. 'bold', 'heavy') or numeric weights to integer values."""
    if isinstance(w, (int, float)):
        return int(w)
    if isinstance(w, str):
        w_lower = w.lower()
        if w_lower in weight_dict:
            return weight_dict[w_lower]
        try:
            return int(w)
        except ValueError:
            pass
    return 400


def set_bold_style():
    """Configure matplotlib with bold typography and S12 styling specifications.
    All text in every figure must be bold: titles, axis labels, tick labels,
    legends, annotations, colorbars, etc.
    """
    global _FONT_REPORTED
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Liberation Sans', 'DejaVu Sans'],
        'font.weight': 'bold',
        'axes.labelweight': 'bold',
        'axes.titleweight': 'bold',
        'figure.titleweight': 'bold',
        'font.size': 9,
        'axes.labelsize': 9,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'axes.linewidth': 1.2,
        'lines.linewidth': 1.5,
        'axes.grid': True,
        'grid.alpha': 0.25,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'savefig.dpi': 300,
    })
    if not _FONT_REPORTED:
        fp = FontProperties(family=['Arial', 'Liberation Sans', 'DejaVu Sans'], weight='bold')
        resolved = findfont(fp)
        font_name = os.path.basename(resolved)
        print(f"[set_bold_style] Matplotlib resolved bold font: {font_name} ({resolved})")
        _FONT_REPORTED = True


def save_fig(fig, path_stem: str):
    """The ONLY way figures are saved. Enforces bold font on every text artist."""
    stem_dir = os.path.dirname(path_stem)
    if stem_dir:
        os.makedirs(stem_dir, exist_ok=True)

    for t in fig.findobj(matplotlib.text.Text):
        t.set_fontweight('bold')
    for t in fig.findobj(matplotlib.text.Text):
        if t.get_text().strip():
            assert weight_value(t.get_fontweight()) >= 700, (path_stem, t.get_text())

    fig.savefig(path_stem + '.png', dpi=300, bbox_inches='tight')
    fig.savefig(path_stem + '.pdf', bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _exp(tag):
    from src.utils.config import find_exp, resolve
    return find_exp(resolve(tag))


def _pred(exp_dir, set_name):
    p = os.path.join(exp_dir, f'predictions_{set_name}.npz')
    if not os.path.isfile(p):
        raise FileNotFoundError(p)
    with np.load(p, allow_pickle=True) as f:
        return np.asarray(f['y_true'], dtype=int), np.asarray(f['y_prob'])


def _test_sets(exp_dir):
    names = []
    for p in sorted(glob.glob(os.path.join(exp_dir, 'predictions_*_test.npz'))):
        n = os.path.basename(p)[len('predictions_'):-len('.npz')]
        if n != 'external_test':
            names.append(n)
    order = ['aptos_test', 'messidor_test', 'ddr_test', 'deepdrid_test', 'idrid_test']
    return sorted(names, key=lambda n: order.index(n) if n in order else 99)


def _label(n):
    from src.data.external import DISPLAY_NAMES
    return DISPLAY_NAMES.get(n, n)


def _new_fig(w=7.2, h=3.0, ncols=1, nrows=1):
    set_bold_style()
    return plt.subplots(nrows, ncols, figsize=(w, h), tight_layout=True, squeeze=False)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_class_distributions(out_dir, cfg):
    p = os.path.join('results', 'phase0_protocol', 'splits_summary.csv')
    d = pd.read_csv(p)
    keep = [s for s in d['split'].unique() if not s.startswith('aptos_fold') and not s.endswith('_train')]
    d = d[d['split'].isin(keep)]
    piv = d.pivot_table(index='split', columns='class', values='count', aggfunc='sum').fillna(0)
    piv = piv.div(piv.sum(axis=1), axis=0) * 100.0
    fig, ax = _new_fig(7.2, 3.2)
    ax = ax[0, 0]
    x = np.arange(5)
    w = 0.8 / max(1, len(piv))
    for i, (name, row) in enumerate(piv.iterrows()):
        ax.bar(x + (i - (len(piv) - 1) / 2) * w, row.values, w, label=_label(name), color=SET_COLORS[i % 7],
               edgecolor='black', lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_ylabel('Proportion (%)')
    ax.legend(frameon=False, fontsize=6.5, ncol=2)
    save_fig(fig, os.path.join(out_dir, 'fig01_class_distributions'))


def fig_training_curves(out_dir, cfg):
    d = _exp(cfg['report_final'])
    files = sorted(glob.glob(os.path.join(d, 'curves', 'fold*.csv')))
    if not files:
        raise FileNotFoundError(os.path.join(d, 'curves'))
    dfs = [pd.read_csv(f) for f in files]
    fig, axes = _new_fig(7.2, 3.0, ncols=2)
    ax1, ax2 = axes[0]
    for df in dfs:
        ax1.plot(df['epoch'] + 1, df['train_loss'], color=COLOR_TRAIN, alpha=0.35)
        ax2.plot(df['epoch'] + 1, df['val_acc'], color=COLOR_VAL, alpha=0.35)
        if 'train_acc' in df.columns:
            ax2.plot(df['epoch'] + 1, df['train_acc'], color=COLOR_TRAIN, alpha=0.35)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.plot([], [], color=COLOR_TRAIN, label='Train (600-image subset)')
    ax2.plot([], [], color=COLOR_VAL, label='Validation fold')
    ax2.legend(frameon=False, fontsize=7)
    save_fig(fig, os.path.join(out_dir, 'fig02_training_curves'))


def fig_confusion(out_dir, cfg):
    d = _exp(cfg['report_final'])
    sets = _test_sets(d)
    fig, axes = _new_fig(2.4 * len(sets), 2.6, ncols=len(sets))
    for ax, n in zip(axes[0], sets):
        y, P = _pred(d, n)
        cm = confusion_matrix(y, P.argmax(1), labels=range(5))
        rs = cm.sum(axis=1, keepdims=True)
        ax.imshow(cm / np.maximum(rs, 1), cmap='Blues', vmin=0, vmax=1)
        for i in range(5):
            for j in range(5):
                ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=6.5,
                        color='white' if cm[i, j] / max(rs[i, 0], 1) > 0.5 else 'black')
        ax.set_xticks(range(5))
        ax.set_yticks(range(5))
        ax.set_xticklabels(['0', '1', '2', '3', '4'])
        ax.set_yticklabels(['0', '1', '2', '3', '4'])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title(_label(n), fontsize=8)
        ax.grid(False)
    save_fig(fig, os.path.join(out_dir, 'fig03_confusion_matrices'))


def fig_roc_pr(out_dir, cfg):
    d = _exp(cfg['report_final'])
    fig, axes = _new_fig(7.2, 3.2, ncols=2)
    ax1, ax2 = axes[0]
    for i, n in enumerate(_test_sets(d)):
        y, P = _pred(d, n)
        yb, s = (y >= 2).astype(int), P[:, 2:].sum(1)
        if len(np.unique(yb)) < 2:
            continue
        fpr, tpr, _ = roc_curve(yb, s)
        pr, rc, _ = precision_recall_curve(yb, s)
        ax1.plot(fpr, tpr, color=SET_COLORS[i % 7], label=_label(n))
        ax2.plot(rc, pr, color=SET_COLORS[i % 7], label=_label(n))
    ax1.plot([0, 1], [0, 1], ':', color='grey')
    ax1.set_xlabel('1 - Specificity')
    ax1.set_ylabel('Sensitivity')
    ax1.legend(frameon=False, fontsize=6.5, loc='lower right')
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.legend(frameon=False, fontsize=6.5, loc='lower left')
    save_fig(fig, os.path.join(out_dir, 'fig04_referable_roc_pr'))


def fig_reliability(out_dir, cfg):
    from src.evaluation.metrics import ece
    d = _exp(cfg['report_final'])
    sets = _test_sets(d)
    fig, axes = _new_fig(2.2 * len(sets), 2.4, ncols=len(sets))
    bins = np.linspace(0, 1, 11)
    centers = (bins[:-1] + bins[1:]) / 2
    for ax, n in zip(axes[0], sets):
        y, P = _pred(d, n)
        conf, corr = P.max(1), (P.argmax(1) == y).astype(float)
        acc = [corr[(conf > lo) & (conf <= hi)].mean() if ((conf > lo) & (conf <= hi)).any() else np.nan
               for lo, hi in zip(bins[:-1], bins[1:])]
        ax.bar(centers, np.nan_to_num(acc), width=0.08, color=COLOR_TRAIN, edgecolor='black')
        ax.plot([0, 1], [0, 1], '--', color='grey')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel('Confidence')
        ax.set_ylabel('Accuracy')
        ax.set_title(f'{_label(n)}  ECE {ece(P, y):.2f}', fontsize=7.5)
    save_fig(fig, os.path.join(out_dir, 'fig05_reliability'))


def fig_risk_coverage(out_dir, cfg):
    d = _exp(cfg['report_final'])
    fig, ax = _new_fig(3.5, 3.0)
    ax = ax[0, 0]
    for i, n in enumerate(_test_sets(d)):
        y, P = _pred(d, n)
        order = np.argsort(-P.max(1))
        corr = (P.argmax(1) == y)[order].astype(float)
        cov = np.arange(1, len(y) + 1) / len(y)
        ax.plot(cov, 1 - np.cumsum(corr) / np.arange(1, len(y) + 1), color=SET_COLORS[i % 7], label=_label(n))
    ax.set_xlabel('Coverage')
    ax.set_ylabel('Selective risk (error rate)')
    ax.legend(frameon=False, fontsize=6.5)
    save_fig(fig, os.path.join(out_dir, 'fig06_risk_coverage'))


def fig_undergrading(out_dir, cfg):
    tags = [t for t in (cfg.get('report_baseline'), cfg.get('report_final')) if t]
    dirs = [_exp(t) for t in tags]
    sets = [n for n in _test_sets(dirs[-1]) if n != 'aptos_test']
    if not sets:
        raise FileNotFoundError('no external test predictions')
    fig, axes = _new_fig(2.4 * len(sets), 2.8, ncols=len(sets))
    x = np.arange(5)
    for ax, n in zip(axes[0], sets):
        y, _ = _pred(dirs[-1], n)
        ax.bar(x - 0.3, np.bincount(y, minlength=5) / len(y) * 100, 0.3, color='#BBBBBB', edgecolor='black',
               label='True')
        for j, (t, d) in enumerate(zip(tags, dirs)):
            yy, P = _pred(d, n)
            ax.bar(x + 0.3 * j, np.bincount(P.argmax(1), minlength=5) / len(yy) * 100, 0.3,
                   color=SET_COLORS[j + 1], edgecolor='black', label=f'Pred {t}')
        ax.set_xticks(x)
        ax.set_xticklabels(['0', '1', '2', '3', '4'])
        ax.set_xlabel('Grade')
        ax.set_ylabel('Share of images (%)')
        ax.set_title(_label(n), fontsize=8)
    axes[0][0].legend(frameon=False, fontsize=6)
    save_fig(fig, os.path.join(out_dir, 'fig07_grade_distribution_shift'))


def fig_phase_forest(out_dir, cfg):
    p = os.path.join('results', 'master', 'master_deltas.csv')
    d = pd.read_csv(p)
    d = d[d.get('decision_original', d.get('decision')) != 'REFERENCE'] if 'decision_original' in d else d
    rows = []
    for _, r in d.iterrows():
        for key, lo, hi, lab in (('d_oof_qwk', 'ci_lo', 'ci_hi', 'APTOS OOF'),
                                 ('d_dev_mean_qwk', 'd_dev_mean_qwk_ci_lo', 'd_dev_mean_qwk_ci_hi', 'External DEV')):
            if key in r and pd.notna(pd.to_numeric(r[key], errors='coerce')):
                rows.append((f"{r['experiment']}", lab, float(r[key]),
                             float(pd.to_numeric(r.get(lo), errors='coerce')),
                             float(pd.to_numeric(r.get(hi), errors='coerce'))))
    if not rows:
        raise FileNotFoundError('no deltas in master_deltas.csv')
    exps = list(dict.fromkeys(r[0] for r in rows))
    fig, ax = _new_fig(7.2, max(2.5, 0.28 * len(exps) + 1))
    ax = ax[0, 0]
    for k, lab in enumerate(('APTOS OOF', 'External DEV')):
        sub = [r for r in rows if r[1] == lab]
        yy = [exps.index(r[0]) + (0.15 if k else -0.15) for r in sub]
        ax.errorbar([r[2] for r in sub], yy,
                    xerr=[[r[2] - r[3] for r in sub], [r[4] - r[2] for r in sub]],
                    fmt='o', color=SET_COLORS[k], label=lab, capsize=2, ms=3)
    ax.axvline(0, color='grey', ls='--')
    ax.set_yticks(range(len(exps)))
    ax.set_yticklabels(exps, fontsize=6.5)
    ax.set_xlabel('Delta QWK vs phase reference (95 % CI)')
    ax.legend(frameon=False, fontsize=7)
    save_fig(fig, os.path.join(out_dir, 'fig08_phase_forest'))


def fig_per_class(out_dir, cfg):
    d = _exp(cfg['report_final'])
    pc = pd.read_csv(os.path.join(d, 'per_class.csv'))
    pc = pc[pc['set'].str.endswith('_test')]
    sets = list(pc['set'].unique())
    fig, axes = _new_fig(7.2, 3.0, ncols=2)
    for ax, metric in zip(axes[0], ('recall', 'f1')):
        x = np.arange(5)
        w = 0.8 / max(1, len(sets))
        for i, n in enumerate(sets):
            v = pd.to_numeric(pc[pc['set'] == n].sort_values('class')[metric], errors='coerce').values
            ax.bar(x + (i - (len(sets) - 1) / 2) * w, v, w, color=SET_COLORS[i % 7], label=_label(n),
                   edgecolor='black', lw=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(CLASS_NAMES, rotation=30, ha='right', fontsize=7)
        ax.set_ylabel(metric.capitalize())
        ax.set_ylim(0, 1)
    axes[0][0].legend(frameon=False, fontsize=6)
    save_fig(fig, os.path.join(out_dir, 'fig09_per_class'))


def fig_fewshot(out_dir, cfg):
    tags = [t for t in (cfg.get('report_baseline'), cfg.get('report_final')) if t]
    frames = {}
    for t in tags:
        from src.evaluation.tables import read_csv_safe
        f = read_csv_safe(os.path.join(_exp(t), 'fewshot.csv'))
        if not f.empty:
            frames[t] = f
    if not frames:
        raise FileNotFoundError('fewshot.csv empty')
    datasets = list(dict.fromkeys(ds for f in frames.values() for ds in f['dataset']))
    fig, axes = _new_fig(2.4 * len(datasets), 2.8, ncols=len(datasets))
    for ax, ds in zip(axes[0], datasets):
        for j, (t, f) in enumerate(frames.items()):
            sub = f[(f['dataset'] == ds) & (f['kind'].isin(['fewshot', 'all']))].sort_values('n')
            base_row = f[(f['dataset'] == ds) & (f['kind'] == 'argmax')]
            if sub.empty:
                continue
            xs = list(sub['n'])
            ax.errorbar(xs, sub['qwk_mean'], yerr=sub['qwk_sd'], marker='o', ms=3, capsize=2,
                        color=SET_COLORS[j], label=t)
            if not base_row.empty:
                ax.axhline(float(base_row['qwk_mean'].iloc[0]), color=SET_COLORS[j], ls=':')
        ax.set_xscale('log')
        ax.set_xlabel('Labelled target images (n)')
        ax.set_ylabel('TEST QWK')
        ax.set_title(ds, fontsize=8)
    axes[0][0].legend(frameon=False, fontsize=6)
    save_fig(fig, os.path.join(out_dir, 'fig10_fewshot_cutpoints'))


DEFAULT_ORDINAL = 'p4j_coralsrc_shared'


def _ordinal_run(cfg):
    """(tag, exp_dir, note) of the coral_src run for the source cut-point figure.

    cfg['report_ordinal'] (default p4j_coralsrc_shared). Only when that run does not exist in this project
    (e.g. a reduced project) is report_final used instead, and the substitution is returned as a note for
    FIGURES_README.txt. A report_ordinal run that exists but is not a coral_src run is an error."""
    tag = cfg.get('report_ordinal', DEFAULT_ORDINAL)
    if tag:
        try:
            return tag, _exp(tag), None
        except FileNotFoundError:
            final = cfg.get('report_final')
            if not final:
                raise FileNotFoundError(f'report_ordinal run {tag!r} not found under results/phase*/')
            return final, _exp(final), f'report_ordinal run {tag!r} not found: report_final {final!r} used'
    final = cfg.get('report_final')
    if not final:
        raise FileNotFoundError('set --set report_ordinal=<coral_src tag>')
    return final, _exp(final), 'report_ordinal is null: report_final used'


def fig_source_thresholds(out_dir, cfg):
    tag, d, note = _ordinal_run(cfg)
    files = sorted(glob.glob(os.path.join(d, 'source_thresholds_fold*.json')))
    if not files:
        who = 'report_ordinal' if note is None else 'report_final'
        raise FileNotFoundError(f'source_thresholds_fold*.json ({who} {tag} is not a coral_src run)')
    data = [json.load(open(f)) for f in files]
    names = list(data[0].keys())
    fig, ax = _new_fig(3.5, 2.8)
    ax = ax[0, 0]
    for i, n in enumerate(names):
        arr = np.array([dd[n] for dd in data])
        m, sd = arr.mean(0), arr.std(0)
        ax.errorbar(np.arange(1, arr.shape[1] + 1) + 0.08 * (i - len(names) / 2), m, yerr=sd, marker='o', ms=3,
                    capsize=2, color=DOMAIN_COLORS.get(n, '#000000'), label=n)
    ax.set_xticks(range(1, len(data[0][names[0]]) + 1))
    ax.set_xticklabels([f'{k}|{k + 1}' for k in range(len(data[0][names[0]]))])
    ax.set_xlabel('Grade boundary')
    ax.set_ylabel('Cut-point on severity score')
    ax.legend(frameon=False, fontsize=6.5)
    save_fig(fig, os.path.join(out_dir, 'fig11_source_cutpoints'))
    return f'drawn from {tag}' + (f' ({note})' if note else ' (report_ordinal)')


def fig_label_ambiguity(out_dir, cfg):
    run = cfg.get('soft_label_run')
    files = sorted(glob.glob(os.path.join('softlabels', str(run), 'outer*.npz'))) if run else []
    if not files:
        raise FileNotFoundError('softlabels/<run>/outer*.npz (set soft_label_run=<run>)')
    alphas, ys = [], []
    for f in files:
        with np.load(f, allow_pickle=True) as z:
            alphas.append(np.asarray(z['alpha']))
            ys.append(np.asarray(z['y']))
    a, y = np.concatenate(alphas), np.concatenate(ys)
    fig, ax = _new_fig(3.5, 2.8)
    ax = ax[0, 0]
    ax.boxplot([a[y == c] for c in range(5)])
    ax.set_xticks(range(1, 6))
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_ylabel('Ambiguity alpha')
    ax.tick_params(axis='x', labelrotation=30)
    save_fig(fig, os.path.join(out_dir, 'fig12_label_ambiguity'))


# needs: False = no model; True = cfg report_final; 'ordinal' = cfg report_ordinal (report_final only when the
# report_ordinal run does not exist, see _ordinal_run)
PAPER_FIGURES = [
    ('fig01_class_distributions', fig_class_distributions, False),
    ('fig02_training_curves', fig_training_curves, True),
    ('fig03_confusion_matrices', fig_confusion, True),
    ('fig04_referable_roc_pr', fig_roc_pr, True),
    ('fig05_reliability', fig_reliability, True),
    ('fig06_risk_coverage', fig_risk_coverage, True),
    ('fig07_grade_distribution_shift', fig_undergrading, True),
    ('fig08_phase_forest', fig_phase_forest, False),
    ('fig09_per_class', fig_per_class, True),
    ('fig10_fewshot_cutpoints', fig_fewshot, True),
    ('fig11_source_cutpoints', fig_source_thresholds, 'ordinal'),
    ('fig12_label_ambiguity', fig_label_ambiguity, False),
]

# Figures written by the stand-alone tools into results/figures/ (never regenerated or deleted by --mode report):
# (glob pattern without extension, producing tool, command that writes it)
TOOL_FIGURES = [
    ('fig_recal_levels', 'tools/recalibration_suite.py', 'python tools/recalibration_suite.py --tags <tags>'),
    ('fig_recal_misspec', 'tools/recalibration_suite.py', 'python tools/recalibration_suite.py --tags <tags>'),
    ('fig_recal_operating_points', 'tools/recalibration_suite.py',
     'python tools/recalibration_suite.py --tags <tags>'),
    ('fig_recal_*', 'tools/recalibration_suite.py', 'python tools/recalibration_suite.py --tags <tags>'),
    ('fig_shift_decomposition', 'tools/shift_decomposition.py', 'python tools/shift_decomposition.py --tags <tags>'),
    ('fig_adaptation*', 'tools/adaptation_curve.py', 'python tools/adaptation_curve.py --groups <groups>'),
]
TOOL_FIG_DIR = os.path.join('results', 'figures')


def tool_figures_status(fig_dir=TOOL_FIG_DIR):
    """[(name, tool, 'present' | 'missing', detail)] for every figure the tools write. Wildcard entries list
    each matching file not already covered by an explicit entry (nothing listed when none exists, except for
    fig_adaptation*, which is reported missing when no file matches)."""
    out, seen = [], set()
    for pat, tool, cmd in TOOL_FIGURES:
        stems = {os.path.splitext(os.path.basename(p))[0] for e in ('png', 'pdf')
                 for p in glob.glob(os.path.join(fig_dir, f'{pat}.{e}'))}
        names = sorted(n for n in stems if n not in seen)
        if not names and pat == 'fig_recal_*':          # catch-all for further suite figures: nothing extra
            continue
        if not names:
            out.append((pat, tool, 'missing', f'run: {cmd}'))
            seen.add(pat)
            continue
        for n in names:
            seen.add(n)
            exts = [e for e in ('png', 'pdf') if os.path.isfile(os.path.join(fig_dir, f'{n}.{e}'))]
            out.append((n, tool, 'present', '+'.join(exts)))
    return out


def build_paper_figures(out_dir: str = 'results/figures', cfg: dict = None):
    """Draw every paper figure whose inputs exist; list skipped figures in FIGURES_README.txt.

    Figures that need a model read cfg['report_final'] (and cfg['report_baseline'] where two models are
    compared); the source cut-point figure reads the ordinal run cfg['report_ordinal']. The XAI grid is
    produced separately by --mode xai (results/figures/xai_grid.*). The recalibration / shift-decomposition /
    adaptation figures written by tools/*.py are only registered (present / missing) in FIGURES_README.txt.
    """
    cfg = cfg or {}
    os.makedirs(out_dir, exist_ok=True)
    for old in glob.glob(os.path.join(out_dir, 'fig[0-9]*.png')) + glob.glob(os.path.join(out_dir, 'fig[0-9]*.pdf')):
        os.remove(old)
    done, skipped, notes = [], [], []
    for name, fn, needs_model in PAPER_FIGURES:
        if needs_model == 'ordinal':
            if not (cfg.get('report_ordinal', DEFAULT_ORDINAL) or cfg.get('report_final')):
                skipped.append(f'{name}: set --set report_ordinal=<coral_src tag>')
                continue
        elif needs_model and not cfg.get('report_final'):
            skipped.append(f'{name}: set --set report_final=<tag> (and report_baseline=<tag>)')
            continue
        try:
            note = fn(out_dir, cfg)
            done.append(name)
            if isinstance(note, str) and note:
                notes.append(f'{name}: {note}')
        except (FileNotFoundError, KeyError, ValueError) as e:
            skipped.append(f'{name}: input missing ({e})')
        finally:
            plt.close('all')
    tools = tool_figures_status()
    with open(os.path.join(out_dir, 'FIGURES_README.txt'), 'w') as f:
        f.write('Paper figures drawn from real results only.\n\nWritten:\n')
        f.writelines(f'  {d}\n' for d in done)
        f.write('\nSkipped:\n')
        f.writelines(f'  {s}\n' for s in skipped)
        if notes:
            f.write('\nNotes:\n')
            f.writelines(f'  {s}\n' for s in notes)
        f.write(f'\nProduced by tools (in {TOOL_FIG_DIR}/; not regenerated by --mode report):\n')
        f.writelines(f'  {n}: produced by {t} ({st}; {det})\n' for n, t, st, det in tools)
    print(f"build_paper_figures: {len(done)} figures written, {len(skipped)} skipped, "
          f"{sum(st == 'present' for _, _, st, _ in tools)} tool figures present -> {out_dir}/FIGURES_README.txt")
    return done, skipped
