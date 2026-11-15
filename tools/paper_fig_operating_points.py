"""Paper figure: referable ROC curve of one model at each held-out site, with the operating points of the
main grading rules (no experiment code names in the figure).

Reads
  results/<phase>/<tag>/predictions_<site>_test.npz   (or the legacy Messidor-2 name)
  results/master/recal/recal_summary.csv               (bin_sens_mean / bin_spec_mean per site and method)
Writes
  <out>.png and <out>.pdf (default results/figures/paper_fig07_operating_points)

The ROC curve uses P(grade >= 2) = p2 + p3 + p4 of the stored (temperature-scaled) probabilities, the same score
as the referable AUROC of the report. Operating points come from recal_summary.csv, so they match Table 14.

Usage
  env/bin/python tools/paper_fig_operating_points.py --exp_dir results/phase3b_joint/p3b_joint --tag p3b_joint
"""
import argparse
import os

import numpy as np

LEGACY = {('messidor', 'test'): 'predictions_external_test.npz'}
SITES = (('messidor', 'Messidor-2'), ('ddr', 'DDR'), ('deepdrid', 'DeepDRiD'), ('idrid', 'IDRiD'))
RULES = (('argmax', 'Argmax', 'X', '#0072B2'),
         ('mlls_bcts_quantile', 'MLLS, quantile rule', 'v', '#D55E00'),
         ('oracle_quantile', 'True prevalence, quantile rule', 's', '#009E73'),
         ('all_dev_cuts', 'DEV cut-points', 'o', '#CC79A7'))


def load_scores(exp_dir, site):
    for name in (f'predictions_{site}_test.npz', LEGACY.get((site, 'test'))):
        if name and os.path.isfile(os.path.join(exp_dir, name)):
            with np.load(os.path.join(exp_dir, name), allow_pickle=True) as f:
                y = np.asarray(f['y_true'] if 'y_true' in f.files else f['y'], dtype=int)
                P = np.asarray(f['y_prob'] if 'y_prob' in f.files else f['P'], dtype=float)
            return (y >= 2).astype(int), P[:, 2:].sum(1)
    return None


def roc_curve(yb, s):
    """(fpr, tpr, auc) with ties handled by grouping equal scores."""
    order = np.argsort(-s, kind='mergesort')
    s, yb = s[order], yb[order]
    distinct = np.r_[np.where(np.diff(s))[0], len(s) - 1]
    tps = np.cumsum(yb)[distinct]
    fps = (distinct + 1) - tps
    P, N = yb.sum(), len(yb) - yb.sum()
    tpr = np.r_[0.0, tps / max(P, 1)]
    fpr = np.r_[0.0, fps / max(N, 1)]
    trapz = getattr(np, 'trapezoid', None) or np.trapz
    return fpr, tpr, float(trapz(tpr, fpr))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--exp_dir', required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--summary', default=os.path.join('results', 'master', 'recal', 'recal_summary.csv'))
    ap.add_argument('--out', default=os.path.join('results', 'figures', 'paper_fig07_operating_points'))
    a = ap.parse_args(argv)

    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False})
    summ = pd.read_csv(a.summary)
    summ = summ[summ['tag'] == a.tag]
    sites = [(k, n) for k, n in SITES if load_scores(a.exp_dir, k) is not None]
    if not sites:
        raise SystemExit(f'no prediction files found in {a.exp_dir}')
    fig, axes = plt.subplots(1, len(sites), figsize=(1.85 * len(sites) + 0.4, 2.6), squeeze=False)
    for ax, (site, name) in zip(axes[0], sites):
        yb, s = load_scores(a.exp_dir, site)
        fpr, tpr, auc = roc_curve(yb, s)
        ax.plot(fpr, tpr, color='black', lw=1.2)
        for m, lab, mk, col in RULES:
            r = summ[(summ['dataset'] == site) & (summ['method'] == m)]
            if len(r) and np.isfinite(r['bin_sens_mean'].iloc[0]):
                ax.scatter(1 - r['bin_spec_mean'].iloc[0], r['bin_sens_mean'].iloc[0], marker=mk, s=42,
                           color=col, edgecolor='black', linewidth=0.4, zorder=3, label=lab)
        ax.set_xlim(-0.01, 0.5)
        ax.set_ylim(0, 1.01)
        ax.set_title(f'{name} (AUROC {auc:.2f})', fontsize=9.5)
        ax.set_xlabel('1 - specificity', fontsize=9.5)
        ax.tick_params(labelsize=8.5)
        ax.grid(True, color='#DDDDDD', lw=0.5)
    axes[0][0].set_ylabel('Sensitivity', fontsize=9.5)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False, fontsize=9)
    fig.tight_layout()
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(f'{a.out}.{ext}', dpi=300, bbox_inches='tight')
    print('paper_fig_operating_points: written', a.out + '.png/.pdf')


if __name__ == '__main__':
    main()
