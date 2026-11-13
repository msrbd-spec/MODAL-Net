"""Label-free detector of when unlabelled prior (label-shift) correction is unsafe (CPU only, exploratory).

Motivation (step R results, 2026-11-13): annotation-free prior estimation (MLLS+BCTS, BBSE) followed by
quantile cut-points LOWERED external QWK, while the shift decomposition showed that MLLS works under a pure
prior shift (scenario A) and fails under acquisition / annotation shift (scenarios B, C). MLLS and BBSE rest on
the same label-shift assumption p_t(x|y) = p_s(x|y) but use the classifier differently (EM on calibrated
posteriors vs. a confusion-matrix inversion), so when that assumption holds both converge to the same target
prior, and when it is violated they are expected to disagree. The disagreement needs no target label.

THIS ANALYSIS IS POST-HOC / EXPLORATORY (it was not in the pre-declared step-R rules); report it as such.

Pre-declared reading for THIS analysis (fixed before running it on the real predictions):
  primary score   D_mb = TV(pi_MLLS, pi_BBSE) on the unlabelled DEV split of a scenario draw
  primary label   unsafe = QWK(mlls_bcts_quantile) - QWK(argmax) < -HARM_TOL on that draw's TEST split
  primary metric  AUROC of D_mb for `unsafe`, tag ALL, stratum C (observed site splits + their stratified
                  bootstrap draws only). Strata 'site_images' (B+C) and 'all' (A0/A/B/C) are secondary: they mix
                  APTOS-OOF single-fold predictions (A0/A) with site 5-fold+TTA ensembles (B/C), so any score that
                  separates "APTOS vs site images" scores high there by construction.
  secondary label unsafe_vs_oracle = QWK(mlls_bcts_quantile) - QWK(oracle_quantile) < -HARM_TOL: harm due to the
                  prior ESTIMATE only (oracle_quantile uses the DEV label shares; outcome side only).
  practical test  gated policy: use the correction when D <= tau, else keep argmax; tau chosen
                  leave-one-site-out (other held-out sites only; fitted on all their scenarios, and as a
                  sensitivity check on their C rows only) to maximise the mean gated QWK.
                  Claim "safe gating" only if, on the C rows of the held-out sites (MEAN over sites), the lower
                  2.5 % point of the per-draw (gated - argmax) spread is >= -0.005.
  Only the primary row is confirmatory; every other score x label x stratum row is descriptive (multiplicity).
  Spearman of a score with the harm is expected to be NEGATIVE (more disagreement -> more harm).
Comparators (also label-free): D_ms = TV(pi_MLLS, pi_S) (how far MLLS moves), D_bs = TV(pi_BBSE, pi_S),
D_conf = |mean max-probability on DEV - on the source half S| (confidence-drop baseline).

Scenarios, source half S, pool T and every resampling are EXACTLY those of tools/shift_decomposition.py (same
functions, same seeds): A0 no shift, A pure prior shift (APTOS images), B site images at the source shares,
C observed site splits (draw -1 = the observed splits, draws 0..R-1 = stratified bootstrap). The detector
only receives the unlabelled DEV probabilities of a draw; TEST labels are used only to score the outcome.

Outputs (--out_dir, default results/master/recal):
  shift_detector_draws.csv    tag, dataset, scenario, draw, n_dev, n_test, scores (D_mb, D_ms, D_bs, D_conf),
                              estimated priors, tv_true_dev (DEV label shares vs pi_S; descriptive only),
                              qwk_<method> on TEST for argmax / mlls_bcts_quantile / mlls_bcts_bayes /
                              bbse_quantile, harm_<correction>, unsafe_<correction>
  shift_detector_summary.csv  tag (each tag and ALL), stratum (C, site_images, all), score, label (unsafe_mlls,
                              unsafe_bbse, unsafe_vs_oracle_mlls, site_images), n, n_pos, n_clusters, auroc with a
                              95 % bootstrap interval resampling CLUSTERS (held-out dataset; A0 = one shared cluster)
                              then draws within them, n_boot_finite, spearman with the harm
  shift_detector_gating.csv   tag, correction, score, heldout dataset (and MEAN), tau (fitted without that site),
                              share of draws gated on, QWK of argmax / always-correct / gated on the observed
                              splits (draw -1) and over the C draws, gated - argmax with the 2.5 / 97.5 % points
                              of its per-draw SPREAD (not a CI of the mean; sites are treated as fixed),
                              tau_fit_on (all_scenarios | C_only), n_tags per site
  shift_detector_notes.csv    everything skipped and why
  results/figures/fig_shift_detector.{png,pdf}

Usage:
  python tools/shift_detector.py --tags p3b_joint,p10d_ce,p10q_ce,p4j_coralsrc_shared [--R 50] [--B 1000]
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation import label_shift as ls  # noqa: E402
from src.evaluation.cutpoints import apply_cutpoints, expected_grade, prevalence_cutpoints  # noqa: E402
from src.utils.config import find_exp  # noqa: E402
from tools import shift_decomposition as sd  # noqa: E402
from tools.adaptation_curve import _heldout, _load  # noqa: E402

K = 5
HARM_TOL = 0.02
SCORES = ('D_mb', 'D_ms', 'D_bs', 'D_conf')
PRIMARY_SCORE = 'D_mb'
CORRECTIONS = {'mlls': 'mlls_bcts_quantile', 'bbse': 'bbse_quantile'}
QWK_METHODS = ('argmax', 'mlls_bcts_quantile', 'mlls_bcts_bayes', 'bbse_quantile', 'oracle_quantile')
SITE_IMAGES = ('B', 'C')                      # scenarios built from the site's own images (by construction)
STRATA = {'C': ('C',), 'site_images': SITE_IMAGES, 'all': ('A0', 'A', 'B', 'C')}
PRIMARY_STRATUM = 'C'
LABELS = {'unsafe_mlls': 'harm_mlls', 'unsafe_bbse': 'harm_bbse',
          'unsafe_vs_oracle_mlls': 'harm_vs_oracle_mlls', 'site_images': 'harm_mlls'}
CAVEAT = ('post-hoc exploratory analysis (not pre-registered); A0/A/B are semi-synthetic resamplings of saved '
          'predictions (tools/shift_decomposition.py); A0/A rows are APTOS-OOF single-fold predictions while B/C '
          'are site 5-fold+TTA ensembles, so strata mixing them are confounded (primary = stratum C); the same '
          'images recur across tags and scenarios (clusters are not independent); only 3-4 held-out sites, so the '
          'effective n is the number of sites; tau is fitted on labelled outcomes of OTHER external sites '
          '(label-free only at the target site); site_images is a scenario indicator, not a true shift label')


# ---------------------------------------------------------------------------
# Detector (unlabelled DEV only) and outcomes (TEST labels only)
# ---------------------------------------------------------------------------

def detector_scores(src, P_dev):
    """Label-free scores of one unlabelled DEV split. `src` = sd.fit_source output (+ 'conf' added here).

    Returns (scores dict, priors dict). Never receives DEV or TEST labels."""
    P_dev = np.asarray(P_dev, dtype=float)
    pi_s = src['pi']
    pi_m = ls.mlls_em(ls.apply_bcts(P_dev, src['bcts']), pi_s)[0]
    pi_b, info = ls.bbse(src['P'], src['y'], P_dev, return_info=True)
    conf_src = float(np.asarray(src['P']).max(1).mean())
    scores = {'D_mb': ls.tv_distance(pi_m, pi_b), 'D_ms': ls.tv_distance(pi_m, pi_s),
              'D_bs': ls.tv_distance(pi_b, pi_s), 'D_conf': abs(float(P_dev.max(1).mean()) - conf_src)}
    return scores, {'mlls': pi_m, 'bbse': pi_b, '_bbse_info': info}


def predict_methods(src, P_dev, P_test, priors, dev_shares=None):
    """TEST grades of the scored methods, fitted on S and the unlabelled DEV split only.

    `dev_shares` (DEV label shares) feeds ONLY the oracle_quantile reference, used as an outcome (the
    unsafe_vs_oracle label), never by the detector or the gated policy."""
    E_dev, E_test = expected_grade(P_dev), expected_grade(P_test)
    Pt_b = ls.apply_bcts(P_test, src['bcts'])
    out = {'argmax': np.asarray(P_test).argmax(1),
           'mlls_bcts_quantile': apply_cutpoints(E_test, prevalence_cutpoints(E_dev, priors['mlls'])),
           'mlls_bcts_bayes': ls.prior_correct(Pt_b, src['pi'], priors['mlls']).argmax(1),
           'bbse_quantile': apply_cutpoints(E_test, prevalence_cutpoints(E_dev, priors['bbse']))}
    if dev_shares is not None:
        out['oracle_quantile'] = apply_cutpoints(E_test, prevalence_cutpoints(E_dev, dev_shares))
    return out


# ---------------------------------------------------------------------------
# Per-tag analysis (loading identical to tools/shift_decomposition.analyse_tag)
# ---------------------------------------------------------------------------

def _prepare(tag, seed, notes):
    try:
        exp_dir = find_exp(tag)
    except FileNotFoundError as e:
        notes.add(tag, '-', '-', str(e), echo=True)
        return None
    heldout = _heldout(exp_dir)
    sites, raw_pairs = {}, []
    for ds in heldout:
        dev, test = _load(exp_dir, ds, 'dev'), _load(exp_dir, ds, 'test')
        if dev is None or test is None:
            notes.add(tag, ds, '-', 'no dev/test predictions: dataset skipped', echo=True)
            continue
        for split in ('dev', 'test'):
            f = sd._load_file(sd._pred_path(exp_dir, ds, split))
            raw_pairs.append((f[3], f[2]))
        sites[ds] = {'P_dev': dev[2], 'y_dev': dev[1], 'P_test': test[2], 'y_test': test[1],
                     'dev_shares': ls.class_shares(dev[1], K)}
    if not sites:
        return None
    src_all = sd.load_source(exp_dir, tag, raw_pairs, notes)
    if src_all is None:
        notes.add(tag, '-', 'source', 'no OOF: run skipped', echo=True)
        return None
    order, _ = sd._oof_order(exp_dir)
    P_oof, y_oof = (src_all['P'], src_all['y']) if order is None else (src_all['P'][order], src_all['y'][order])
    iS, iT = sd.split_source(y_oof, seed)
    src = sd.fit_source(P_oof[iS], y_oof[iS])
    pool = {'P': P_oof[iT], 'y': y_oof[iT]}
    return {'sites': sites, 'src': src, 'pool': pool, 'heldout': heldout}


def _unsafe(harm):
    """1 if the harm is below -HARM_TOL, 0 otherwise, NaN when the harm is not finite (e.g. NaN QWK)."""
    return float(harm < -HARM_TOL) if np.isfinite(harm) else np.nan


def analyse_tag(tag, R, seed, notes):
    prep = _prepare(tag, seed, notes)
    if prep is None:
        return []
    src, pool = prep['src'], prep['pool']
    rows = []
    for ds, site in prep['sites'].items():
        for scen in sd.SCENARIOS:
            draws = ([-1] if scen == 'C' else []) + list(range(int(R)))
            for r in draws:
                Pd, yd, Pt, scorer = sd.build_scenario(scen, r, seed, ds, site, pool, src['pi'], notes, tag)
                scores, priors = detector_scores(src, Pd)
                binfo = priors.pop('_bbse_info')
                if binfo['degenerate']:
                    notes.add(tag, ds, f'scenario_{scen}', f'draw {r}: BBSE degenerate (all weights clipped): '
                              'pi_BBSE = pi_S, so D_mb = D_ms')
                pred = predict_methods(src, Pd, Pt, priors, dev_shares=ls.class_shares(yd, K))
                q = {m: scorer.score(pred[m])['qwk'] for m in QWK_METHODS}
                row = {'tag': tag, 'dataset': ds, 'scenario': scen, 'draw': int(r), 'n_dev': len(yd),
                       'n_test': scorer.n, **scores,
                       'tv_true_dev': ls.tv_distance(ls.class_shares(yd, K), src['pi']),
                       'bbse_cond': float(binfo['cond']), 'bbse_degenerate': int(binfo['degenerate'])}
                for name, pi in priors.items():
                    row.update({f'pi_{name}_{k}': float(v) for k, v in enumerate(pi)})
                row.update({f'qwk_{m}': q[m] for m in QWK_METHODS})
                for c, m in CORRECTIONS.items():
                    row[f'harm_{c}'] = q[m] - q['argmax']
                    row[f'unsafe_{c}'] = _unsafe(row[f'harm_{c}'])
                row['harm_vs_oracle_mlls'] = q['mlls_bcts_quantile'] - q['oracle_quantile']
                row['unsafe_vs_oracle_mlls'] = _unsafe(row['harm_vs_oracle_mlls'])
                row['site_images'] = int(scen in SITE_IMAGES)
                rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def auroc(score, label):
    """Mann-Whitney AUROC (ties count 1/2); NaN unless both classes are present."""
    s, y = np.asarray(score, dtype=float), np.asarray(label, dtype=float)
    ok = np.isfinite(s) & np.isfinite(y)
    s, y = s[ok], y[ok].astype(int)
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return np.nan
    ranks = pd.Series(s).rank(method='average').values
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _spearman(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
        return np.nan
    return float(pd.Series(a[ok]).rank().corr(pd.Series(b[ok]).rank()))


def _cluster_key(df):
    """Bootstrap cluster of each row: the held-out dataset (all its scenarios and tags share images), except A0,
    whose rows of every site and tag come from the one shared APTOS pool / source half -> one cluster."""
    return np.where(df['scenario'].values == 'A0', 'A0_pool', df['dataset'].values.astype(str))


def _cluster_boot(df, B, rng):
    """Bootstrap row indices: resample clusters with replacement, then rows (draws) within each picked cluster."""
    key = _cluster_key(df)
    cells = [np.flatnonzero(key == k) for k in pd.unique(key)]
    for _ in range(int(B)):
        pick = rng.integers(0, len(cells), len(cells))
        yield np.concatenate([rng.choice(cells[i], size=len(cells[i]), replace=True) for i in pick])


def summarise(draws, B, seed):
    rows = []
    groups = [(t, draws[draws['tag'] == t]) for t in draws['tag'].unique()] + [('ALL', draws)]
    for tag, g0 in groups:
        for stratum, scen in STRATA.items():
            g = g0[g0['scenario'].isin(scen)].reset_index(drop=True)
            if g.empty:
                continue
            boots = list(_cluster_boot(g, B, np.random.default_rng([int(seed), 77]))) if B > 0 else []
            n_cl = len(pd.unique(_cluster_key(g)))
            for lab, harm in LABELS.items():
                if lab == 'site_images' and stratum != 'all':
                    continue                                   # constant within C / B+C
                for s in SCORES:
                    a = auroc(g[s], g[lab])
                    bs = np.asarray([auroc(g[s].values[i], g[lab].values[i]) for i in boots], dtype=float)
                    bs = bs[np.isfinite(bs)]
                    lo, hi = (np.percentile(bs, [2.5, 97.5]) if len(bs) else (np.nan, np.nan))
                    rows.append({'tag': tag, 'stratum': stratum, 'score': s, 'label': lab,
                                 'primary': int(s == PRIMARY_SCORE and lab == 'unsafe_mlls'
                                                and stratum == PRIMARY_STRATUM and tag == 'ALL'),
                                 'n': int(g[lab].notna().sum()), 'n_pos': int(np.nansum(g[lab])),
                                 'n_clusters': n_cl, 'auroc': a, 'auroc_lo': float(lo), 'auroc_hi': float(hi),
                                 'n_boot_finite': int(len(bs)), 'spearman_vs_harm': _spearman(g[s], g[harm]),
                                 'spearman_vs_n_dev': _spearman(g[s], g['n_dev']), 'harm_column': harm,
                                 'caveat': CAVEAT})
    return pd.DataFrame(rows)


def fit_tau(score, q_arg, q_cor):
    """Threshold maximising the mean gated QWK (correct when score <= tau, else argmax).

    Candidates: -inf (never correct) and every observed score; ties broken towards the smaller tau."""
    s = np.asarray(score, dtype=float)
    q_arg, q_cor = np.asarray(q_arg, dtype=float), np.asarray(q_cor, dtype=float)
    cands = np.concatenate([[-np.inf], np.unique(s[np.isfinite(s)])])
    best, best_v = -np.inf, -np.inf
    for t in cands:
        v = float(np.nanmean(np.where(s <= t, q_cor, q_arg)))
        if v > best_v + 1e-12:
            best, best_v = t, v
    return float(best)


GATING_COLS = ['tag', 'correction', 'score', 'tau_fit_on', 'dataset', 'n_tags', 'tau', 'gated_on_share_C',
               'gated_on_observed', 'qwk_argmax_observed', 'qwk_always_observed', 'qwk_gated_observed',
               'qwk_argmax_C', 'qwk_always_C', 'qwk_gated_C', 'gated_minus_argmax_C',
               'gated_minus_argmax_C_spread_lo', 'gated_minus_argmax_C_spread_hi', 'caveat']
_MEAN_KEYS = ('gated_on_share_C', 'gated_on_observed', 'qwk_argmax_observed', 'qwk_always_observed',
              'qwk_gated_observed', 'qwk_argmax_C', 'qwk_always_C', 'qwk_gated_C', 'gated_minus_argmax_C')


def _nanmean(v):
    v = np.asarray(v, dtype=float)
    return float(np.nanmean(v)) if np.isfinite(v).any() else np.nan


def gating(draws, B, seed):
    """Leave-one-site-out gating: tau fitted on the OTHER held-out datasets only (every tag, in ALL), either on all
    their scenarios ('all_scenarios', default) or on their C rows only ('C_only', sensitivity); scored on the
    held-out site's C rows. Intervals are 2.5 / 97.5 % points of the per-draw spread, not CIs of the mean."""
    rows = []
    rng = np.random.default_rng([int(seed), 99])
    groups = [(t, draws[draws['tag'] == t]) for t in draws['tag'].unique()] + [('ALL', draws)]
    for tag, g in groups:
        for c, m in CORRECTIONS.items():
            for s in SCORES:
                for fit_on in ('all_scenarios', 'C_only'):
                    per_site = []
                    for ds in g['dataset'].unique():
                        train, test = g[g['dataset'] != ds], g[g['dataset'] == ds]
                        if fit_on == 'C_only':
                            train = train[train['scenario'] == 'C']
                        if train.empty:
                            continue
                        tau = fit_tau(train[s], train['qwk_argmax'], train[f'qwk_{m}'])
                        obs = test[(test['scenario'] == 'C') & (test['draw'] < 0)]
                        rep = test[(test['scenario'] == 'C') & (test['draw'] >= 0)]

                        def gate(d):
                            return np.where(d[s].values <= tau, d[f'qwk_{m}'].values, d['qwk_argmax'].values)

                        d = (gate(rep) - rep['qwk_argmax'].values) if len(rep) else np.array([])
                        r = {'tag': tag, 'correction': m, 'score': s, 'tau_fit_on': fit_on, 'dataset': ds,
                             'n_tags': int(test['tag'].nunique()), 'tau': tau,
                             'gated_on_share_C': _nanmean(rep[s] <= tau) if len(rep) else np.nan,
                             'gated_on_observed': _nanmean(obs[s] <= tau) if len(obs) else np.nan,
                             'qwk_argmax_observed': _nanmean(obs['qwk_argmax']),
                             'qwk_always_observed': _nanmean(obs[f'qwk_{m}']),
                             'qwk_gated_observed': _nanmean(gate(obs)) if len(obs) else np.nan,
                             'qwk_argmax_C': _nanmean(rep['qwk_argmax']), 'qwk_always_C': _nanmean(rep[f'qwk_{m}']),
                             'qwk_gated_C': _nanmean(gate(rep)) if len(rep) else np.nan,
                             'gated_minus_argmax_C': _nanmean(d)}
                        fin = d[np.isfinite(d)]
                        r['gated_minus_argmax_C_spread_lo'], r['gated_minus_argmax_C_spread_hi'] = (
                            np.percentile(fin, [2.5, 97.5]) if len(fin) else (np.nan, np.nan))
                        per_site.append((r, fin))
                        rows.append(r)
                    if len(per_site) > 1:
                        mean = {'tag': tag, 'correction': m, 'score': s, 'tau_fit_on': fit_on, 'dataset': 'MEAN',
                                'n_tags': int(g['tag'].nunique()), 'tau': np.nan}
                        for k in _MEAN_KEYS:
                            mean[k] = _nanmean([p[k] for p, _ in per_site])
                        fins = [f for _, f in per_site if len(f)]
                        if fins and B > 0:                 # mean over sites of one resampled draw per site
                            bs = [np.mean([f[rng.integers(0, len(f))] for f in fins]) for _ in range(int(B))]
                            mean['gated_minus_argmax_C_spread_lo'], mean['gated_minus_argmax_C_spread_hi'] = \
                                np.percentile(bs, [2.5, 97.5])
                        rows.append(mean)
    out = pd.DataFrame(rows, columns=GATING_COLS[:-1])
    out['caveat'] = CAVEAT
    return out[GATING_COLS]


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

SCEN_COLORS = {'A0': '#9e9e9e', 'A': '#2a78d6', 'B': '#eb6834', 'C': '#c0392b'}


def plot(draws, gate, stem):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'font.weight': 'bold', 'axes.labelweight': 'bold',
                         'axes.titleweight': 'bold'})
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    ax = axes[0]
    data = [draws.loc[draws['scenario'] == s, PRIMARY_SCORE].dropna().values for s in sd.SCENARIOS]
    bp = ax.boxplot(data, patch_artist=True, showfliers=False)
    for patch, s in zip(bp['boxes'], sd.SCENARIOS):
        patch.set_facecolor(SCEN_COLORS[s])
        patch.set_alpha(0.75)
    ax.set_xticks(range(1, len(sd.SCENARIOS) + 1))
    ax.set_xticklabels(['A0 no shift', 'A prior only', 'B image/annot.', 'C observed'], fontsize=8)
    ax.set_ylabel('MLLS-BBSE disagreement TV(pi_MLLS, pi_BBSE)')
    ax.set_title('Disagreement by scenario')
    ax = axes[1]
    for s in sd.SCENARIOS:
        d = draws[draws['scenario'] == s]
        ax.scatter(d[PRIMARY_SCORE], d['harm_mlls'], s=8, alpha=0.5, color=SCEN_COLORS[s], label=s)
    ax.axhline(-HARM_TOL, color='#555555', lw=0.8, ls='--')
    ax.set_xlabel('TV(pi_MLLS, pi_BBSE)')
    ax.set_ylabel('QWK(MLLS quantile) - QWK(argmax)')
    ax.set_title('Disagreement vs harm of MLLS correction')
    ax.legend(frameon=False, fontsize=8)
    ax = axes[2]
    g = gate[(gate['tag'] == 'ALL') & (gate['dataset'] != 'MEAN') & (gate['score'] == PRIMARY_SCORE)
             & (gate['tau_fit_on'] == 'all_scenarios')]
    labels, vals = [], []
    for m in CORRECTIONS.values():
        t = g[g['correction'] == m]
        for _, r in t.iterrows():
            labels.append(f"{r['dataset']}\n{m.split('_')[0]}")
            vals.append((r['qwk_argmax_observed'], r['qwk_always_observed'], r['qwk_gated_observed']))
    if vals:
        v = np.asarray(vals, dtype=float)
        x = np.arange(len(labels))
        for i, (name, col) in enumerate((('argmax', '#9e9e9e'), ('always correct', '#c0392b'),
                                         ('gated (LOSO tau)', '#1baf7a'))):
            ax.bar(x + (i - 1) * 0.27, v[:, i], 0.25, color=col, label=name)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_ylabel('QWK on observed held-out TEST')
        ax.legend(frameon=False, fontsize=8)
    ax.set_title('Gating the correction (ALL tags pooled)')
    fig.text(0.5, -0.02, 'Post-hoc exploratory analysis. A0, A, B: semi-synthetic resamplings of saved predictions; '
             'tau chosen leave-one-site-out.', ha='center', fontsize=7)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    os.makedirs(os.path.dirname(stem) or '.', exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(f'{stem}.{ext}', dpi=200, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tags', required=True)
    ap.add_argument('--R', type=int, default=50, help='resampling draws per scenario (as shift_decomposition)')
    ap.add_argument('--B', type=int, default=1000, help='bootstrap resamples for the AUROC / gating intervals')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out_dir', default=os.path.join('results', 'master', 'recal'))
    ap.add_argument('--fig_stem', default=os.path.join('results', 'figures', 'fig_shift_detector'))
    a = ap.parse_args(argv)
    if a.R < 1:
        ap.error('--R must be >= 1')
    if a.B < 0:
        ap.error('--B must be >= 0')
    tags = [t.strip() for t in a.tags.split(',') if t.strip()]
    notes = sd.Notes()
    rows = [r for t in tags for r in analyse_tag(t, a.R, a.seed, notes)]
    if not rows:
        print('shift_detector: no analysable run (held-out dev/test predictions + oof.npz) for any tag')
        sys.exit(1)
    draws = pd.DataFrame(rows)
    summ = summarise(draws, a.B, a.seed)
    gate = gating(draws, a.B, a.seed)
    os.makedirs(a.out_dir, exist_ok=True)
    p = lambda f: os.path.join(a.out_dir, f)                          # noqa: E731
    draws.round(6).to_csv(p('shift_detector_draws.csv'), index=False)
    summ.round(4).to_csv(p('shift_detector_summary.csv'), index=False)
    gate.round(4).to_csv(p('shift_detector_gating.csv'), index=False)
    pd.DataFrame(notes.rows, columns=['tag', 'dataset', 'method', 'note']).to_csv(
        p('shift_detector_notes.csv'), index=False)
    plot(draws, gate, a.fig_stem)

    with pd.option_context('display.width', 200, 'display.max_rows', 200):
        for stratum in STRATA:
            sub = summ[(summ['tag'] == 'ALL') & (summ['stratum'] == stratum)]
            if len(sub):
                print(f'\nAUROC, ALL tags pooled, stratum {stratum} (rows: label-free score; columns: label):')
                print(sub.pivot_table(index='score', columns='label', values='auroc').round(3).to_string())
        pr = summ[summ['primary'] == 1]
        if len(pr):
            pr = pr.iloc[0]
            print(f"\nPRIMARY (ALL, stratum C)  D_mb -> unsafe_mlls: AUROC {pr['auroc']:.3f} "
                  f"[{pr['auroc_lo']:.3f}, {pr['auroc_hi']:.3f}] ({pr['n_clusters']} clusters, "
                  f"{pr['n_boot_finite']} finite resamples), Spearman with harm {pr['spearman_vs_harm']:.3f} "
                  f"(expected < 0), n={pr['n']}, unsafe={pr['n_pos']}")
            base = summ[(summ['tag'] == 'ALL') & (summ['stratum'] == 'C') & (summ['score'] == 'D_conf')
                        & (summ['label'] == 'unsafe_mlls')]
            if len(base):
                print(f"         baseline D_conf on the same rows: AUROC {base['auroc'].iloc[0]:.3f}")
        else:
            print('\nPRIMARY row not available (no C rows with both classes).')
        gm = gate[(gate['dataset'] == 'MEAN') & (gate['score'] == PRIMARY_SCORE)]
        cols = ['tag', 'correction', 'tau_fit_on', 'qwk_argmax_observed', 'qwk_always_observed', 'qwk_gated_observed',
                'gated_minus_argmax_C', 'gated_minus_argmax_C_spread_lo', 'gated_minus_argmax_C_spread_hi']
        if len(gm):
            print('\nGating with D_mb, MEAN over held-out sites (tau leave-one-site-out):')
            print(gm[cols].round(3).to_string(index=False))
        else:
            print('\nGating: fewer than 2 held-out sites, no MEAN rows.')
    print(f'\nwritten: {p("shift_detector_draws.csv")}, shift_detector_summary.csv, shift_detector_gating.csv, '
          f'shift_detector_notes.csv; {a.fig_stem}.png/.pdf (R={a.R}, B={a.B}, seed={a.seed})')


if __name__ == '__main__':
    main()
