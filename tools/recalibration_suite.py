"""Recalibration / label-shift suite on saved predictions (CPU only; the networks are never touched).

Question: how much target-site information does a model need to grade a new site well, measured with
REALISTIC information levels and the standard label-shift baselines? (The E4 "prevalence" level of
tools/adaptation_curve.py takes the grade shares from the site's own DEV labels: an ORACLE.)

Data per (tag, held-out dataset): temperature-scaled class probabilities P (key y_prob) of the site's DEV
and TEST splits (sorted by image id, exactly as tools/adaptation_curve._load), and the source = the run's
APTOS out-of-fold predictions (oof.npz; raw probabilities there, so the same temperature is applied: T is
recovered from the (y_prob_raw, y_prob) pairs of the prediction files, or refitted on the OOF exactly like
evaluate.py when the files carry no raw probabilities). Severity score E = sum_k k P_k. Only datasets that
are NOT training sources of the run are used.

Methods and the target information each uses (`info` column). TEST labels are used for metrics only.

  info=none
    argmax                  the model's own prediction
    default_cuts            cut-points 0.5, 1.5, 2.5, 3.5 on E
    source_cuts             QWK-optimal cut-points fitted on the APTOS OOF scores
  info=unlabelled_target    (unlabelled target DEV images; never a label)
    upa                     UPA-style quantile alignment: map fitted on unlabelled DEV E -> source OOF E,
                            applied to TEST E, then source_cuts
    mlls_bcts_bayes         BCTS fitted on OOF; EM (Saerens) on BCTS-calibrated DEV -> pi_hat; Bayes prior
                            correction of BCTS-calibrated TEST probabilities, argmax
    mlls_bcts_quantile      pi_hat as above; cut-points = quantiles of unlabelled DEV E reproducing pi_hat
    em_raw_bayes/_quantile  the same EM on the temperature-scaled probabilities without BCTS
    bbse_bayes/_quantile    pi_hat from black-box shift estimation (soft confusion on OOF); Bayes correction
                            applied to the BCTS-calibrated TEST probabilities
    {mlls_bcts,bbse}_bagged_quantile
                            uncertainty-aware prevalence: the unlabelled DEV images are bootstrapped (--B_pi
                            resamples, default 200; cluster-level with the `group` column of
                            splits/external/<ds>_dev.csv, else image-level; IDENTICAL resamples for MLLS and
                            BBSE, for every tag and every seed-equal run), the prior is re-estimated on each
                            resample (pi_b), cut-points = element-wise median over b of
                            prevalence_cutpoints(E_dev, pi_b) on ALL unlabelled DEV scores (only the prior
                            varies), kept strictly increasing
    {mlls_bcts,bbse}_shrunk_quantile
                            prior shrunk towards the source prior pi_src (James-Stein style):
                              pi_shrunk = pi_src + w (pi_hat - pi_src),
                              w = 1 / (1 + tr(Cov_boot(pi_hat)) / ||pi_hat - pi_src||^2), clipped to [0, 1]
                            (pi_hat = full-DEV estimate; Cov_boot over the same --B_pi resamples): a noisy
                            estimate of a small shift moves less. Then quantile cut-points on unlabelled DEV E,
                            evaluated exactly like mlls_bcts_quantile. Per-class 95 % percentile intervals of
                            pi_hat and w are written to prevalence_estimates.csv and recal_fitted.csv
    bn_adapt_*, tent_*      test-time adaptation predictions written by another tool
                            (results/<phase>/<tag>/target_adapt/{bn,tent}/predictions_<ds>_test.npz), scored
                            with argmax and default_cuts; skipped with a note when absent
  info=prevalence_oracle    (grade shares of the site's DEV labels ~ TEST shares: UPPER BOUND, not label-free)
    oracle_quantile         prevalence cut-points on unlabelled DEV E with the DEV label shares
    oracle_bayes            Bayes correction of BCTS-calibrated TEST probabilities with the DEV label shares
  info=prevalence_other_sites
    other_sites_quantile    mean DEV grade shares of the OTHER held-out sites of the run (leave-one-site-out;
                            never the target's own shares)
  info=prevalence_misspecified
    misspecified_quantile_nodr{+-0.05,+-0.10}  oracle shares with that absolute No-DR mass moved
    misspecified_quantile_tv{0.05,0.10}        oracle shares perturbed at total variation exactly delta
                                               (random Dirichlet directions, --misspec_draws draws)
  info=n_labels             (n labelled DEV images; draws identical to tools/adaptation_curve.py)
    fewshot_plain           QWK-optimal cut-points on the n images
    fewshot_shrink          shrunk towards the default cut-points, lam by leave-one-site-out over the
                            other held-out sites' DEV splits (tools/adaptation_curve logic, imported). Besides
                            the n target labels this ONE scalar uses the labelled DEV splits of the other
                            held-out sites (recorded per row in column lam_from); never the target's
    fewshot_prev_quantile   Dirichlet(1)-smoothed shares of the n labels + quantiles of all unlabelled DEV E
    fewshot_offset          one common shift of the default cut-points, QWK-optimal on the n images
    fewshot_temp            temperature fitted on the n images, argmax (identical to argmax: temperature
                            scaling never changes the argmax; kept as the explicit null baseline)
    fewshot_temp_cuts       the same temperature, default cut-points on the re-scaled E
    fewshot_vector          vector scaling (per-class scale + bias, L2 to identity, reg 0.1), argmax
    (+ rows of results/master/recal/fewshot_head_draws.csv when present: last-layer fine-tuning, merged)
  info=all_dev_labels       (every labelled DEV image; n = -1)
    all_dev_cuts, all_dev_temp, all_dev_temp_cuts

Outputs (--out_dir, default results/master/recal):
  recal_draws.csv              one row per (tag, dataset, method, info, n, draw); schema shared with
                               tools/fewshot_head.py (first 14 columns fixed)
  recal_summary.csv            mean / SD over draws, bootstrap 95 % interval of QWK for deterministic methods,
                               every full_metrics column for deterministic methods; dataset 'MEAN' = unweighted
                               mean over the run's held-out datasets (only when all are present; these differ
                               between tags); dataset 'MEAN_COMMON' = unweighted mean over the held-out datasets
                               COMMON to every analysed tag (intersection; like-for-like across tags). Column
                               `sites` lists the datasets averaged (the dataset itself on per-site rows)
  recal_compare.csv            paired differences vs argmax and mlls_bcts_bayes (and fewshot_plain at the
                               same n), per site, MEAN and MEAN_COMMON (column `sites`):
                               deterministic -> paired cluster bootstrap on TEST (unit in bootstrap_unit;
                               interval_source 'TEST bootstrap');
                               draw-based (n labels, random prior perturbations) -> PRIMARY diff_lo / diff_hi /
                               diff_mean / p_two_sided are JOINT over draws and TEST: for b = 0..B-1 the draw
                               (b mod D) (D = number of draws) is scored on the b-th TEST resample (the SAME
                               resamples as every deterministic method, so a deterministic reference is paired
                               resample by resample, and fewshot_plain is paired draw AND resample) and the
                               paired difference taken; percentile interval over b (columns diff_lo_joint /
                               diff_hi_joint; interval_source 'joint ...'). The draws-only interval (TEST held
                               fixed) is kept in diff_lo_draws / diff_hi_draws / p_two_sided_draws. Rows without
                               stored TEST predictions (merged fewshot_head rows) or with --B 0 fall back to the
                               draws-only interval (interval_source says so). MEAN / MEAN_COMMON: average over
                               datasets per draw (draws only) and per resample index b (joint)
  prevalence_estimates.csv     estimated priors vs DEV and TEST shares (TEST shares: reporting only); for
                               mlls_bcts / bbse also pi_lo_k / pi_hi_k (bootstrap 95 % over unlabelled DEV
                               resamples), the shrinkage weight and the shrunk priors (rows *_shrunk)
  binary_operating_points.csv  referable operating points (DEV-chosen thresholds; implied by each method;
                               TEST-oracle ROC sensitivity, reporting only, with the specificity it actually
                               achieves, >= the target) with PPV/NPV at 5/10/20/30 %
  ordinal_discrimination.csv   ranking metrics (independent of every cut-point)
  indist_control.csv           the same recalibrations inside APTOS (OOF -> APTOS test) vs external gains,
                               plus an estimator-bias control (MLLS/BBSE prior error on APTOS test)
  aptos_dup_sensitivity.csv    APTOS test with / without near-duplicates of the run's training images
                               (all training sources when leakage_check rows exist for the tag)
  recal_fitted.csv             every fitted quantity (cut-points, priors, temperatures, thresholds, ...)
  recal_notes.csv              everything that could not be computed (written NaN) and why
  results/figures/fig_recal_levels, fig_recal_misspec, fig_recal_operating_points (.png/.pdf)

Usage:
  python tools/recalibration_suite.py --tags p3b_joint,p10d_ce,p10q_ce,p4j_coralsrc_shared [--B_pi 200]
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation import clinical as cl  # noqa: E402
from src.evaluation import label_shift as ls  # noqa: E402
from src.evaluation.cutpoints import (DEFAULT_CUTS, apply_cutpoints, expected_grade, fit_cutpoints,  # noqa: E402
                                      grade_metrics, prevalence_cutpoints)
from src.utils.config import find_exp  # noqa: E402
from tools.adaptation_curve import (DATASETS, _draw_indices, _heldout, _inner_table, _load,  # noqa: E402
                                    _pick_lambda)

K = 5
INFO_LEVELS = ('none', 'unlabelled_target', 'prevalence_oracle', 'prevalence_other_sites', 'prevalence_estimated',
               'prevalence_misspecified', 'n_labels', 'all_dev_labels')
DRAW_COLS = ['tag', 'dataset', 'method', 'info', 'n', 'draw', 'n_test', 'qwk', 'acc', 'f1_macro', 'mild_recall',
             'bin_sens', 'bin_spec', 'pred_nodr_share']
DRAW_METRICS = DRAW_COLS[7:]
CMP_METRICS = ('qwk', 'mild_recall', 'bin_sens', 'bin_spec', 'f1_macro')
REFERENCES = ('argmax', 'mlls_bcts_bayes')
SPECS = (0.90, 0.95, 0.97, 0.99)
NODR_DELTAS = (-0.10, -0.05, 0.05, 0.10)
TV_DELTAS = (0.05, 0.10)
TTA_METHODS = (('bn_adapt', 'bn'), ('tent', 'tent'))
PRIOR_BOOT = ('mlls_bcts', 'bbse')          # estimators with a bootstrap of the unlabelled DEV images
AGGREGATES = ('MEAN', 'MEAN_COMMON')        # pseudo-datasets of the summary / compare tables
LEAKAGE_PAIRS = os.path.join('results', 'master', 'leakage_pairs.csv')
ORD_NOTE = 'ranking metrics of E and P(grade>=2): independent of every cut-point / recalibration method'


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _stem(x):
    return os.path.splitext(os.path.basename(str(x)))[0]


def _load_file(path):
    """(ids, y, P, P_raw or None) sorted by id, the same order as adaptation_curve._load."""
    with np.load(path, allow_pickle=True) as f:
        y = np.asarray(f['y_true'] if 'y_true' in f.files else f['y'], dtype=int)
        P = np.asarray(f['y_prob'] if 'y_prob' in f.files else f['P'], dtype=float)
        raw = np.asarray(f['y_prob_raw'], dtype=float) if 'y_prob_raw' in f.files else None
        ids = np.array([str(i) for i in f['ids']])
    order = np.argsort(ids, kind='stable')
    return ids[order], y[order], P[order], (raw[order] if raw is not None and raw.shape == P.shape else None)


def _pred_path(exp_dir, ds, split):
    from tools.adaptation_curve import LEGACY
    legacy = dict(LEGACY)
    legacy[('aptos', 'test')] = 'predictions_test.npz'
    for name in (f'predictions_{ds}_{split}.npz', legacy.get((ds, split))):
        if name and os.path.isfile(os.path.join(exp_dir, name)):
            return os.path.join(exp_dir, name)
    return None


def _temperature_from_pair(raw, scaled):
    """1/T from log(scaled) = log(raw) / T + c (row-wise), least squares on non-underflowing entries."""
    if raw is None:
        return None
    lr, lsc = np.log(np.clip(raw, 1e-300, None)), np.log(np.clip(scaled, 1e-300, None))
    j = raw.argmax(1)
    r = np.arange(len(raw))
    dr, ds_ = lr - lr[r, j][:, None], lsc - lsc[r, j][:, None]
    ok = (raw > 1e-12) & (scaled > 1e-12)
    ok[r, j] = False
    if ok.sum() < 5 or (dr[ok] ** 2).sum() <= 0:
        return None
    inv = float((dr[ok] * ds_[ok]).sum() / (dr[ok] ** 2).sum())
    return 1.0 / inv if inv > 0 else None


class Notes:
    def __init__(self):
        self.rows = []

    def add(self, tag, dataset, method, note, echo=False):
        self.rows.append({'tag': tag, 'dataset': dataset, 'method': method, 'note': note})
        if echo:
            print(f'recalibration_suite: {tag} {dataset} {method}: {note}')


def load_source(exp_dir, tag, raw_pairs, notes):
    """APTOS OOF on the scale of the target y_prob files: {'P', 'y', 'E', 'pi', 'T', 'T_source'} or None."""
    p = os.path.join(exp_dir, 'oof.npz')
    if not os.path.isfile(p):
        notes.add(tag, 'aptos_oof', 'source', 'oof.npz missing: every source-fitted method is NaN', echo=True)
        return None
    with np.load(p, allow_pickle=True) as f:
        y = np.asarray(f['y_true'] if 'y_true' in f.files else f['y'], dtype=int)
        P_raw = np.asarray(f['y_prob'] if 'y_prob' in f.files else f['P'], dtype=float)
    from src.evaluation.metrics import apply_temperature, fit_temperature
    T_refit = float(fit_temperature(np.log(np.clip(P_raw, 1e-15, 1.0)), y))
    est = [t for t in (_temperature_from_pair(r, s) for r, s in raw_pairs) if t is not None]
    if est:
        T, src = float(np.median(est)), 'prediction_files'
        if abs(T - T_refit) > 0.02 * T:
            notes.add(tag, 'aptos_oof', 'source', f'T from prediction files {T:.4f} != OOF refit {T_refit:.4f}; '
                      'the file T is used (it is the T applied to the target probabilities)')
    else:
        T, src = T_refit, 'refit_oof'
    P = apply_temperature(P_raw, T)
    return {'P': P, 'y': y, 'E': expected_grade(P), 'pi': ls.class_shares(y, K), 'T': T, 'T_source': src}


class Site:
    """FIT VIEW of one target site: DEV images + DEV labels, TEST images. Holds no TEST label."""

    def __init__(self, ds, dev, test):
        self.ds = ds
        self.ids_dev, self.y_dev, self.P_dev = dev[0], dev[1], dev[2]
        self.ids_test, self.P_test = test[0], test[2]
        self.E_dev, self.E_test = expected_grade(self.P_dev), expected_grade(self.P_test)
        self.pref_dev, self.pref_test = self.P_dev[:, 2:].sum(1), self.P_test[:, 2:].sum(1)
        self.dev_shares = ls.class_shares(self.y_dev, K)


class Scorer:
    """The ONLY holder of TEST labels. Every method passes hard predictions / scores here; nothing fitted
    is ever derived from it."""

    def __init__(self, y_test, W, unit):
        self._y = np.asarray(y_test, dtype=int)
        self._W = W
        self.unit = unit
        self.n = len(self._y)

    def grade(self, pred):
        return grade_metrics(self._y, pred, K)

    def full(self, pred, score=None, pref=None):
        return cl.full_metrics(self._y, pred, score, pref, K)

    def boot(self, pred):
        return cl.batch_metrics_from_weights(self._y, pred, self._W, K)

    @property
    def B(self):
        return int(self._W.shape[0])

    def boot_cycled(self, preds):
        """Joint draws x TEST bootstrap: resample b (row b of the SAME multiplicity matrix as boot(), hence
        the same resamples as every deterministic method) scored with preds[b mod D], D = len(preds).
        Returns a dict of (B,) arrays (keys of batch_metrics_from_weights)."""
        B, D = self.B, len(preds)
        out = None
        for d in range(min(D, B)):
            rows = np.arange(d, B, D)
            bm = cl.batch_metrics_from_weights(self._y, preds[d], self._W[rows], K)
            if out is None:
                out = {m: np.full(B, np.nan) for m in bm}
            for m, v in bm.items():
                out[m][rows] = v
        return out

    def shares(self):
        return ls.class_shares(self._y, K)

    def binary(self, score, t):
        r = cl.binary_at_threshold(score, self._y >= 2, t)
        pos = np.asarray(score) >= t
        r['missed_per_1000'] = 1000.0 * float(((self._y >= 2) & ~pos).mean())
        r['prevalence'] = float((self._y >= 2).mean())
        return r

    def oracle_sens(self, score, spec):
        """(sens, achieved spec) of the TEST-label oracle operating point (reporting only)."""
        return cl.sens_at_spec_oracle(score, self._y >= 2, spec, return_spec=True)

    def roc(self, score):
        return cl.roc_points(score, self._y >= 2)

    def auroc(self, score):
        return cl._auroc(self._y >= 2, score)

    def referable_prevalence(self):
        return float((self._y >= 2).mean())


def _split_groups(ds, ids, split, tag, notes, what='bootstrap'):
    """Bootstrap clusters of a split from the `group` column of splits/external/<ds>_<split>.csv; the image
    ids themselves (image-level bootstrap) when the file / column / ids are missing."""
    p = os.path.join('splits', 'external', f'{ds}_{split}.csv')
    if not os.path.isfile(p):
        return np.asarray(ids), 'image (no split file)'
    d = pd.read_csv(p, dtype={'id_code': str, 'group': str}, keep_default_na=False)
    if 'group' not in d.columns:
        return np.asarray(ids), 'image (split file has no group column)'
    g = dict(zip(d['id_code'].astype(str), d['group'].astype(str)))
    g.update({_stem(k): v for k, v in list(g.items()) if _stem(k) not in g})
    out = [g.get(str(i), g.get(_stem(i))) for i in ids]
    if any(v is None or v == '' for v in out):
        notes.add(tag, ds, what, f'{split.upper()} ids not all found in the split file: image-level bootstrap')
        return np.asarray(ids), 'image (ids not matched in split file)'
    out = np.asarray(out)
    if len(np.unique(out)) == len(out):
        return out, 'image'
    return out, f'cluster (group column; {len(np.unique(out))} groups / {len(out)} images)'


def _test_groups(ds, ids_test, tag, notes):
    return _split_groups(ds, ids_test, 'test', tag, notes)


# ---------------------------------------------------------------------------
# Per-tag analysis
# ---------------------------------------------------------------------------

class TagResult:
    def __init__(self, tag):
        self.tag = tag
        self.draws = []          # recal_draws rows
        self.full = {}           # (ds, method) -> full_metrics (deterministic methods)
        self.boot = {}           # (ds, method) -> dict of (B,) arrays
        self.units = {}          # ds -> bootstrap unit
        self.fitted = []
        self.prev = []
        self.binary = []
        self.ordinal = []
        self.roc = {}            # ds -> (fpr, tpr, auroc)
        self.heldout = []
        self.sites = []
        self.source = None
        self.c_src = None
        self.pred = {}           # (ds, method) -> TEST prediction of a deterministic method
        self.dpred = {}          # (ds, method, n) -> {draw: TEST prediction} of a draw-based method
        self.scorers = {}        # ds -> Scorer (metrics only; never used by a fit)
        self.seconds = {}        # stage -> wall-clock seconds


def analyse_tag(tag, a, ns, notes):
    R = TagResult(tag)
    try:
        exp_dir = find_exp(tag)
    except FileNotFoundError as e:
        notes.add(tag, '-', '-', str(e), echo=True)
        return None
    R.exp_dir = exp_dir
    R.heldout = _heldout(exp_dir)
    sites, scorers, raw_pairs = {}, {}, []
    for ds in R.heldout:
        dev, test = _load(exp_dir, ds, 'dev'), _load(exp_dir, ds, 'test')
        if dev is None or test is None:
            notes.add(tag, ds, '-', 'no dev/test predictions: dataset skipped (MEAN rows not written)', echo=True)
            continue
        for split in ('dev', 'test'):
            f = _load_file(_pred_path(exp_dir, ds, split))
            raw_pairs.append((f[3], f[2]))
        sites[ds] = Site(ds, dev, test)
        groups, unit = _test_groups(ds, test[0], tag, notes)
        W = cl.cluster_bootstrap_weights(groups, a.B, np.random.default_rng([a.seed, sum(map(ord, ds)), 101]))
        scorers[ds] = Scorer(test[1], W, unit)
        R.units[ds] = unit
    R.sites = list(sites)
    R.scorers = scorers
    if not sites:
        return R
    src = load_source(exp_dir, tag, raw_pairs, notes)
    R.source = src
    fit = R.fitted

    def rec(ds, method, params, n=0, draw=0):
        for k, v in params.items():
            fit.append({'tag': tag, 'dataset': ds, 'method': method, 'n': int(n), 'draw': int(draw), 'param': k,
                        'value': float(v)})

    if src is not None:
        c_src = fit_cutpoints(src['E'], src['y'])
        bcts = ls.fit_bcts(src['P'], src['y'])
        if not bcts['converged']:
            notes.add(tag, 'aptos_oof', 'bcts', 'L-BFGS-B did not report convergence (parameters still used)')
        rec('aptos_oof', 'source', {'T_oof': src['T'], 'T_from_files': float(src['T_source'] == 'prediction_files'),
                                    **{f'pi_src_{k}': v for k, v in enumerate(src['pi'])}})
        rec('aptos_oof', 'source_cuts', {f'c{j + 1}': c for j, c in enumerate(c_src)})
        rec('aptos_oof', 'bcts', {'T': bcts['T'], **{f'b{k}': v for k, v in enumerate(bcts['b'])}})
        R.c_src = c_src
    else:
        c_src, bcts = None, None
    Q = _inner_table({d: {'P_dev': s.P_dev, 'y_dev': s.y_dev} for d, s in sites.items()}, ns, max(5, a.draws // 2),
                     a.seed) if len(sites) > 1 else {}
    tta_missing = []
    for ds, S in sites.items():
        sc = scorers[ds]
        others = [o for o in sites if o != ds]

        def row(method, info, mets, n=0, draw=0, **extra):
            assert info in INFO_LEVELS, info
            r = {'tag': tag, 'dataset': ds, 'method': method, 'info': info, 'n': int(n), 'draw': int(draw),
                 'n_test': sc.n}
            for m in DRAW_METRICS:
                r[m] = float(mets[m]) if mets is not None else float('nan')
            r.update(extra)
            R.draws.append(r)

        def det(method, info, pred, score=None, pref=None, n=0, **extra):
            """Deterministic method: one row + full metrics + bootstrap. pred None -> NaN row."""
            if pred is None:
                row(method, info, None, n=n, **extra)
                R.full[(ds, method)] = None
                return
            row(method, info, sc.grade(pred), n=n, **extra)
            R.full[(ds, method)] = sc.full(pred, score, pref)
            R.boot[(ds, method)] = sc.boot(pred)
            R.pred[(ds, method)] = np.asarray(pred, dtype=np.int8)

        def drow(method, info, pred, n, draw, **extra):
            """Draw-based method: one row per draw; the TEST prediction is kept for the joint interval."""
            row(method, info, sc.grade(pred), n, draw, **extra)
            R.dpred.setdefault((ds, method, int(n)), {})[int(draw)] = np.asarray(pred, dtype=np.int8)

        def cuts_rec(method, c, n=0, draw=0, extra=None):
            rec(ds, method, {**{f'c{j + 1}': v for j, v in enumerate(c)}, **(extra or {})}, n, draw)

        # ---- Level 0: no target information --------------------------------------------------------------
        det('argmax', 'none', S.P_test.argmax(1), S.E_test, S.pref_test)
        det('default_cuts', 'none', apply_cutpoints(S.E_test, DEFAULT_CUTS), S.E_test, S.pref_test)
        if c_src is not None:
            det('source_cuts', 'none', apply_cutpoints(S.E_test, c_src), S.E_test, S.pref_test)
            qmap = ls.fit_quantile_map(S.E_dev, src['E'])                 # unlabelled DEV only
            E_up = qmap(S.E_test)
            rec(ds, 'upa', {f'map_at_{v:.2f}': qmap(np.array([v]))[0] for v in np.linspace(0, 4, 17)})
            det('upa', 'unlabelled_target', apply_cutpoints(E_up, c_src), E_up, S.pref_test)
        else:
            for m, info in (('source_cuts', 'none'), ('upa', 'unlabelled_target')):
                det(m, info, None)
                notes.add(tag, ds, m, 'no source OOF: NaN')

        # ---- unlabelled target: estimated prevalence ------------------------------------------------------
        est, prior_ci = {}, {}
        if src is not None:
            Pd_b, Pt_b = ls.apply_bcts(S.P_dev, bcts), ls.apply_bcts(S.P_test, bcts)
            est['mlls_bcts'] = ls.mlls_em(Pd_b, src['pi'])[0]
            est['em_raw'] = ls.mlls_em(S.P_dev, src['pi'])[0]
            pi_b, info_b = ls.bbse(src['P'], src['y'], S.P_dev, return_info=True)
            est['bbse'] = pi_b
            if info_b['degenerate']:
                notes.add(tag, ds, 'bbse', 'every BBSE weight clipped to 0: pi_src returned')
            for name, pi in est.items():
                rec(ds, name, {f'pi_{k}': v for k, v in enumerate(pi)})
                P_for_bayes = S.P_test if name == 'em_raw' else Pt_b
                Pc = ls.prior_correct(P_for_bayes, src['pi'], pi)
                det(f'{name}_bayes', 'unlabelled_target', Pc.argmax(1), expected_grade(Pc), Pc[:, 2:].sum(1))
                c = prevalence_cutpoints(S.E_dev, pi)
                cuts_rec(f'{name}_quantile', c)
                det(f'{name}_quantile', 'unlabelled_target', apply_cutpoints(S.E_test, c), S.E_test, S.pref_test)
            # uncertainty-aware prevalence: bootstrap of the UNLABELLED DEV images (no label of any kind);
            # identical resamples for MLLS and BBSE and for every tag (seed depends on the dataset only)
            t0 = time.perf_counter()
            dev_groups, _ = _split_groups(ds, S.ids_dev, 'dev', tag, notes, what='prior_bootstrap')
            idx, dev_unit = ls.bootstrap_indices(len(S.P_dev), a.B_pi,
                                                 np.random.default_rng([a.seed, sum(map(ord, ds)), 202]), dev_groups)
            pboot = ls.bootstrap_label_shift_priors(S.P_dev, src['P'], src['y'], bcts=bcts, indices=idx)
            for name in PRIOR_BOOT:
                bt, pi_hat = pboot[name], est[name]
                assert np.allclose(bt['pi'], pi_hat, atol=1e-12, rtol=0)        # same full-DEV point estimate
                pi_sh, w = ls.shrink_prior(pi_hat, src['pi'], bt['cov_trace'])
                shift_sq = float(np.sum((pi_hat - src['pi']) ** 2))
                prior_ci[name] = {'lo': bt['lo'], 'hi': bt['hi'], 'w': w, 'B': bt['B'], 'unit': dev_unit,
                                  'shrunk': pi_sh}
                rec(ds, f'{name}_prior_boot', {'B_pi': bt['B'], 'cluster': float(dev_unit.startswith('cluster')),
                                               'tr_cov': bt['cov_trace'], 'shift_sq': shift_sq, 'shrink_w': w,
                                               **{f'pi_lo_{k}': v for k, v in enumerate(bt['lo'])},
                                               **{f'pi_hi_{k}': v for k, v in enumerate(bt['hi'])},
                                               **({'n_degenerate': bt['n_degenerate']} if 'n_degenerate' in bt
                                                  else {})})
                if bt.get('n_degenerate'):
                    notes.add(tag, ds, f'{name}_prior_boot', f'{bt["n_degenerate"]} of {bt["B"]} resamples: every '
                              'BBSE weight clipped to 0 (pi_src used for those resamples)')
                if bt['B'] < 2:
                    for v in ('bagged', 'shrunk'):
                        det(f'{name}_{v}_quantile', 'unlabelled_target', None)
                    notes.add(tag, ds, name, f'--B_pi {a.B_pi} < 2: bagged / shrunk quantile methods NaN')
                    continue
                c = ls.bagged_prevalence_cutpoints(S.E_dev, bt['draws'], K)
                cuts_rec(f'{name}_bagged_quantile', c, extra={'B_pi': bt['B']})
                det(f'{name}_bagged_quantile', 'unlabelled_target', apply_cutpoints(S.E_test, c), S.E_test,
                    S.pref_test)
                c = prevalence_cutpoints(S.E_dev, pi_sh)
                cuts_rec(f'{name}_shrunk_quantile', c, extra={'w': w, **{f'pi_{k}': v for k, v in enumerate(pi_sh)}})
                det(f'{name}_shrunk_quantile', 'unlabelled_target', apply_cutpoints(S.E_test, c), S.E_test,
                    S.pref_test)
            for name in PRIOR_BOOT:
                if name in prior_ci:
                    est[f'{name}_shrunk'] = prior_ci[name]['shrunk']
            R.seconds['prior_bootstrap'] = R.seconds.get('prior_bootstrap', 0.0) + time.perf_counter() - t0
        else:
            Pt_b = None
            for name in ('mlls_bcts', 'em_raw', 'bbse'):
                for v in ('bayes', 'quantile') + (('bagged_quantile', 'shrunk_quantile') if name in PRIOR_BOOT
                                                  else ()):
                    det(f'{name}_{v}', 'unlabelled_target', None)
                notes.add(tag, ds, name, 'no source OOF (needed for pi_src / calibration): NaN')
        for name, sub in TTA_METHODS:
            p = os.path.join(exp_dir, 'target_adapt', sub, f'predictions_{ds}_test.npz')
            if not os.path.isfile(p):
                tta_missing.append(f'{sub}/{ds}')
                continue
            ids, _, P_tta, _ = _load_file(p)
            if not np.array_equal(ids, S.ids_test):
                for v in ('argmax', 'default_cuts'):
                    det(f'{name}_{v}', 'unlabelled_target', None)
                notes.add(tag, ds, name, f'{p}: TEST ids differ from the main prediction file: NaN', echo=True)
                continue
            E_tta = expected_grade(P_tta)
            det(f'{name}_argmax', 'unlabelled_target', P_tta.argmax(1), E_tta, P_tta[:, 2:].sum(1))
            det(f'{name}_default_cuts', 'unlabelled_target', apply_cutpoints(E_tta, DEFAULT_CUTS), E_tta,
                P_tta[:, 2:].sum(1))

        # ---- prior-informed (no image labels) -------------------------------------------------------------
        oracle = S.dev_shares
        rec(ds, 'oracle_dev_shares', {f'pi_{k}': v for k, v in enumerate(oracle)})
        c = prevalence_cutpoints(S.E_dev, oracle)
        cuts_rec('oracle_quantile', c)
        det('oracle_quantile', 'prevalence_oracle', apply_cutpoints(S.E_test, c), S.E_test, S.pref_test)
        if Pt_b is not None:
            Pc = ls.prior_correct(Pt_b, src['pi'], oracle)
            det('oracle_bayes', 'prevalence_oracle', Pc.argmax(1), expected_grade(Pc), Pc[:, 2:].sum(1))
        else:
            det('oracle_bayes', 'prevalence_oracle', None)
        if others:
            pi_o = other_sites_prior({o: sites[o].dev_shares for o in sites}, ds)
            est['other_sites'] = pi_o
            rec(ds, 'other_sites', {f'pi_{k}': v for k, v in enumerate(pi_o)})
            c = prevalence_cutpoints(S.E_dev, pi_o)
            cuts_rec('other_sites_quantile', c)
            det('other_sites_quantile', 'prevalence_other_sites', apply_cutpoints(S.E_test, c), S.E_test,
                S.pref_test, other_sites_from=','.join(others))
        else:
            det('other_sites_quantile', 'prevalence_other_sites', None)
            notes.add(tag, ds, 'other_sites_quantile', 'no other held-out site with DEV labels: NaN')
        for delta in NODR_DELTAS:
            m = f'misspecified_quantile_nodr{delta:+.2f}'
            try:
                q = ls.perturb_prevalence(oracle, 'nodr_shift', delta)
            except ValueError as e:
                det(m, 'prevalence_misspecified', None, perturb_kind='nodr_shift', perturb_delta=delta)
                notes.add(tag, ds, m, f'{e}: NaN')
                continue
            c = prevalence_cutpoints(S.E_dev, q)
            cuts_rec(m, c, extra={f'pi_{k}': v for k, v in enumerate(q)})
            det(m, 'prevalence_misspecified', apply_cutpoints(S.E_test, c), S.E_test, S.pref_test,
                perturb_kind='nodr_shift', perturb_delta=delta, perturb_tv=ls.tv_distance(q, oracle))
        for delta in TV_DELTAS:
            m = f'misspecified_quantile_tv{delta:.2f}'
            rng = np.random.default_rng([a.seed, int(round(delta * 1000)), sum(map(ord, ds)), 17])
            for d in range(int(a.misspec_draws)):
                try:
                    q = ls.perturb_prevalence(oracle, 'tv_random', delta, rng)
                except ValueError as e:
                    notes.add(tag, ds, m, f'draw {d}: {e}: skipped')
                    continue
                c = prevalence_cutpoints(S.E_dev, q)
                cuts_rec(m, c, draw=d, extra={f'pi_{k}': v for k, v in enumerate(q)})
                drow(m, 'prevalence_misspecified', apply_cutpoints(S.E_test, c), 0, d,
                     perturb_kind='tv_random', perturb_delta=delta, perturb_tv=ls.tv_distance(q, oracle))

        # ---- few labels ------------------------------------------------------------------------------------
        for n in ns:
            if n >= len(S.y_dev):
                notes.add(tag, ds, 'fewshot', f'n={n} >= DEV size {len(S.y_dev)}: skipped')
                continue
            lam = _pick_lambda(Q, ds, others, n) if others else 0.0
            lam_from = [o for o in others if (o, n, 0.0) in Q]       # sites that actually scored this n
            rec(ds, 'fewshot_shrink_lambda', {'lam': lam}, n)
            pis = []
            for k, i in enumerate(_draw_indices(len(S.y_dev), n, a.draws, ds, a.seed)):
                Ei, yi, Pi = S.E_dev[i], S.y_dev[i], S.P_dev[i]
                c = fit_cutpoints(Ei, yi)
                cuts_rec('fewshot_plain', c, n, k)
                drow('fewshot_plain', 'n_labels', apply_cutpoints(S.E_test, c), n, k)
                c = fit_cutpoints(Ei, yi, prior=DEFAULT_CUTS, lam=lam)
                cuts_rec('fewshot_shrink', c, n, k, {'lam': lam})
                drow('fewshot_shrink', 'n_labels', apply_cutpoints(S.E_test, c), n, k, lam=lam,
                     lam_from=','.join(lam_from))
                pi = ls.smoothed_prevalence(yi, K)
                pis.append(pi)
                c = prevalence_cutpoints(S.E_dev, pi)
                cuts_rec('fewshot_prev_quantile', c, n, k, {f'pi_{j}': v for j, v in enumerate(pi)})
                drow('fewshot_prev_quantile', 'n_labels', apply_cutpoints(S.E_test, c), n, k)
                s = ls.fit_cut_offset(Ei, yi)
                rec(ds, 'fewshot_offset', {'offset': s}, n, k)
                drow('fewshot_offset', 'n_labels', apply_cutpoints(S.E_test, DEFAULT_CUTS + s), n, k)
                tp = ls.temperature_scale_fit(Pi, yi)
                rec(ds, 'fewshot_temp', tp, n, k)
                Pt = ls.temperature_scale_apply(S.P_test, tp)
                drow('fewshot_temp', 'n_labels', Pt.argmax(1), n, k)
                drow('fewshot_temp_cuts', 'n_labels', apply_cutpoints(expected_grade(Pt), DEFAULT_CUTS), n, k)
                vp = ls.vector_scale_fit(Pi, yi)
                rec(ds, 'fewshot_vector', {**{f'a{j}': v for j, v in enumerate(vp['a'])},
                                           **{f'b{j}': v for j, v in enumerate(vp['b'])}}, n, k)
                drow('fewshot_vector', 'n_labels', ls.vector_scale_apply(S.P_test, vp).argmax(1), n, k)
            R.prev.append(_prev_row(tag, ds, f'fewshot_smoothed_n{n}', 'n_labels', np.mean(pis, 0), oracle,
                                    sc.shares(), per_draw=pis))

        # ---- all DEV labels --------------------------------------------------------------------------------
        c = fit_cutpoints(S.E_dev, S.y_dev)
        cuts_rec('all_dev_cuts', c, -1)
        det('all_dev_cuts', 'all_dev_labels', apply_cutpoints(S.E_test, c), S.E_test, S.pref_test, n=-1)
        tp = ls.temperature_scale_fit(S.P_dev, S.y_dev)
        rec(ds, 'all_dev_temp', tp, -1)
        Pt = ls.temperature_scale_apply(S.P_test, tp)
        det('all_dev_temp', 'all_dev_labels', Pt.argmax(1), expected_grade(Pt), Pt[:, 2:].sum(1), n=-1)
        det('all_dev_temp_cuts', 'all_dev_labels', apply_cutpoints(expected_grade(Pt), DEFAULT_CUTS),
            expected_grade(Pt), Pt[:, 2:].sum(1), n=-1)

        # ---- prevalence estimates --------------------------------------------------------------------------
        test_sh = sc.shares()
        R.prev.append(_prev_row(tag, ds, 'dev_labels_oracle', 'prevalence_oracle', oracle, oracle, test_sh))
        if src is not None:
            R.prev.append(_prev_row(tag, ds, 'source_prior', 'none', src['pi'], oracle, test_sh))
        for name, pi in est.items():
            base = name[:-len('_shrunk')] if name.endswith('_shrunk') else name
            extra = {}
            if base in prior_ci:
                ci = prior_ci[base]
                if name == base:                       # bootstrap interval of the full-DEV estimate itself
                    extra.update({f'pi_lo_{k}': float(v) for k, v in enumerate(ci['lo'])})
                    extra.update({f'pi_hi_{k}': float(v) for k, v in enumerate(ci['hi'])})
                extra.update({'shrink_weight': ci['w'], 'prior_boot_B': ci['B'], 'prior_boot_unit': ci['unit']})
            R.prev.append(_prev_row(tag, ds, name, 'prevalence_other_sites' if name == 'other_sites'
                                    else 'unlabelled_target', pi, oracle, test_sh, extra=extra))

        # ---- binary operating points -----------------------------------------------------------------------
        ybin_dev = S.y_dev >= 2
        for spec in SPECS:
            t = cl.threshold_for_specificity(S.pref_dev, ybin_dev, spec)
            rec(ds, f'dev_threshold_spec{spec:.2f}', {'threshold': t})
            dv = cl.binary_at_threshold(S.pref_dev, ybin_dev, t)
            ts = sc.binary(S.pref_test, t)
            R.binary.append(_bin_row(tag, ds, 'dev_threshold_pref', 'all_dev_labels', -1, spec, t, dv, ts['sens'],
                                     ts['spec'], ts['prevalence'], 1000 * ts['referral_share'], ts['missed_per_1000']))
            o_sens, o_spec = sc.oracle_sens(S.pref_test, spec)        # achieved spec (>= target), not the target
            R.binary.append(_bin_row(tag, ds, 'test_oracle_roc', 'test_labels_oracle_reporting_only', 0, spec,
                                     np.nan, None, o_sens, o_spec, ts['prevalence'], np.nan, np.nan))
        fpr, tpr = sc.roc(S.pref_test)
        R.roc[ds] = (fpr, tpr, sc.auroc(S.pref_test))

        # ---- ordinal discrimination (cut-point independent) ------------------------------------------------
        fm = sc.full(S.P_test.argmax(1), S.E_test, S.pref_test)
        R.ordinal.append({'tag': tag, 'dataset': ds, 'n_test': sc.n,
                          **{k: v for k, v in fm.items() if k.startswith(('c_index', 'auroc_'))}, 'note': ORD_NOTE})
    if tta_missing:
        print(f'recalibration_suite: {tag}: no target-adaptation predictions for {", ".join(tta_missing)} '
              f'(bn_adapt / tent rows skipped)')
    return R


def other_sites_prior(dev_shares, target):
    """Leave-one-site-out prior: unweighted mean of the DEV grade shares of every site except `target`.
    The target's own shares are removed before averaging (never used)."""
    others = [np.asarray(v, dtype=float) for d, v in dev_shares.items() if d != target]
    if not others:
        raise ValueError('no other site')
    return ls._as_simplex(np.mean(others, axis=0))


def _prev_row(tag, ds, estimator, info, pi, dev, test, per_draw=None, extra=None):
    r = {'tag': tag, 'dataset': ds, 'estimator': estimator, 'info': info}
    r.update({f'pi_hat_{k}': float(v) for k, v in enumerate(pi)})
    r.update(extra or {})
    r.update({f'dev_share_{k}': float(v) for k, v in enumerate(dev)})
    r.update({f'test_share_{k}': float(v) for k, v in enumerate(test)})
    draws = per_draw if per_draw is not None else [pi]
    for ref, name in ((dev, 'dev'), (test, 'test')):
        r[f'l1_vs_{name}'] = float(np.mean([np.abs(np.asarray(p) - ref).sum() for p in draws]))
        r[f'tv_vs_{name}'] = float(np.mean([ls.tv_distance(p, ref) for p in draws]))
    r['draws'] = len(draws)
    return r


def _bin_row(tag, ds, method, info, n, target_spec, t, dev, sens, spec, prev, refer, missed):
    r = {'tag': tag, 'dataset': ds, 'method': method, 'info': info, 'n': n, 'target_spec': target_spec,
         'threshold': t, 'dev_sens': dev['sens'] if dev else np.nan, 'dev_spec': dev['spec'] if dev else np.nan,
         'sens': sens, 'spec': spec, 'test_referable_prevalence': prev, 'referrals_per_1000': refer,
         'missed_referable_per_1000': missed}
    for p in cl.PREVALENCES:
        ppv, npv = cl.ppv_npv(sens, spec, p)
        r[f'ppv_at_{p:.2f}'], r[f'npv_at_{p:.2f}'] = ppv, npv
    return r


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _complete(R, g):
    """True when a method has a row for every held-out dataset of the run (MEAN rows only then)."""
    return len(R.sites) == len(R.heldout) and set(g['dataset']) == set(R.heldout)


def common_sites(results):
    """Held-out datasets analysed for EVERY tag (intersection), in DATASETS order."""
    if not results:
        return []
    s = set(results[0].sites)
    for R in results[1:]:
        s &= set(R.sites)
    return [d for d in DATASETS if d in s] + sorted(d for d in s if d not in DATASETS)


def _aggregates(R, common):
    """[(label, sites)]: 'MEAN' over the run's held-out datasets (only when all were analysed) and
    'MEAN_COMMON' over the datasets common to every tag (when non-empty)."""
    out = []
    if len(R.sites) == len(R.heldout):
        out.append(('MEAN', list(R.heldout)))
    if common and set(common) <= set(R.sites):
        out.append(('MEAN_COMMON', list(common)))
    return out


def _mean_row(draws, R, label, sites, method, info, n, g):
    """Unweighted mean over `sites` of one (tag, method, info, n); g = its per-site summary rows."""
    tag = R.tag
    d = draws[(draws['tag'] == tag) & (draws['method'] == method) & (draws['n'] == n)
              & draws['dataset'].isin(sites)]
    per_draw = d.groupby('draw')
    cnt = per_draw['dataset'].nunique()
    ok_draws = cnt[cnt == len(sites)].index
    r = {'tag': tag, 'dataset': label, 'method': method, 'info': info, 'n': n, 'draws': len(ok_draws),
         'n_test': int(g['n_test'].sum()), 'sites': ','.join(sites)}
    dd = d[d['draw'].isin(ok_draws)].groupby('draw')[list(DRAW_METRICS)].mean()
    for m in DRAW_METRICS:
        v = dd[m].values.astype(float)
        r[f'{m}_mean'] = float(np.mean(v)) if len(v) else np.nan
        r[f'{m}_sd'] = float(np.std(v)) if len(v) > 1 else 0.0
    boots = [R.boot.get((ds, method)) for ds in sites]
    if len(g) == len(sites) and (g['draws'] == 1).all() and all(b is not None for b in boots) \
            and all(np.isfinite(b['qwk']).any() for b in boots):
        mq = np.mean([b['qwk'] for b in boots], axis=0)
        r['qwk_lo'], r['qwk_hi'] = np.nanpercentile(mq, [2.5, 97.5])
        r['interval'] = 'bootstrap 95% (mean of independent per-dataset resamples)'
    elif len(dd) > 1:
        r['qwk_lo'], r['qwk_hi'] = np.percentile(dd['qwk'].values, [2.5, 97.5])
        r['interval'] = 'percentiles over draws'
    for c in g.columns:
        if c not in r and pd.api.types.is_numeric_dtype(g[c]):
            r[c] = float(g[c].mean())
    return r


def summarise(draws, results, common=None):
    """Per-site rows, then 'MEAN' rows (per-tag held-out sets), then 'MEAN_COMMON' rows (datasets common to
    every tag; `common` defaults to common_sites(results))."""
    rows = []
    by_tag = {R.tag: R for R in results}
    common = common_sites(results) if common is None else list(common)
    keys = ['tag', 'dataset', 'method', 'info', 'n']
    for key, g in draws.groupby(keys, sort=False):
        r = dict(zip(keys, key))
        r['draws'] = len(g)
        r['n_test'] = int(g['n_test'].iloc[0])
        r['sites'] = r['dataset']
        for m in DRAW_METRICS:
            v = g[m].astype(float).values
            r[f'{m}_mean'] = float(np.nanmean(v)) if np.isfinite(v).any() else np.nan
            r[f'{m}_sd'] = float(np.nanstd(v)) if len(v) > 1 and np.isfinite(v).any() else 0.0
        R = by_tag.get(r['tag'])
        if R is not None and len(g) == 1 and (r['dataset'], r['method']) in R.full:
            b = R.boot.get((r['dataset'], r['method']))
            if b is not None and np.isfinite(b['qwk']).any():           # --B 0: no bootstrap interval
                r['qwk_lo'], r['qwk_hi'] = np.nanpercentile(b['qwk'], [2.5, 97.5])
                r['interval'] = 'bootstrap 95% (' + R.units[r['dataset']] + ')'
            fm = R.full[(r['dataset'], r['method'])] or {}
            for k, v in fm.items():
                if k not in r and k not in DRAW_METRICS:
                    r[k] = v
        elif len(g) > 1:
            r['qwk_lo'], r['qwk_hi'] = np.nanpercentile(g['qwk'].astype(float), [2.5, 97.5])
            r['interval'] = 'percentiles over draws'
        rows.append(r)
    summ = pd.DataFrame(rows)
    agg_rows = {label: [] for label in AGGREGATES}
    for (tag, method, info, n), g_all in summ.groupby(['tag', 'method', 'info', 'n'], sort=False):
        R = by_tag.get(tag)
        if R is None:
            continue
        for label, sites in _aggregates(R, common):
            g = g_all[g_all['dataset'].isin(sites)]
            if set(g['dataset']) != set(sites):
                continue
            agg_rows[label].append(_mean_row(draws, R, label, sites, method, info, n, g))
    return pd.concat([summ] + [pd.DataFrame(agg_rows[label]) for label in AGGREGATES], ignore_index=True,
                     sort=False)


def _ptwo(d):
    d = d[np.isfinite(d)]
    if not len(d):
        return np.nan
    return float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))


SRC_TEST = 'TEST bootstrap'
SRC_JOINT = 'joint: draw (b mod D) x TEST resample b'
SRC_DRAWS = 'draws only'


def _pct(d):
    """(mean, 2.5 %, 97.5 %, two-sided p, number of finite values) of a resample distribution."""
    d = np.asarray(d, dtype=float)
    fin = d[np.isfinite(d)]
    if not len(fin):
        return np.nan, np.nan, np.nan, np.nan, 0
    return float(fin.mean()), float(np.percentile(fin, 2.5)), float(np.percentile(fin, 97.5)), _ptwo(d), len(fin)


def _cmp_row(ref, tag, ds, method, info, n, metric, point, d, unit, sites=None, draw_based=False, d_joint=None,
             no_joint_reason=''):
    """One comparison row. Deterministic (draw_based=False): d = paired TEST-bootstrap differences.
    Draw-based: d = per-draw differences on the full TEST set (draws-only interval) and d_joint = the joint
    draws x TEST-resample differences; the primary diff_mean / diff_lo / diff_hi / p_two_sided /
    n_resamples are the joint ones whenever d_joint has a finite value, else the draws-only ones."""
    m, lo, hi, p, nf = _pct(d)
    r = {'reference': ref, 'tag': tag, 'dataset': ds, 'method': method, 'info': info, 'n': int(n),
         'metric': metric, 'diff_point': float(point), 'diff_mean': m, 'diff_lo': lo, 'diff_hi': hi,
         'p_two_sided': p, 'n_resamples': int(nf), 'bootstrap_unit': unit,
         'sites': ds if sites is None else sites, 'interval_source': SRC_TEST,
         'diff_lo_draws': np.nan, 'diff_hi_draws': np.nan, 'p_two_sided_draws': np.nan,
         'diff_lo_joint': np.nan, 'diff_hi_joint': np.nan, 'p_two_sided_joint': np.nan}
    if not draw_based:
        return r
    r.update({'diff_lo_draws': lo, 'diff_hi_draws': hi, 'p_two_sided_draws': p})
    jm, jlo, jhi, jp, jn = _pct(d_joint) if d_joint is not None else (np.nan, np.nan, np.nan, np.nan, 0)
    if jn:
        r.update({'diff_mean': jm, 'diff_lo': jlo, 'diff_hi': jhi, 'p_two_sided': jp, 'n_resamples': int(jn),
                  'diff_lo_joint': jlo, 'diff_hi_joint': jhi, 'p_two_sided_joint': jp,
                  'interval_source': SRC_JOINT})
    else:
        r['interval_source'] = SRC_DRAWS + (f' ({no_joint_reason})' if no_joint_reason else '')
    return r


def compare(draws, results, common=None):
    """Paired differences (see the module docstring, recal_compare.csv)."""
    rows = []
    common = common_sites(results) if common is None else list(common)
    for R in results:
        tag = R.tag
        aggs = _aggregates(R, common)
        dt = draws[draws['tag'] == tag]
        counts = dt.groupby(['dataset', 'method', 'n'])['draw'].count()
        # deterministic methods: paired cluster bootstrap on TEST
        det_methods = sorted({m for (ds, m) in R.boot}, key=str)
        for ref in REFERENCES:
            for method in det_methods:
                if method == ref:
                    continue
                info = dt.loc[dt['method'] == method, 'info'].iloc[0]
                nval = int(dt.loc[dt['method'] == method, 'n'].iloc[0])
                for metric in CMP_METRICS:
                    per = {}
                    for ds in R.sites:
                        a, b = R.boot.get((ds, method)), R.boot.get((ds, ref))
                        if a is None or b is None:
                            continue
                        per[ds] = a[metric] - b[metric]
                        fa, fb = R.full[(ds, method)], R.full[(ds, ref)]
                        rows.append(_cmp_row(ref, tag, ds, method, info, nval, metric, fa[metric] - fb[metric],
                                             per[ds], R.units[ds]))
                    for label, sites in aggs:
                        if not set(sites) <= set(per):
                            continue
                        point = np.mean([R.full[(ds, method)][metric] - R.full[(ds, ref)][metric] for ds in sites])
                        rows.append(_cmp_row(ref, tag, label, method, info, nval, metric, point,
                                             np.mean([per[ds] for ds in sites], axis=0),
                                             'mean of independent per-dataset resamples', sites=','.join(sites)))

        # draw-based methods: over identical draws (draws only) and jointly over draws x TEST resamples
        jcache = {}

        def joint(ds, method, n, keys):
            """(B,) metric arrays: TEST resample b scored with the prediction of draw keys[b mod len(keys)]."""
            k = (ds, method, int(n), tuple(keys))
            if k not in jcache:
                preds = R.dpred.get((ds, method, int(n)), {})
                sc = R.scorers.get(ds)
                ok = sc is not None and sc.B > 0 and len(keys) and all(kk in preds for kk in keys)
                jcache[k] = sc.boot_cycled([preds[kk] for kk in keys]) if ok else None
            return jcache[k]

        def why_not(ds, method, n):
            if (ds, method, int(n)) not in R.dpred:
                return 'no stored TEST predictions'
            sc = R.scorers.get(ds)
            return '--B 0' if sc is not None and sc.B == 0 else 'joint resamples unavailable'

        draw_keys = sorted({(m, int(n)) for (ds, m, n), c in counts.items() if c > 1 or int(n) > 0}, key=str)
        for method, n in draw_keys:
            info = dt.loc[(dt['method'] == method) & (dt['n'] == n), 'info'].iloc[0]
            refs = list(REFERENCES) + (['fewshot_plain'] if n > 0 and method != 'fewshot_plain' else [])
            for ref in refs:
                for metric in CMP_METRICS:
                    per, perj = {}, {}
                    for ds in R.sites:
                        a = dt[(dt['dataset'] == ds) & (dt['method'] == method) & (dt['n'] == n)].set_index('draw')
                        if a.empty:
                            continue
                        if ref == 'fewshot_plain':
                            b = dt[(dt['dataset'] == ds) & (dt['method'] == ref) & (dt['n'] == n)].set_index('draw')
                            common_d = a.index.intersection(b.index)
                            d = a.loc[common_d, metric].astype(float) - b.loc[common_d, metric].astype(float)
                            keys = sorted(int(x) for x in common_d)
                            ja, jb = joint(ds, method, n, keys), joint(ds, ref, n, keys)
                            dj = ja[metric] - jb[metric] if ja is not None and jb is not None else None
                        else:
                            b = dt[(dt['dataset'] == ds) & (dt['method'] == ref) & (dt['n'] == 0)]
                            if b.empty:
                                continue
                            d = a[metric].astype(float) - float(b[metric].iloc[0])
                            ja = joint(ds, method, n, sorted(int(x) for x in a.index))
                            bb = R.boot.get((ds, ref))                   # the SAME TEST resamples b
                            dj = ja[metric] - bb[metric] if ja is not None and bb is not None else None
                        if not len(d):
                            continue
                        per[ds], perj[ds] = d, dj
                        unit = 'paired over identical draws' if ref == 'fewshot_plain' else (
                            'perturbation draws vs deterministic reference' if n == 0 else
                            'draws vs deterministic reference')
                        if dj is not None:
                            unit += f'; joint with TEST resamples ({R.units[ds]})'
                        rows.append(_cmp_row(ref, tag, ds, method, info, n, metric, float(np.nanmean(d.values)),
                                             d.values, unit, draw_based=True, d_joint=dj,
                                             no_joint_reason='' if dj is not None else why_not(ds, method, n)))
                    for label, sites in aggs:
                        if not set(sites) <= set(per):
                            continue
                        md = pd.concat([per[ds] for ds in sites], axis=1, join='inner').mean(axis=1).values
                        has_j = all(perj[ds] is not None for ds in sites)
                        dj = np.mean([perj[ds] for ds in sites], axis=0) if has_j else None
                        unit = 'mean over datasets per draw' + (
                            '; joint: mean over datasets per resample index b' if has_j else '')
                        reason = '' if has_j else '; '.join(sorted({why_not(ds, method, n) for ds in sites
                                                                    if perj[ds] is None}))
                        rows.append(_cmp_row(ref, tag, label, method, info, n, metric, float(np.nanmean(md)), md,
                                             unit, sites=','.join(sites), draw_based=True, d_joint=dj,
                                             no_joint_reason=reason))
    cols = ['reference', 'tag', 'dataset', 'method', 'info', 'n', 'metric', 'diff_point', 'diff_mean', 'diff_lo',
            'diff_hi', 'p_two_sided', 'n_resamples', 'bootstrap_unit', 'interval_source', 'sites',
            'diff_lo_draws', 'diff_hi_draws', 'p_two_sided_draws', 'diff_lo_joint', 'diff_hi_joint',
            'p_two_sided_joint']
    return pd.DataFrame(rows, columns=cols)


def implied_binary(summary, results):
    """Referable sens/spec implied by every 5-class method (pred >= 2) on TEST, with PPV/NPV."""
    rows = []
    for R in results:
        s = summary[(summary['tag'] == R.tag) & ~summary['dataset'].isin(AGGREGATES)]
        for _, r in s.iterrows():
            fm = R.full.get((r['dataset'], r['method'])) or {}
            rows.append(_bin_row(R.tag, r['dataset'], r['method'], r['info'], int(r['n']), np.nan, np.nan, None,
                                 r['bin_sens_mean'], r['bin_spec_mean'], fm.get('referable_prevalence', np.nan),
                                 fm.get('referrals_per_1000', np.nan), fm.get('missed_referable_per_1000', np.nan)))
    return rows


def _mean_binary(df, results):
    out = []
    by_tag = {R.tag: R for R in results}
    for (tag, method, info, n, spec), g in df.groupby(['tag', 'method', 'info', 'n', 'target_spec'], sort=False,
                                                      dropna=False):
        R = by_tag[tag]
        if not _complete(R, g):
            continue
        r = {'tag': tag, 'dataset': 'MEAN', 'method': method, 'info': info, 'n': n, 'target_spec': spec}
        for c in g.columns:
            if c not in r and pd.api.types.is_numeric_dtype(g[c]):
                r[c] = float(g[c].mean())
        r['threshold'] = np.nan
        out.append(r)
    return out


def indist_control(results, summary, notes):
    rows = []
    for R in results:
        src = R.source
        p = _pred_path(R.exp_dir, 'aptos', 'test')
        if src is None or p is None:
            notes.add(R.tag, 'aptos_test', 'indist_control', 'APTOS test predictions or OOF missing: NaN')
            rows.append({'tag': R.tag, 'method': 'n/a', 'note': 'APTOS test predictions or OOF missing'})
            continue
        ids, y_at, P_at, _ = _load_file(p)
        E_at = expected_grade(P_at)
        q_arg = grade_metrics(y_at, P_at.argmax(1))['qwk']
        internal = {'default_cuts': apply_cutpoints(E_at, DEFAULT_CUTS),
                    'source_cuts': apply_cutpoints(E_at, R.c_src),
                    'oracle_quantile': apply_cutpoints(E_at, prevalence_cutpoints(src['E'], src['pi']))}
        desc = {'default_cuts': 'cut-points 0.5..3.5', 'source_cuts': 'cut-points fitted on APTOS OOF',
                'oracle_quantile': 'quantile cut-points: OOF E with OOF grade shares (in-distribution oracle)'}
        # Estimator-bias control (audit recommendation): the label-shift estimators are fitted on single-fold OOF
        # probabilities but applied to 5-fold ensembles. APTOS test has (almost) no shift relative to the OOF, so
        # any prior the estimators "find" there, and any QWK change they cause, is estimator bias, not adaptation.
        # The estimators see the APTOS-test images unlabelled (transductive control; labels only for metrics).
        bcts = ls.fit_bcts(src['P'], src['y'])
        P_at_b = ls.apply_bcts(P_at, bcts)
        pi_true = ls.class_shares(y_at, K)
        est = {'mlls_bcts': ls.mlls_em(P_at_b, src['pi'])[0], 'bbse': ls.bbse(src['P'], src['y'], P_at)}
        pi_err = {}
        for name, pi in est.items():
            pi_err[name] = float(0.5 * np.abs(np.asarray(pi) - pi_true).sum())
            internal[f'{name}_quantile'] = apply_cutpoints(E_at, prevalence_cutpoints(E_at, pi))
            internal[f'{name}_bayes'] = ls.prior_correct(P_at_b, src['pi'], pi).argmax(1)
            desc[f'{name}_quantile'] = (f'{name} prior estimated on unlabelled APTOS-test images, quantile cut-points '
                                        '(estimator-bias control)')
            desc[f'{name}_bayes'] = f'{name} prior estimated on unlabelled APTOS-test images, Bayes correction'
        mean = summary[(summary['tag'] == R.tag) & (summary['dataset'] == 'MEAN')].set_index('method')
        for m, pred in internal.items():
            qi = grade_metrics(y_at, pred)['qwk']
            if m in mean.index and 'argmax' in mean.index:
                ext = float(mean.loc[m, 'qwk_mean'] - mean.loc['argmax', 'qwk_mean'])
                note = ''
            else:
                ext, note = np.nan, 'external MEAN missing (not every held-out dataset present)'
            est_name = m.rsplit('_', 1)[0] if m.endswith(('_quantile', '_bayes')) else ''
            rows.append({'tag': R.tag, 'method': m, 'internal_definition': desc[m], 'aptos_test_n': len(y_at),
                         'prior_tv_error_vs_aptos_test': pi_err.get(est_name, np.nan),
                         'aptos_test_qwk_argmax': q_arg, 'aptos_test_qwk_method': qi, 'internal_gain': qi - q_arg,
                         'external_gain': ext, 'net_external_gain': ext - (qi - q_arg),
                         'external_sites': ','.join(R.heldout), 'note': note})
    return pd.DataFrame(rows)


def aptos_dup_sensitivity(results, notes):
    rows = []
    pairs = None
    cols = ['tag', 'set', 'eval_id', 'train_dataset', 'train_id']
    if os.path.isfile(LEAKAGE_PAIRS):
        try:
            pairs = pd.read_csv(LEAKAGE_PAIRS, dtype=str)
        except pd.errors.EmptyDataError:          # leakage_check writes an empty file when nothing matched
            pairs = pd.DataFrame(columns=cols)
        if len(pairs) and any(c not in pairs.columns for c in cols):
            notes.add('-', 'aptos_test', 'aptos_dup_sensitivity', f'{LEAKAGE_PAIRS} lacks columns {cols}', echo=True)
            pairs = None
        elif not len(pairs):
            pairs = pd.DataFrame(columns=cols)
    for R in results:
        p = _pred_path(R.exp_dir, 'aptos', 'test')
        if p is None:
            rows.append({'tag': R.tag, 'subset': 'n/a', 'note': 'no APTOS test predictions'})
            continue
        if pairs is None:
            rows.append({'tag': R.tag, 'subset': 'n/a',
                         'note': f'{LEAKAGE_PAIRS} absent: run tools/leakage_check.py first'})
            notes.add(R.tag, 'aptos_test', 'aptos_dup_sensitivity', f'{LEAKAGE_PAIRS} absent')
            continue
        ids, y, P, _ = _load_file(p)
        sub = pairs[(pairs['set'] == 'aptos_test') & (pairs['train_dataset'] == 'APTOS')]
        own_all = pairs[(pairs['set'] == 'aptos_test') & (pairs['tag'] == R.tag)]
        own = sub[sub['tag'] == R.tag]
        if len(own_all):
            # every training source of this run (APTOS train, EyePACS, extra sources), not only APTOS train
            dup = {_stem(i) for i in own_all['eval_id']}
            src = (f'{LEAKAGE_PAIRS} rows of tag {R.tag}, all training sources '
                   f'({",".join(sorted(set(own_all["train_dataset"])))})')
        elif len(own):
            dup, src = {_stem(i) for i in own['eval_id']}, f'{LEAKAGE_PAIRS} rows of tag {R.tag}'
        else:
            dup = {_stem(i) for i in sub['eval_id']} if len(sub) else set()
            src = (f'{LEAKAGE_PAIRS} rows of all tags (no rows for {R.tag}; the APTOS train pool is identical '
                   'for every run)')
        keep = np.array([_stem(i) not in dup for i in ids])
        E = expected_grade(P)
        if not len(own_all):
            notes.add(R.tag, 'aptos_test', 'aptos_dup_sensitivity', 'no leakage_check rows for this tag: only '
                      'APTOS-train duplicates excluded; run tools/leakage_check.py --tags ' + R.tag)
        sub_name = 'excl_dups_of_training_pool' if len(own_all) else 'excl_dups_of_aptos_train'
        for subset, m in (('all', np.ones(len(ids), bool)), (sub_name, keep)):
            for method, pred in (('argmax', P.argmax(1)), ('default_cuts', apply_cutpoints(E, DEFAULT_CUTS))):
                fm = cl.full_metrics(y[m], pred[m], E[m], P[m, 2:].sum(1))
                rows.append({'tag': R.tag, 'subset': subset, 'method': method, 'n_excluded': int((~keep).sum()),
                             'pairs_from': src, **fm, 'note': ''})
    return pd.DataFrame(rows)


def merge_fewshot_head(draws, out_dir, results, notes):
    p = os.path.join(out_dir, 'fewshot_head_draws.csv')
    if not os.path.isfile(p):
        return draws
    h = pd.read_csv(p)
    missing = [c for c in DRAW_COLS if c not in h.columns]
    if missing:
        notes.add('-', '-', 'fewshot_head', f'{p} lacks columns {missing}: not merged', echo=True)
        return draws
    ok = set()
    for R in results:
        ok |= {(R.tag, ds) for ds in R.sites}
    keep = np.array([(t, d) in ok for t, d in zip(h['tag'], h['dataset'])], dtype=bool)
    if (~keep).any():
        notes.add('-', '-', 'fewshot_head', f'{int((~keep).sum())} rows for tags/datasets not analysed: dropped')
    bad = ~h['info'].isin(INFO_LEVELS)
    if (keep & bad).any():
        notes.add('-', '-', 'fewshot_head', 'rows with an unknown info level dropped', echo=True)
        keep &= ~bad
    print(f'recalibration_suite: merged {int(keep.sum())} rows of {p}')
    return pd.concat([draws, h[keep]], ignore_index=True, sort=False)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

INFO_COLORS = {'none': '#7f7f7f', 'unlabelled_target': '#1f77b4', 'prevalence_oracle': '#d62728',
               'prevalence_other_sites': '#9467bd', 'prevalence_estimated': '#17becf',
               'prevalence_misspecified': '#ff7f0e', 'n_labels': '#2ca02c', 'all_dev_labels': '#8c564b'}


def _plt():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'font.weight': 'bold', 'axes.labelweight': 'bold',
                         'axes.titleweight': 'bold'})
    return plt


def _save(fig, stem):
    os.makedirs(os.path.dirname(stem), exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(f'{stem}.{ext}', dpi=200, bbox_inches='tight')


def plot_levels(summary, tags, fig_n, stem):
    plt = _plt()
    order = list(INFO_LEVELS)
    fig, axes = plt.subplots(1, len(tags), figsize=(5.6 * len(tags), 9.0), squeeze=False)
    for ax, tag in zip(axes[0], tags):
        g = summary[(summary['tag'] == tag) & (summary['dataset'] == 'MEAN')]
        g = g[(g['n'] <= 0) | (g['n'] == fig_n)]
        g = g[~g['method'].str.startswith('misspecified_quantile_nodr')]
        if g.empty:
            ax.set_title(f'{tag}: no complete MEAN')
            ax.axis('off')
            continue
        g = g.assign(_o=g['info'].map(order.index)).sort_values(['_o', 'qwk_mean'])
        y = np.arange(len(g))
        lo = (g['qwk_mean'] - g['qwk_lo']).clip(lower=0).fillna(0)
        hi = (g['qwk_hi'] - g['qwk_mean']).clip(lower=0).fillna(0)
        ax.barh(y, g['qwk_mean'], xerr=[lo, hi], color=[INFO_COLORS[i] for i in g['info']], capsize=2)
        for yi, v, h in zip(y, g['qwk_mean'], hi):
            if np.isfinite(v):
                ax.text(v + h + 0.004, yi, f'{v:.2f}', va='center', fontsize=7)
        labels = [m if n <= 0 else f'{m} (n={int(n)})' for m, n in zip(g['method'], g['n'])]
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=7)
        fin = g['qwk_mean'][np.isfinite(g['qwk_mean'])]
        if len(fin):
            ax.set_xlim(max(0.0, fin.min() - 0.1), min(1.02, float(np.nanmax(g['qwk_hi'].fillna(fin.max()))) + 0.05))
        ax.set_xlabel('QWK on held-out TEST (mean over sites)')
        ax.set_title(tag)
    used = [i for i in INFO_COLORS if i in set(summary['info'])]
    handles = [plt.Rectangle((0, 0), 1, 1, color=INFO_COLORS[i]) for i in used]
    fig.legend(handles, used, loc='lower center', ncol=4, fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, 0.025))
    fig.text(0.5, 0.005, 'error bars: 95% bootstrap interval (deterministic methods) or 2.5-97.5 percentiles over '
             'draws (few-label / random perturbation)', ha='center', fontsize=7)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    _save(fig, stem)
    plt.close(fig)


def plot_misspec(summary, tags, stem):
    plt = _plt()
    fig, axes = plt.subplots(1, len(tags), figsize=(4.6 * len(tags), 3.8), squeeze=False)
    for ax, tag in zip(axes[0], tags):
        g = summary[(summary['tag'] == tag) & (summary['dataset'] == 'MEAN')].set_index('method')
        if 'oracle_quantile' not in g.index:
            ax.axis('off')
            continue
        q0 = g.loc['oracle_quantile', 'qwk_mean']
        xs, ys = [0.0], [q0]
        for d in NODR_DELTAS:
            m = f'misspecified_quantile_nodr{d:+.2f}'
            if m in g.index:
                xs.append(d)
                ys.append(g.loc[m, 'qwk_mean'])
        o = np.argsort(xs)
        ax.plot(np.array(xs)[o], np.array(ys)[o], marker='o', label='No-DR share shifted by delta')
        tx, ty, te = [0.0], [q0], [0.0]
        for d in TV_DELTAS:
            m = f'misspecified_quantile_tv{d:.2f}'
            if m in g.index:
                tx.append(d)
                ty.append(g.loc[m, 'qwk_mean'])
                te.append(g.loc[m, 'qwk_sd'])
        ax.errorbar(tx, ty, yerr=te, marker='s', capsize=3, label='random TV = delta (mean +- SD)')
        for m, ls_, lab in (('argmax', ':', 'argmax'), ('mlls_bcts_quantile', '--', 'MLLS-BCTS quantile')):
            if m in g.index and np.isfinite(g.loc[m, 'qwk_mean']):
                ax.axhline(g.loc[m, 'qwk_mean'], ls=ls_, color='k', lw=1, label=f'{lab} {g.loc[m, "qwk_mean"]:.2f}')
        for x, yv in zip(np.array(xs)[o], np.array(ys)[o]):
            ax.annotate(f'{yv:.2f}', (x, yv), textcoords='offset points', xytext=(0, 5), fontsize=7, ha='center')
        ax.set_xticks([-0.10, -0.05, 0.0, 0.05, 0.10])
        ax.set_xticklabels(['-0.10', '-0.05', '0.00', '0.05', '0.10'])
        ax.set_xlabel('prevalence perturbation delta (0 = oracle DEV shares)')
        ax.set_ylabel('QWK (mean over held-out TEST)')
        ax.set_title(tag)
        ax.legend(fontsize=6)
    fig.tight_layout()
    _save(fig, stem)
    plt.close(fig)


def plot_operating_points(results, summary, binary, stem):
    plt = _plt()
    res = [R for R in results if R.roc]
    if not res:
        return
    ncol = max(len(R.sites) for R in res)
    fig, axes = plt.subplots(len(res), ncol, figsize=(3.3 * ncol, 3.2 * len(res)), squeeze=False)
    marks = (('argmax', 'x', 'argmax'), ('oracle_quantile', 's', 'oracle quantile'),
             ('mlls_bcts_bayes', '^', 'MLLS-BCTS Bayes'))
    for row_axes, R in zip(axes, res):
        s = summary[(summary['tag'] == R.tag)]
        b = binary[(binary['tag'] == R.tag) & (binary['method'] == 'dev_threshold_pref')
                   & np.isclose(binary['target_spec'].astype(float), 0.95)]
        for ax in row_axes[len(R.sites):]:
            ax.axis('off')
        for ax, ds in zip(row_axes, R.sites):
            fpr, tpr, auc = R.roc[ds]
            ax.plot(fpr, tpr, color='k', lw=1.2, label=f'P(grade>=2) AUROC {auc:.2f}')
            for m, mk, lab in marks:
                r = s[(s['dataset'] == ds) & (s['method'] == m)]
                if len(r) and np.isfinite(r['bin_sens_mean'].iloc[0]):
                    ax.scatter(1 - r['bin_spec_mean'].iloc[0], r['bin_sens_mean'].iloc[0], marker=mk, s=40,
                               zorder=3, label=f'{lab} ({r["bin_sens_mean"].iloc[0]:.2f}/{r["bin_spec_mean"].iloc[0]:.2f})')
            r = b[b['dataset'] == ds]
            if len(r) and np.isfinite(r['sens'].iloc[0]):
                ax.scatter(1 - r['spec'].iloc[0], r['sens'].iloc[0], marker='o', s=40, zorder=3,
                           label=f'DEV thr spec.95 ({r["sens"].iloc[0]:.2f}/{r["spec"].iloc[0]:.2f})')
            ys_ = [c.get_offsets()[0][1] for c in ax.collections if len(c.get_offsets())]
            ax.set_xlim(-0.01, 0.6)
            ax.set_ylim(min([0.4] + [v - 0.05 for v in ys_ if np.isfinite(v)]), 1.01)
            ax.set_xlabel('1 - specificity')
            ax.set_ylabel('sensitivity (referable)')
            ax.set_title(f'{R.tag} / {ds}', fontsize=8)
            ax.legend(fontsize=5.5, loc='lower right')
    fig.tight_layout()
    _save(fig, stem)
    plt.close(fig)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tags', required=True)
    ap.add_argument('--ns', default='10,25,50,100')
    ap.add_argument('--draws', type=int, default=20)
    ap.add_argument('--B', type=int, default=1000)
    ap.add_argument('--misspec_draws', type=int, default=50)
    ap.add_argument('--B_pi', type=int, default=200,
                    help='bootstrap resamples of the unlabelled DEV images for the prior uncertainty '
                         '(bagged / shrunk quantile methods)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--fig_n', type=int, default=25, help='n shown for few-label methods in fig_recal_levels')
    ap.add_argument('--out_dir', default=os.path.join('results', 'master', 'recal'))
    a = ap.parse_args(argv)
    tags = [t.strip() for t in a.tags.split(',') if t.strip()]
    ns = [int(x) for x in a.ns.split(',') if x.strip()]
    notes = Notes()
    results = []
    t_start = time.perf_counter()
    timing = []
    for tag in tags:
        t0 = time.perf_counter()
        R = analyse_tag(tag, a, ns, notes)
        if R is not None and R.sites:
            results.append(R)
            dt_ = time.perf_counter() - t0
            timing.append((f'analyse {tag}', dt_))
            print(f'recalibration_suite: {tag}: {len(R.sites)} held-out sites analysed in {dt_:.1f} s '
                  f'(prior bootstrap {R.seconds.get("prior_bootstrap", 0.0):.1f} s)', flush=True)
    if not results:
        print('recalibration_suite: no held-out dev/test predictions found for any tag')
        sys.exit(1)
    os.makedirs(a.out_dir, exist_ok=True)
    common = common_sites(results)
    rtags = [R.tag for R in results]
    dropped = [t for t in tags if t not in rtags]
    notes.add(','.join(rtags), 'MEAN_COMMON', '-',
              'MEAN_COMMON = unweighted mean over the held-out datasets analysed for EVERY tag (intersection): '
              + (','.join(common) if common else 'EMPTY (no MEAN_COMMON rows)') + '; per-tag held-out sets: '
              + '; '.join(f'{R.tag}: {",".join(R.sites)}' for R in results)
              + (f'; tags not analysed (excluded from the intersection): {",".join(dropped)}' if dropped else ''))
    draws = pd.DataFrame([r for R in results for r in R.draws])
    draws = draws[DRAW_COLS + [c for c in draws.columns if c not in DRAW_COLS]]
    draws.round(6).to_csv(os.path.join(a.out_dir, 'recal_draws.csv'), index=False)
    draws_all = merge_fewshot_head(draws, a.out_dir, results, notes)
    t0 = time.perf_counter()
    summary = summarise(draws_all, results, common)
    summary.round(6).to_csv(os.path.join(a.out_dir, 'recal_summary.csv'), index=False)
    timing.append(('summarise', time.perf_counter() - t0))
    t0 = time.perf_counter()
    cmp = compare(draws_all, results, common)
    cmp.round(6).to_csv(os.path.join(a.out_dir, 'recal_compare.csv'), index=False)
    timing.append(('compare (incl. joint draws x TEST bootstrap)', time.perf_counter() - t0))
    t0 = time.perf_counter()
    pd.DataFrame([r for R in results for r in R.prev]).round(6).to_csv(
        os.path.join(a.out_dir, 'prevalence_estimates.csv'), index=False)
    bdf = pd.DataFrame([r for R in results for r in R.binary] + implied_binary(summary, results))
    bdf = pd.concat([bdf, pd.DataFrame(_mean_binary(bdf, results))], ignore_index=True, sort=False)
    bdf.round(6).to_csv(os.path.join(a.out_dir, 'binary_operating_points.csv'), index=False)
    od = pd.DataFrame([r for R in results for r in R.ordinal])
    od_mean = []
    for R in results:
        g = od[od['tag'] == R.tag]
        if _complete(R, g):
            od_mean.append({'tag': R.tag, 'dataset': 'MEAN', **g.mean(numeric_only=True).to_dict(), 'note': ORD_NOTE})
    pd.concat([od, pd.DataFrame(od_mean)], ignore_index=True, sort=False).round(6).to_csv(
        os.path.join(a.out_dir, 'ordinal_discrimination.csv'), index=False)
    indist_control(results, summary, notes).round(6).to_csv(os.path.join(a.out_dir, 'indist_control.csv'),
                                                            index=False)
    aptos_dup_sensitivity(results, notes).round(6).to_csv(os.path.join(a.out_dir, 'aptos_dup_sensitivity.csv'),
                                                          index=False)
    pd.DataFrame([r for R in results for r in R.fitted]).to_csv(os.path.join(a.out_dir, 'recal_fitted.csv'),
                                                                index=False, float_format='%.10g')
    pd.DataFrame(notes.rows, columns=['tag', 'dataset', 'method', 'note']).to_csv(
        os.path.join(a.out_dir, 'recal_notes.csv'), index=False)
    timing.append(('other tables', time.perf_counter() - t0))
    t0 = time.perf_counter()
    fig_dir = os.path.join('results', 'figures')
    plot_levels(summary, rtags, a.fig_n, os.path.join(fig_dir, 'fig_recal_levels'))
    plot_misspec(summary, rtags, os.path.join(fig_dir, 'fig_recal_misspec'))
    plot_operating_points(results, summary, bdf, os.path.join(fig_dir, 'fig_recal_operating_points'))
    timing.append(('figures', time.perf_counter() - t0))

    for label, title in (('MEAN', 'MEAN external QWK (held-out TEST; per-tag held-out sets) by method:'),
                         ('MEAN_COMMON', f'MEAN_COMMON external QWK (datasets common to every tag: '
                                         f'{",".join(common)}) by method:')):
        show = summary[(summary['dataset'] == label) & ((summary['n'] <= 0) | (summary['n'] == max(ns)))]
        show = show[~show['method'].str.startswith('misspecified_quantile_nodr')]
        if label == 'MEAN_COMMON' and all(set(R.sites) == set(common) for R in results):
            print('\nMEAN_COMMON = MEAN for every tag (all tags share their held-out sets)')
            continue
        if len(show):
            pv = show.set_index(['info', 'method', 'n', 'tag'])['qwk_mean'].unstack('tag')
            pv = pv.reindex(pd.MultiIndex.from_frame(show[['info', 'method', 'n']].drop_duplicates()))
            with pd.option_context('display.width', 220, 'display.max_rows', 200):
                print('\n' + title)
                print(pv.round(3).to_string())
    if notes.rows:
        print(f'\n{len(notes.rows)} notes (NaN / skipped items) -> {a.out_dir}/recal_notes.csv')
    print(f'written: {a.out_dir}/recal_draws.csv, recal_summary.csv, recal_compare.csv, prevalence_estimates.csv, '
          f'binary_operating_points.csv, ordinal_discrimination.csv, indist_control.csv, '
          f'aptos_dup_sensitivity.csv, recal_fitted.csv, recal_notes.csv; '
          f'{fig_dir}/fig_recal_levels, fig_recal_misspec, fig_recal_operating_points (.png/.pdf)')
    print('timing: ' + '; '.join(f'{k} {v:.1f} s' for k, v in timing)
          + f'; total {time.perf_counter() - t_start:.1f} s')


if __name__ == '__main__':
    main()
