"""Controlled decomposition of the external recalibration gain: prior (label) shift vs cut-point /
annotation shift, on the REAL saved predictions (CPU only; no network is touched, nothing is retrained).

Reviewer question: when re-fitting the 4 ordinal cut-points improves external grading, is that only the
correction of a different grade distribution (prior shift: p(x|y) unchanged, only the class shares move),
or is there a genuine cut-point / annotation shift (the same image receives another grade at another
site: p(y|x) changes)?

Data per (tag, held-out dataset d) -- only datasets that are NOT training sources of the run:
  * target: temperature-scaled class probabilities (key y_prob) of d's DEV and TEST splits, sorted by image
    id exactly as tools/adaptation_curve._load;
  * source: the run's APTOS out-of-fold predictions (oof.npz, raw probabilities), put on the SAME
    temperature as the target files (T recovered from the (y_prob_raw, y_prob) pairs of the prediction
    files, else refitted on the OOF) with tools/recalibration_suite.load_source (imported; an identical
    local copy is used only if that module cannot be imported, which is written to the notes file).
    The OOF images are sorted by id and split ONCE, stratified by grade, 50/50 with a fixed seed into a
    pseudo-source half S and a pseudo-target pool T. Everything source-side (BCTS calibration, the
    source prior pi_S, source cut-points -- recorded for reference only) is fitted on S only, in every
    scenario, so all scenarios share one source.

Scenarios (each yields a DEV split used for fitting and a TEST split used for metrics only). They form a
2 x 2 design, images {APTOS pool T, site d} x grade shares {source S, site d}:
  A0 no shift at all (reference added to the requested A/B/C; semi-synthetic): exactly as A but resampled
     to the shares of S. Needed because the generic gain of QWK-optimal cut-points over argmax (and of
     BCTS over the raw argmax) depends on the grade distribution itself, so B (source shares) must be
     compared with A0 (source shares), and C (site shares) with A (site shares).
  A  pure prior shift, in-distribution images (semi-synthetic). DEV and TEST are drawn WITH replacement
     from the pool T so that their grade counts reproduce the grade shares of d's DEV labels (shares used
     only to build the scenario); sizes |d DEV| and |d TEST|. Per draw T is split again, stratified, into
     two disjoint halves (DEV from one, TEST from the other) -- impossible only for a grade with a single
     image in T (then that image is shared; noted). Here p(x|y) is APTOS's: only the priors differ.
  B  acquisition / annotation shift WITHOUT prior shift (semi-synthetic). d's own DEV and TEST are each
     resampled with replacement, within the split, to the grade shares of S. Images and labels are d's,
     the priors equal the source's.
  C  observed. d's DEV / TEST as they are. Point estimate on the observed splits (`*_point` columns)
     and R stratified bootstrap draws (each split resampled within grade to its OWN counts) so that the
     interval is built exactly like those of A0, A and B (B and C then differ only in the shares).
  Grade counts of a resampled split are exact (largest-remainder rounding of n * shares); a grade with
  positive target share but absent from the pool is dropped and the other shares renormalised (noted).

Methods (fitted on S and the scenario's DEV only; `info` column = target information used):
  argmax              none                 the model's own prediction
  default_cuts        none                 cut-points 0.5, 1.5, 2.5, 3.5 on E = sum_k k P_k
  oracle_bayes        prevalence_oracle    BCTS(S)-calibrated TEST probabilities, Bayes prior correction
                                           pi_S -> DEV label shares (ORACLE: true shares), argmax
  oracle_quantile     prevalence_oracle    cut-points = quantiles of unlabelled DEV E reproducing the DEV
                                           label shares (ORACLE)
  mlls_bcts_bayes     unlabelled_target    Saerens EM on BCTS-calibrated unlabelled DEV -> pi_hat; Bayes
                                           correction of BCTS TEST probabilities, argmax
  mlls_bcts_quantile  unlabelled_target    pi_hat as above; quantile cut-points on unlabelled DEV E
  all_dev_cuts        all_dev_labels       QWK-optimal cut-points on E with every DEV label
  all_dev_bayes_cuts  all_dev_labels       oracle_bayes correction of DEV and TEST, then QWK-optimal
                                           cut-points on the corrected E with every DEV label: does
                                           re-fitting cut-points add anything beyond prior correction?
Metrics on TEST: QWK, mild recall (grade 1), referable (grade >= 2) sensitivity / specificity, macro-F1
(macro-F1 averages over the grades PRESENT in that TEST split; mild recall is NaN without grade-1 images;
src.evaluation.cutpoints.grade_metrics otherwise). Mean and 2.5 / 97.5 percentiles over the R draws
(percentiles of the per-draw distribution: the spread of one replicate of the scenario, not an interval
for the mean).

Contrasts (per draw, then summarised; also for every metric):
  gain_bayes              = oracle_bayes - argmax (includes the BCTS(S) recalibration itself, which is
                            measured alone in A0 / B where the DEV shares equal pi_S up to rounding)
  gain_cuts_over_bayes    = all_dev_bayes_cuts - oracle_bayes
  gain_quantile           = oracle_quantile - argmax
  gain_cuts_over_quantile = all_dev_cuts - oracle_quantile
  gain_mlls_bayes         = mlls_bcts_bayes - argmax
  mlls_recovery           = (mlls_bcts_bayes - argmax) / (oracle_bayes - argmax); `mean` = ratio of the
                            mean gains (NaN when |mean oracle gain| < 0.01), interval = percentiles of the
                            per-draw ratios over draws with |oracle gain| >= 0.01 (n_valid)
  vs_argmax:<method>      = <method> - argmax (used by the figure)
  cross-scenario rows (scenario column 'X-Y' = the contrast in X minus in Y; per-draw differences of
  independent draws, paired by index only):
    'B-A0'  site images vs APTOS images at the SOURCE shares  -> acquisition / annotation effect
    'C-A'   site images vs APTOS images at the SITE shares    -> acquisition / annotation effect
    'A-A0'  site shares vs source shares on APTOS images      -> pure prior-shift effect
    'C-B'   site shares vs source shares on site images       -> prior-shift effect on site images

HYPOTHESES (how the outputs are meant to be read; not conclusions -- they are tested by the outputs):
  H1  If in B the cut-point refit beyond prior correction (gain_cuts_over_bayes) still gives a clear gain,
      and more than in A0 (row 'B-A0'; likewise 'C-A'), the external error contains annotation /
      cut-point shift beyond prevalence.
  H2  If the gains concentrate in A (gain_bayes, gain_quantile large; 'A-A0' > 0) and vanish in B
      ('B-A0' ~ 0), the external shift is mainly a prior shift.
  H3  MLLS should recover the prior-shift part well in A (mlls_recovery near 1) but cannot recover the
      annotation part of B / C (it re-estimates priors only; p(y|x) changes are invisible to it).

LIMITATIONS (also written into the `caveat` column):
  * A0, A and B are semi-synthetic resamplings of saved predictions, not new sites.
  * A0 / A are conditional on the one fixed pool T (half of the APTOS OOF): the sampling error of that
    pool itself (e.g. which few rare-grade images it holds) is not propagated into their intervals, so
    an A-vs-C difference of the order of a rare-grade pool effect (~0.01 QWK in the simulations of
    tests/test_shift_decomposition.py) is not evidence of anything.
  * Resampling with replacement from a finite pool repeats the few images of rare grades, so the variance
    of rare-grade metrics (and of fits that depend on them) is understated; exact grade counts (no
    multinomial count noise) understate it further.
  * In A0 / A, p(x|y) is that of the APTOS OOF predictions (one fold model per image), whereas the external
    predictions are 5-fold (+TTA) ensembles: A-vs-C differences also contain this ensembling difference.
  * The source half S has half the OOF size, so BCTS / pi_S are noisier than in tools/recalibration_suite.
  * Scenario TEST composition in B / C is built from TEST labels (stratified resampling); this defines
    the evaluation set, never a fitted quantity (DEV and TEST draws use separate random streams; the
    test suite proves fitted quantities are invariant to the TEST labels).

Outputs (--out_dir, default results/master/recal):
  shift_decomposition.csv            tag, dataset (incl. MEAN = unweighted mean over the run's held-out
                                     datasets, per draw), scenario, method, info, R, n_dev, n_test,
                                     <metric>_mean/_lo/_hi/_point, caveat
  shift_decomposition_contrasts.csv  tag, dataset, scenario, contrast, metric, mean, lo, hi, point,
                                     n_valid, R, note
  shift_decomposition_draws.csv      one row per (tag, dataset, scenario, draw, method); draw -1 = observed
  shift_decomposition_fitted.csv     every fitted quantity (BCTS, priors, cut-points) per draw
  shift_decomposition_notes.csv      everything skipped / adjusted and why
  results/figures/fig_shift_decomposition.{png,pdf}

Usage:
  python tools/shift_decomposition.py --tags p3b_joint,p10d_ce,p10q_ce [--R 50] [--seed 0]
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation import label_shift as ls  # noqa: E402
from src.evaluation.cutpoints import (DEFAULT_CUTS, apply_cutpoints, expected_grade, fit_cutpoints,  # noqa: E402
                                      grade_metrics, prevalence_cutpoints)
from src.utils.config import find_exp  # noqa: E402
from tools.adaptation_curve import _heldout, _load  # noqa: E402

K = 5
METHODS = ('argmax', 'default_cuts', 'oracle_bayes', 'oracle_quantile', 'mlls_bcts_bayes', 'mlls_bcts_quantile',
           'all_dev_cuts', 'all_dev_bayes_cuts')
METHOD_INFO = {'argmax': 'none', 'default_cuts': 'none', 'oracle_bayes': 'prevalence_oracle',
               'oracle_quantile': 'prevalence_oracle', 'mlls_bcts_bayes': 'unlabelled_target',
               'mlls_bcts_quantile': 'unlabelled_target', 'all_dev_cuts': 'all_dev_labels',
               'all_dev_bayes_cuts': 'all_dev_labels'}
METRICS = ('qwk', 'mild_recall', 'bin_sens', 'bin_spec', 'f1_macro')
SCENARIOS = ('A0', 'A', 'B', 'C')
SCENARIO_DESC = {'A0': 'no shift: APTOS OOF pool resampled to source (S) shares (semi-synthetic reference)',
                 'A': 'pure prior shift: APTOS OOF pool resampled to site DEV shares (semi-synthetic)',
                 'B': 'site images resampled to source (S) shares: no prior shift (semi-synthetic)',
                 'C': 'observed site DEV/TEST (point) + stratified bootstrap draws'}
SCEN_CODE = {'A': 1, 'B': 2, 'C': 3, 'A0': 4}
CONTRASTS = {'gain_bayes': ('oracle_bayes', 'argmax'),
             'gain_cuts_over_bayes': ('all_dev_bayes_cuts', 'oracle_bayes'),
             'gain_quantile': ('oracle_quantile', 'argmax'),
             'gain_cuts_over_quantile': ('all_dev_cuts', 'oracle_quantile'),
             'gain_mlls_bayes': ('mlls_bcts_bayes', 'argmax')}
CROSS_CONTRASTS = ('gain_bayes', 'gain_cuts_over_bayes', 'gain_quantile', 'gain_cuts_over_quantile')
CROSS_PAIRS = (('B', 'A0'), ('C', 'A'), ('A', 'A0'), ('C', 'B'))
RATIO_MIN = 0.01
SPLIT_SEED = 4242
CAVEAT_SEMI = ('semi-synthetic resampling with replacement of saved predictions (exact grade counts): '
               'understates the variance of rare grades')
CAVEAT_C = 'observed splits; interval from a stratified bootstrap (understates rare-grade variance)'


# ---------------------------------------------------------------------------
# Loading (reuses tools/recalibration_suite when importable)
# ---------------------------------------------------------------------------

class Notes:
    def __init__(self):
        self.rows = []

    def add(self, tag, dataset, method, note, echo=False):
        self.rows.append({'tag': tag, 'dataset': dataset, 'method': method, 'note': note})
        if echo:
            print(f'shift_decomposition: {tag} {dataset} {method}: {note}')


def _load_file_local(path):
    """(ids, y, P, P_raw or None) sorted by id (copy of recalibration_suite._load_file)."""
    with np.load(path, allow_pickle=True) as f:
        y = np.asarray(f['y_true'] if 'y_true' in f.files else f['y'], dtype=int)
        P = np.asarray(f['y_prob'] if 'y_prob' in f.files else f['P'], dtype=float)
        raw = np.asarray(f['y_prob_raw'], dtype=float) if 'y_prob_raw' in f.files else None
        ids = np.array([str(i) for i in f['ids']])
    order = np.argsort(ids, kind='stable')
    return ids[order], y[order], P[order], (raw[order] if raw is not None and raw.shape == P.shape else None)


def _pred_path_local(exp_dir, ds, split):
    from tools.adaptation_curve import LEGACY
    for name in (f'predictions_{ds}_{split}.npz', LEGACY.get((ds, split))):
        if name and os.path.isfile(os.path.join(exp_dir, name)):
            return os.path.join(exp_dir, name)
    return None


def _temperature_from_pair_local(raw, scaled):
    """Copy of recalibration_suite._temperature_from_pair: 1/T from log(scaled) = log(raw)/T + c."""
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


def _load_source_local(exp_dir, tag, raw_pairs, notes):
    """Copy of recalibration_suite.load_source (same temperature logic)."""
    p = os.path.join(exp_dir, 'oof.npz')
    if not os.path.isfile(p):
        notes.add(tag, 'aptos_oof', 'source', 'oof.npz missing', echo=True)
        return None
    with np.load(p, allow_pickle=True) as f:
        y = np.asarray(f['y_true'] if 'y_true' in f.files else f['y'], dtype=int)
        P_raw = np.asarray(f['y_prob'] if 'y_prob' in f.files else f['P'], dtype=float)
    from src.evaluation.metrics import apply_temperature, fit_temperature
    T_refit = float(fit_temperature(np.log(np.clip(P_raw, 1e-15, 1.0)), y))
    est = [t for t in (_temperature_from_pair_local(r, s) for r, s in raw_pairs) if t is not None]
    if est:
        T, src = float(np.median(est)), 'prediction_files'
        if abs(T - T_refit) > 0.02 * T:
            notes.add(tag, 'aptos_oof', 'source', f'T from prediction files {T:.4f} != OOF refit {T_refit:.4f}; '
                      'the file T is used (it is the T applied to the target probabilities)')
    else:
        T, src = T_refit, 'refit_oof'
    P = apply_temperature(P_raw, T)
    return {'P': P, 'y': y, 'E': expected_grade(P), 'pi': ls.class_shares(y, K), 'T': T, 'T_source': src}


try:
    from tools.recalibration_suite import _load_file, _pred_path, load_source  # noqa: E402
    HELPERS = 'tools.recalibration_suite'
except Exception:                                     # pragma: no cover - only if that module is broken
    _load_file, _pred_path, load_source = _load_file_local, _pred_path_local, _load_source_local
    HELPERS = 'local copy (tools.recalibration_suite not importable)'


def _oof_order(exp_dir):
    """Permutation sorting oof.npz by image id (load_source returns arrays in file order)."""
    with np.load(os.path.join(exp_dir, 'oof.npz'), allow_pickle=True) as f:
        ids = np.array([str(i) for i in f['ids']]) if 'ids' in f.files else None
    if ids is None:
        return None, None
    o = np.argsort(ids, kind='stable')
    return o, ids[o]


def split_source(y, seed):
    """Stratified 50/50 split of the (id-sorted) OOF into pseudo-source S and pseudo-target pool T.
    Depends only on the labels and the seed (identical across runs with the same OOF images)."""
    y = np.asarray(y, dtype=int)
    rng = np.random.default_rng([int(seed), SPLIT_SEED])
    S, T = [], []
    for k in range(K):
        idx = rng.permutation(np.flatnonzero(y == k))
        h = (len(idx) + 1) // 2
        S.append(idx[:h])
        T.append(idx[h:])
    return np.sort(np.concatenate(S)).astype(int), np.sort(np.concatenate(T)).astype(int)


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------

def _alloc(shares, n):
    """Exact integer counts summing to n, largest-remainder rounding of n * shares (ties -> lower grade)."""
    q = np.asarray(shares, dtype=float)
    x = q * int(n)
    c = np.floor(x + 1e-9).astype(int)
    rest = int(n) - int(c.sum())
    if rest > 0:
        order = sorted(range(len(q)), key=lambda k: (-(x[k] - c[k]), k))
        for k in order[:rest]:
            c[k] += 1
    return c


def stratified_resample(y_pool, shares, n, rng):
    """Indices into y_pool (with replacement) whose grade counts are _alloc(shares, n).

    A grade with positive share but no image in the pool is dropped and the remaining shares renormalised.
    Returns (idx, missing_grades). Consumes rng identically for identical (counts, pool sizes)."""
    y_pool = np.asarray(y_pool, dtype=int)
    present = np.bincount(y_pool, minlength=K)[:K] > 0
    q = np.clip(np.asarray(shares, dtype=float), 0.0, None)
    missing = [k for k in range(K) if q[k] > 0 and not present[k]]
    q = np.where(present, q, 0.0)
    if q.sum() <= 0:
        raise ValueError('no grade of the target shares is present in the pool')
    counts = _alloc(q / q.sum(), n)
    parts = [rng.choice(np.flatnonzero(y_pool == k), size=int(c), replace=True)
             for k, c in enumerate(counts) if c > 0]
    return np.concatenate(parts).astype(int), missing


def _halves(y, rng):
    """Per-draw stratified split of the pool into two disjoint halves; a grade with one image is shared."""
    a, b, shared = [], [], []
    for k in range(K):
        idx = rng.permutation(np.flatnonzero(y == k))
        if len(idx) == 1:
            a.append(idx)
            b.append(idx)
            shared.append(k)
            continue
        h = len(idx) // 2
        a.append(idx[:h])
        b.append(idx[h:])
    return np.sort(np.concatenate(a)).astype(int), np.sort(np.concatenate(b)).astype(int), shared


def _rng(seed, r, scen, ds, part):
    return np.random.default_rng([int(seed), int(r), SCEN_CODE[scen], sum(map(ord, ds)), int(part)])


# ---------------------------------------------------------------------------
# Fitting (never receives TEST labels) and scoring (the only holder of TEST labels)
# ---------------------------------------------------------------------------

def fit_source(P_S, y_S):
    """Everything source-side, fitted on the pseudo-source half S only."""
    P_S = np.asarray(P_S, dtype=float)
    y_S = np.asarray(y_S, dtype=int)
    E = expected_grade(P_S)
    return {'P': P_S, 'y': y_S, 'E': E, 'pi': ls.class_shares(y_S, K), 'bcts': ls.fit_bcts(P_S, y_S),
            'c_src': fit_cutpoints(E, y_S)}


def fit_predict(src, P_dev, y_dev, P_test):
    """Fit every method on (S, DEV images, DEV labels where the method says so) and predict TEST grades.

    TEST labels are deliberately not an argument. Returns (pred: method -> (N_test,) grades,
    fitted: method -> {param: value})."""
    P_dev, P_test = np.asarray(P_dev, dtype=float), np.asarray(P_test, dtype=float)
    y_dev = np.asarray(y_dev, dtype=int)
    E_dev, E_test = expected_grade(P_dev), expected_grade(P_test)
    pi_s, bcts = src['pi'], src['bcts']
    pi_dev = ls.class_shares(y_dev, K)
    Pd_b, Pt_b = ls.apply_bcts(P_dev, bcts), ls.apply_bcts(P_test, bcts)
    pred, fit = {}, {}

    def cuts(c):
        return {f'c{j + 1}': float(v) for j, v in enumerate(c)}

    def pis(p):
        return {f'pi_{k}': float(v) for k, v in enumerate(p)}

    pred['argmax'] = P_test.argmax(1)
    pred['default_cuts'] = apply_cutpoints(E_test, DEFAULT_CUTS)
    Pc_t = ls.prior_correct(Pt_b, pi_s, pi_dev)
    pred['oracle_bayes'] = Pc_t.argmax(1)
    fit['oracle_bayes'] = pis(pi_dev)
    c = prevalence_cutpoints(E_dev, pi_dev)
    pred['oracle_quantile'] = apply_cutpoints(E_test, c)
    fit['oracle_quantile'] = cuts(c)
    pi_hat = ls.mlls_em(Pd_b, pi_s)[0]
    pred['mlls_bcts_bayes'] = ls.prior_correct(Pt_b, pi_s, pi_hat).argmax(1)
    fit['mlls_bcts_bayes'] = pis(pi_hat)
    c = prevalence_cutpoints(E_dev, pi_hat)
    pred['mlls_bcts_quantile'] = apply_cutpoints(E_test, c)
    fit['mlls_bcts_quantile'] = cuts(c)
    c = fit_cutpoints(E_dev, y_dev)
    pred['all_dev_cuts'] = apply_cutpoints(E_test, c)
    fit['all_dev_cuts'] = cuts(c)
    Pc_d = ls.prior_correct(Pd_b, pi_s, pi_dev)
    c = fit_cutpoints(expected_grade(Pc_d), y_dev)
    pred['all_dev_bayes_cuts'] = apply_cutpoints(expected_grade(Pc_t), c)
    fit['all_dev_bayes_cuts'] = cuts(c)
    return pred, fit


def score_metrics(y, pred):
    """QWK, mild recall, referable sens / spec (grade_metrics) and macro-F1 over the grades present in y."""
    y, pred = np.asarray(y, dtype=int), np.asarray(pred, dtype=int)
    g = grade_metrics(y, pred, K)
    cm = np.bincount(y * K + pred, minlength=K * K).reshape(K, K).astype(float)
    tp, row, col = np.diag(cm), cm.sum(1), cm.sum(0)
    rec = np.divide(tp, row, out=np.zeros(K), where=row > 0)
    prec = np.divide(tp, col, out=np.zeros(K), where=col > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(K), where=(prec + rec) > 0)
    present = row > 0
    return {'qwk': g['qwk'], 'mild_recall': g['mild_recall'] if present[1] else np.nan,
            'bin_sens': g['bin_sens'], 'bin_spec': g['bin_spec'],
            'f1_macro': float(f1[present].mean()) if present.any() else np.nan}


class Scorer:
    """The ONLY holder of a scenario's TEST labels; receives predictions, never returns anything fitted."""

    def __init__(self, y_test):
        self._y = np.asarray(y_test, dtype=int)
        self.n = len(self._y)

    def score(self, pred):
        return score_metrics(self._y, pred)


# ---------------------------------------------------------------------------
# Scenario construction
# ---------------------------------------------------------------------------

def build_scenario(scen, r, seed, ds, site, pool, pi_s, notes, tag):
    """(P_dev, y_dev, P_test, Scorer) of scenario `scen`, draw r (r = -1: observed splits, C only).

    DEV and TEST are drawn from separate random streams, so the DEV draw (and hence every fitted quantity)
    never depends on TEST labels."""
    n_dev, n_test = len(site['y_dev']), len(site['y_test'])
    if scen == 'C' and r < 0:
        return site['P_dev'], site['y_dev'], site['P_test'], Scorer(site['y_test'])
    if scen in ('A', 'A0'):
        a, b, shared = _halves(pool['y'], _rng(seed, r, scen, ds, 9))
        if shared and r == 0:
            notes.add(tag, ds, f'scenario_{scen}', f'grade(s) {shared} have one image in the pool T: shared by '
                      'pseudo-DEV and pseudo-TEST')
        q = site['dev_shares'] if scen == 'A' else pi_s
        i_dev, m1 = stratified_resample(pool['y'][a], q, n_dev, _rng(seed, r, scen, ds, 0))
        i_test, m2 = stratified_resample(pool['y'][b], q, n_test, _rng(seed, r, scen, ds, 1))
        i_dev, i_test = a[i_dev], b[i_test]
        Pd, yd, Pt, yt = pool['P'][i_dev], pool['y'][i_dev], pool['P'][i_test], pool['y'][i_test]
    else:
        q_dev = pi_s if scen == 'B' else ls.class_shares(site['y_dev'], K)
        q_test = pi_s if scen == 'B' else ls.class_shares(site['y_test'], K)
        i_dev, m1 = stratified_resample(site['y_dev'], q_dev, n_dev, _rng(seed, r, scen, ds, 0))
        i_test, m2 = stratified_resample(site['y_test'], q_test, n_test, _rng(seed, r, scen, ds, 1))
        Pd, yd = site['P_dev'][i_dev], site['y_dev'][i_dev]
        Pt, yt = site['P_test'][i_test], site['y_test'][i_test]
    if (m1 or m2) and r == 0:
        notes.add(tag, ds, f'scenario_{scen}', f'grades absent from the pool, share redistributed: DEV {m1}, '
                  f'TEST {m2}')
    return Pd, yd, Pt, Scorer(yt)


# ---------------------------------------------------------------------------
# Per-tag analysis
# ---------------------------------------------------------------------------

def analyse_tag(tag, R, seed, notes):
    """Draw rows and fitted rows of one run; None when the run cannot be analysed."""
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
            notes.add(tag, ds, '-', 'no dev/test predictions: dataset skipped (MEAN rows not written)', echo=True)
            continue
        for split in ('dev', 'test'):
            f = _load_file(_pred_path(exp_dir, ds, split))
            raw_pairs.append((f[3], f[2]))
        sites[ds] = {'P_dev': dev[2], 'y_dev': dev[1], 'P_test': test[2], 'y_test': test[1],
                     'dev_shares': ls.class_shares(dev[1], K)}
    if not sites:
        return None
    src_all = load_source(exp_dir, tag, raw_pairs, notes)
    if src_all is None:
        notes.add(tag, '-', 'source', 'no OOF: run skipped', echo=True)
        return None
    order, _ = _oof_order(exp_dir)
    P_oof, y_oof = (src_all['P'], src_all['y']) if order is None else (src_all['P'][order], src_all['y'][order])
    iS, iT = split_source(y_oof, seed)
    src = fit_source(P_oof[iS], y_oof[iS])
    pool = {'P': P_oof[iT], 'y': y_oof[iT]}
    fitted, rows = [], []

    def rec(ds, scen, draw, method, params):
        for k, v in params.items():
            fitted.append({'tag': tag, 'dataset': ds, 'scenario': scen, 'draw': int(draw), 'method': method,
                           'param': k, 'value': float(v)})

    rec('aptos_oof_S', 'source', -1, 'source', {'T_oof': src_all['T'], 'n_S': len(iS), 'n_T_pool': len(iT),
                                                'T_from_files': float(src_all['T_source'] == 'prediction_files'),
                                                **{f'pi_S_{k}': v for k, v in enumerate(src['pi'])}})
    rec('aptos_oof_S', 'source', -1, 'bcts', {'T': src['bcts']['T'],
                                              **{f'b{k}': v for k, v in enumerate(src['bcts']['b'])}})
    rec('aptos_oof_S', 'source', -1, 'source_cuts_reference_only',
        {f'c{j + 1}': v for j, v in enumerate(src['c_src'])})
    if not src['bcts']['converged']:
        notes.add(tag, 'aptos_oof_S', 'bcts', 'L-BFGS-B did not report convergence (parameters still used)')
    for ds, site in sites.items():
        for scen in SCENARIOS:
            draws = ([-1] if scen == 'C' else []) + list(range(int(R)))
            for r in draws:
                Pd, yd, Pt, sc = build_scenario(scen, r, seed, ds, site, pool, src['pi'], notes, tag)
                pred, fit = fit_predict(src, Pd, yd, Pt)
                for m, params in fit.items():
                    rec(ds, scen, r, m, params)
                for m in METHODS:
                    rows.append({'tag': tag, 'dataset': ds, 'scenario': scen, 'draw': int(r), 'method': m,
                                 'info': METHOD_INFO[m], 'n_dev': len(yd), 'n_test': sc.n, **sc.score(pred[m])})
    return {'tag': tag, 'heldout': heldout, 'sites': list(sites), 'rows': rows, 'fitted': fitted}


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def add_mean_rows(draws, results):
    """dataset 'MEAN' = unweighted mean over the run's held-out datasets, per draw (only when all present)."""
    out = []
    for res in results:
        if set(res['sites']) != set(res['heldout']):
            continue
        d = draws[draws['tag'] == res['tag']]
        g = d.groupby(['scenario', 'draw', 'method'], sort=False)
        m = g[list(METRICS)].mean().reset_index()            # pandas mean skips NaN (e.g. no grade-1 images)
        cnt = g['dataset'].nunique().values
        sz = g[['n_dev', 'n_test']].sum().reset_index(drop=True)
        m = m[cnt == len(res['heldout'])]
        m = m.assign(tag=res['tag'], dataset='MEAN', info=m['method'].map(METHOD_INFO),
                     n_dev=sz.loc[m.index, 'n_dev'].values, n_test=sz.loc[m.index, 'n_test'].values)
        out.append(m)
    return pd.concat([draws] + out, ignore_index=True, sort=False)[list(draws.columns)]


def _stats(v):
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    if not len(v):
        return np.nan, np.nan, np.nan
    lo, hi = np.percentile(v, [2.5, 97.5])
    return float(v.mean()), float(lo), float(hi)


def summarise(draws):
    rows = []
    for (tag, ds, scen, method), g in draws.groupby(['tag', 'dataset', 'scenario', 'method'], sort=False):
        rep = g[g['draw'] >= 0]
        pt = g[g['draw'] < 0]
        r = {'tag': tag, 'dataset': ds, 'scenario': scen, 'scenario_desc': SCENARIO_DESC[scen], 'method': method,
             'info': METHOD_INFO[method], 'R': len(rep), 'n_dev': int(g['n_dev'].iloc[-1]),
             'n_test': int(g['n_test'].iloc[-1])}
        for m in METRICS:
            r[f'{m}_mean'], r[f'{m}_lo'], r[f'{m}_hi'] = _stats(rep[m])
            r[f'{m}_point'] = float(pt[m].iloc[0]) if len(pt) else np.nan
        r['caveat'] = CAVEAT_C if scen == 'C' else CAVEAT_SEMI
        rows.append(r)
    return pd.DataFrame(rows)


def _crow(tag, ds, scen, name, metric, mean, lo, hi, point, n_valid, R, note=''):
    return {'tag': tag, 'dataset': ds, 'scenario': scen, 'contrast': name, 'metric': metric, 'mean': mean,
            'lo': lo, 'hi': hi, 'point': point, 'n_valid': int(n_valid), 'R': int(R), 'note': note}


def contrasts(draws):
    """Per-draw contrasts between methods of the same scenario draw, and B-A / C-A differences."""
    rows = []
    per = {}                                              # (tag, ds, scen, name, metric) -> (draw vals, point)
    keys = ['tag', 'dataset', 'scenario', 'draw']
    for metric in METRICS:
        wide = draws.set_index(keys + ['method'])[metric].unstack('method')
        for (tag, ds, scen), w in wide.groupby(level=[0, 1, 2], sort=False):
            w = w.droplevel([0, 1, 2]).sort_index()
            rep, pt = w[w.index >= 0], w[w.index < 0]
            R = len(rep)
            named = dict(CONTRASTS)
            named.update({f'vs_argmax:{m}': (m, 'argmax') for m in METHODS if m != 'argmax'})
            for name, (a, b) in named.items():
                d = (rep[a] - rep[b]).values.astype(float)
                p = float(pt[a].iloc[0] - pt[b].iloc[0]) if len(pt) else np.nan
                per[(tag, ds, scen, name, metric)] = (d, p)
                mu, lo, hi = _stats(d)
                rows.append(_crow(tag, ds, scen, name, metric, mu, lo, hi, p, np.isfinite(d).sum(), R))
            num = (rep['mlls_bcts_bayes'] - rep['argmax']).values.astype(float)
            den = (rep['oracle_bayes'] - rep['argmax']).values.astype(float)
            ok = np.isfinite(num) & np.isfinite(den) & (np.abs(den) >= RATIO_MIN)
            mden = np.nanmean(den) if np.isfinite(den).any() else np.nan
            note = ''
            if np.isfinite(mden) and abs(mden) >= RATIO_MIN:
                mu = float(np.nanmean(num) / mden)
            else:
                mu, note = np.nan, f'|mean oracle_bayes gain| < {RATIO_MIN}: recovery undefined'
            lo, hi = (np.percentile(num[ok] / den[ok], [2.5, 97.5]) if ok.any() else (np.nan, np.nan))
            p = np.nan
            if len(pt):
                pn = float(pt['mlls_bcts_bayes'].iloc[0] - pt['argmax'].iloc[0])
                pd_ = float(pt['oracle_bayes'].iloc[0] - pt['argmax'].iloc[0])
                p = pn / pd_ if np.isfinite(pd_) and abs(pd_) >= RATIO_MIN else np.nan
            rows.append(_crow(tag, ds, scen, 'mlls_recovery', metric, mu, float(lo), float(hi), p, ok.sum(), R,
                              note))
    for (tag, ds, scen, name, metric), (d, _) in list(per.items()):
        if name not in CROSS_CONTRASTS:
            continue
        for x, ref in CROSS_PAIRS:
            if scen != x or (tag, ds, ref, name, metric) not in per:
                continue
            dr = per[(tag, ds, ref, name, metric)][0]
            n = min(len(d), len(dr))
            diff = d[:n] - dr[:n]
            _, lo, hi = _stats(diff)
            ok = np.isfinite(d).any() and np.isfinite(dr).any()
            mu = float(np.nanmean(d) - np.nanmean(dr)) if ok else np.nan
            rows.append(_crow(tag, ds, f'{x}-{ref}', name, metric, mu, lo, hi, np.nan, np.isfinite(diff).sum(), n,
                              'difference of independent draws (paired by index only)'))
    cols = ['tag', 'dataset', 'scenario', 'contrast', 'metric', 'mean', 'lo', 'hi', 'point', 'n_valid', 'R', 'note']
    return pd.DataFrame(rows, columns=cols)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

TAG_COLORS = ('#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300')     # fixed categorical order
FIG_METHODS = tuple(m for m in METHODS if m != 'argmax')


def plot(con, tags, stem):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'font.weight': 'bold', 'axes.labelweight': 'bold',
                         'axes.titleweight': 'bold'})
    g = con[(con['dataset'] == 'MEAN') & (con['metric'] == 'qwk') & con['contrast'].str.startswith('vs_argmax:')]
    fig, axes = plt.subplots(1, len(SCENARIOS), figsize=(4.6 * len(SCENARIOS), 4.4), sharey=True, squeeze=False)
    x = np.arange(len(FIG_METHODS))
    width = 0.8 / max(1, len(tags))
    for ax, scen in zip(axes[0], SCENARIOS):
        s = g[g['scenario'] == scen]
        for i, tag in enumerate(tags):
            t = s[s['tag'] == tag].set_index('contrast')
            if t.empty:
                continue
            v = np.array([t['mean'].get(f'vs_argmax:{m}', np.nan) for m in FIG_METHODS], dtype=float)
            lo = np.array([t['lo'].get(f'vs_argmax:{m}', np.nan) for m in FIG_METHODS], dtype=float)
            hi = np.array([t['hi'].get(f'vs_argmax:{m}', np.nan) for m in FIG_METHODS], dtype=float)
            err = np.vstack([np.clip(v - lo, 0, None), np.clip(hi - v, 0, None)])
            ax.bar(x + (i - (len(tags) - 1) / 2) * width, v, width * 0.92, yerr=np.nan_to_num(err),
                   color=TAG_COLORS[i % len(TAG_COLORS)], capsize=2, error_kw={'lw': 0.8}, label=tag)
        ax.axhline(0, color='#555555', lw=0.8)
        ax.grid(axis='y', color='#dddddd', lw=0.5)
        ax.set_axisbelow(True)
        ax.set_xticks(x)
        ax.set_xticklabels(FIG_METHODS, rotation=40, ha='right', fontsize=7)
        ax.set_title(f'{scen}: {SCENARIO_DESC[scen].split(":")[0]}', fontsize=9)
        if scen == SCENARIOS[0]:
            ax.set_ylabel('QWK gain over argmax (mean over held-out sites)')
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='upper center', ncol=len(labels), frameon=False, fontsize=8,
                   bbox_to_anchor=(0.5, 1.02))
    fig.text(0.5, -0.02, 'bars: mean over draws; error bars: 2.5-97.5 percentiles over draws. A0, A and B are '
             'semi-synthetic resamplings (with replacement: rare-grade variance understated).',
             ha='center', fontsize=7)
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
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
    ap.add_argument('--R', type=int, default=50, help='resampling draws per scenario')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out_dir', default=os.path.join('results', 'master', 'recal'))
    ap.add_argument('--fig_stem', default=os.path.join('results', 'figures', 'fig_shift_decomposition'))
    a = ap.parse_args(argv)
    if a.R < 1:
        ap.error('--R must be >= 1')
    tags = [t.strip() for t in a.tags.split(',') if t.strip()]
    notes = Notes()
    notes.add('-', '-', 'helpers', f'loading / temperature helpers: {HELPERS}')
    results = [res for res in (analyse_tag(t, a.R, a.seed, notes) for t in tags) if res is not None]
    if not results:
        print('shift_decomposition: no analysable run (held-out dev/test predictions + oof.npz) for any tag')
        sys.exit(1)
    os.makedirs(a.out_dir, exist_ok=True)
    draws = pd.DataFrame([r for res in results for r in res['rows']])
    draws = add_mean_rows(draws, results)
    summ = summarise(draws)
    con = contrasts(draws)
    p = lambda f: os.path.join(a.out_dir, f)                          # noqa: E731
    draws.round(6).to_csv(p('shift_decomposition_draws.csv'), index=False)
    summ.round(6).to_csv(p('shift_decomposition.csv'), index=False)
    con.round(6).to_csv(p('shift_decomposition_contrasts.csv'), index=False)
    pd.DataFrame([r for res in results for r in res['fitted']]).to_csv(
        p('shift_decomposition_fitted.csv'), index=False, float_format='%.10g')
    pd.DataFrame(notes.rows, columns=['tag', 'dataset', 'method', 'note']).to_csv(
        p('shift_decomposition_notes.csv'), index=False)
    rtags = [res['tag'] for res in results]
    plot(con, rtags, a.fig_stem)

    show = con[(con['metric'] == 'qwk') & (con['dataset'] == 'MEAN')
               & con['contrast'].isin(['gain_bayes', 'gain_cuts_over_bayes', 'gain_quantile', 'mlls_recovery'])]
    if len(show):
        pv = show.pivot_table(index=['tag', 'scenario'], columns='contrast', values='mean', aggfunc='first')
        with pd.option_context('display.width', 200, 'display.max_rows', 200):
            print('\nQWK contrasts, MEAN over held-out sites (mean over draws; A0/A/B semi-synthetic):')
            print(pv.round(3).to_string())
    print(f'\nwritten: {p("shift_decomposition.csv")}, shift_decomposition_contrasts.csv, '
          f'shift_decomposition_draws.csv, shift_decomposition_fitted.csv, shift_decomposition_notes.csv; '
          f'{a.fig_stem}.png/.pdf (R={a.R}, seed={a.seed})')


if __name__ == '__main__':
    main()
