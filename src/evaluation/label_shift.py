"""Label-shift estimation and recalibration baselines on saved class probabilities (pure numpy / scipy).

Notation: P (N, K) class probabilities of one model, E = sum_k k P_k its expected grade, pi a grade
distribution (length K, on the simplex). "Source" = the data a calibration map / source prior is fitted on
(in this project: the APTOS out-of-fold predictions); "target" = a new site.

  fit_bcts / apply_bcts           bias-corrected temperature scaling (Alexandari et al., ICML 2020)
  mlls_em                         maximum-likelihood label-shift EM (Saerens, Latinne & Decaestecker 2002)
  bbse                            black-box shift estimation (Lipton, Wang & Smola, ICML 2018), soft confusion
  prior_correct                   Bayes prior correction P'_k ~ P_k pi_tgt_k / pi_src_k
  fit_quantile_map / quantile_align
                                  UPA-style monotone score alignment (Roschewitz et al. 2023): empirical CDF
                                  of target scores -> source score quantiles
  temperature_scale_fit / _apply  scalar temperature on n labelled images (few-label baseline)
  vector_scale_fit / _apply       per-class scale + bias with fixed L2 shrinkage to the identity map
  fit_cut_offset                  one common shift of the default cut-points, QWK-optimal on labelled images
  perturb_prevalence              deterministic No-DR shift / random total-variation perturbation of a prior
  smoothed_prevalence             Dirichlet(alpha)-smoothed grade shares from n labels
  bootstrap_indices               image- or cluster-level bootstrap resamples of unlabelled target images
  bootstrap_prior                 any prior estimator re-run on bootstrap resamples: point, percentile CI, draws
  bootstrap_label_shift_priors    MLLS+BCTS and BBSE on IDENTICAL resamples of the unlabelled target images
  shrinkage_weight / shrink_prior James-Stein-style shrinkage of an estimated prior towards the source prior
  bagged_prevalence_cutpoints     element-wise median of quantile cut-points over bootstrap priors

Every function is deterministic (given its rng), clips probabilities away from 0 where it takes logs or
ratios, and never returns NaN for finite inputs.
"""

import numpy as np
from scipy.optimize import minimize

EPS = 1e-12
K_DEFAULT = 5


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _as_prob(P):
    P = np.clip(np.asarray(P, dtype=float), 0.0, None)
    s = P.sum(1, keepdims=True)
    K = P.shape[1]
    return np.where(s > 0, P / np.where(s > 0, s, 1.0), 1.0 / K)


def _as_simplex(pi):
    pi = np.clip(np.asarray(pi, dtype=float), 0.0, None)
    s = pi.sum()
    return pi / s if s > 0 else np.full(len(pi), 1.0 / len(pi))


def _softmax(S):
    S = S - S.max(1, keepdims=True)
    e = np.exp(S)
    return e / e.sum(1, keepdims=True)


def _log_softmax(S):
    S = S - S.max(1, keepdims=True)
    return S - np.log(np.exp(S).sum(1, keepdims=True))


def _logits(P):
    return np.log(np.clip(np.asarray(P, dtype=float), EPS, 1.0))


def class_shares(y, K=K_DEFAULT):
    """Empirical grade shares of integer labels (uniform when y is empty)."""
    y = np.asarray(y, dtype=int)
    if len(y) == 0:
        return np.full(K, 1.0 / K)
    return np.bincount(y, minlength=K)[:K] / float(len(y))


def tv_distance(p, q):
    return 0.5 * float(np.abs(np.asarray(p, float) - np.asarray(q, float)).sum())


# ---------------------------------------------------------------------------
# Bias-corrected temperature scaling
# ---------------------------------------------------------------------------

def fit_bcts(P_src, y_src, max_iter=500):
    """Bias-corrected temperature scaling (BCTS; Alexandari, Kundaje & Shrikumar, ICML 2020).

    z = log(P_src + eps); minimise the mean NLL of softmax(z / T + b) on the labelled source images over a
    scalar T > 0 and a class bias b (length K). b is identifiable only up to a constant and is returned
    centred (mean 0). The bias terms absorb a mismatch between the training prior and the calibration-set
    prior, which is what makes EM label-shift estimation well specified (the calibrated probabilities then
    reflect the calibration-set prior pi_src = class shares of y_src).

    Returns {'T': float, 'b': (K,) array, 'nll': float, 'converged': bool}.
    """
    z = _logits(P_src)
    y = np.asarray(y_src, dtype=int)
    N, K = z.shape
    Y = np.zeros((N, K))
    Y[np.arange(N), y] = 1.0

    def f(theta):
        logT, b = theta[0], theta[1:]
        T = np.exp(logT)
        S = z / T + b
        ls = _log_softmax(S)
        nll = -float((ls * Y).sum()) / N
        G = (np.exp(ls) - Y) / N                       # d nll / d S
        g_b = G.sum(0)
        g_logT = float((G * (-z / T)).sum())
        return nll, np.concatenate([[g_logT], g_b])

    theta0 = np.zeros(K + 1)
    bounds = [(np.log(0.05), np.log(20.0))] + [(-20.0, 20.0)] * K
    res = minimize(f, theta0, jac=True, method='L-BFGS-B', bounds=bounds, options={'maxiter': max_iter})
    b = res.x[1:] - res.x[1:].mean()
    return {'T': float(np.exp(res.x[0])), 'b': b, 'nll': float(res.fun), 'converged': bool(res.success)}


def apply_bcts(P, params):
    """softmax(log(P + eps) / T + b) with the parameters returned by fit_bcts."""
    return _softmax(_logits(P) / float(params['T']) + np.asarray(params['b'], dtype=float))


# ---------------------------------------------------------------------------
# Prior estimation
# ---------------------------------------------------------------------------

def prior_correct(P, pi_src, pi_tgt):
    """Bayes prior correction: P'_k proportional to P_k * pi_tgt_k / pi_src_k (rows renormalised).

    P must be calibrated with respect to pi_src. Zero-mass classes are floored at eps in both priors, so
    every row keeps positive mass and the result is never NaN.
    """
    P = _as_prob(P)
    r = np.clip(_as_simplex(pi_tgt), EPS, None) / np.clip(_as_simplex(pi_src), EPS, None)
    Q = P * r[None, :]
    s = Q.sum(1, keepdims=True)
    return np.where(s > 0, Q / np.where(s > 0, s, 1.0), P)


def mlls_em(P_tgt, pi_src, max_iter=1000, tol=1e-8, init=None):
    """Saerens et al. (2002) EM for the target prior under label shift.

    E-step: posteriors re-weighted by pi_t / pi_src (prior_correct); M-step: pi_t = mean posterior.
    Starts at pi_src (or at `init`, a warm start used by the bootstrap; the log-likelihood is concave in pi,
    so the fixed point is the same up to tol); stops when max |pi_t^(m+1) - pi_t^(m)| < tol. P_tgt must be
    calibrated to pi_src (e.g. BCTS fitted on the source). Returns (pi_tgt_hat, P_tgt_adjusted).
    """
    P = _as_prob(P_tgt)
    pi_s = _as_simplex(pi_src)
    pi = pi_s.copy() if init is None else _as_simplex(init)
    for _ in range(int(max_iter)):
        Q = prior_correct(P, pi_s, pi)
        new = _as_simplex(Q.mean(0))
        done = np.abs(new - pi).max() < tol
        pi = new
        if done:
            break
    return pi, prior_correct(P, pi_s, pi)


def bbse(P_src, y_src, P_tgt, return_info=False):
    """Black-box shift estimation (Lipton et al. 2018) with soft predictions.

    C[i, j] = (1/n_src) sum_{source images with y = j} P(yhat = i)   (joint soft confusion; column sums =
              source prior pi_src)
    mu[i]   = mean over target images of P(yhat = i)
    Solve C w = mu by least squares (numpy lstsq, minimum-norm for rank-deficient C), clip w >= 0 and return
    pi_tgt = w * pi_src renormalised. If every weight is clipped to 0 (degenerate), pi_src is returned and
    info['degenerate'] = True. With return_info=True also returns {'w', 'cond', 'degenerate'}.
    """
    P_s, P_t = _as_prob(P_src), _as_prob(P_tgt)
    y = np.asarray(y_src, dtype=int)
    N, K = P_s.shape
    C = np.zeros((K, K))
    for j in range(K):
        m = y == j
        if m.any():
            C[:, j] = P_s[m].sum(0) / N
    pi_s = C.sum(0)
    mu = P_t.mean(0)
    w = np.linalg.lstsq(C, mu, rcond=None)[0]
    w = np.clip(np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)
    est = w * pi_s
    degenerate = not est.sum() > 0
    pi_t = _as_simplex(pi_s) if degenerate else est / est.sum()
    if return_info:
        with np.errstate(all='ignore'):
            cond = float(np.linalg.cond(C))
        return pi_t, {'w': w, 'cond': cond, 'degenerate': bool(degenerate)}
    return pi_t


def smoothed_prevalence(y_labels, K=K_DEFAULT, alpha=1.0):
    """Dirichlet(alpha)-smoothed grade shares (counts + alpha) / (n + K alpha) from n labels."""
    y = np.asarray(y_labels, dtype=int)
    c = np.bincount(y, minlength=K)[:K].astype(float) if len(y) else np.zeros(K)
    return (c + alpha) / (c.sum() + K * alpha) if (c.sum() + K * alpha) > 0 else np.full(K, 1.0 / K)


# ---------------------------------------------------------------------------
# Score alignment (UPA-style)
# ---------------------------------------------------------------------------

def fit_quantile_map(E_tgt_ref, E_src):
    """Monotone (non-decreasing) map sending the target score distribution onto the source distribution.

    u(e) = mid-rank empirical CDF of the reference target scores (ties share their mean CDF position;
    linear interpolation between distinct values, clamped to [0.5/n, 1 - 0.5/n] outside their range);
    map(e) = source empirical quantile at u(e) (linear interpolation). Fitted on reference (e.g. unlabelled
    DEV) target images only; apply the returned callable to new target images.
    """
    ref = np.sort(np.asarray(E_tgt_ref, dtype=float))
    src = np.sort(np.asarray(E_src, dtype=float))
    n = len(ref)
    vals, first, counts = np.unique(ref, return_index=True, return_counts=True)
    cdf = (first + 0.5 * counts) / n

    def _map(E):
        E = np.asarray(E, dtype=float)
        if len(vals) == 1:
            u = np.full(E.shape, 0.5)
        else:
            u = np.interp(E, vals, cdf)
        return np.quantile(src, np.clip(u, 0.0, 1.0))

    return _map


def quantile_align(E_tgt, E_src, E_new=None):
    """UPA-style alignment: map fitted on E_tgt (target reference scores) onto E_src, applied to E_new
    (default: E_tgt itself)."""
    m = fit_quantile_map(E_tgt, E_src)
    return m(E_tgt if E_new is None else E_new)


# ---------------------------------------------------------------------------
# Few-label calibration baselines
# ---------------------------------------------------------------------------

def temperature_scale_fit(P, y):
    """Scalar temperature (NLL-optimal, bounded 0.1..10) on labelled images; same routine as the
    project's OOF temperature (src.evaluation.metrics.fit_temperature)."""
    from src.evaluation.metrics import fit_temperature
    return {'T': float(fit_temperature(_logits(P), np.asarray(y, dtype=int)))}


def temperature_scale_apply(P, params):
    from src.evaluation.metrics import apply_temperature
    return apply_temperature(np.asarray(P, dtype=float), float(params['T']))


VECTOR_REG = 0.1


def vector_scale_fit(P, y, reg=VECTOR_REG, max_iter=500):
    """Vector scaling: softmax(a * log P + b), a, b in R^K, minimising mean NLL + reg (||a - 1||^2 + ||b||^2).

    The fixed L2 term (reg = 0.1, not tuned) shrinks towards the identity map, so the fit is defined with
    very few labels and with grades absent from the labelled sample.
    """
    z = _logits(P)
    y = np.asarray(y, dtype=int)
    N, K = z.shape
    Y = np.zeros((N, K))
    Y[np.arange(N), y] = 1.0

    def f(theta):
        a, b = theta[:K], theta[K:]
        ls = _log_softmax(z * a + b)
        nll = -float((ls * Y).sum()) / max(N, 1)
        G = (np.exp(ls) - Y) / max(N, 1)
        pen = reg * (float(((a - 1) ** 2).sum()) + float((b ** 2).sum()))
        g = np.concatenate([(G * z).sum(0) + 2 * reg * (a - 1), G.sum(0) + 2 * reg * b])
        return nll + pen, g

    theta0 = np.concatenate([np.ones(K), np.zeros(K)])
    res = minimize(f, theta0, jac=True, method='L-BFGS-B', options={'maxiter': max_iter})
    return {'a': res.x[:K].copy(), 'b': res.x[K:].copy(), 'reg': float(reg), 'converged': bool(res.success)}


def vector_scale_apply(P, params):
    return _softmax(_logits(P) * np.asarray(params['a'], float) + np.asarray(params['b'], float))


def fit_cut_offset(E, y, base=None, K=K_DEFAULT):
    """One common shift s of the default cut-points (0.5..3.5 + s), QWK-optimal on labelled images.

    Candidates: s = 0 and every s that places one cut-point exactly midway between two consecutive distinct
    labelled scores (so every distinct partition reachable by a common shift is tried). Ties -> smallest |s|,
    then smaller s. Returns s (0.0 when fewer than two grades are present).
    """
    from src.evaluation.cutpoints import DEFAULT_CUTS, apply_cutpoints, qwk_from_pred
    base = DEFAULT_CUTS[:K - 1] if base is None else np.asarray(base, dtype=float)
    E = np.asarray(E, dtype=float)
    y = np.asarray(y, dtype=int)
    if len(y) < 2 or len(np.unique(y)) < 2:
        return 0.0
    u = np.unique(E)
    mids = np.concatenate([[u[0] - 1e-6], (u[:-1] + u[1:]) / 2.0, [u[-1] + 1e-6]])
    cand = np.unique(np.concatenate([[0.0], (mids[:, None] - base[None, :]).ravel()]))
    best_s, best_q = 0.0, -np.inf
    for s in sorted(cand, key=lambda v: (abs(v), v)):
        q = qwk_from_pred(y, apply_cutpoints(E, base + s), K)
        if q > best_q + 1e-12:
            best_s, best_q = float(s), q
    return best_s


# ---------------------------------------------------------------------------
# Prior perturbation (misspecification analysis)
# ---------------------------------------------------------------------------

def perturb_prevalence(pi, kind, delta, rng=None, max_tries=10000):
    """Perturbed copy of a grade distribution.

    kind='nodr_shift': add the signed absolute mass `delta` to grade 0 and rescale grades 1..K-1
        proportionally (TV distance = |delta|). Raises ValueError when infeasible (grade-0 share would leave
        [0, 1], or there is no mass outside grade 0 to rescale).
    kind='tv_random': pi + v with sum(v) = 0 and TV(pi, pi + v) = delta exactly; the direction is the
        difference of two Dirichlet(1) draws, scaled to total variation delta; draws leaving the simplex are
        rejected and redrawn (ValueError after max_tries rejections). Needs rng (numpy Generator).
    """
    pi = _as_simplex(pi)
    delta = float(delta)
    if kind == 'nodr_shift':
        new0 = pi[0] + delta
        rest = 1.0 - pi[0]
        if new0 < -1e-12 or new0 > 1 + 1e-12 or (rest <= 0 and abs(delta) > 0):
            raise ValueError(f'nodr_shift {delta:+.3f} infeasible for grade-0 share {pi[0]:.3f}')
        out = pi.copy()
        out[0] = min(max(new0, 0.0), 1.0)
        if rest > 0:
            out[1:] = pi[1:] * (1.0 - out[0]) / rest
        return out
    if kind == 'tv_random':
        if rng is None:
            raise ValueError('tv_random needs an rng')
        if delta == 0:
            return pi.copy()
        K = len(pi)
        for _ in range(int(max_tries)):
            g = rng.dirichlet(np.ones(K)) - rng.dirichlet(np.ones(K))
            l1 = np.abs(g).sum()
            if l1 <= 0:
                continue
            v = g * (2.0 * delta / l1)
            out = pi + v
            if out.min() >= -1e-12:
                return np.clip(out, 0.0, None)
        raise ValueError(f'tv_random {delta}: no feasible draw in {max_tries} tries')
    raise ValueError(f'unknown perturbation kind {kind!r}')


# ---------------------------------------------------------------------------
# Uncertainty of annotation-free prior estimates (bootstrap of unlabelled target images)
# ---------------------------------------------------------------------------

def bootstrap_indices(n, B, rng, groups=None):
    """B bootstrap resamples of n unlabelled images as index arrays.

    groups=None, or every group unique -> image-level (n images drawn with replacement). Otherwise
    cluster-level: G = number of distinct groups are drawn with replacement and every image of a drawn group
    enters the resample (as often as the group was drawn) - the same scheme as
    src.evaluation.clinical.cluster_bootstrap_weights. Returns (list of index arrays, unit string).
    """
    n, B = int(n), int(B)
    if groups is None:
        codes = np.arange(n)
    else:
        groups = np.asarray([str(g) for g in groups])
        if len(groups) != n:
            raise ValueError('groups must have one entry per image')
        codes = np.unique(groups, return_inverse=True)[1]
    G = int(codes.max()) + 1 if n else 0
    unit = 'image' if G == n else f'cluster ({G} groups / {n} images)'
    if G == n:
        return [rng.integers(0, n, size=n) for _ in range(B)], unit
    members = [np.flatnonzero(codes == g) for g in range(G)]
    out = []
    for _ in range(B):
        drawn = rng.integers(0, G, size=G)
        out.append(np.concatenate([members[g] for g in drawn]) if G else np.zeros(0, int))
    return out, unit


def bootstrap_prior(P, estimator, B=200, rng=None, groups=None, alpha=0.05, indices=None, point=None):
    """Bootstrap uncertainty of a prior estimated from unlabelled images.

    estimator: callable P_subset (m, K) -> prior (K,). The point estimate is estimator(P) on all images
    (or `point`, when the caller already has it);
    the same estimator is re-run on B bootstrap resamples (bootstrap_indices; or `indices`, a precomputed
    list, so several estimators can share IDENTICAL resamples). Returns
      {'pi': point estimate, 'draws': (B, K), 'lo', 'hi': per-class (alpha/2, 1-alpha/2) percentiles of the
       draws, 'cov': (K, K) bootstrap covariance (ddof 1), 'cov_trace': tr(cov), 'B': B, 'unit': str}.
    With B = 0 (or fewer than 2 resamples) lo/hi = pi and cov = 0.
    """
    P = _as_prob(P)
    pi = _as_simplex(estimator(P) if point is None else point)
    unit = 'given indices'
    if indices is None:
        if rng is None:
            raise ValueError('bootstrap_prior needs an rng or precomputed indices')
        indices, unit = bootstrap_indices(len(P), B, rng, groups)
    draws = np.array([_as_simplex(estimator(P[i])) for i in indices]).reshape(-1, len(pi))
    if len(draws) >= 2:
        lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)], axis=0)
        cov = np.cov(draws, rowvar=False, ddof=1)
    else:
        lo, hi, cov = pi.copy(), pi.copy(), np.zeros((len(pi), len(pi)))
    return {'pi': pi, 'draws': draws, 'lo': lo, 'hi': hi, 'cov': cov, 'cov_trace': float(np.trace(cov)),
            'B': int(len(draws)), 'unit': unit}


def bootstrap_label_shift_priors(P_tgt, P_src, y_src, B=200, rng=None, groups=None, bcts=None, alpha=0.05,
                                 indices=None):
    """MLLS+BCTS and BBSE prior estimates with bootstrap uncertainty on IDENTICAL resamples of the unlabelled
    target images P_tgt (the source P_src / y_src - labelled calibration data - are held fixed).

    mlls_bcts: BCTS fitted on (P_src, y_src) unless given; EM (Saerens) on the BCTS-calibrated target rows,
               pi_src = class shares of y_src; each resample warm-starts EM at the full-sample estimate.
    bbse:      soft-confusion BBSE (Lipton et al. 2018); 'n_degenerate' counts resamples with every weight
               clipped to 0 (pi_src returned there).
    Returns {'mlls_bcts': bootstrap_prior dict, 'bbse': bootstrap_prior dict + 'n_degenerate', 'unit'}.
    """
    P_tgt = _as_prob(P_tgt)
    y_src = np.asarray(y_src, dtype=int)
    K = P_tgt.shape[1]
    pi_src = class_shares(y_src, K)
    bcts = fit_bcts(P_src, y_src) if bcts is None else bcts
    unit = 'given indices'
    if indices is None:
        if rng is None:
            raise ValueError('bootstrap_label_shift_priors needs an rng or precomputed indices')
        indices, unit = bootstrap_indices(len(P_tgt), B, rng, groups)
    Pb = apply_bcts(P_tgt, bcts)
    pi0 = mlls_em(Pb, pi_src)[0]
    out = {'unit': unit}
    # the BCTS calibration is row-wise, so resampling calibrated rows == calibrating resampled rows
    out['mlls_bcts'] = bootstrap_prior(Pb, lambda Q: mlls_em(Q, pi_src, init=pi0)[0] if len(Q) else pi0,
                                       alpha=alpha, indices=indices, point=pi0)
    degenerate = []

    def _bbse(Q):
        pi_b, info = bbse(P_src, y_src, Q, return_info=True)
        degenerate.append(info['degenerate'])
        return pi_b

    out['bbse'] = bootstrap_prior(P_tgt, _bbse, alpha=alpha, indices=indices)
    out['bbse']['n_degenerate'] = int(np.sum(degenerate[1:]))          # [0] is the full-sample fit
    for k in ('mlls_bcts', 'bbse'):
        out[k]['unit'] = unit
    return out


def shrinkage_weight(pi_hat, pi_src, cov_trace):
    """James-Stein-style weight on the estimated shift (pi_hat - pi_src):

        w = 1 / (1 + tr(Cov_boot(pi_hat)) / ||pi_hat - pi_src||^2),  clipped to [0, 1]

    tr(Cov) = total bootstrap variance of the estimate (noise), ||pi_hat - pi_src||^2 = squared size of the
    estimated shift (signal). w -> 1 when the estimate is precise relative to the shift (keep the estimate);
    w -> 0 when the shift is small relative to its noise (fall back to the source prior). No shift and no
    noise -> w = 1 (nothing to shrink); no shift but positive noise -> w = 0.
    """
    d2 = float(np.sum((np.asarray(pi_hat, float) - np.asarray(pi_src, float)) ** 2))
    v = max(float(cov_trace), 0.0)
    if d2 <= 0.0:
        return 1.0 if v <= 0.0 else 0.0
    return float(np.clip(1.0 / (1.0 + v / d2), 0.0, 1.0))


def shrink_prior(pi_hat, pi_src, cov_trace):
    """(pi_src + w (pi_hat - pi_src), w) with w = shrinkage_weight(...). A convex combination of two points
    of the simplex, hence itself on the simplex."""
    w = shrinkage_weight(pi_hat, pi_src, cov_trace)
    p = _as_simplex(pi_src) + w * (_as_simplex(pi_hat) - _as_simplex(pi_src))
    return _as_simplex(p), w


def bagged_prevalence_cutpoints(E_unlabelled, prior_draws, K=K_DEFAULT):
    """Element-wise median over bootstrap priors pi_b of prevalence_cutpoints(E_unlabelled, pi_b), on the
    SAME unlabelled scores (only the prior varies), kept strictly increasing (ties -> +1e-6, as in
    prevalence_cutpoints)."""
    from src.evaluation.cutpoints import prevalence_cutpoints
    C = np.array([prevalence_cutpoints(E_unlabelled, p, K) for p in np.atleast_2d(prior_draws)])
    c = np.median(C, axis=0)
    for k in range(1, len(c)):
        if c[k] <= c[k - 1]:
            c[k] = c[k - 1] + 1e-6
    return c
