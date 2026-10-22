"""Calibration and recalibration utilities: temperature scaling and devcut."""

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from sklearn.metrics import cohen_kappa_score

from src.evaluation.metrics import ece, nll, brier, fit_temperature, apply_temperature


def fit_devcut(y_dev: np.ndarray, P_dev: np.ndarray):
    """Fit 2-parameter linear mapping E' = a * E + b on Messidor-DEV to maximise QWK.

    Args:
        y_dev: (N,) ground truth classes 0..4 on DEV.
        P_dev: (N, 5) predicted probabilities on DEV.

    Returns:
        tuple (a, b): optimal slope and intercept.
    """
    K = P_dev.shape[1]
    grades = np.arange(K)
    E = (P_dev * grades).sum(axis=1)

    cutpoints = np.array([0.5, 1.5, 2.5, 3.5])

    def obj(params):
        a, b = params
        e_adj = a * E + b
        preds = np.digitize(e_adj, cutpoints)
        preds = np.clip(preds, 0, K - 1)
        qwk = cohen_kappa_score(y_dev, preds, weights='quadratic')
        return -qwk  # minimize negative QWK

    res = minimize(obj, x0=[1.0, 0.0], method='Nelder-Mead')
    a_opt, b_opt = float(res.x[0]), float(res.x[1])
    return a_opt, b_opt


def apply_devcut(P: np.ndarray, a: float, b: float) -> np.ndarray:
    """Apply devcut recalibration to probabilities P.

    Returns:
        (N, 5) one-hot or discrete probabilities matching the digitized predictions.
    """
    K = P.shape[1]
    grades = np.arange(K)
    E = (P * grades).sum(axis=1)
    e_adj = a * E + b
    cutpoints = np.array([0.5, 1.5, 2.5, 3.5])
    preds = np.digitize(e_adj, cutpoints)
    preds = np.clip(preds, 0, K - 1)
    return np.eye(K)[preds]
