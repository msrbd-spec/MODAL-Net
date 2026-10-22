"""Ambiguity-Aware Soft Labels (AASL) with Nested Cross-Validation Protocol.

Implements S7k:
  - build_soft_labels(P, y, K=5, eps=0.1, sigma=0.7, alpha_max=0.4):
      Computes confident-learning thresholds t_y, ambiguity parameter alpha,
      and softened target distribution Q where argmax(Q) == y always.
  - make_softlabels(cfg):
      Executes 5 x 4 inner cross-validation on outer training pools T_k,
      enforcing strict nested isolation (outer fold k is never in inner training).
      Saves softlabels/<tag>/outer<k>.npz with keys (ids, P, Q, alpha, y).
"""

import os
import json
import yaml
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.data.splits import fold_frames
from src.training.trainer import train_fold
from src.utils.logger import get_logger


def build_soft_labels(P: np.ndarray, y: np.ndarray, K: int = 5,
                       eps: float = 0.1, sigma: float = 0.7,
                       alpha_max: float = 0.4):
    """Compute ambiguity-aware soft labels Q and ambiguity scores alpha.

    Args:
        P: (N, K) predicted probability distribution from cross-validation.
        y: (N,) ground-truth class integer indices in [0, K-1].
        K: Number of classes (default: 5).
        eps: Ordinal smoothing label noise budget (default: 0.1).
        sigma: Ordinal Gaussian dispersion width (default: 0.7).
        alpha_max: Maximum softening weight (default: 0.4).

    Returns:
        Q: (N, K) normalized soft label distribution where argmax(Q) == y.
        a: (N,) sample-wise ambiguity parameter in [0, alpha_max].
    """
    n = len(y)
    y = np.asarray(y, dtype=int)
    P = np.asarray(P, dtype=float)

    # 1. Base target distribution with ordinal Gaussian smoothing
    oh = np.eye(K)[y]
    kk = np.arange(K)[None, :]
    g = np.exp(-((kk - y[:, None]) ** 2) / (2.0 * (sigma ** 2)))
    g /= g.sum(axis=1, keepdims=True)
    base = (1.0 - eps) * oh + eps * g

    # 2. Confident-learning class-wise thresholds
    s = P[np.arange(n), y]
    t = np.array([
        s[y == j].mean() if (y == j).any() else 1.0
        for j in range(K)
    ])

    # 3. Sample ambiguity score: zero for confident samples, up to alpha_max
    a = np.clip((t[y] - s) / np.maximum(t[y], 1e-6), 0.0, 1.0) * alpha_max

    # 4. Blend base and predicted distribution
    Q = (1.0 - a)[:, None] * base + a[:, None] * P

    # 5. Row-normalize
    Q = Q / Q.sum(axis=1, keepdims=True)

    return Q, a


def make_softlabels(cfg: dict):
    """Generate nested soft labels for all outer folds.

    Protocol (S7k):
      For outer fold k:
        - Outer training pool T_k = the other 4 folds
        - Outer validation pool V_k = fold k
        - Assert V_k is completely disjoint from T_k and any inner training set
        - Inner StratifiedKFold(inner_k=4, shuffle=True, seed=42+k) on T_k
        - 4 inner models (inner_epochs, mixstyle/amp off, soft_labels none)
        - Collect inner-OOF P for every sample in T_k
        - build_soft_labels -> softlabels/<tag>/outer<k>.npz (ids, P, Q, alpha, y)
    """
    tag = cfg.get('tag') or 'default_sl'
    logger = get_logger(tag, 'make_softlabels')
    logger.info(f"Starting make_softlabels for tag: {tag}")

    # Snapshot configuration
    ckpt_dir = os.path.join('checkpoints', tag)
    os.makedirs(ckpt_dir, exist_ok=True)
    snap_path = os.path.join(ckpt_dir, 'config_snapshot.yaml')
    with open(snap_path, 'w') as f_snap:
        yaml.dump(cfg, f_snap)
    logger.info(f"Saved config snapshot to {snap_path}")

    sl_dir = os.path.join('softlabels', tag)
    os.makedirs(sl_dir, exist_ok=True)

    n_folds = int(cfg.get('n_folds', 5))
    inner_k = int(cfg.get('inner_k', 4))
    inner_epochs = int(cfg.get('inner_epochs', 15))
    K = int(cfg.get('num_classes', 5))
    eps = float(cfg.get('ordinal_eps', 0.1))
    sigma = float(cfg.get('ordinal_sigma', 0.7))
    alpha_max = float(cfg.get('alpha_max', 0.4))
    phase = cfg.get('phase', 'phase4_label_loss')

    for k in range(n_folds):
        sl_file = os.path.join(sl_dir, f'outer{k}.npz')
        if os.path.isfile(sl_file):
            logger.info(f"Outer fold {k} soft labels already exist: {sl_file}. Skipping.")
            continue

        tr_df, va_df = fold_frames(cfg, k)
        outer_val_ids = set(va_df['id_code'].astype(str))
        outer_tr_ids = set(tr_df['id_code'].astype(str))

        # Check isolation: outer fold k must be disjoint from training pool
        assert outer_val_ids.isdisjoint(outer_tr_ids), (
            f"Isolation failure in fold {k}: outer val ids overlap with outer train ids!"
        )

        logger.info(f"=== Outer Fold {k}/{n_folds} (Pool size: {len(tr_df)} images) ===")
        skf = StratifiedKFold(n_splits=inner_k, shuffle=True, random_state=42 + k)
        inner_splits = list(skf.split(tr_df, tr_df['diagnosis']))

        inner_oof_dict = {}

        for j, (in_tr_idx, in_va_idx) in enumerate(inner_splits):
            in_tr_df = tr_df.iloc[in_tr_idx].reset_index(drop=True)
            in_va_df = tr_df.iloc[in_va_idx].reset_index(drop=True)

            # Strict nested isolation check: outer fold k must NEVER occur in inner training
            in_tr_ids = set(in_tr_df['id_code'].astype(str))
            assert outer_val_ids.isdisjoint(in_tr_ids), (
                f"Leakage in outer fold {k}, inner fold {j}: outer fold {k} ID leaked into inner training set!"
            )

            # Inner configuration overrides
            inner_cfg = dict(cfg)
            inner_cfg['epochs'] = inner_epochs
            inner_cfg['mixstyle'] = False
            inner_cfg['amp_mix'] = False
            inner_cfg['soft_labels'] = 'none'
            inner_cfg['tag'] = tag
            inner_cfg['phase'] = phase

            inner_ckpt_path = os.path.join('checkpoints', tag, 'inner', f'outer{k}_inner{j}.pth')
            logger.info(f"Training inner fold {j}/{inner_k} for outer {k} -> {inner_ckpt_path}")

            train_fold(
                inner_cfg,
                k=j,
                tag=tag,
                logger=logger,
                train_df=in_tr_df,
                val_df=in_va_df,
                out=inner_ckpt_path
            )

            # Read inner OOF predictions
            inner_stem = f'outer{k}_inner{j}'
            inner_oof_path = os.path.join('results', phase, tag, f'{inner_stem}_oof.npz')
            assert os.path.isfile(inner_oof_path), f"Inner OOF file missing: {inner_oof_path}"

            with np.load(inner_oof_path, allow_pickle=True) as f_oof:
                oof_ids = [str(x) for x in f_oof['ids']]
                oof_p = f_oof['P']
                for sample_id, p_vec in zip(oof_ids, oof_p):
                    inner_oof_dict[sample_id] = p_vec

        # Verify all images in tr_df have received inner-OOF predictions
        all_tr_ids = [str(x) for x in tr_df['id_code']]
        assert all(s_id in inner_oof_dict for s_id in all_tr_ids), (
            f"Missing inner-OOF predictions for outer fold {k}!"
        )

        P_Tk = np.array([inner_oof_dict[s_id] for s_id in all_tr_ids])
        y_Tk = tr_df['diagnosis'].values.astype(int)
        ids_Tk = np.array(all_tr_ids)

        # Build soft labels
        Q_Tk, alpha_Tk = build_soft_labels(
            P=P_Tk,
            y=y_Tk,
            K=K,
            eps=eps,
            sigma=sigma,
            alpha_max=alpha_max
        )

        # Assert contract: rows sum to 1 and argmax(Q) == y
        assert np.allclose(Q_Tk.sum(axis=1), np.ones(len(Q_Tk)), atol=1e-5), "Q rows do not sum to 1!"
        assert np.all(np.argmax(Q_Tk, axis=1) == y_Tk), "Softening flipped argmax from true label!"

        np.savez(
            sl_file,
            ids=ids_Tk,
            P=P_Tk,
            Q=Q_Tk,
            alpha=alpha_Tk,
            y=y_Tk
        )
        logger.info(f"Saved {len(ids_Tk)} soft labels for outer fold {k} to {sl_file}")

    logger.info(f"Nested soft labels complete for tag: {tag}")
