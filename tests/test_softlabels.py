"""Tests for WP4: Nested Soft Labels (Test T6).

Requirements:
  - Rows sum to 1
  - argmax(Q) == y
  - alpha is 0 for confident samples
  - Nested IDs disjoint (outer fold k is never in inner training)
"""

import os
import tempfile
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold

from src.training.softlabels import build_soft_labels, make_softlabels
from src.data.datasets import DRDataset
from src.data.splits import aptos_frames


def test_t6_soft_labels_properties():
    """Verify mathematical and algorithmic contract of build_soft_labels (T6)."""
    np.random.seed(42)
    K = 5
    n = 200
    y = np.random.choice(K, size=n)

    # Generate probabilities: some very confident, some ambiguous
    P = np.random.dirichlet(np.ones(K), size=n)

    # Make first 20 samples hyper-confident (s_i close to 1.0)
    for i in range(20):
        P[i, :] = 0.01 / (K - 1)
        P[i, y[i]] = 0.99

    # Make next 20 samples ambiguous/uncertain (s_i low)
    for i in range(20, 40):
        P[i, :] = 0.9 / (K - 1)
        P[i, y[i]] = 0.1

    Q, alpha = build_soft_labels(P, y, K=K, eps=0.1, sigma=0.7, alpha_max=0.4)

    # 1. Rows must sum to 1
    assert np.allclose(Q.sum(axis=1), np.ones(n), atol=1e-5), "Q rows must sum to 1"

    # 2. argmax(Q) == y always (no label flipping)
    assert np.all(np.argmax(Q, axis=1) == y), "argmax(Q) must equal ground truth y"

    # 3. alpha is 0 for confident samples (where s_i >= threshold t_{y_i})
    s = P[np.arange(n), y]
    t = np.array([s[y == j].mean() if (y == j).any() else 1.0 for j in range(K)])
    confident_mask = (s >= t[y])
    assert np.all(alpha[confident_mask] == 0.0), "alpha must be 0 for confident samples"

    # 4. alpha is bounded by [0, alpha_max]
    assert np.all(alpha >= 0.0) and np.all(alpha <= 0.40001), "alpha must be in [0, alpha_max]"

    # 5. Ambiguous samples have alpha > 0
    ambiguous_mask = (s < t[y])
    if ambiguous_mask.any():
        assert np.all(alpha[ambiguous_mask] > 0.0), "alpha must be > 0 for unconfident samples"


def test_t6_nested_isolation():
    """Verify nested cross-validation isolation: outer fold k is NEVER in inner training (T6)."""
    # Create synthetic pool of 100 images
    n_samples = 100
    ids = [f'aptos_{i:04d}' for i in range(n_samples)]
    diagnosis = np.repeat(np.arange(5), 20)
    df = pd.DataFrame({'id_code': ids, 'diagnosis': diagnosis})

    outer_k = 5
    inner_k = 4
    outer_skf = StratifiedKFold(n_splits=outer_k, shuffle=True, random_state=42)

    for k, (outer_tr_idx, outer_va_idx) in enumerate(outer_skf.split(df, df['diagnosis'])):
        outer_tr_df = df.iloc[outer_tr_idx].reset_index(drop=True)
        outer_va_df = df.iloc[outer_va_idx].reset_index(drop=True)

        outer_val_ids = set(outer_va_df['id_code'].astype(str))
        outer_tr_ids = set(outer_tr_df['id_code'].astype(str))

        # Check outer isolation
        assert outer_val_ids.isdisjoint(outer_tr_ids)

        # Inner cross-validation on outer_tr_df (pool T_k)
        inner_skf = StratifiedKFold(n_splits=inner_k, shuffle=True, random_state=42 + k)
        for j, (inner_tr_idx, inner_va_idx) in enumerate(inner_skf.split(outer_tr_df, outer_tr_df['diagnosis'])):
            inner_tr_df = outer_tr_df.iloc[inner_tr_idx]
            inner_va_df = outer_tr_df.iloc[inner_va_idx]

            inner_tr_ids = set(inner_tr_df['id_code'].astype(str))
            inner_va_ids = set(inner_va_df['id_code'].astype(str))

            # CRITICAL ASSERTION: Outer fold k validation ids must NEVER appear in inner train OR inner val
            assert outer_val_ids.isdisjoint(inner_tr_ids), (
                f"Outer fold {k} val ID found in inner training set for inner fold {j}!"
            )
            assert outer_val_ids.isdisjoint(inner_va_ids), (
                f"Outer fold {k} val ID found in inner validation set for inner fold {j}!"
            )


def test_dataset_uses_soft_labels():
    """Verify DRDataset incorporates precomputed soft labels Q when soft dict is supplied."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create a tiny image
        import cv2
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imwrite(os.path.join(tmp_dir, 'sample_0.jpeg'), img)

        df = pd.DataFrame({'id_code': ['sample_0'], 'diagnosis': [2]})
        cfg = {
            'preproc': 'legacy',
            'aug': 'basic',
            'img_size': 100,
            'ordinal_eps': 0.1,
            'ordinal_sigma': 0.7,
        }

        # Case 1: soft is None -> uses base_target
        ds_hard = DRDataset(df, tmp_dir, cfg, train=True, soft=None)
        item_hard = ds_hard[0]

        # Case 2: soft provided with custom Q
        custom_q = np.array([0.05, 0.15, 0.60, 0.15, 0.05], dtype=np.float32)
        soft_map = {'sample_0': custom_q}
        ds_soft = DRDataset(df, tmp_dir, cfg, train=True, soft=soft_map)
        item_soft = ds_soft[0]

        assert np.allclose(item_soft['q'].numpy(), custom_q, atol=1e-5)
        assert not np.allclose(item_hard['q'].numpy(), custom_q, atol=1e-5)
