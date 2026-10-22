"""Tests for WP8 optional extensions: SAM and Feature Balancing (S7n)."""

import os
import shutil
import tempfile
import numpy as np
import pytest
import torch
import torch.nn as nn

from src.training.sam import SAM
from src.training.feature_balance import (
    rebalance_smote,
    rebalance_gauss,
    rebalance_lfsd,
    train_balanced_head,
    predict_with_head,
    BalancedHead,
    LFSD,
    run_feature_balance_experiment,
)


def test_sam_optimizer():
    """Verify SAM two-pass optimization on a synthetic linear problem."""
    torch.manual_seed(42)
    model = nn.Linear(10, 2)
    optimizer = SAM(model.parameters(), torch.optim.SGD, rho=0.05, lr=0.1)

    x = torch.randn(16, 10)
    y = torch.randint(0, 2, (16,))
    criterion = nn.CrossEntropyLoss()

    # Pass 1: standard step via closure
    initial_loss = criterion(model(x), y).item()

    def closure():
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        return loss

    optimizer.step(closure)
    new_loss = criterion(model(x), y).item()
    assert new_loss < initial_loss, f"Expected loss to decrease: initial={initial_loss}, new={new_loss}"

    # Pass 2: manual first_step and second_step
    optimizer.zero_grad()
    loss1 = criterion(model(x), y)
    loss1.backward()
    optimizer.first_step(zero_grad=True)

    loss2 = criterion(model(x), y)
    loss2.backward()
    optimizer.second_step(zero_grad=True)


def test_rebalance_smote():
    """Verify SMOTE balances minority class counts and preserves feature dimension."""
    np.random.seed(42)
    # 5 classes with highly skewed counts
    X = np.random.randn(100, 32).astype(np.float32)
    # counts: 50, 20, 15, 10, 5
    y = np.array([0] * 50 + [1] * 20 + [2] * 15 + [3] * 10 + [4] * 5, dtype=np.int64)

    X_bal, y_bal = rebalance_smote(X, y, k_neighbors=3, seed=42)
    assert X_bal.shape[1] == 32
    assert len(X_bal) == len(y_bal)

    classes, counts = np.unique(y_bal, return_counts=True)
    assert len(classes) == 5
    for cnt in counts:
        assert cnt == 50, f"Expected count 50 for all classes, got {cnt}"


def test_rebalance_gauss():
    """Verify class-conditional Gaussian sampling with shrinkage covariance."""
    np.random.seed(42)
    X = np.random.randn(80, 16).astype(np.float32)
    y = np.array([0] * 40 + [1] * 20 + [2] * 10 + [3] * 6 + [4] * 4, dtype=np.int64)

    X_bal, y_bal = rebalance_gauss(X, y, shrinkage=0.2, seed=42)
    assert X_bal.shape[1] == 16
    assert len(X_bal) == len(y_bal)

    classes, counts = np.unique(y_bal, return_counts=True)
    assert len(classes) == 5
    for cnt in counts:
        assert cnt == 40, f"Expected count 40 for all classes, got {cnt}"


def test_rebalance_lfsd_diffusion():
    """Verify 1-D DDPM (LFSD) training and minority sampling."""
    torch.manual_seed(42)
    np.random.seed(42)
    X = np.random.randn(50, 16).astype(np.float32)
    y = np.array([0] * 25 + [1] * 10 + [2] * 8 + [3] * 5 + [4] * 2, dtype=np.int64)

    # Test direct model methods
    lfsd = LFSD(in_dim=16, num_classes=5, hidden_dim=64, T=10)
    bx = torch.randn(8, 16)
    by = torch.randint(0, 5, (8,))
    loss = lfsd.loss(bx, by)
    assert loss.dim() == 0
    assert not torch.isnan(loss)

    samples = lfsd.sample(num_samples=4, target_class=2, device='cpu')
    assert samples.shape == (4, 16)

    # Test rebalance_lfsd function (short epochs for test)
    X_bal, y_bal = rebalance_lfsd(X, y, epochs=5, batch_size=16, device='cpu', seed=42)
    assert X_bal.shape[1] == 16
    classes, counts = np.unique(y_bal, return_counts=True)
    for cnt in counts:
        assert cnt == 25


def test_balanced_head_training_and_inference():
    """Verify BalancedHead training with reweight (logit adjustment) and plain inference."""
    torch.manual_seed(42)
    np.random.seed(42)
    X = np.random.randn(100, 32).astype(np.float32)
    y = np.random.randint(0, 5, (100,))

    head, log_prior = train_balanced_head(
        X, y, method='reweight', tau=1.0, epochs=5, batch_size=32, device='cpu', seed=42
    )
    assert isinstance(head, BalancedHead)
    assert log_prior is not None
    assert len(log_prior) == 5

    probs = predict_with_head(head, X[:10], device='cpu')
    assert probs.shape == (10, 5)
    np.testing.assert_allclose(probs.sum(axis=1), np.ones(10), atol=1e-5)


def test_feature_balance_experiment_mock():
    """Verify end-to-end execution of run_feature_balance_experiment with mock cache."""
    tmp_dir = tempfile.mkdtemp()
    try:
        base_tag = 'test_owner_model'
        exp_phase = 'phase7_feature_balance'
        fb_tag = 'test_fb_reweight'

        # Set up mock base checkpoints and extracted features
        owner_ckpt_dir = os.path.join('checkpoints', base_tag)
        feat_cache_dir = os.path.join(owner_ckpt_dir, 'extracted_features')
        os.makedirs(feat_cache_dir, exist_ok=True)

        for k in range(5):
            # Dummy checkpoint file
            torch.save({'dummy': 1}, os.path.join(owner_ckpt_dir, f'fold{k}.pth'))
            # Mock extracted features
            np.savez(
                os.path.join(feat_cache_dir, f'fold{k}_features.npz'),
                oof_feat=np.random.randn(20, 16).astype(np.float32),
                oof_y=np.random.randint(0, 5, 20),
                oof_ids=np.array([f"oof_{k}_{i}" for i in range(20)]),
                test_feat=np.random.randn(15, 16).astype(np.float32),
                test_y=np.random.randint(0, 5, 15),
                test_ids=np.array([f"test_{i}" for i in range(15)]),
                dev_feat=np.random.randn(10, 16).astype(np.float32),
                dev_y=np.random.randint(0, 5, 10),
                dev_ids=np.array([f"dev_{i}" for i in range(10)]),
                mtest_feat=np.random.randn(10, 16).astype(np.float32),
                mtest_y=np.random.randint(0, 5, 10),
                mtest_ids=np.array([f"mtest_{i}" for i in range(10)]),
            )

        # Mock base experiment results so find_exp finds it
        base_exp_dir = os.path.join('results', 'phase6_domain_gen', base_tag)
        os.makedirs(base_exp_dir, exist_ok=True)
        with open(os.path.join(base_exp_dir, 'summary.csv'), 'w') as f_s:
            f_s.write("experiment,test_qwk,oof_qwk\n" f"{base_tag},0.85,0.84\n")

        cfg = {
            'tag': fb_tag,
            'phase': exp_phase,
            'base': base_tag,
            'method': 'reweight',
            'tau': 1.0,
            'seed': 42,
            'paths': {
                'aptos': {'train_images': tmp_dir, 'val_images': tmp_dir, 'splits_dir': 'data/splits'},
                'messidor2': {'images': tmp_dir, 'splits_dir': 'data/splits'},
            },
        }

        summary = run_feature_balance_experiment(cfg)
        assert summary is not None

        exp_res_dir = os.path.join('results', exp_phase, fb_tag)
        assert os.path.isfile(os.path.join(exp_res_dir, 'summary.csv'))
        assert os.path.isfile(os.path.join(exp_res_dir, 'oof.npz'))
        assert os.path.isfile(os.path.join(exp_res_dir, 'predictions_test.npz'))
        assert os.path.isfile(os.path.join(exp_res_dir, 'thresholds.json'))
        assert os.path.isfile(os.path.join('checkpoints', fb_tag, 'config_snapshot.yaml'))

        # Check that ckpt_from points to owner
        import yaml
        snap = yaml.safe_load(open(os.path.join('checkpoints', fb_tag, 'config_snapshot.yaml')))
        assert snap.get('ckpt_from') == base_tag

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        # Clean up temporary test directories
        shutil.rmtree(os.path.join('checkpoints', base_tag), ignore_errors=True)
        shutil.rmtree(os.path.join('checkpoints', fb_tag), ignore_errors=True)
        shutil.rmtree(os.path.join('results', 'phase6_domain_gen', base_tag), ignore_errors=True)
        shutil.rmtree(os.path.join('results', exp_phase, fb_tag), ignore_errors=True)
