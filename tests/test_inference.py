"""Unit tests for WP6: Inference, TTA, Conformal Prediction, and Selective Classification."""

import os
import tempfile
import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from src.evaluation.predict import apply_view, rescale, TTA_CONFIGS
from src.evaluation.conformal import (
    interval_scores,
    compute_qhat,
    interval_set,
    predict_intervals,
    selective_prediction_metrics,
    evaluate_conformal_predictions,
)
from src.evaluation.calibration import fit_devcut, apply_devcut
from src.evaluation.experiment import (
    run_inference_experiment,
    run_ensemble_experiment,
    run_calibrate_conformal,
)
from src.utils.config import ckpt_owner


def test_t10_conformal_exchangeable_coverage():
    """Test T10: Empirical coverage >= 1 - alpha - 0.02 on synthetic exchangeable data."""
    rng = np.random.default_rng(42)
    N = 3000
    K = 5
    alpha = 0.10

    # Generate synthetic Dirichlet probability distributions
    raw_alphas = np.array([2.0, 1.5, 1.0, 0.8, 0.5])
    P = rng.dirichlet(raw_alphas, size=N)

    # Sample true labels y according to the distribution P_i
    y = np.array([rng.choice(K, p=P[i]) for i in range(N)])

    # Split into calibration and test sets (exchangeable split)
    n_cal = 1500
    P_cal, y_cal = P[:n_cal], y[:n_cal]
    P_test, y_test = P[n_cal:], y[n_cal:]

    # Compute conformal quantile on calibration set
    scores_cal = interval_scores(P_cal, y_cal)
    qhat = compute_qhat(scores_cal, alpha=alpha)

    # Predict intervals on test set
    intervals = predict_intervals(P_test, qhat)
    lo = intervals[:, 0]
    hi = intervals[:, 1]

    # Check empirical coverage
    covered = (y_test >= lo) & (y_test <= hi)
    emp_coverage = float(np.mean(covered))

    target_coverage = 1.0 - alpha
    # T10 spec: empirical coverage >= 1 - alpha - 0.02
    assert emp_coverage >= target_coverage - 0.02, (
        f"Empirical coverage {emp_coverage:.4f} is below target {target_coverage - 0.02:.4f}"
    )


def test_conformal_interval_properties():
    """Verify mathematical properties of ordinal-interval conformal sets."""
    # 1. Monotonicity: larger score means more mass needed
    P = np.array([
        [0.8, 0.1, 0.05, 0.03, 0.02],
        [0.05, 0.1, 0.7, 0.1, 0.05],
    ])
    y_easy = np.array([0, 2])
    y_hard = np.array([4, 0])

    scores_easy = interval_scores(P, y_easy)
    scores_hard = interval_scores(P, y_hard)

    assert np.isclose(scores_easy[0], 0.8)
    assert np.isclose(scores_easy[1], 0.7)
    assert np.isclose(scores_hard[0], 1.0)
    assert np.isclose(scores_hard[1], 0.95)

    # 2. Interval set includes argmax always
    for p in P:
        lo, hi = interval_set(p, qhat=0.5)
        argmax = int(np.argmax(p))
        assert lo <= argmax <= hi
        assert lo <= hi

    # 3. Referable abstention logic
    # Sample with interval [1, 2] crosses grade 1|2 boundary -> must abstain
    y_test = np.array([1, 3])
    P_test = np.array([
        [0.1, 0.45, 0.45, 0.0, 0.0],  # interval will cover 1 and 2
        [0.0, 0.0, 0.1, 0.8, 0.1],     # interval will cover [3, 3] or [2, 3]
    ])
    qhat = 0.85
    eval_res = evaluate_conformal_predictions(P_test, y_test, qhat, alpha=0.10)
    assert float(eval_res['abstention_referable']) == 50.0  # 1 out of 2 samples


def test_selective_prediction():
    """Verify selective classification metrics and risk-coverage relationship."""
    rng = np.random.default_rng(123)
    N = 200
    y = rng.choice(5, size=N)
    # Generate probabilities: high confidence for correct, low confidence for incorrect
    P = np.zeros((N, 5))
    for i in range(N):
        if i < 150:  # 75% correct
            P[i, y[i]] = rng.uniform(0.7, 0.99)
            rem = (1.0 - P[i, y[i]]) / 4
            for c in range(5):
                if c != y[i]:
                    P[i, c] = rem
        else:  # incorrect
            wrong = (y[i] + 1) % 5
            P[i, wrong] = rng.uniform(0.3, 0.5)
            rem = (1.0 - P[i, wrong]) / 4
            for c in range(5):
                if c != wrong:
                    P[i, c] = rem

    metrics = selective_prediction_metrics(P, y)
    assert 0.0 <= float(metrics['aurc']) <= 1.0
    # Higher confidence threshold (lower coverage) should have higher accuracy
    assert float(metrics['acc_at_cov70']) >= float(metrics['acc_at_cov90']) - 1.0


def test_predict_tta_and_rescale():
    """Verify apply_view rotations/flips and multi-scale interpolation."""
    B, C, H, W = 2, 3, 32, 32
    x = torch.randn(B, C, H, W)

    # 1. apply_view
    for r in range(4):
        for h in (0, 1):
            for v in (0, 1):
                out = apply_view(x, r, h, v)
                assert out.shape == (B, C, H, W)

    # Identity check
    assert torch.allclose(apply_view(x, 0, 0, 0), x)

    # 2. rescale
    # Scale factor 1.0 is exact identity
    assert torch.allclose(rescale(x, 1.0), x)

    # Scale factors 0.9 and 1.1 preserve shape exactly
    for s in (0.9, 1.1):
        x_scaled = rescale(x, s)
        assert x_scaled.shape == (B, C, H, W)


def test_ckpt_owner_resolution():
    """Verify ckpt_owner traces back through ckpt_from and base chains."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create owner folder with dummy fold0.pth
        owner_dir = os.path.join('checkpoints', 'owner_test_tag')
        os.makedirs(owner_dir, exist_ok=True)
        with open(os.path.join(owner_dir, 'fold0.pth'), 'w') as f:
            f.write('dummy')

        # Create intermediate tag pointing to owner
        inter_dir = os.path.join('checkpoints', 'inter_test_tag')
        os.makedirs(inter_dir, exist_ok=True)
        with open(os.path.join(inter_dir, 'config_snapshot.yaml'), 'w') as f:
            yaml.safe_dump({'ckpt_from': 'owner_test_tag'}, f)

        # Query ckpt_owner with child config pointing to inter
        child_cfg = {'ckpt_from': 'inter_test_tag'}
        resolved = ckpt_owner(child_cfg)
        assert resolved == 'owner_test_tag'

        # Cleanup
        import shutil
        shutil.rmtree(owner_dir)
        shutil.rmtree(inter_dir)


def test_calibrate_conformal_workflow(monkeypatch):
    """Verify run_calibrate_conformal generates conformal.csv and figures."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        phase = 'phase8_inference'
        base_tag = 'p8_base_test'
        tag = 'p8_conf_test'

        base_dir = os.path.join('results', phase, base_tag)
        os.makedirs(base_dir, exist_ok=True)

        y = np.array([0, 1, 2, 3, 4] * 20)
        P = np.eye(5)[y]
        P[0] = [0.4, 0.4, 0.2, 0.0, 0.0]  # slight ambiguity

        np.savez(os.path.join(base_dir, 'oof.npz'), ids=np.arange(100), y=y, P=P)
        np.savez(os.path.join(base_dir, 'predictions_test.npz'), ids=np.arange(100), y=y, y_true=y, y_prob=P, P=P)
        np.savez(os.path.join(base_dir, 'predictions_external_test.npz'), ids=np.arange(100), y=y, y_true=y, y_prob=P, P=P)

        cfg = {
            'tag': tag,
            'phase': phase,
            'base': base_tag,
            'conformal_alpha': 0.10,
        }

        conf_test, conf_ext = run_calibrate_conformal(cfg)

        out_csv = os.path.join('results', phase, tag, 'conformal.csv')
        assert os.path.isfile(out_csv)
        df = pd.read_csv(out_csv)
        assert len(df) == 2
        assert 'empirical_coverage' in df.columns
        assert 'abstention_referable' in df.columns

        # Verify figures
        fig_png = os.path.join('results', phase, tag, 'figures', 'conformal_test.png')
        assert os.path.isfile(fig_png)

        # Cleanup
        import shutil
        shutil.rmtree(base_dir)
        shutil.rmtree(os.path.join('results', phase, tag))
        shutil.rmtree(os.path.join('checkpoints', tag), ignore_errors=True)


def test_ensemble_experiment_workflow():
    """Verify run_ensemble_experiment averages member probabilities and sets winner_eligible=false."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        phase = 'phase5_architecture'
        m1_tag = 'm1_test'
        m2_tag = 'm2_test'
        ens_tag = 'ens_test'

        y = np.array([0, 1, 2, 3, 4] * 20)
        P1 = np.eye(5)[y]
        P2 = np.ones((100, 5)) / 5.0

        for m_tag, P in [(m1_tag, P1), (m2_tag, P2)]:
            m_dir = os.path.join('results', phase, m_tag)
            os.makedirs(m_dir, exist_ok=True)
            np.savez(os.path.join(m_dir, 'oof.npz'), ids=np.arange(100), y=y, P=P)
            np.savez(os.path.join(m_dir, 'predictions_test.npz'), ids=np.arange(100), y=y, y_true=y, y_prob=P, P=P)
            np.savez(os.path.join(m_dir, 'predictions_external_dev.npz'), ids=np.arange(100), y=y, y_true=y, y_prob=P, P=P)
            np.savez(os.path.join(m_dir, 'predictions_external_test.npz'), ids=np.arange(100), y=y, y_true=y, y_prob=P, P=P)

        cfg = {
            'tag': ens_tag,
            'phase': phase,
            'members': f'{m1_tag},{m2_tag}',
        }

        sum_row = run_ensemble_experiment(cfg)

        ens_dir = os.path.join('results', phase, ens_tag)
        assert os.path.isfile(os.path.join(ens_dir, 'summary.csv'))
        df = pd.read_csv(os.path.join(ens_dir, 'summary.csv'))
        assert not bool(df.loc[0, 'winner_eligible'])
        assert df.loc[0, 'protocol'] == 'ensemble_of_models'

        # Verify averaged probabilities
        pred_test = np.load(os.path.join(ens_dir, 'predictions_test.npz'))
        expected_avg = (P1 + P2) / 2.0
        assert np.allclose(pred_test['y_prob'], expected_avg)

        # Cleanup
        import shutil
        shutil.rmtree(os.path.join('results', phase, m1_tag))
        shutil.rmtree(os.path.join('results', phase, m2_tag))
        shutil.rmtree(ens_dir)
        shutil.rmtree(os.path.join('checkpoints', ens_tag), ignore_errors=True)


def test_inference_experiment_workflow(monkeypatch):
    """Verify run_inference_experiment resolves owner, saves config_snapshot with ckpt_from, and writes results."""
    import shutil
    import json
    phase = 'phase8_inference'
    owner_tag = 'p6_winner_mock'
    inf_tag = 'p8_tta_flip_mock'

    # Setup owner checkpoints and results
    owner_ckpt_dir = os.path.join('checkpoints', owner_tag)
    owner_res_dir = os.path.join('results', 'phase6_domain_gen', owner_tag)
    os.makedirs(owner_ckpt_dir, exist_ok=True)
    os.makedirs(owner_res_dir, exist_ok=True)

    # 5 dummy checkpoint files
    for k in range(5):
        with open(os.path.join(owner_ckpt_dir, f'fold{k}.pth'), 'w') as f:
            f.write('dummy')

    y = np.array([0, 1, 2, 3, 4] * 10)
    P = np.eye(5)[y]
    np.savez(os.path.join(owner_res_dir, 'oof.npz'), ids=np.arange(50), y=y, P=P)
    with open(os.path.join(owner_ckpt_dir, 'thresholds.json'), 'w') as f:
        json.dump({'thr_spec90': 0.5, 'thr_spec95': 0.7, 'temperature': 1.0}, f)

    # Mock GradeNet and predict_probs
    monkeypatch.setattr('src.evaluation.experiment.GradeNet', lambda cfg: torch.nn.Linear(2, 5))
    monkeypatch.setattr('torch.load', lambda *args, **kwargs: {'weight': torch.randn(5, 2), 'bias': torch.zeros(5)})
    monkeypatch.setattr(
        'src.evaluation.experiment.predict_probs',
        lambda models, loader, cfg, **kwargs: (
            list(range(len(loader.dataset))),
            loader.dataset.df['diagnosis'].values,
            np.eye(5)[loader.dataset.df['diagnosis'].values],
            [np.eye(5)[loader.dataset.df['diagnosis'].values] for _ in range(5)]
        )
    )

    # Mock dataset loaders
    dummy_df = pd.DataFrame({'id_code': [f'img_{i}' for i in range(20)], 'diagnosis': [i % 5 for i in range(20)]})
    monkeypatch.setattr('src.evaluation.experiment.test_frame', lambda cfg: dummy_df)
    monkeypatch.setattr('src.evaluation.experiment.messidor_frame', lambda cfg, split: dummy_df)
    monkeypatch.setattr('src.data.datasets.cv2.imread', lambda path: np.zeros((32, 32, 3), dtype=np.uint8))

    cfg = {
        'tag': inf_tag,
        'phase': phase,
        'base': owner_tag,
        'tta': 'flip',
        'n_folds': 5,
        'img_size': 512,
        'preproc': 'scaled',
        'aug': 'basic',
        'paths': {
            'aptos': {'test_images': 'dummy'},
            'messidor': {'images': 'dummy'},
        }
    }

    sum_row = run_inference_experiment(cfg)

    # Assertions
    inf_res_dir = os.path.join('results', phase, inf_tag)
    assert os.path.isfile(os.path.join(inf_res_dir, 'summary.csv'))
    assert os.path.isfile(os.path.join(inf_res_dir, 'predictions_test.npz'))
    assert os.path.isfile(os.path.join(inf_res_dir, 'predictions_external_test.npz'))
    assert os.path.isfile(os.path.join(inf_res_dir, 'oof.npz'))

    # Check config_snapshot.yaml has ckpt_from: owner_tag
    snap_path = os.path.join('checkpoints', inf_tag, 'config_snapshot.yaml')
    assert os.path.isfile(snap_path)
    with open(snap_path) as f:
        snap_data = yaml.safe_load(f)
    assert snap_data['ckpt_from'] == owner_tag

    # Cleanup
    shutil.rmtree(owner_ckpt_dir)
    shutil.rmtree(owner_res_dir)
    shutil.rmtree(inf_res_dir)
    shutil.rmtree(os.path.join('checkpoints', inf_tag))


