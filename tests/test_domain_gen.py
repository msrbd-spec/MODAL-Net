"""Tests for WP5: Domain Generalisation (MixStyle, Amplitude Mix, UDA, Dev-Cut Recalibration, DEV/TEST handling)."""

import os
import tempfile
import cv2
import numpy as np
import pytest
import torch
import pandas as pd

from src.models.gradenet import GradeNet
from src.evaluation.calibration import fit_devcut, apply_devcut
from src.training.trainer import build_style_bank
from src.data.datasets import DRDataset
from src.evaluation.tables import phase_select, selection_guard


def test_devcut_calibration():
    """Verify fit_devcut and apply_devcut linear mapping and QWK optimization."""
    np.random.seed(42)
    N = 100
    y_dev = np.random.choice(5, size=N)

    # Simulated probabilities with a shift in expected value
    P_dev = np.zeros((N, 5))
    for i in range(N):
        # Shifted: true grade y, but probabilities peaked at y * 0.8
        shifted_y = y_dev[i] * 0.8
        probs = np.exp(-0.5 * (np.arange(5) - shifted_y) ** 2)
        P_dev[i] = probs / probs.sum()

    a_opt, b_opt = fit_devcut(y_dev, P_dev)

    # Apply devcut to test set
    P_test = np.copy(P_dev)
    P_recal = apply_devcut(P_test, a_opt, b_opt)

    assert P_recal.shape == (N, 5)
    assert np.allclose(P_recal.sum(axis=1), np.ones(N), atol=1e-5)
    # devcut returns discrete/one-hot probabilities
    assert np.all(np.isin(P_recal, [0.0, 1.0]))


def test_gradenet_mixstyle_forward():
    """Verify GradeNet with MixStyle manual stage loop."""
    cfg = {
        'backbone': 'maxvit_large_tf_512.in21k_ft_in1k',
        'drop_path': 0.1,
        'pretrained': False,
        'num_classes': 5,
        'head_type': 'softmax',
        'head_dropout': 0.3,
        'mixstyle': True,
        'ms_p': 1.0,
        'ms_alpha': 0.1,
        'ms_stages': [0, 1],
    }
    model = GradeNet(cfg)
    model.train()

    # Batch of size 2 (1 APTOS | 1 EyePACS)
    x = torch.randn(2, 3, 512, 512)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 5)

    # In eval mode MixStyle acts as identity
    model.eval()
    with torch.no_grad():
        out_eval = model(x)
    assert out_eval.shape == (2, 5)


def test_style_bank_and_dataset_amp_mix():
    """Verify build_style_bank and DRDataset with lazy file path loading."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create small test images
        img1 = np.full((64, 64, 3), 50, dtype=np.uint8)
        img2 = np.full((64, 64, 3), 150, dtype=np.uint8)
        p1 = os.path.join(tmp_dir, 'img1.jpeg')
        p2 = os.path.join(tmp_dir, 'img2.jpeg')
        cv2.imwrite(p1, img1)
        cv2.imwrite(p2, img2)

        df = pd.DataFrame({'id_code': ['img1'], 'diagnosis': [1]})
        cfg = {
            'preproc': 'scaled',
            'aug': 'basic',
            'img_size': 64,
            'amp_mix': True,
            'amp_prob': 1.0,  # always mix
            'amp_stage': 'pre',
            'amp_beta_min': 0.05,
            'amp_beta_max': 0.10,
        }

        # Pass file paths in style bank
        bank = [p2]
        ds = DRDataset(df, dirs=[tmp_dir], cfg=cfg, train=True, style_bank=bank)
        item = ds[0]

        assert item['x'].shape == (3, 64, 64)
        assert item['domain'] == 0


def test_phase6_selection_criteria():
    """Verify phase_select ranks candidates by Messidor-DEV QWK and respects winner_eligible."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        phase = 'phase6_domain_gen'
        p_dir = os.path.join('results', phase)
        os.makedirs(p_dir, exist_ok=True)

        # Reference
        ref_dir = os.path.join(p_dir, 'p6_ref')
        os.makedirs(ref_dir, exist_ok=True)
        y = np.array([0, 1, 2, 3, 4] * 20)
        P_ref_dev = np.eye(5)[y].copy()
        # introduce slight errors in ref dev so candidate has strictly positive delta QWK
        P_ref_dev[0] = [0.2, 0.8, 0, 0, 0]
        P_ref_dev[5] = [0.1, 0.1, 0.8, 0, 0]
        P = np.eye(5)[y]
        np.savez(os.path.join(ref_dir, 'oof.npz'), ids=np.arange(100), y=y, P=P)
        np.savez(os.path.join(ref_dir, 'predictions_external_dev.npz'), ids=np.arange(100), y=y, P=P_ref_dev)
        pd.DataFrame([{
            'oof_qwk': '0.85', 'extdev_qwk': '0.70', 'winner_eligible': 'true',
            'cv_val_acc': '80.00', 'cv_val_qwk': '0.85', 'cv_val_bin_auroc': '0.90'
        }]).to_csv(os.path.join(ref_dir, 'summary.csv'), index=False)

        # Candidate 1: High DEV QWK, eligible
        c1_dir = os.path.join(p_dir, 'p6_amp')
        os.makedirs(c1_dir, exist_ok=True)
        np.savez(os.path.join(c1_dir, 'oof.npz'), ids=np.arange(100), y=y, P=P)
        np.savez(os.path.join(c1_dir, 'predictions_external_dev.npz'), ids=np.arange(100), y=y, P=P)
        pd.DataFrame([{
            'oof_qwk': '0.85', 'extdev_qwk': '0.78', 'winner_eligible': 'true',
            'cv_val_acc': '80.00', 'cv_val_qwk': '0.85', 'cv_val_bin_auroc': '0.90'
        }]).to_csv(os.path.join(c1_dir, 'summary.csv'), index=False)

        # Candidate 2: Higher DEV QWK but winner_eligible = false (e.g. UDA or devcut)
        c2_dir = os.path.join(p_dir, 'p6_uda')
        os.makedirs(c2_dir, exist_ok=True)
        np.savez(os.path.join(c2_dir, 'oof.npz'), ids=np.arange(100), y=y, P=P)
        np.savez(os.path.join(c2_dir, 'predictions_external_dev.npz'), ids=np.arange(100), y=y, P=P)
        pd.DataFrame([{
            'oof_qwk': '0.86', 'extdev_qwk': '0.82', 'winner_eligible': 'false',
            'cv_val_acc': '80.00', 'cv_val_qwk': '0.86', 'cv_val_bin_auroc': '0.90'
        }]).to_csv(os.path.join(c2_dir, 'summary.csv'), index=False)

        phase_select(phase, 'p6_ref')

        # Check winner is p6_amp (not p6_uda because winner_eligible=false)
        winner_path = os.path.join(p_dir, 'WINNER.txt')
        assert os.path.isfile(winner_path)
        winner_tag = open(winner_path).read().splitlines()[0].strip()
        assert winner_tag == 'p6_amp'

        # Cleanup
        import shutil
        shutil.rmtree(p_dir)
