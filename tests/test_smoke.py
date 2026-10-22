"""Test T11: End-to-end smoke test for experiment mode and per-experiment file outputs."""

import os
import shutil
import cv2
import numpy as np
import pandas as pd
import pytest
import yaml

from src.evaluation.experiment import run_experiment


@pytest.fixture
def smoke_env(tmp_path):
    """Create minimal synthetic environment for smoke testing."""
    root = tmp_path / "smoke_data"
    root.mkdir()

    aptos_tr = root / "aptos_train"
    aptos_tr.mkdir()
    aptos_val = root / "aptos_val"
    aptos_val.mkdir()
    aptos_te = root / "aptos_test"
    aptos_te.mkdir()
    mess_dir = root / "mess_images"
    mess_dir.mkdir()

    # Generate dummy fundus images
    def _save_img(path):
        im = np.zeros((64, 64, 3), dtype=np.uint8)
        cv2.circle(im, (32, 32), 25, (30, 80, 180), -1)
        cv2.imwrite(str(path), im)

    # 16 train images across 5 classes
    train_rows = []
    for i in range(16):
        name = f"tr_{i:02d}.png"
        _save_img(aptos_tr / name)
        train_rows.append({"id_code": f"tr_{i:02d}", "diagnosis": i % 5})
    train_df = pd.DataFrame(train_rows)
    train_csv = root / "train_1.csv"
    train_df.to_csv(train_csv, index=False)

    # 8 val images
    val_rows = []
    for i in range(8):
        name = f"val_{i:02d}.png"
        _save_img(aptos_val / name)
        val_rows.append({"id_code": f"val_{i:02d}", "diagnosis": i % 5})
    val_df = pd.DataFrame(val_rows)
    val_csv = root / "valid.csv"
    val_df.to_csv(val_csv, index=False)

    # 8 test images
    test_rows = []
    for i in range(8):
        name = f"te_{i:02d}.png"
        _save_img(aptos_te / name)
        test_rows.append({"id_code": f"te_{i:02d}", "diagnosis": i % 5})
    test_df = pd.DataFrame(test_rows)
    test_csv = root / "test.csv"
    test_df.to_csv(test_csv, index=False)

    # 8 messidor images
    mess_rows = []
    for i in range(8):
        name = f"mess_{i:02d}_0100_PP.png"
        _save_img(mess_dir / name)
        mess_rows.append({
            "id_code": f"mess_{i:02d}_0100_PP",
            "diagnosis": i % 5,
            "adjudicated_dme": 0,
            "adjudicated_gradable": 1,
        })
    mess_df = pd.DataFrame(mess_rows)
    mess_csv = root / "messidor.csv"
    mess_df.to_csv(mess_csv, index=False)

    # Setup splits directory for messidor dev/test
    os.makedirs("splits", exist_ok=True)
    import json
    with open("splits/messidor_dev_ids.json", "w") as f:
        json.dump([f"mess_{i:02d}_0100_PP" for i in range(4)], f)
    with open("splits/messidor_test_ids.json", "w") as f:
        json.dump([f"mess_{i:02d}_0100_PP" for i in range(4, 8)], f)

    cfg = yaml.safe_load(open("configs/config.yaml"))
    cfg.update({
        'phase': 'phase0_protocol',
        'tag': 'smoke',
        'backbone': 'resnet18',  # lightweight backbone for fast CPU smoke test
        'pretrained': False,
        'img_size': 64,
        'epochs': 1,
        'n_folds': 2,
        'batch_size': 4,
        'num_workers': 0,
        'mixstyle': False,
        'amp_mix': False,
        'paths': {
            'aptos': {
                'train_images': str(aptos_tr),
                'val_images': str(aptos_val),
                'test_images': str(aptos_te),
                'train_csv': str(train_csv),
                'val_csv': str(val_csv),
                'test_csv': str(test_csv),
            },
            'messidor': {
                'images': str(mess_dir),
                'csv': str(mess_csv),
            },
            'eyepacs': {
                'images': str(root),
                'labels_csv': str(train_csv),
                'lesion_labels': str(root),
            },
        },
    })
    return cfg


def test_t11_smoke_run(smoke_env):
    """T11: Run 1-epoch 2-fold experiment and verify all S5 files appear, then cleanup."""
    cfg = smoke_env
    tag = cfg['tag']
    phase = cfg['phase']

    # Execute experiment
    summary_row = run_experiment(cfg)
    assert summary_row is not None

    # Verify Checkpoints folder files
    ckpt_dir = os.path.join("checkpoints", tag)
    assert os.path.isfile(os.path.join(ckpt_dir, "fold0.pth"))
    assert os.path.isfile(os.path.join(ckpt_dir, "fold1.pth"))
    assert os.path.isfile(os.path.join(ckpt_dir, "fold0_val_metrics.json"))
    assert os.path.isfile(os.path.join(ckpt_dir, "fold1_val_metrics.json"))
    assert os.path.isfile(os.path.join(ckpt_dir, "thresholds.json"))
    assert os.path.isfile(os.path.join(ckpt_dir, "config_snapshot.yaml"))

    # Verify Results folder files
    res_dir = os.path.join("results", phase, tag)
    expected_files = [
        "summary.csv",
        "per_fold.csv",
        "per_class.csv",
        "binary.csv",
        "confusion_test.csv",
        "confusion_external.csv",
        "calibration.csv",
        "config.yaml",
        "predictions_test.npz",
        "predictions_external_dev.npz",
        "predictions_external_test.npz",
        "oof.npz",
        "curves/fold0.csv",
        "curves/fold1.csv",
        "figures/curves.png",
        "figures/cm_test.png",
        "figures/cm_external.png",
        "figures/roc.png",
        "figures/pr.png",
        "figures/reliability.png",
    ]
    for rel_path in expected_files:
        full_path = os.path.join(res_dir, rel_path)
        assert os.path.isfile(full_path), f"Missing S5 file: {full_path}"

    # Cleanup smoke outputs as requested by T11
    shutil.rmtree(ckpt_dir, ignore_errors=True)
    shutil.rmtree(res_dir, ignore_errors=True)
    shutil.rmtree(os.path.join("logs", tag), ignore_errors=True)
