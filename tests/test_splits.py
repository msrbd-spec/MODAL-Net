"""Test T1: Data splits reproduce StratifiedKFold(5, True, 42), disjoint test/dev sets, exam-id grouping."""

import os
import json
import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.model_selection import StratifiedKFold

from src.data.splits import (
    aptos_frames,
    fold_frames,
    test_frame as get_test_frame,
    make_messidor_split,
    messidor_frame,
    make_splits,
    _exam_groups,
)


@pytest.fixture
def synthetic_data_env(tmp_path, monkeypatch):
    """Create a temporary dataset environment matching Part 1 S4 schema."""
    aptos_dir = tmp_path / "APTOS_19"
    aptos_dir.mkdir()
    messidor_dir = tmp_path / "Messidor_2"
    messidor_dir.mkdir()
    eyepacs_dir = tmp_path / "EyePACS" / "trainLabels.csv"
    eyepacs_dir.mkdir(parents=True)

    # 1. APTOS train_1, valid, test
    rng = np.random.default_rng(42)
    # 200 train images
    train_ids = [f"train_{i:04d}" for i in range(200)]
    train_diag = rng.integers(0, 5, size=200)
    train_df = pd.DataFrame({"id_code": train_ids, "diagnosis": train_diag})
    train_df.to_csv(aptos_dir / "train_1.csv", index=False)

    # 50 valid images
    val_ids = [f"valid_{i:04d}" for i in range(50)]
    val_diag = rng.integers(0, 5, size=50)
    val_df = pd.DataFrame({"id_code": val_ids, "diagnosis": val_diag})
    val_df.to_csv(aptos_dir / "valid.csv", index=False)

    # 40 test images (completely disjoint ids)
    test_ids = [f"test_{i:04d}" for i in range(40)]
    test_diag = rng.integers(0, 5, size=40)
    test_df = pd.DataFrame({"id_code": test_ids, "diagnosis": test_diag})
    test_df.to_csv(aptos_dir / "test.csv", index=False)

    # 2. Messidor-2 data with exam grouping (e.g. 2 eyes per exam)
    mess_rows = []
    for exam_idx in range(60):
        exam_prefix = f"20051020_{exam_idx:05d}"
        for eye in ["0100_PP", "0200_PP"]:
            img_id = f"{exam_prefix}_{eye}"
            diag = int(rng.integers(0, 5))
            mess_rows.append({
                "id_code": img_id,
                "diagnosis": diag,
                "adjudicated_dme": 0,
                "adjudicated_gradable": 1,
            })
    mess_df = pd.DataFrame(mess_rows)
    mess_df.to_csv(messidor_dir / "messidor_data.csv", index=False)

    # 3. EyePACS labels
    eyepacs_rows = pd.DataFrame({
        "image": [f"eye_{i:04d}" for i in range(50)],
        "level": rng.integers(0, 5, size=50),
    })
    eyepacs_rows.to_csv(eyepacs_dir / "trainLabels.csv", index=False)

    # Synthetic config
    cfg = yaml.safe_load(open("configs/config.yaml"))
    cfg["paths"]["aptos"]["train_csv"] = str(aptos_dir / "train_1.csv")
    cfg["paths"]["aptos"]["val_csv"] = str(aptos_dir / "valid.csv")
    cfg["paths"]["aptos"]["test_csv"] = str(aptos_dir / "test.csv")
    cfg["paths"]["messidor"]["csv"] = str(messidor_dir / "messidor_data.csv")
    cfg["paths"]["eyepacs"]["labels_csv"] = str(eyepacs_dir / "trainLabels.csv")

    # make_splits() writes to hardcoded relative paths ("splits/...",
    # "results/phase0_protocol/..."). Run the test from tmp_path so the
    # synthetic artifacts never overwrite the real ones in the repo.
    monkeypatch.chdir(tmp_path)

    return cfg


def test_t1_splits_reproducibility_and_isolation(synthetic_data_env):
    """T1: Verify StratifiedKFold reproduction, disjoint test ids, and exam grouping."""
    cfg = synthetic_data_env

    # 1. Verify folds reproduce StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    merged, splits = aptos_frames(cfg)
    assert len(merged) == 250  # 200 train + 50 val

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    expected_splits = list(skf.split(merged, merged['diagnosis']))

    for k in range(5):
        tr_act, va_act = splits[k]
        tr_exp, va_exp = expected_splits[k]
        np.testing.assert_array_equal(tr_act, tr_exp)
        np.testing.assert_array_equal(va_act, va_exp)

    # 2. Check fold_frames returns correct partitions
    all_val_indices = []
    for k in range(5):
        train_f, val_f = fold_frames(cfg, k)
        assert len(train_f) + len(val_f) == len(merged)
        all_val_indices.extend(splits[k][1])
    assert sorted(all_val_indices) == list(range(len(merged)))

    # 3. Test make_splits procedure
    make_splits(cfg)

    # Check splits/aptos_folds.json
    assert os.path.isfile("splits/aptos_folds.json")
    with open("splits/aptos_folds.json") as f:
        aptos_folds_data = json.load(f)

    test_df = get_test_frame(cfg)
    test_id_set = set(test_df['id_code'].astype(str))

    for k in range(5):
        val_ids = set(aptos_folds_data[f"fold{k}_val"])
        train_ids = set(aptos_folds_data[f"fold{k}_train"])
        assert val_ids.isdisjoint(train_ids)
        assert val_ids.isdisjoint(test_id_set)
        assert train_ids.isdisjoint(test_id_set)

    # 4. Check Messidor dev/test disjoint by exam id
    assert os.path.isfile("splits/messidor_dev_ids.json")
    assert os.path.isfile("splits/messidor_test_ids.json")
    dev_ids = set(json.load(open("splits/messidor_dev_ids.json")))
    test_ids_m = set(json.load(open("splits/messidor_test_ids.json")))

    assert dev_ids.isdisjoint(test_ids_m)

    # Check exam grouping: no exam shared between dev and test
    dev_df = messidor_frame(cfg, 'dev')
    test_df_m = messidor_frame(cfg, 'test')
    dev_exams = set(_exam_groups(dev_df))
    test_exams = set(_exam_groups(test_df_m))
    assert dev_exams.isdisjoint(test_exams)

    # 5. Check counts printed to splits_summary.csv
    summary_path = "results/phase0_protocol/splits_summary.csv"
    assert os.path.isfile(summary_path)
    summary_df = pd.read_csv(summary_path)
    assert not summary_df.empty
    assert "split" in summary_df.columns
    assert "class" in summary_df.columns
    assert "count" in summary_df.columns
    assert summary_df["count"].sum() > 0
