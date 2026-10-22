"""Data splits: APTOS k-fold, Messidor-2 DEV/TEST, EyePACS frame."""

import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold


# ── APTOS ─────────────────────────────────────────────────────────────────

def aptos_frames(cfg):
    """Return (merged_df, splits) reproducing the legacy StratifiedKFold.

    merged = concat(train_1.csv, valid.csv) with ignore_index=True (3296 rows).
    splits = list of (train_idx, val_idx) arrays.
    """
    P = cfg['paths']['aptos']
    train_df = pd.read_csv(P['train_csv'])
    val_df = pd.read_csv(P['val_csv'])
    merged = pd.concat([train_df, val_df], ignore_index=True)
    skf = StratifiedKFold(
        n_splits=cfg['n_folds'], shuffle=True, random_state=cfg['kfold_seed'])
    splits = list(skf.split(merged, merged['diagnosis']))
    return merged, splits


def fold_frames(cfg, k):
    """Return (train_df, val_df) for APTOS fold k."""
    m, sp = aptos_frames(cfg)
    train_idx, val_idx = sp[k]
    return m.iloc[train_idx].reset_index(drop=True), \
           m.iloc[val_idx].reset_index(drop=True)


def test_frame(cfg):
    """Return the APTOS test dataframe (366 rows)."""
    return pd.read_csv(cfg['paths']['aptos']['test_csv'])


# ── Messidor-2 ────────────────────────────────────────────────────────────

def messidor_all(cfg):
    """Return all gradable Messidor-2 rows."""
    d = pd.read_csv(cfg['paths']['messidor']['csv'])
    return d[d['adjudicated_gradable'] == 1].reset_index(drop=True)


def _exam_groups(d):
    """Extract exam-id groups from id_code (first two '_'-separated tokens).

    ASSUMPTION: Messidor-2 id_code format is like '20051020_56234_0100_PP'
    where the exam id is the first two tokens joined by '_'.
    Verify against the CSV; ask the owner if the pattern fails.
    """
    return d['id_code'].astype(str).str.split('_').str[:2].str.join('_')


def make_messidor_split(cfg):
    """Create Messidor-2 DEV (~20%) / TEST (~80%) split.

    Uses StratifiedGroupKFold(5, shuffle=True, random_state=42);
    fold 0 test indices -> DEV, everything else -> TEST.
    Saves splits/messidor_dev_ids.json, splits/messidor_test_ids.json.
    """
    d = messidor_all(cfg)
    g = _exam_groups(d)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    # fold 0: train_idx, test_idx  —  test_idx is DEV
    _, dev_idx = next(iter(sgkf.split(d, d['diagnosis'], g)))
    dev = set(d.loc[dev_idx, 'id_code'].astype(str))
    all_ids = set(d['id_code'].astype(str))
    test = all_ids - dev

    os.makedirs('splits', exist_ok=True)
    with open('splits/messidor_dev_ids.json', 'w') as f:
        json.dump(sorted(dev), f)
    with open('splits/messidor_test_ids.json', 'w') as f:
        json.dump(sorted(test), f)

    return dev, test


def messidor_frame(cfg, split):
    """Return Messidor-2 dataframe for split='dev' or 'test'."""
    d = messidor_all(cfg)
    path = os.path.join('splits', f'messidor_{split}_ids.json')
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f'{path} not found; run --mode make_splits first')
    ids = set(json.load(open(path)))
    return d[d['id_code'].astype(str).isin(ids)].reset_index(drop=True)


# ── EyePACS ───────────────────────────────────────────────────────────────

def eyepacs_frame(cfg, exclude_ids=()):
    """Return EyePACS dataframe (id_code, diagnosis), excluding duplicates."""
    d = pd.read_csv(cfg['paths']['eyepacs']['labels_csv'])
    d = d.rename(columns={'image': 'id_code', 'level': 'diagnosis'})
    if exclude_ids:
        d = d[~d['id_code'].isin(exclude_ids)]
    return d.reset_index(drop=True)


# ── make_splits: the full Phase-0 split procedure ─────────────────────────

def make_splits(cfg):
    """Run all split procedures and write artifacts.

    Writes:
      - splits/aptos_folds.json  (ids per fold)
      - splits/messidor_dev_ids.json, splits/messidor_test_ids.json
      - results/phase0_protocol/splits_summary.csv
    Asserts:
      - test ids are disjoint from all fold ids
      - folds partition the merged pool
    """
    # ── APTOS folds ───────────────────────────────────────────────────
    merged, splits = aptos_frames(cfg)
    test_df = test_frame(cfg)

    # Save per-fold ids
    folds_json = {}
    all_fold_ids = set()
    for k, (tr, va) in enumerate(splits):
        fold_val_ids = merged.iloc[va]['id_code'].astype(str).tolist()
        fold_train_ids = merged.iloc[tr]['id_code'].astype(str).tolist()
        folds_json[f'fold{k}_val'] = fold_val_ids
        folds_json[f'fold{k}_train'] = fold_train_ids
        all_fold_ids |= set(fold_val_ids) | set(fold_train_ids)

    os.makedirs('splits', exist_ok=True)
    with open('splits/aptos_folds.json', 'w') as f:
        json.dump(folds_json, f, indent=2)

    # Assertions
    test_ids = set(test_df['id_code'].astype(str))
    assert test_ids.isdisjoint(all_fold_ids), \
        'APTOS test ids overlap with fold ids!'

    # Check folds partition the pool
    pool_ids = set(merged['id_code'].astype(str))
    assert all_fold_ids == pool_ids, \
        'Folds do not partition the merged pool!'

    # ── Messidor split ────────────────────────────────────────────────
    dev_ids, test_ids_m = make_messidor_split(cfg)

    # Verify DEV and TEST are disjoint by exam id
    d_all = messidor_all(cfg)
    g_all = _exam_groups(d_all)
    dev_exams = set(g_all[d_all['id_code'].astype(str).isin(dev_ids)])
    test_exams = set(g_all[d_all['id_code'].astype(str).isin(test_ids_m)])
    assert dev_exams.isdisjoint(test_exams), \
        'Messidor DEV and TEST share exam ids!'

    # ── Summary CSV ───────────────────────────────────────────────────
    rows = []
    # APTOS folds
    for k, (tr, va) in enumerate(splits):
        for cls in range(5):
            ct = int((merged.iloc[va]['diagnosis'] == cls).sum())
            rows.append({'split': f'aptos_fold{k}_val', 'class': cls, 'count': ct})
            ct = int((merged.iloc[tr]['diagnosis'] == cls).sum())
            rows.append({'split': f'aptos_fold{k}_train', 'class': cls, 'count': ct})
    # APTOS test
    for cls in range(5):
        ct = int((test_df['diagnosis'] == cls).sum())
        rows.append({'split': 'aptos_test', 'class': cls, 'count': ct})
    # Messidor
    d_dev = messidor_frame(cfg, 'dev')
    d_test = messidor_frame(cfg, 'test')
    for cls in range(5):
        rows.append({'split': 'messidor_dev', 'class': cls,
                     'count': int((d_dev['diagnosis'] == cls).sum())})
        rows.append({'split': 'messidor_test', 'class': cls,
                     'count': int((d_test['diagnosis'] == cls).sum())})

    out_dir = os.path.join('results', 'phase0_protocol')
    os.makedirs(out_dir, exist_ok=True)
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(out_dir, 'splits_summary.csv'), index=False)

    print(f'make_splits: APTOS pool={len(merged)}, test={len(test_df)}, '
          f'Messidor DEV={len(dev_ids)}, TEST={len(test_ids_m)}')
    print(f'  Saved: splits/aptos_folds.json, splits/messidor_*.json, '
          f'results/phase0_protocol/splits_summary.csv')
