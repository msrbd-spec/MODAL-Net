"""Evaluation datasets and extra training sources: Messidor-2, DDR, DeepDRiD, IDRiD.

Every evaluation set has a frozen split file written by `--mode make_external_splits`:

    splits/external/<dataset>_<split>.csv      columns: id_code, diagnosis, path, group
    splits/external/exclusions.csv             evaluation images removed as duplicates of training images
    splits/external/manifest.json              counts per split and the rules that produced them

Splits (official splits wherever the dataset ships them):
    messidor : dev / test        = the exam-grouped DEV (~20 %) / TEST (~80 %) of splits/messidor_*_ids.json
                                    (never filtered: every finished run was reported on exactly these images)
    ddr      : train / dev / test = official DR_grading train / valid / test, grade 5 (ungradable) removed
    deepdrid : train / dev / test = official regular-fundus training / validation / Challenge-1 evaluation set,
                                    one label per IMAGE from the eye-level column (left_eye_DR_Level or right_eye_DR_Level)
    idrid    : dev / test        = official Disease-Grading training set (413) / testing set (103)

The `train` split exists only for datasets that can be added as a training source (extra_sources).
The APTOS test set (366 images) and Messidor-2 DEV / TEST stay exactly as before and are never filtered;
their overlap with each run's training images is reported by tools/leakage_check.py instead.
Duplicate exclusion applies to DDR, DeepDRiD and IDRiD (dev and test), which no finished run was scored on.
"""

import glob
import json
import os
import re
import zipfile
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from src.utils.config import EXTERNAL_DATASETS

SPLIT_DIR = os.path.join('splits', 'external')
AUDIT_DIR = os.path.join('results', 'phase0_protocol', 'audit')

# Domain ids carried by every sample (DRDataset.domain); used by multi-source sampling and coral_src offsets.
DOMAIN_IDS = {'aptos': 0, 'eyepacs': 1, 'messidor': 2, 'ddr': 3, 'deepdrid': 4, 'idrid': 5}

# Which splits each external dataset provides.
DATASET_SPLITS = {
    'messidor': ('dev', 'test'),
    'ddr': ('train', 'dev', 'test'),
    'deepdrid': ('train', 'dev', 'test'),
    'idrid': ('dev', 'test'),
}

# Dataset labels used inside the duplicate-audit CSVs.
AUDIT_LABELS = {'aptos': 'APTOS', 'eyepacs': 'EyePACS', 'messidor': 'Messidor',
                'ddr': 'DDR', 'deepdrid': 'DeepDRiD', 'idrid': 'IDRiD'}

# Evaluation sets that are never filtered (protocol frozen before the external datasets were added).
UNFILTERED = ('messidor',)

DISPLAY_NAMES = {'aptos_test': 'APTOS test', 'oof': 'APTOS OOF',
                 'messidor_dev': 'Messidor-2 DEV', 'messidor_test': 'Messidor-2 TEST',
                 'ddr_dev': 'DDR dev', 'ddr_test': 'DDR test',
                 'deepdrid_dev': 'DeepDRiD dev', 'deepdrid_test': 'DeepDRiD test',
                 'idrid_dev': 'IDRiD dev', 'idrid_test': 'IDRiD test'}


# ---------------------------------------------------------------------------
# Minimal .xlsx reader (the cluster has no openpyxl; DeepDRiD ships its test labels as .xlsx)
# ---------------------------------------------------------------------------

_NS = {'m': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}


def _col_index(cell_ref):
    letters = re.match(r'([A-Z]+)', cell_ref).group(1)
    idx = 0
    for ch in letters:
        idx = idx * 26 + (ord(ch) - ord('A') + 1)
    return idx - 1


def read_xlsx_first_sheet(path):
    """Read the first worksheet of an .xlsx file into a DataFrame (first row = header).

    Supports shared strings, inline strings and numeric cells, which is what DeepDRiD uses.
    """
    with zipfile.ZipFile(path) as z:
        shared = []
        if 'xl/sharedStrings.xml' in z.namelist():
            root = ET.fromstring(z.read('xl/sharedStrings.xml'))
            for si in root.findall('m:si', _NS):
                shared.append(''.join(t.text or '' for t in si.iter('{%s}t' % _NS['m'])))
        sheets = sorted(n for n in z.namelist() if re.match(r'xl/worksheets/sheet\d+\.xml$', n))
        if not sheets:
            raise ValueError(f'{path}: no worksheet found')
        root = ET.fromstring(z.read(sheets[0]))
    rows = []
    for r in root.iter('{%s}row' % _NS['m']):
        vals = {}
        for c in r.findall('m:c', _NS):
            ref, typ = c.get('r'), c.get('t')
            v = c.find('m:v', _NS)
            if typ == 's' and v is not None:
                val = shared[int(v.text)]
            elif typ == 'inlineStr':
                val = ''.join(t.text or '' for t in c.iter('{%s}t' % _NS['m']))
            elif v is not None:
                txt = v.text
                try:
                    f = float(txt)
                    val = int(f) if f.is_integer() else f
                except (TypeError, ValueError):
                    val = txt
            else:
                val = None
            vals[_col_index(ref)] = val
        if vals:
            width = max(vals) + 1
            rows.append([vals.get(i) for i in range(width)])
    if not rows:
        return pd.DataFrame()
    header = [str(h).strip() for h in rows[0]]
    body = [row + [None] * (len(header) - len(row)) for row in rows[1:]]
    return pd.DataFrame([row[:len(header)] for row in body], columns=header)


# ---------------------------------------------------------------------------
# Raw frames, built from the files each dataset ships
# ---------------------------------------------------------------------------

def _frame(ids, labels, paths, groups):
    df = pd.DataFrame({'id_code': [str(i) for i in ids], 'diagnosis': [int(v) for v in labels],
                       'path': [str(p) for p in paths], 'group': [str(g) for g in groups]})
    assert df['id_code'].is_unique, 'duplicate id_code inside one split'
    assert df['diagnosis'].between(0, 4).all(), 'labels outside 0..4'
    return df.reset_index(drop=True)


def _messidor_raw(cfg, split):
    from src.data.splits import messidor_frame, _exam_groups
    d = messidor_frame(cfg, split)
    return _frame(d['id_code'], d['diagnosis'], [''] * len(d), _exam_groups(d))


def _ddr_raw(cfg, split):
    root = cfg['paths']['ddr']['root']
    official = {'train': 'train', 'dev': 'valid', 'test': 'test'}[split]
    txt = os.path.join(root, f'{official}.txt')
    if not os.path.isfile(txt):
        raise FileNotFoundError(f'DDR label file not found: {txt}')
    ids, labels, paths = [], [], []
    for line in open(txt):
        parts = line.strip().split()
        if len(parts) != 2:
            continue
        fname, grade = parts[0], int(parts[1])
        if grade == 5:          # ungradable
            continue
        ids.append(os.path.splitext(fname)[0])
        labels.append(grade)
        paths.append(os.path.join(root, official, fname))
    return _frame(ids, labels, paths, ids)


def _deepdrid_raw(cfg, split):
    P = cfg['paths']['deepdrid']
    root = P['root']
    if split in ('train', 'dev'):
        sub = P['train_dir'] if split == 'train' else P['val_dir']
        csv = os.path.join(root, sub, P['train_csv'] if split == 'train' else P['val_csv'])
        if not os.path.isfile(csv):
            raise FileNotFoundError(f'DeepDRiD csv not found: {csv}')
        d = pd.read_csv(csv)
        img = d['image_id'].astype(str)
        left, right = d['left_eye_DR_Level'], d['right_eye_DR_Level']
        # Each row carries exactly one eye-level label. The file name normally says which eye
        # (_l / _r), but a few rows (e.g. 56_l3) have the label in the other column, so the
        # non-empty column is used and rows with two different labels are rejected.
        both = left.notna() & right.notna() & (left != right)
        if both.any():
            raise ValueError(f'{csv}: rows with two different eye-level labels: {img[both].tolist()[:5]}')
        lab = left.where(left.notna(), right)
        if lab.isna().any():
            raise ValueError(f'{csv}: images without an eye-level DR label: {img[lab.isna()].tolist()[:5]}')
        lab = lab.to_numpy()
        pats = d['patient_id'].astype(str)
        paths = [os.path.join(root, sub, 'Images', p, f'{i}.jpg') for p, i in zip(pats, img)]
        return _frame(img, lab.astype(int), paths, pats)
    sub = P['test_dir']
    xl = os.path.join(root, sub, P['test_labels'])
    if not os.path.isfile(xl):
        raise FileNotFoundError(f'DeepDRiD test labels not found: {xl}')
    d = read_xlsx_first_sheet(xl)
    lab_col = [c for c in d.columns if c.lower().startswith('dr_level')][0]
    d = d.dropna(subset=['image_id', lab_col])
    img = d['image_id'].astype(str)
    pats = img.str.split('_').str[0]
    paths = [os.path.join(root, sub, 'Images', p, f'{i}.jpg') for p, i in zip(pats, img)]
    return _frame(img, d[lab_col].astype(int), paths, pats)


def _idrid_raw(cfg, split):
    P = cfg['paths']['idrid']
    root = P['root']
    csv = os.path.join(root, P['train_csv'] if split == 'dev' else P['test_csv'])
    img_dir = os.path.join(root, P['train_images'] if split == 'dev' else P['test_images'])
    if not os.path.isfile(csv):
        raise FileNotFoundError(f'IDRiD label file not found: {csv}')
    d = pd.read_csv(csv)
    d = d.loc[:, ~d.columns.str.startswith('Unnamed')]
    name_col = [c for c in d.columns if c.strip().lower() == 'image name'][0]
    lab_col = [c for c in d.columns if c.strip().lower() == 'retinopathy grade'][0]
    d = d.dropna(subset=[name_col])
    prefix = 'tr_' if split == 'dev' else 'te_'
    names = d[name_col].astype(str).str.strip()
    # train and test folders reuse the same file names, so the id carries the split prefix
    return _frame(prefix + names, d[lab_col], [os.path.join(img_dir, f'{n}.jpg') for n in names], prefix + names)


_RAW = {'messidor': _messidor_raw, 'ddr': _ddr_raw, 'deepdrid': _deepdrid_raw, 'idrid': _idrid_raw}


def raw_frame(cfg, name, split):
    """Frame built directly from the dataset's own files (no duplicate exclusion)."""
    if split not in DATASET_SPLITS[name]:
        raise ValueError(f'{name} has no {split!r} split (available: {DATASET_SPLITS[name]})')
    return _RAW[name](cfg, split)


# ---------------------------------------------------------------------------
# Duplicate-based exclusion of evaluation images
# ---------------------------------------------------------------------------

def _stem(x):
    return os.path.splitext(os.path.basename(str(x)))[0]


def training_pool_ids(cfg, available):
    """Audit ids of every image that ANY run could train on: APTOS pool, labelled EyePACS, DDR/DeepDRiD train."""
    from src.data.splits import aptos_frames, eyepacs_frame
    pool = {}
    try:
        merged, _ = aptos_frames(cfg)
        pool['APTOS'] = {_stem(i) for i in merged['id_code']}
    except FileNotFoundError:
        pool['APTOS'] = set()
    try:
        pool['EyePACS'] = {_stem(i) for i in eyepacs_frame(cfg)['id_code']}
    except FileNotFoundError:
        pool['EyePACS'] = set()
    for name in ('ddr', 'deepdrid'):
        if name in available:
            pool[AUDIT_LABELS[name]] = set(raw_frame(cfg, name, 'train')['id_code'])
    return pool


def load_audit_pairs():
    """All candidate duplicate pairs from the original and the external audit (empty frame if none ran)."""
    frames = []
    for fname in ('duplicates.csv', 'duplicates_external.csv'):
        p = os.path.join(AUDIT_DIR, fname)
        if os.path.isfile(p):
            frames.append(pd.read_csv(p, dtype=str))
    if not frames:
        return pd.DataFrame(columns=['dataset_a', 'id_a', 'dataset_b', 'id_b', 'hamming_distance'])
    d = pd.concat(frames, ignore_index=True)
    d['hamming_distance'] = pd.to_numeric(d['hamming_distance'], errors='coerce')
    return d


def find_exclusions(cfg, name, split, frame, pairs, pool, max_dist):
    """Rows of `frame` (an evaluation split of dataset `name`) that duplicate a training-pool image."""
    label = AUDIT_LABELS[name]
    ids = {_stem(i): i for i in frame['id_code']}
    out = []
    pairs = pairs[pairs['hamming_distance'] <= max_dist]
    for side, other in (('a', 'b'), ('b', 'a')):
        sub = pairs[pairs[f'dataset_{side}'] == label]
        for _, r in sub.iterrows():
            sid = _stem(r[f'id_{side}'])
            if sid not in ids:
                continue
            o_ds, o_id = r[f'dataset_{other}'], _stem(r[f'id_{other}'])
            if o_id in pool.get(o_ds, set()) and not (o_ds == label and o_id == sid):
                out.append({'dataset': name, 'split': split, 'id_code': ids[sid], 'matched_dataset': o_ds,
                            'matched_id': o_id, 'hamming_distance': int(r['hamming_distance'])})
    return pd.DataFrame(out, columns=['dataset', 'split', 'id_code', 'matched_dataset', 'matched_id',
                                      'hamming_distance'])


def available_datasets(cfg):
    """External datasets whose label files exist on this machine."""
    out = []
    for name in EXTERNAL_DATASETS:
        try:
            raw_frame(cfg, name, DATASET_SPLITS[name][-1])
            out.append(name)
        except (FileNotFoundError, KeyError):
            continue
    return out


def make_external_splits(cfg):
    """Freeze every evaluation / extra-source split to splits/external/*.csv (see module docstring).

    Raises if a dataset listed in cfg['external_sets'] or cfg['extra_sources'] is missing.
    Evaluation images that duplicate a training-pool image (pHash Hamming <= dedup_exclude_max in the
    audit CSVs) are removed and listed in splits/external/exclusions.csv.
    """
    wanted = set(cfg.get('external_sets') or []) | {s for s in (cfg.get('extra_sources') or []) if s != 'eyepacs'}
    available = available_datasets(cfg)
    missing = sorted(wanted - set(available))
    if missing:
        raise FileNotFoundError(
            f'make_external_splits: datasets listed in external_sets/extra_sources are missing on disk: {missing}. '
            f'Check cfg paths (configs/config.yaml -> paths) or remove them from external_sets.')
    max_dist = int(cfg.get('dedup_exclude_max', 4))
    pairs = load_audit_pairs()
    pool = training_pool_ids(cfg, available)
    os.makedirs(SPLIT_DIR, exist_ok=True)
    manifest = {'dedup_exclude_max': max_dist,
                'audit_files': [f for f in ('duplicates.csv', 'duplicates_external.csv')
                                if os.path.isfile(os.path.join(AUDIT_DIR, f))],
                'splits': {}}
    excl_all = []
    for name in available:
        for split in DATASET_SPLITS[name]:
            fr = raw_frame(cfg, name, split)
            n_raw = len(fr)
            if split != 'train' and name not in UNFILTERED:
                ex = find_exclusions(cfg, name, split, fr, pairs, pool, max_dist)
                if len(ex):
                    excl_all.append(ex)
                    fr = fr[~fr['id_code'].isin(set(ex['id_code']))].reset_index(drop=True)
            missing_files = [p for p in fr['path'] if p and not os.path.isfile(p)]
            if missing_files:
                raise FileNotFoundError(f'{name}/{split}: {len(missing_files)} image files missing, '
                                        f'first: {missing_files[0]}')
            fr.to_csv(os.path.join(SPLIT_DIR, f'{name}_{split}.csv'), index=False)
            counts = fr['diagnosis'].value_counts().reindex(range(5), fill_value=0).astype(int).tolist()
            manifest['splits'][f'{name}_{split}'] = {'n_raw': n_raw, 'n': len(fr), 'class_counts': counts}
            share = (n_raw - len(fr)) / max(1, n_raw)
            msg = f'{name}_{split}: {len(fr)} images (removed as duplicates: {n_raw - len(fr)})'
            if share > 0.02:
                msg += '  <-- more than 2 % removed: review the pairs with tools/dup_review.py'
            print('make_external_splits:', msg)
    ex_df = pd.concat(excl_all, ignore_index=True) if excl_all else pd.DataFrame(
        columns=['dataset', 'split', 'id_code', 'matched_dataset', 'matched_id', 'hamming_distance'])
    ex_df.to_csv(os.path.join(SPLIT_DIR, 'exclusions.csv'), index=False)
    if not manifest['audit_files']:
        manifest['warning'] = 'no duplicate audit found: nothing was excluded (run dedup_audit / dedup_external first)'
        print('make_external_splits: WARNING no duplicate audit CSV found, no exclusions applied')
    with open(os.path.join(SPLIT_DIR, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)
    _append_split_summary(manifest)
    return manifest


def _append_split_summary(manifest):
    """Add the external split counts to results/phase0_protocol/splits_summary.csv (rows replaced, not duplicated)."""
    out = os.path.join('results', 'phase0_protocol', 'splits_summary.csv')
    # Messidor rows written by make_splits are kept; only the new datasets are added / refreshed.
    rows = [{'split': k, 'class': c, 'count': n}
            for k, v in manifest['splits'].items() if not k.startswith(UNFILTERED)
            for c, n in enumerate(v['class_counts'])]
    new = pd.DataFrame(rows)
    if os.path.isfile(out):
        old = pd.read_csv(out)
        old = old[~old['split'].isin(new['split'].unique())]
        new = pd.concat([old, new], ignore_index=True)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    new.to_csv(out, index=False)


# ---------------------------------------------------------------------------
# Frozen frames used by training and evaluation
# ---------------------------------------------------------------------------

def external_frame(cfg, name, split):
    """Frozen split of an external dataset (raises with instructions when make_external_splits has not run)."""
    p = os.path.join(SPLIT_DIR, f'{name}_{split}.csv')
    if not os.path.isfile(p):
        raise FileNotFoundError(
            f'{p} not found. Run: python main.py --mode make_external_splits  (after the duplicate audit).')
    d = pd.read_csv(p, dtype={'id_code': str, 'path': str, 'group': str}, keep_default_na=False)
    d['diagnosis'] = d['diagnosis'].astype(int)
    return d


def eval_set_names(cfg):
    """Ordered evaluation sets of an experiment: APTOS test, then dev/test of every external dataset."""
    names = ['aptos_test']
    for d in (cfg.get('external_sets') or []):
        for s in ('dev', 'test'):
            names.append(f'{d}_{s}')
    return names


def split_set_name(set_name):
    """'ddr_test' -> ('ddr', 'test'); 'aptos_test' -> ('aptos', 'test')."""
    ds, split = set_name.rsplit('_', 1)
    return ds, split


def eval_frame(cfg, set_name):
    """(frame, image_dirs, domain_id) for an evaluation set name."""
    from src.data.splits import test_frame
    if set_name == 'aptos_test':
        d = test_frame(cfg)
        return d, [cfg['paths']['aptos']['test_images']], DOMAIN_IDS['aptos']
    ds, split = split_set_name(set_name)
    d = external_frame(cfg, ds, split)
    dirs = [cfg['paths']['messidor']['images']] if ds == 'messidor' else []
    return d, dirs, DOMAIN_IDS[ds]


def source_frame(cfg, name):
    """Training frame of an extra source: EyePACS labelled images or the frozen DDR / DeepDRiD train split."""
    if name == 'eyepacs':
        from src.data.splits import eyepacs_frame
        return eyepacs_frame(cfg), [cfg['paths']['eyepacs']['images']]
    return external_frame(cfg, name, 'train'), []


def check_eval_inputs(cfg):
    """Fail fast (before any training) when an evaluation or source split is missing."""
    problems = []
    for s in eval_set_names(cfg):
        try:
            d, dirs, _ = eval_frame(cfg, s)
            if len(d) == 0:
                problems.append(f'{s}: empty split')
                continue
            first = d.iloc[0]
            path = str(first.get('path', '') or '')
            if path:
                if not os.path.isfile(path):
                    problems.append(f'{s}: image missing {path}')
            elif not any(glob.glob(os.path.join(dd, str(first['id_code']) + '*')) for dd in dirs):
                problems.append(f'{s}: image {first["id_code"]} not found in {dirs}')
        except FileNotFoundError as e:
            problems.append(f'{s}: {e}')
    for s in (cfg.get('extra_sources') or []):
        if s == 'eyepacs':
            continue
        try:
            external_frame(cfg, s, 'train')
        except FileNotFoundError as e:
            problems.append(f'{s}_train: {e}')
    if problems:
        raise FileNotFoundError('Evaluation inputs missing:\n  ' + '\n  '.join(problems))
