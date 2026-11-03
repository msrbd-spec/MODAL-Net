"""Visual review of the duplicate pairs that removed evaluation images (contact sheet).

Reads splits/external/exclusions.csv (written by --mode make_external_splits) and draws each excluded
evaluation image next to the training-pool image it matched, with the pHash distance, so a human can
confirm that the pairs are true duplicates before the paper states the counts. If many pairs are
different images, lower dedup_exclude_max and re-run make_external_splits.

Usage:
  python tools/dup_review.py                 # first 40 pairs
  python tools/dup_review.py --max-pairs 120
Output: results/phase0_protocol/audit/dup_review_<k>.png (20 pairs per sheet)
"""

import argparse
import glob
import os
import sys

import cv2
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.external import AUDIT_LABELS, DATASET_SPLITS, available_datasets, raw_frame, _stem


def image_index(cfg):
    P = cfg['paths']
    idx = {}
    for label, dirs in (('APTOS', [P['aptos']['train_images'], P['aptos']['val_images'], P['aptos']['test_images']]),
                        ('Messidor', [P['messidor']['images']]), ('EyePACS', [P['eyepacs']['images']])):
        for d in dirs:
            for p in glob.glob(os.path.join(d, '*')):
                idx[(label, os.path.splitext(os.path.basename(p))[0])] = p
    for name in available_datasets(cfg):
        if name == 'messidor':
            continue
        for split in DATASET_SPLITS[name]:
            fr = raw_frame(cfg, name, split)
            for i, p in zip(fr['id_code'], fr['path']):
                idx[(AUDIT_LABELS[name], str(i))] = p
    return idx


def _thumb(path, size=220):
    im = cv2.imread(path) if path else None
    if im is None:
        im = np.full((size, size, 3), 255, np.uint8)
        cv2.putText(im, 'missing', (40, size // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return im
    h, w = im.shape[:2]
    s = size / max(h, w)
    im = cv2.resize(im, (int(w * s), int(h * s)))
    canvas = np.full((size, size, 3), 255, np.uint8)
    y0, x0 = (size - im.shape[0]) // 2, (size - im.shape[1]) // 2
    canvas[y0:y0 + im.shape[0], x0:x0 + im.shape[1]] = im
    return canvas


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--max-pairs', type=int, default=40)
    a = ap.parse_args()
    ex_path = os.path.join('splits', 'external', 'exclusions.csv')
    if not os.path.isfile(ex_path):
        sys.exit(f'{ex_path} not found: run --mode make_external_splits first')
    ex = pd.read_csv(ex_path, dtype=str)
    if ex.empty:
        print('no excluded evaluation images: nothing to review')
        return
    cfg = yaml.safe_load(open('configs/config.yaml'))
    idx = image_index(cfg)
    ex = ex.sort_values('hamming_distance').head(a.max_pairs)
    out_dir = os.path.join('results', 'phase0_protocol', 'audit')
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for _, r in ex.iterrows():
        left = _thumb(idx.get((AUDIT_LABELS[r['dataset']], r['id_code'])) or
                      idx.get((AUDIT_LABELS[r['dataset']], _stem(r['id_code']))))
        right = _thumb(idx.get((r['matched_dataset'], r['matched_id'])) or
                       idx.get((r['matched_dataset'], _stem(r['matched_id']))))
        pair = np.hstack([left, np.full((220, 10, 3), 255, np.uint8), right])
        bar = np.full((30, pair.shape[1], 3), 255, np.uint8)
        txt = f"{r['dataset']}/{r['split']} {r['id_code']}  vs  {r['matched_dataset']} {r['matched_id']}  d={r['hamming_distance']}"
        cv2.putText(bar, txt[:70], (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1)
        rows.append(np.vstack([bar, pair]))
    for k in range(0, len(rows), 20):
        chunk = rows[k:k + 20]
        while len(chunk) % 2:
            chunk.append(np.full_like(rows[0], 255))
        grid = np.vstack([np.hstack(chunk[i:i + 2]) for i in range(0, len(chunk), 2)])
        out = os.path.join(out_dir, f'dup_review_{k // 20}.png')
        cv2.imwrite(out, grid)
        print('written:', out)


if __name__ == '__main__':
    main()
