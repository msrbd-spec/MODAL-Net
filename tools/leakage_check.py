"""Leakage check: does any evaluation image duplicate an image a given run actually trained on?

For every tag, the training images are reconstructed exactly as the trainer draws them (APTOS CV pool;
the EyePACS sample of joint_eyepacs runs = eyepacs_frame().sample(n=eyepacs_max, random_state=seed);
the sampled extra_sources). Every evaluation set (APTOS test and the frozen external dev/test splits)
is then checked against the duplicate audit (pHash Hamming <= dedup_exclude_max).

make_external_splits already removes external evaluation images that duplicate ANY training-pool
image, so the external columns should read 0; the APTOS test set is never filtered, so its count is
the number to state in the paper.

Usage:
  python tools/leakage_check.py --tags p2b_none,p3b_joint
Output: results/master/leakage_check.csv (+ leakage_pairs.csv with the matching pairs)
"""

import argparse
import os
import sys

import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.external import AUDIT_LABELS, eval_frame, eval_set_names, load_audit_pairs, source_frame, _stem
from src.data.splits import aptos_frames, eyepacs_frame


def training_images(cfg):
    merged, _ = aptos_frames(cfg)
    imgs = {'APTOS': {_stem(i) for i in merged['id_code']}}
    if cfg.get('joint_eyepacs'):
        eye = eyepacs_frame(cfg)
        cap = int(cfg.get('eyepacs_max', 20000))
        if len(eye) > cap:
            eye = eye.sample(n=cap, random_state=int(cfg.get('seed', 0)))
        imgs['EyePACS'] = {_stem(i) for i in eye['id_code']}
    for name in (cfg.get('extra_sources') or []):
        fr, _ = source_frame(cfg, name)
        cap = int(cfg.get('source_max', 20000))
        if len(fr) > cap:
            fr = fr.sample(n=cap, random_state=int(cfg.get('seed', 0)))
        imgs[AUDIT_LABELS[name]] = {_stem(i) for i in fr['id_code']}
    return imgs


def check(tag, max_dist, pairs):
    defaults = yaml.safe_load(open('configs/config.yaml'))
    snap = yaml.safe_load(open(os.path.join('checkpoints', tag, 'config_snapshot.yaml'))) or {}
    cfg = {**defaults, **snap}
    train = training_images(cfg)
    rows, hits = [], []
    pairs = pairs[pairs['hamming_distance'] <= max_dist]
    for set_name in eval_set_names(cfg):
        fr = eval_frame(cfg, set_name)[0]
        ds = 'aptos' if set_name == 'aptos_test' else set_name.rsplit('_', 1)[0]
        label = AUDIT_LABELS[ds]
        ids = {_stem(i) for i in fr['id_code']}
        found = set()
        for side, other in (('a', 'b'), ('b', 'a')):
            sub = pairs[pairs[f'dataset_{side}'] == label]
            for _, r in sub.iterrows():
                sid, o_ds, o_id = _stem(r[f'id_{side}']), r[f'dataset_{other}'], _stem(r[f'id_{other}'])
                if sid in ids and o_id in train.get(o_ds, set()) and not (o_ds == label and o_id == sid):
                    found.add(sid)
                    hits.append({'tag': tag, 'set': set_name, 'eval_id': sid, 'train_dataset': o_ds,
                                 'train_id': o_id, 'hamming_distance': r['hamming_distance']})
        rows.append({'tag': tag, 'set': set_name, 'n_images': len(ids), 'n_duplicated_in_training': len(found),
                     'share_pct': round(100.0 * len(found) / max(1, len(ids)), 2),
                     'training_sources_checked': ','.join(sorted(train))})
    return rows, hits


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tags', required=True)
    ap.add_argument('--max-dist', type=int, default=None,
                    help='default: the dedup_exclude_max recorded in splits/external/manifest.json')
    a = ap.parse_args()
    manifest = os.path.join('splits', 'external', 'manifest.json')
    if a.max_dist is not None:
        max_dist = a.max_dist
    elif os.path.isfile(manifest):
        import json
        max_dist = int(json.load(open(manifest))['dedup_exclude_max'])   # the threshold the frozen splits used
    else:
        max_dist = int(yaml.safe_load(open('configs/config.yaml'))['dedup_exclude_max'])
    print(f'leakage_check: pHash distance threshold {max_dist}')
    pairs = load_audit_pairs()
    if pairs.empty:
        sys.exit('no duplicate audit found: run --mode dedup_audit (and --mode dedup_external) first')
    rows, hits = [], []
    for t in [x.strip() for x in a.tags.split(',') if x.strip()]:
        r, h = check(t, max_dist, pairs)
        rows += r
        hits += h
    os.makedirs(os.path.join('results', 'master'), exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join('results', 'master', 'leakage_check.csv'), index=False)
    pd.DataFrame(hits).to_csv(os.path.join('results', 'master', 'leakage_pairs.csv'), index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print('written: results/master/leakage_check.csv, results/master/leakage_pairs.csv')


if __name__ == '__main__':
    main()
