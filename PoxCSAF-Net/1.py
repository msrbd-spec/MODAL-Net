#!/usr/bin/env python3
"""
check_focal_alpha_sources.py  (finding 01, CPU only, no training, runs in seconds)

Verifies, straight from your dataset folders, the numbers quoted in the finding-01 edits:
  1. per-class test counts of every evaluated fold, compared with the fixed counts (42, 94, 35, 116)
  2. focal alpha computed from raw (pre-augmentation) training counts of each fold
  3. focal alpha computed from post-augmentation training counts, as in the focal_train_dist ablation
  4. distances between these vectors and the fixed vector, and between uniform and the fixed vector

Run from the folder that contains datasets/.  Output: focal_alpha_check.json and a printed table.
"""
import os, json
import numpy as np

DATA_ROOT = 'datasets/final_5_fold_pruned/'
RUN       = 'FL_Run1_Uniform'          # Valid/Test are identical across Run1, Run2, Run3
FOLDS     = [f'Fold_{i}' for i in range(1, 6)]
CLASSES   = ['Chickenpox', 'Healthy', 'Measles', 'Monkeypox']
NUM_CLIENTS = 5
FIXED_COUNTS = np.array([42, 94, 35, 116], dtype=float)

IMG_EXT_ALL      = ('.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tif', '.tiff')
IMG_EXT_ABLATION = ('.png', '.jpg', '.jpeg', '.bmp')   # exactly what build_train_strategy_variants() counted


def alpha_from_counts(counts):
    inv = 1.0 / (np.asarray(counts, dtype=float) + 1e-6)
    return inv / inv.sum()


def count_files(fold, split, cls, mode):
    """mode: 'all_ablation_ext' | 'raw' | 'all'"""
    n = 0
    for c in range(1, NUM_CLIENTS + 1):
        d = os.path.join(DATA_ROOT, fold, RUN, f'Client_{c}', split, cls)
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            low = f.lower()
            if mode == 'all_ablation_ext':
                n += low.endswith(IMG_EXT_ABLATION)
            elif mode == 'raw':
                n += (low.endswith(IMG_EXT_ALL) and not f.startswith('aug_'))
            else:
                n += low.endswith(IMG_EXT_ALL)
    return int(n)


fixed_alpha = alpha_from_counts(FIXED_COUNTS)
uniform     = np.ones(4) / 4
report = {'fixed_counts': FIXED_COUNTS.tolist(), 'fixed_alpha': fixed_alpha.tolist(), 'folds': {}}

print(f'fixed counts {FIXED_COUNTS.astype(int).tolist()}  ->  alpha {np.round(fixed_alpha, 3).tolist()}')
print(f'uniform vs fixed: max|diff|={np.abs(uniform - fixed_alpha).max():.3f}  L1={np.abs(uniform - fixed_alpha).sum():.3f}\n')

for fold in FOLDS:
    test_c    = [count_files(fold, 'Test',  c, 'all')              for c in CLASSES]
    raw_tr    = [count_files(fold, 'Train', c, 'raw')              for c in CLASSES]
    post_aug  = [count_files(fold, 'Train', c, 'all_ablation_ext') for c in CLASSES]
    a_raw, a_post = alpha_from_counts(raw_tr), alpha_from_counts(post_aug)
    report['folds'][fold] = {
        'test_counts': test_c, 'raw_train_counts': raw_tr, 'post_aug_train_counts': post_aug,
        'alpha_raw_train': a_raw.tolist(), 'alpha_post_aug': a_post.tolist(),
        'maxabs_raw_vs_fixed': float(np.abs(a_raw - fixed_alpha).max()),
        'l1_raw_vs_fixed': float(np.abs(a_raw - fixed_alpha).sum()),
        'maxabs_postaug_vs_uniform': float(np.abs(a_post - uniform).max()),
        'fixed_counts_equal_fold_test_counts': bool(np.array_equal(FIXED_COUNTS, np.array(test_c, dtype=float))),
    }
    r = report['folds'][fold]
    print(f'{fold}: test={test_c}  rawTrain={raw_tr}  postAugTrain={post_aug}')
    print(f'   alpha(raw train) = {np.round(a_raw, 3).tolist()}   max|diff| vs fixed = {r["maxabs_raw_vs_fixed"]:.3f}  (L1 {r["l1_raw_vs_fixed"]:.3f})')
    print(f'   alpha(post-aug)  = {np.round(a_post, 3).tolist()}   max|diff| vs uniform = {r["maxabs_postaug_vs_uniform"]:.4f}')
    print(f'   fixed counts equal this fold test counts? {r["fixed_counts_equal_fold_test_counts"]}\n')

report['worst_maxabs_raw_vs_fixed'] = max(v['maxabs_raw_vs_fixed'] for v in report['folds'].values())
report['any_fold_test_equals_fixed'] = any(v['fixed_counts_equal_fold_test_counts'] for v in report['folds'].values())
print(f'Worst-case max|diff| between fixed alpha and raw-train alpha over folds: {report["worst_maxabs_raw_vs_fixed"]:.3f}')
print(f'Does any evaluated fold have test counts equal to the fixed counts? {report["any_fold_test_equals_fixed"]}')

with open('focal_alpha_check.json', 'w') as f:
    json.dump(report, f, indent=2)
print('Saved: focal_alpha_check.json')