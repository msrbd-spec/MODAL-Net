"""Duplicate audit: pHash-based cross-dataset deduplication.

64-bit DCT pHash of the green channel of a central crop after CLAHE.
Hamming distance <= 4 => candidate duplicate pair.
Searches across/inside APTOS, EyePACS, Messidor-2.

Output: results/phase0_protocol/audit/duplicates.csv
"""

import os
import cv2
import numpy as np
from glob import glob
from itertools import combinations
from tqdm import tqdm
try:
    from scipy.fftpack import dct
except ImportError:
    from scipy.fft import dct


# ---------------------------------------------------------------------------
# pHash computation
# ---------------------------------------------------------------------------

def _crop_fundus_green(img_bgr, thr=10, margin=5):
    """Crop fundus bounding box and return green channel."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, t = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY)
    cs, _ = cv2.findContours(t, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cs:
        x, y, w, h = cv2.boundingRect(max(cs, key=cv2.contourArea))
        x, y = max(0, x - margin), max(0, y - margin)
        w = min(img_bgr.shape[1] - x, w + 2 * margin)
        h = min(img_bgr.shape[0] - y, h + 2 * margin)
        img_bgr = img_bgr[y:y + h, x:x + w]
    return img_bgr[:, :, 1]  # green channel


def _clahe_green(green):
    """Apply CLAHE to a single-channel image."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(green)


def phash_dct64(img_bgr, hash_size=8, highfreq_factor=4):
    """Compute a 64-bit DCT perceptual hash.

    Steps:
      1. Crop fundus, extract green channel.
      2. CLAHE enhancement.
      3. Resize to (hash_size * highfreq_factor) x (hash_size * highfreq_factor).
      4. 2-D DCT, keep top-left hash_size x hash_size block.
      5. Median threshold -> 64 bits packed into an int.

    Returns:
        int: 64-bit hash value.
    """
    green = _crop_fundus_green(img_bgr)
    green = _clahe_green(green)
    img_size = hash_size * highfreq_factor  # default 32
    resized = cv2.resize(green, (img_size, img_size),
                         interpolation=cv2.INTER_AREA).astype(np.float64)
    # 2-D DCT (apply 1-D DCT along rows, then columns)
    d = dct(dct(resized, axis=0, norm='ortho'), axis=1, norm='ortho')
    low = d[:hash_size, :hash_size]
    med = np.median(low)
    bits = (low > med).flatten()  # 64 bools
    # Pack into integer
    h = 0
    for b in bits:
        h = (h << 1) | int(b)
    return h


def hamming(a, b):
    """Hamming distance between two 64-bit integers."""
    return bin(a ^ b).count('1')


# ---------------------------------------------------------------------------
# Batch hashing
# ---------------------------------------------------------------------------

def _collect_images(dirs, extensions=('.png', '.jpg', '.jpeg')):
    """Return list of (path, basename_no_ext) for all images in dirs."""
    items = []
    if isinstance(dirs, str):
        dirs = [dirs]
    for d in dirs:
        if not os.path.isdir(d):
            continue
        for ext in extensions:
            for p in glob(os.path.join(d, f'*{ext}')):
                items.append((p, os.path.splitext(os.path.basename(p))[0]))
            for p in glob(os.path.join(d, f'*{ext.upper()}')):
                items.append((p, os.path.splitext(os.path.basename(p))[0]))
    return items


def _hash_all(image_items, label=''):
    """Compute pHash for a list of (path, id) tuples. Returns list of (id, hash)."""
    results = []
    for path, img_id in tqdm(image_items, desc=f'pHash {label}'):
        img = cv2.imread(path)
        if img is None:
            continue
        try:
            h = phash_dct64(img)
            results.append((img_id, h))
        except Exception:
            continue
    return results


# ---------------------------------------------------------------------------
# Main audit function
# ---------------------------------------------------------------------------

def run_dedup_audit(cfg):
    """Run the full duplicate audit across APTOS, EyePACS, Messidor-2.

    Writes results/phase0_protocol/audit/duplicates.csv with columns:
      dataset_a, id_a, dataset_b, id_b, hamming_distance
    """
    P = cfg['paths']

    # Collect images per dataset
    aptos_dirs = [P['aptos']['train_images'], P['aptos']['val_images'],
                  P['aptos']['test_images']]
    aptos_items = _collect_images(aptos_dirs)
    mess_items = _collect_images(P['messidor']['images'])
    eyepacs_items = _collect_images(P['eyepacs']['images'],
                                   extensions=('.jpeg', '.jpg', '.png'))

    print(f'dedup_audit: APTOS={len(aptos_items)}, '
          f'EyePACS={len(eyepacs_items)}, Messidor={len(mess_items)}')

    # Compute hashes
    aptos_hashes = _hash_all(aptos_items, 'APTOS')
    mess_hashes = _hash_all(mess_items, 'Messidor')
    eyepacs_hashes = _hash_all(eyepacs_items, 'EyePACS')

    # Compare all pairs with Hamming <= 4
    pairs = []

    def _compare(set_a, label_a, set_b, label_b, same_set=False):
        """Compare hashes between two sets. If same_set, skip (i, i)."""
        for i, (id_a, h_a) in enumerate(set_a):
            start = i + 1 if same_set else 0
            for j in range(start, len(set_b)):
                id_b, h_b = set_b[j]
                d = hamming(h_a, h_b)
                if d <= 4:
                    pairs.append({
                        'dataset_a': label_a, 'id_a': id_a,
                        'dataset_b': label_b, 'id_b': id_b,
                        'hamming_distance': d
                    })

    # Within-dataset
    _compare(aptos_hashes, 'APTOS', aptos_hashes, 'APTOS', same_set=True)
    _compare(mess_hashes, 'Messidor', mess_hashes, 'Messidor', same_set=True)
    _compare(eyepacs_hashes, 'EyePACS', eyepacs_hashes, 'EyePACS',
             same_set=True)

    # Cross-dataset
    _compare(aptos_hashes, 'APTOS', mess_hashes, 'Messidor')
    _compare(aptos_hashes, 'APTOS', eyepacs_hashes, 'EyePACS')
    _compare(mess_hashes, 'Messidor', eyepacs_hashes, 'EyePACS')

    # Write output
    import pandas as pd
    out_dir = os.path.join('results', 'phase0_protocol', 'audit')
    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame(pairs) if pairs else pd.DataFrame(
        columns=['dataset_a', 'id_a', 'dataset_b', 'id_b', 'hamming_distance'])
    df.to_csv(os.path.join(out_dir, 'duplicates.csv'), index=False)
    print(f'dedup_audit: {len(pairs)} candidate pairs -> '
          f'results/phase0_protocol/audit/duplicates.csv')
