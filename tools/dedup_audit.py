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
# Vectorized pairwise comparison (numpy) — replaces the O(n^2) pure-Python
# double loop, which was intractable for EyePACS (~4.4 billion pairs).
# ---------------------------------------------------------------------------

# Precomputed popcount for every byte value.
_POPCOUNT8 = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(axis=1).astype(np.uint8)


def _hashes_to_bits(hashes):
    """Convert a sequence of 64-bit int hashes to an (N, 8) uint8 byte array."""
    if len(hashes) == 0:
        return np.zeros((0, 8), dtype=np.uint8)
    arr = np.asarray(hashes, dtype=np.uint64)
    # (N, 8) view of each 64-bit hash as 8 bytes (little-endian on x86)
    return arr.view(np.uint8).reshape(len(arr), 8).copy()


def _popcount_xor(bits_a, bits_b):
    """Pairwise Hamming distances between rows of bits_a and bits_b.

    bits_a: (n, 8) uint8, bits_b: (m, 8) uint8. Returns (n, m) uint8 distances.
    """
    xor = bits_a[:, None, :] ^ bits_b[None, :, :]   # (n, m, 8)
    return _POPCOUNT8[xor].sum(axis=2).astype(np.uint8)  # (n, m)


def _compare_vectorized(set_a, label_a, set_b, label_b, max_dist=4,
                        same_set=False, chunk_a=1024, chunk_b=4096, pbar=None):
    """Vectorized pair search: all (i, j) with Hamming(a_i, b_j) <= max_dist.

    If same_set, the upper triangle is compared (i < j) and set_b is set_a.
    Returns a list of pair dicts. chunk_a/chunk_b bound the working array
    size to chunk_a * chunk_b * 8 bytes.
    """
    n = len(set_a)
    pairs = []
    ids_a = [x[0] for x in set_a]
    bits_a = _hashes_to_bits([x[1] for x in set_a])
    if same_set:
        ids_b, bits_b = ids_a, bits_a
    else:
        ids_b = [x[0] for x in set_b]
        bits_b = _hashes_to_bits([x[1] for x in set_b])
    if n == 0 or len(ids_b) == 0:
        return pairs

    for i0 in range(0, n, chunk_a):
        i1 = min(i0 + chunk_a, n)
        # For same_set, skip blocks entirely below the diagonal
        j_start = i0 if same_set else 0
        for j0 in range(j_start, len(ids_b), chunk_b):
            j1 = min(j0 + chunk_b, len(ids_b))
            dist = _popcount_xor(bits_a[i0:i1], bits_b[j0:j1])
            if same_set:
                # mask out j <= i within the diagonal block
                rows = np.arange(i0, i1)[:, None]
                cols = np.arange(j0, j1)[None, :]
                dist = np.where(cols > rows, dist, 255)
            hits = np.argwhere(dist <= max_dist)
            for r, c in hits:
                pairs.append({
                    'dataset_a': label_a, 'id_a': ids_a[i0 + int(r)],
                    'dataset_b': label_b, 'id_b': ids_b[j0 + int(c)],
                    'hamming_distance': int(dist[r, c])
                })
        if pbar is not None:
            pbar.update(i1 - i0)
    return pairs


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


def _hash_all(image_items, label='', cache_dir=None):
    """Compute pHash for a list of (path, id) tuples. Returns list of (id, hash).

    If cache_dir is given, results are cached in <cache_dir>/phash_<label>.npz
    so re-running the audit skips the (expensive) image hashing step.
    """
    cache_path = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f'phash_{label}.npz')
        if os.path.isfile(cache_path):
            try:
                z = np.load(cache_path, allow_pickle=False)
                cached = list(zip(z['ids'].tolist(), z['hashes'].tolist()))
                if len(cached) == len(image_items):
                    print(f'pHash {label}: loaded {len(cached)} hashes from cache')
                    return cached
            except Exception:
                pass  # cache unreadable -> recompute

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

    if cache_path and results:
        try:
            ids = np.array([r[0] for r in results])
            hashes = np.array([r[1] for r in results], dtype=np.uint64)
            np.savez_compressed(cache_path, ids=ids, hashes=hashes)
        except Exception as e:
            print(f'pHash {label}: cache write failed ({e})')
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

    # Compute hashes (cached under results/phase0_protocol/audit/)
    cache_dir = os.path.join('results', 'phase0_protocol', 'audit')
    aptos_hashes = _hash_all(aptos_items, 'APTOS', cache_dir=cache_dir)
    mess_hashes = _hash_all(mess_items, 'Messidor', cache_dir=cache_dir)
    eyepacs_hashes = _hash_all(eyepacs_items, 'EyePACS', cache_dir=cache_dir)

    # Compare all pairs with Hamming <= 4 (vectorized; progress over rows of A)
    def _compare(set_a, label_a, set_b, label_b, same_set=False):
        n = len(set_a)
        with tqdm(total=n, desc=f'dupes {label_a} vs {label_b}') as pbar:
            return _compare_vectorized(set_a, label_a, set_b, label_b,
                                       max_dist=4, same_set=same_set,
                                       pbar=pbar)

    pairs = []
    # Within-dataset
    pairs += _compare(aptos_hashes, 'APTOS', aptos_hashes, 'APTOS', same_set=True)
    pairs += _compare(mess_hashes, 'Messidor', mess_hashes, 'Messidor', same_set=True)
    pairs += _compare(eyepacs_hashes, 'EyePACS', eyepacs_hashes, 'EyePACS',
                      same_set=True)

    # Cross-dataset
    pairs += _compare(aptos_hashes, 'APTOS', mess_hashes, 'Messidor')
    pairs += _compare(aptos_hashes, 'APTOS', eyepacs_hashes, 'EyePACS')
    pairs += _compare(mess_hashes, 'Messidor', eyepacs_hashes, 'EyePACS')

    # Write output
    import pandas as pd
    out_dir = os.path.join('results', 'phase0_protocol', 'audit')
    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame(pairs) if pairs else pd.DataFrame(
        columns=['dataset_a', 'id_a', 'dataset_b', 'id_b', 'hamming_distance'])
    df.to_csv(os.path.join(out_dir, 'duplicates.csv'), index=False)
    print(f'dedup_audit: {len(pairs)} candidate pairs -> '
          f'results/phase0_protocol/audit/duplicates.csv')
