#!/usr/bin/env python3
"""
external_dev_overlap_audit.py   (finding 08, CPU only, no GPU, no training)

WHAT IT DOES
Audits whether MPox-Vision (external set) shares exact or near-duplicate images with the development pool
(Intermediate_Merged_Clean_Dataset, the deduplicated pool of 1,463 images that every fold is cut from).
  1. Exact duplicates: SHA-256 of the file bytes.
  2. Near duplicates: 64-bit perceptual hash (pHash) and difference hash (dHash). Every external image is compared with
     every development image under all 8 flips and 90-degree rotations, so flipped or rotated copies are caught.
  3. For every external image the closest development image is exported, with a side-by-side montage for each close pair so
     the manual review outcome can be documented. SSIM is added for the flagged pairs when scikit-image is installed.
  4. Optional residual check inside the development pool: how many pairs are still close after deduplication.
Hashes are robust to resizing and mild JPEG recompression. They are NOT robust to heavy cropping or watermarks, which is stated
as a limitation of the audit.

Run from the folder that contains datasets/ .   pip install imagehash pillow numpy scikit-image   (scikit-image is optional)
"""
import os, sys, json, csv, hashlib, warnings
import numpy as np
from PIL import Image, ImageOps, ImageDraw
warnings.filterwarnings('ignore')
try:
    import imagehash
except ImportError:
    raise SystemExit('imagehash is required. Run: pip install imagehash')
try:
    from skimage.metrics import structural_similarity as _ssim
except Exception:
    _ssim = None

DEV_DIR   = 'datasets/Intermediate_Merged_Clean_Dataset'
EXT_DIR   = 'datasets/external_mpox_vision'
OUT_DIR   = 'outputs/overlap_audit'
IMG_EXT   = ('.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tif', '.tiff')
HASH_SIZE = 8                                   # 8x8 = 64-bit hashes
THRESHOLDS = [0, 2, 4, 6, 8, 10, 12, 16, 20, 25]
REVIEW_T  = 10                                  # pairs with pHash or dHash <= this get a montage for manual review
MAX_MONTAGES = 80
RUN_WITHIN_POOL = True                          # residual near-duplicate check inside the development pool
os.makedirs(os.path.join(OUT_DIR, 'review_montages'), exist_ok=True)

try:
    _T = Image.Transpose
    ROT90, ROT180, ROT270, FLIP_LR = _T.ROTATE_90, _T.ROTATE_180, _T.ROTATE_270, _T.FLIP_LEFT_RIGHT
except AttributeError:
    ROT90, ROT180, ROT270, FLIP_LR = Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270, Image.FLIP_LEFT_RIGHT

POP = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)


def list_images(root):
    items = []
    for dp, _, files in os.walk(root):
        for f in sorted(files):
            if f.lower().endswith(IMG_EXT):
                rel = os.path.relpath(dp, root)
                cls = rel.split(os.sep)[0] if rel != '.' else ''
                items.append((os.path.join(dp, f), cls))
    return sorted(items)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def dihedral_variants(img):
    """The 8 images of the dihedral group: 4 rotations of the image and 4 rotations of its mirror image."""
    out = []
    for base in (img, img.transpose(FLIP_LR)):
        out += [base, base.transpose(ROT90), base.transpose(ROT180), base.transpose(ROT270)]
    return out


def to_u64(h):
    return np.uint64(int(str(h), 16))


def hashes_of(img, with_variants):
    imgs = dihedral_variants(img) if with_variants else [img]
    ph = np.array([to_u64(imagehash.phash(v, hash_size=HASH_SIZE)) for v in imgs], dtype=np.uint64)
    dh = np.array([to_u64(imagehash.dhash(v, hash_size=HASH_SIZE)) for v in imgs], dtype=np.uint64)
    return ph, dh


def hamming_matrix(A, B):
    """A: (n,) uint64, B: (m,) uint64 -> (n, m) Hamming distances."""
    x = np.ascontiguousarray(np.bitwise_xor(A[:, None], B[None, :]))
    n, m = x.shape
    return POP[x.view(np.uint8).reshape(n, m, 8)].sum(axis=2).astype(np.int16)


def closest_with_variants(Q, base):
    """Q: (n,8) query hashes (8 variants), base: (m,) hashes. Returns best distance, best index, best variant per query."""
    n = Q.shape[0]
    best_d = np.full(n, 10 ** 6, dtype=np.int32)
    best_i = np.zeros(n, dtype=np.int32)
    best_v = np.zeros(n, dtype=np.int32)
    for v in range(Q.shape[1]):
        D = hamming_matrix(Q[:, v], base)
        mn, ix = D.min(axis=1), D.argmin(axis=1)
        upd = mn < best_d
        best_d[upd], best_i[upd], best_v[upd] = mn[upd], ix[upd], v
    return best_d, best_i, best_v


def cumulative_counts(dists):
    return {str(t): int((np.asarray(dists) <= t).sum()) for t in THRESHOLDS}


def load_rgb(path):
    return Image.open(path).convert('RGB')


def ssim_of(img_a, img_b):
    if _ssim is None:
        return None
    a = np.asarray(img_a.convert('L').resize((128, 128)), dtype=np.float64)
    b = np.asarray(img_b.convert('L').resize((128, 128)), dtype=np.float64)
    return float(_ssim(a, b, data_range=255.0))


def montage(ext_img, dev_img, caption, path):
    tile = 256
    canvas = Image.new('RGB', (tile * 2 + 10, tile + 42), (255, 255, 255))
    canvas.paste(ext_img.resize((tile, tile)), (0, 0))
    canvas.paste(dev_img.resize((tile, tile)), (tile + 10, 0))
    d = ImageDraw.Draw(canvas)
    d.text((4, tile + 4), caption[:70], fill=(0, 0, 0))
    d.text((4, tile + 22), caption[70:140], fill=(0, 0, 0))
    canvas.save(path)


if __name__ == '__main__':
    dev = list_images(DEV_DIR)
    ext = list_images(EXT_DIR)
    if not dev or not ext:
        raise SystemExit(f'Found {len(dev)} development and {len(ext)} external images. Check DEV_DIR and EXT_DIR.')
    print(f'Development pool: {len(dev)} images   External set: {len(ext)} images   hash size {HASH_SIZE}x{HASH_SIZE}')

    # ── hashes ──
    dev_sha, ext_sha = [], []
    dev_ph, dev_dh, ext_ph, ext_dh = [], [], [], []
    for k, (p, _) in enumerate(dev):
        dev_sha.append(sha256_file(p))
        ph, dh = hashes_of(load_rgb(p), with_variants=RUN_WITHIN_POOL)
        dev_ph.append(ph); dev_dh.append(dh)
        if (k + 1) % 300 == 0: print(f'  hashed {k + 1}/{len(dev)} development images')
    for k, (p, _) in enumerate(ext):
        ext_sha.append(sha256_file(p))
        ph, dh = hashes_of(load_rgb(p), with_variants=True)
        ext_ph.append(ph); ext_dh.append(dh)
        if (k + 1) % 300 == 0: print(f'  hashed {k + 1}/{len(ext)} external images')
    dev_ph, dev_dh = np.array(dev_ph, dtype=np.uint64), np.array(dev_dh, dtype=np.uint64)
    ext_ph, ext_dh = np.array(ext_ph, dtype=np.uint64), np.array(ext_dh, dtype=np.uint64)

    # ── 1. exact duplicates ──
    sha_to_dev = {}
    for i, s in enumerate(dev_sha):
        sha_to_dev.setdefault(s, []).append(i)
    exact_ext = [j for j, s in enumerate(ext_sha) if s in sha_to_dev]
    dev_exact_groups = [v for v in sha_to_dev.values() if len(v) > 1]
    print(f'\nExact SHA-256 matches, external vs development: {len(exact_ext)}')
    print(f'Exact SHA-256 duplicate groups inside the development pool: {len(dev_exact_groups)}')

    # ── 2. near duplicates, external vs development (base hashes of the development images) ──
    dev_ph_base, dev_dh_base = dev_ph[:, 0], dev_dh[:, 0]
    pd_, pi_, pv_ = closest_with_variants(ext_ph, dev_ph_base)
    dd_, di_, dv_ = closest_with_variants(ext_dh, dev_dh_base)
    print('\nCumulative number of external images whose closest development image is within a Hamming distance (64-bit hashes):')
    print('  threshold : ' + ' '.join(f'{t:>5}' for t in THRESHOLDS))
    print('  pHash     : ' + ' '.join(f'{c:>5}' for c in cumulative_counts(pd_).values()))
    print('  dHash     : ' + ' '.join(f'{c:>5}' for c in cumulative_counts(dd_).values()))
    print(f'Closest pair overall: pHash {int(pd_.min())}, dHash {int(dd_.min())}')

    # ── 3. closest pairs, review montages, SSIM ──
    flagged = sorted([j for j in range(len(ext)) if min(pd_[j], dd_[j]) <= REVIEW_T or j in exact_ext],
                     key=lambda j: min(pd_[j], dd_[j]))
    rows = []
    for j, (ep, ecls) in enumerate(ext):
        use_ph = pd_[j] <= dd_[j]
        di = int(pi_[j] if use_ph else di_[j]); dvar = int(pv_[j] if use_ph else dv_[j])
        rows.append({'external_path': ep, 'external_class': ecls, 'dev_path': dev[di][0], 'dev_class': dev[di][1],
                     'phash_dist': int(pd_[j]), 'dhash_dist': int(dd_[j]), 'closest_by': 'phash' if use_ph else 'dhash',
                     'variant_index': dvar, 'sha256_exact_match': j in exact_ext, 'ssim': None})
    n_montage = 0
    for j in flagged:
        r = rows[j]
        e_img, d_img = load_rgb(r['external_path']), load_rgb(r['dev_path'])
        aligned = dihedral_variants(e_img)[r['variant_index']]
        r['ssim'] = ssim_of(aligned, d_img)
        if n_montage < MAX_MONTAGES:
            cap = (f'ext {os.path.basename(r["external_path"])} [{r["external_class"]}]  vs  dev '
                   f'{os.path.basename(r["dev_path"])} [{r["dev_class"]}]   pHash {r["phash_dist"]} dHash {r["dhash_dist"]} '
                   f'SSIM {r["ssim"] if r["ssim"] is None else round(r["ssim"], 3)}')
            montage(aligned, d_img, cap, os.path.join(OUT_DIR, 'review_montages', f'pair_{n_montage:03d}.png'))
            n_montage += 1
    print(f'\nPairs flagged for manual review (pHash or dHash <= {REVIEW_T}, or exact match): {len(flagged)}  '
          f'(montages saved: {n_montage}, SSIM {"on" if _ssim else "off, install scikit-image"})')
    with open(os.path.join(OUT_DIR, 'external_vs_dev_closest_pairs.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    summary = {'n_dev': len(dev), 'n_external': len(ext), 'hash_bits': HASH_SIZE * HASH_SIZE,
               'exact_sha256_external_vs_dev': len(exact_ext),
               'exact_sha256_groups_inside_dev': len(dev_exact_groups),
               'phash_cumulative_counts': cumulative_counts(pd_), 'dhash_cumulative_counts': cumulative_counts(dd_),
               'closest_phash': int(pd_.min()), 'closest_dhash': int(dd_.min()),
               'review_threshold': REVIEW_T, 'pairs_flagged_for_manual_review': len(flagged),
               'variants_compared': 'identity, 3 rotations, mirror, 3 rotations of the mirror',
               'limitation': 'perceptual hashes do not detect heavy cropping or watermark-dominated copies'}

    # ── 4. residual near duplicates inside the development pool ──
    if RUN_WITHIN_POOL:
        n = len(dev)
        iu = np.triu_indices(n, k=1)
        cls_arr = np.array([c for _, c in dev])
        same_cls = cls_arr[iu[0]] == cls_arr[iu[1]]
        sym = {}
        for name, H in (('phash', dev_ph), ('dhash', dev_dh)):
            best = np.full((n, n), 10 ** 4, dtype=np.int32)
            for v in range(H.shape[1]):
                D = hamming_matrix(H[:, v], H[:, 0]).astype(np.int32)
                np.fill_diagonal(D, 10 ** 4)
                best = np.minimum(best, D)
            sym[name] = np.minimum(best, best.T)
        res = {}
        for name in ('phash', 'dhash'):
            up = sym[name][iu]
            res[name] = {'cumulative_pairs': cumulative_counts(up),
                         'cumulative_pairs_different_class': {str(t): int(((up <= t) & ~same_cls).sum()) for t in THRESHOLDS}}
        up_ph = sym['phash'][iu]
        order = np.argsort(up_ph)[:100]
        pair_rows = []
        for o in order:
            a, b = int(iu[0][o]), int(iu[1][o])
            pair_rows.append({'path_a': dev[a][0], 'class_a': dev[a][1], 'path_b': dev[b][0], 'class_b': dev[b][1],
                              'phash_dist': int(sym['phash'][a, b]), 'dhash_dist': int(sym['dhash'][a, b])})
        summary['within_pool_residual_pairs'] = res
        print('\nResidual pairs INSIDE the development pool (after deduplication), cumulative by Hamming distance:')
        print('  threshold        : ' + ' '.join(f'{t:>5}' for t in THRESHOLDS))
        for name in ('phash', 'dhash'):
            print(f'  {name:<16} : ' + ' '.join(f'{c:>5}' for c in res[name]['cumulative_pairs'].values()))
            print(f'  {name + " diff-class":<16} : ' + ' '.join(f'{c:>5}' for c in res[name]['cumulative_pairs_different_class'].values()))
        with open(os.path.join(OUT_DIR, 'within_pool_closest_pairs.csv'), 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['path_a', 'class_a', 'path_b', 'class_b', 'phash_dist', 'dhash_dist'])
            w.writeheader(); w.writerows(pair_rows)

    with open(os.path.join(OUT_DIR, 'overlap_audit_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\nSaved to {OUT_DIR}: overlap_audit_summary.json, external_vs_dev_closest_pairs.csv, review_montages/')
    print('Next: open the montages, record for each flagged pair whether it is a true duplicate, then send me the printed tables.')
