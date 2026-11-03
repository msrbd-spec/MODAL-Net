"""Expert-mask localisation of the model's attention (--mode xai_localization).

Uses the pixel-level lesion annotations shipped with DDR (lesion_segmentation: MA, HE, EX, SE; 757
images) and IDRiD (A. Segmentation: MA, HE, EX, SE; 81 images). These masks were drawn by graders,
unlike the heuristic LesionDetector masks used for the qualitative XAI grid.

For every annotated image with at least one lesion pixel:
  * CAM = fold-averaged Grad-CAM++ (cfg xai_loc_method) for the most likely DR grade
    (target class = 1 + argmax P[1:]), mapped to the same 512 x 512 geometry as the model input.
  * Masks are cropped / resized with the identical fundus crop and resize (nearest neighbour) and
    dilated by cfg xai_loc_dilate pixels (annotations are thin, CAMs are coarse).
Per image metrics (inside the fundus mask):
  energy_in_mask  CAM mass inside the dilated lesion mask / total CAM mass
  enrichment      energy_in_mask / (lesion area / fundus area); 1 = no better than uniform attention
  pointing_game   1 if the CAM maximum lies inside the dilated lesion mask
  pixel_auroc     AUROC of CAM values for lesion vs non-lesion pixels (128 x 128 grid)
Reported per dataset and per lesion type (mean with 95 % bootstrap CI) in
results/<phase>/<tag>/xai/xai_localization.csv; per-image values in xai_localization_images.csv.
If DDR is a training source of the evaluated model, only DDR images of the frozen DDR TEST split are used.
"""

import glob
import os

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from src.models.gradenet import GradeNet
from src.utils.config import ckpt_owner, find_exp, train_sources
from src.xai.cams import compute_model_cam
from src.xai.grid import load_display_and_input

LESIONS = ('MA', 'HE', 'EX', 'SE')
IDRID_DIRS = {'MA': '1. Microaneurysms', 'HE': '2. Haemorrhages', 'EX': '3. Hard Exudates', 'SE': '4. Soft Exudates'}


def fundus_box(img, thr=10, margin=5):
    """Bounding box (x, y, w, h) used by src.data.preprocess.crop_fundus (same thresholds and margin)."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, t = cv2.threshold(g, thr, 255, cv2.THRESH_BINARY)
    cs, _ = cv2.findContours(t, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs:
        return 0, 0, img.shape[1], img.shape[0]
    x, y, w, h = cv2.boundingRect(max(cs, key=cv2.contourArea))
    x, y = max(0, x - margin), max(0, y - margin)
    w, h = min(img.shape[1] - x, w + 2 * margin), min(img.shape[0] - y, h + 2 * margin)
    return x, y, w, h


def mask_to_display(mask, box, preproc, size=512):
    """Apply the display / model geometry (crop -> stretch or pad-square resize) to a binary mask."""
    x, y, w, h = box
    m = (mask[y:y + h, x:x + w] > 0).astype(np.uint8) * 255
    if preproc == 'legacy':
        out = cv2.resize(m, (size, size), interpolation=cv2.INTER_NEAREST)
    else:
        s = max(m.shape)
        canvas = np.zeros((s, s), dtype=np.uint8)
        y0, x0 = (s - m.shape[0]) // 2, (s - m.shape[1]) // 2
        canvas[y0:y0 + m.shape[0], x0:x0 + m.shape[1]] = m
        out = cv2.resize(canvas, (size, size), interpolation=cv2.INTER_NEAREST)
    return out > 0


def _read_mask(path, shape):
    if not path or not os.path.isfile(path):
        return np.zeros(shape, dtype=bool)
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        return np.zeros(shape, dtype=bool)
    if m.shape != shape:
        m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return m > 0


def annotated_images(cfg):
    """List of dicts: dataset, id, image path, mask path per lesion type."""
    items = []
    ddr_root = cfg['paths']['ddr'].get('lesion_root', '')
    if ddr_root and os.path.isdir(ddr_root):
        restrict = None
        if 'ddr' in train_sources(cfg):
            from src.data.external import external_frame
            restrict = set(external_frame(cfg, 'ddr', 'test')['id_code'].astype(str))
        for split, ann in (('train', 'train'), ('val', 'val'), ('test', 'tet')):
            ann_dir = os.path.join(ddr_root, 'annotations', ann)
            if not os.path.isdir(ann_dir):
                ann_dir = os.path.join(ddr_root, 'annotations', split)
            for img in sorted(glob.glob(os.path.join(ddr_root, 'images', split, '*.jpg'))):
                iid = os.path.splitext(os.path.basename(img))[0]
                if restrict is not None and iid not in restrict:
                    continue
                items.append({'dataset': 'ddr', 'id': iid, 'image': img,
                              **{les: os.path.join(ann_dir, les, iid + '.tif') for les in LESIONS}})
    idrid_root = cfg['paths']['idrid'].get('lesion_root', '')
    if idrid_root and os.path.isdir(idrid_root):
        for sub in ('a. Training Set', 'b. Testing Set'):
            for img in sorted(glob.glob(os.path.join(idrid_root, '1. Original Images', sub, '*.jpg'))):
                iid = os.path.splitext(os.path.basename(img))[0]
                gt = os.path.join(idrid_root, '2. All Segmentation Groundtruths', sub)
                items.append({'dataset': 'idrid', 'id': f'{sub[0]}_{iid}', 'image': img,
                              **{les: os.path.join(gt, IDRID_DIRS[les], f'{iid}_{les}.tif') for les in LESIONS}})
    return items


def _metrics(cam, lesion, fundus):
    inside = fundus & (cam >= 0)
    total = float(cam[inside].sum())
    area_f = float(fundus.sum())
    area_l = float((lesion & fundus).sum())
    if total <= 1e-8 or area_l == 0 or area_f == 0:
        return None
    energy = float(cam[lesion & fundus].sum()) / total
    peak = np.unravel_index(np.argmax(np.where(fundus, cam, -1)), cam.shape)
    small_c = cv2.resize(cam, (128, 128), interpolation=cv2.INTER_AREA)
    small_l = cv2.resize(lesion.astype(np.uint8), (128, 128), interpolation=cv2.INTER_NEAREST) > 0
    small_f = cv2.resize(fundus.astype(np.uint8), (128, 128), interpolation=cv2.INTER_NEAREST) > 0
    yl = small_l[small_f]
    auc = float(roc_auc_score(yl, small_c[small_f])) if 0 < yl.sum() < len(yl) else np.nan
    return {'energy_in_mask': energy, 'enrichment': energy / (area_l / area_f),
            'pointing_game': float(lesion[peak]), 'pixel_auroc': auc, 'lesion_area_frac': area_l / area_f}


def _unpack(bits, shape):
    return np.unpackbits(bits)[:shape[0] * shape[1]].reshape(shape).astype(bool)


def run_xai_localization(cfg):
    tag = cfg['tag']
    exp_dir = find_exp(tag)
    out_dir = os.path.join(exp_dir, 'xai')
    os.makedirs(out_dir, exist_ok=True)
    items = annotated_images(cfg)
    if not items:
        raise FileNotFoundError('No expert lesion masks found: check paths.ddr.lesion_root / paths.idrid.lesion_root')
    owner = ckpt_owner(cfg)
    n_folds = int(cfg.get('n_folds', 5))
    method = cfg.get('xai_loc_method', 'gradcampp')
    head_type = cfg.get('head_type', 'softmax')
    dil = int(cfg.get('xai_loc_dilate', 15))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dil + 1, 2 * dil + 1)) if dil > 0 else None
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    prepared = []
    for it in items:
        raw = cv2.imread(it['image'])
        if raw is None:
            raise FileNotFoundError(it['image'])
        _, x, fundus = load_display_and_input(raw, cfg)
        box = fundus_box(raw)
        masks = {}
        for les in LESIONS:
            m = mask_to_display(_read_mask(it[les], raw.shape[:2]), box, cfg.get('preproc', 'legacy'))
            masks[les] = cv2.dilate(m.astype(np.uint8), kernel) > 0 if kernel is not None else m
        union = np.any(np.stack(list(masks.values())), axis=0)
        if not (union & fundus).any():
            continue
        # compact storage: float16 input, bit-packed masks, one running CAM sum per image
        prepared.append({'dataset': it['dataset'], 'id': it['id'], 'x': x.half(), 'shape': fundus.shape,
                         'fundus': np.packbits(fundus), 'masks': {k: np.packbits(v) for k, v in masks.items()},
                         'union': np.packbits(union), 'cam_sum': None, 'probs': []})

    from src.models.gradenet import probs_from_logits

    def _model(k):
        m = GradeNet(dict(cfg, pretrained=False, pretrain_path=None))
        m.load_state_dict(torch.load(os.path.join('checkpoints', owner, f'fold{k}.pth'), map_location='cpu'))
        return m.to(device).eval().float()

    # Pass 1: ensemble probabilities fix one target class per image (the most likely DR grade)
    for k in range(n_folds):
        model = _model(k)
        with torch.no_grad():
            for p in prepared:
                p['probs'].append(probs_from_logits(model(p['x'].float().to(device)), head_type).cpu().numpy()[0])
        del model
    for p in prepared:
        p['target'] = 1 + int(np.argmax(np.mean(p['probs'], axis=0)[1:]))
    # Pass 2: CAM of every fold model for that target, averaged later
    for k in range(n_folds):
        model = _model(k)
        with torch.set_grad_enabled(True):
            for p in prepared:
                fundus = _unpack(p['fundus'], p['shape'])
                c = compute_model_cam(model, p['x'].float().to(device), target_class=p['target'],
                                      method=method, head_type=head_type, fundus_mask=fundus)
                p['cam_sum'] = c if p['cam_sum'] is None else p['cam_sum'] + c
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    rows = []
    for p in prepared:
        cam = p['cam_sum'] / float(n_folds)
        rng = cam.max() - cam.min()
        cam = (cam - cam.min()) / rng if rng > 1e-8 else np.zeros_like(cam)
        fundus = _unpack(p['fundus'], p['shape'])
        cam = cam * fundus
        pm = np.mean(p['probs'], axis=0)
        masks = [('any', _unpack(p['union'], p['shape']))] + \
            [(k, _unpack(v, p['shape'])) for k, v in p['masks'].items()]
        for les, m in masks:
            r = _metrics(cam, m, fundus)
            if r is not None:
                rows.append({'dataset': p['dataset'], 'id': p['id'], 'lesion': les, 'pred_grade': int(np.argmax(pm)),
                             'target_class': p['target'], **r})
    cols = ['dataset', 'id', 'lesion', 'pred_grade', 'target_class', 'energy_in_mask', 'enrichment',
            'pointing_game', 'pixel_auroc', 'lesion_area_frac']
    per_image = pd.DataFrame(rows, columns=cols)
    if per_image.empty:
        print('xai_localization: WARNING every CAM was empty (all-zero maps); no metrics computed')
    per_image.round(4).to_csv(os.path.join(out_dir, 'xai_localization_images.csv'), index=False)

    out = []
    rng_b = np.random.default_rng(0)
    for (ds, les), g in per_image.groupby(['dataset', 'lesion']):
        for metric in ('energy_in_mask', 'enrichment', 'pointing_game', 'pixel_auroc', 'lesion_area_frac'):
            v = g[metric].dropna().to_numpy(dtype=float)
            if len(v) == 0:
                continue
            boots = [rng_b.choice(v, len(v), replace=True).mean() for _ in range(1000)]
            out.append({'dataset': ds, 'lesion': les, 'metric': metric, 'n_images': len(v),
                        'mean': round(float(v.mean()), 4), 'ci_lo': round(float(np.percentile(boots, 2.5)), 4),
                        'ci_hi': round(float(np.percentile(boots, 97.5)), 4)})
    res = pd.DataFrame(out, columns=['dataset', 'lesion', 'metric', 'n_images', 'mean', 'ci_lo', 'ci_hi'])
    res.to_csv(os.path.join(out_dir, 'xai_localization.csv'), index=False)
    with open(os.path.join(out_dir, 'xai_localization_notes.txt'), 'w') as f:
        f.write(__doc__)
        f.write(f'\nMethod: {method}; dilation: {dil} px; folds averaged: {n_folds}; images used: {len(prepared)}\n')
    print(f'xai_localization: {len(prepared)} annotated images -> {os.path.join(out_dir, "xai_localization.csv")}')
    return res
