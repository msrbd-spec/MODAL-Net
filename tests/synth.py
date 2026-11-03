"""Builder of a miniature synthetic project (all seven datasets) for the end-to-end tests.

Layouts mirror the real datasets exactly (see configs/config.yaml -> paths), only the images are
small synthetic fundus discs. Nothing here is used outside the test-suite.
"""

import os
import zipfile

import cv2
import numpy as np
import pandas as pd
import yaml


def fundus(path, grade, seed, size=64):
    rng = np.random.default_rng(seed)
    im = np.zeros((size, size + 8, 3), dtype=np.uint8)
    cv2.circle(im, (size // 2 + 4, size // 2), size // 2 - 3, (30, 70 + 10 * grade, 160), -1)
    for _ in range(grade * 3):
        x, y = rng.integers(16, size - 16, 2)
        cv2.circle(im, (int(x) + 4, int(y)), 2, (20, 20, 200 - 20 * grade), -1)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, im)


def write_xlsx(path, header, rows):
    """Minimal .xlsx (inline strings, numeric cells) readable by src.data.external.read_xlsx_first_sheet."""
    def col(i):
        s = ''
        i += 1
        while i:
            i, r = divmod(i - 1, 26)
            s = chr(65 + r) + s
        return s

    def cell(r, c, v):
        ref = f'{col(c)}{r}'
        if isinstance(v, (int, float, np.integer)):
            return f'<c r="{ref}"><v>{v}</v></c>'
        return f'<c r="{ref}" t="inlineStr"><is><t>{v}</t></is></c>'

    xml_rows = []
    for r, vals in enumerate([header] + rows, start=1):
        xml_rows.append(f'<row r="{r}">' + ''.join(cell(r, c, v) for c, v in enumerate(vals)) + '</row>')
    sheet = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
             '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>'
             + ''.join(xml_rows) + '</sheetData></worksheet>')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with zipfile.ZipFile(path, 'w') as z:
        z.writestr('[Content_Types].xml', '<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"/>')
        z.writestr('xl/workbook.xml', '<?xml version="1.0"?><workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"/>')
        z.writestr('xl/worksheets/sheet1.xml', sheet)


def build(root, n_aptos_per_class=6, seed=0):
    """Create every dataset under root/datasets and return a config dict whose paths point there."""
    D = os.path.join(root, 'datasets')
    k = 0

    def img(path, grade):
        nonlocal k
        k += 1
        fundus(path, grade, seed * 100000 + k)

    # APTOS
    A = os.path.join(D, 'APTOS_19')
    tr, va, te = [], [], []
    for c in range(5):
        for j in range(n_aptos_per_class):
            i = f'tr{c}{j:02d}'
            img(os.path.join(A, 'train_images', i + '.png'), c)
            tr.append({'id_code': i, 'diagnosis': c})
        for j in range(2):
            i = f'va{c}{j}'
            img(os.path.join(A, 'val_images', i + '.png'), c)
            va.append({'id_code': i, 'diagnosis': c})
            i = f'te{c}{j}'
            img(os.path.join(A, 'test_images', i + '.png'), c)
            te.append({'id_code': i, 'diagnosis': c})
    pd.DataFrame(tr).to_csv(os.path.join(A, 'train_1.csv'), index=False)
    pd.DataFrame(va).to_csv(os.path.join(A, 'valid.csv'), index=False)
    pd.DataFrame(te).to_csv(os.path.join(A, 'test.csv'), index=False)

    # Messidor-2: 40 images = 20 exams x 2 eyes, grades cycle
    M = os.path.join(D, 'Messidor_2')
    rows = []
    for e in range(20):
        for eye in ('0100', '0200'):
            i = f'20051020_{e:05d}_{eye}_PP'
            g = (e + (eye == '0200')) % 5
            img(os.path.join(M, 'images', i + '.png'), g)
            rows.append({'id_code': i, 'diagnosis': g, 'adjudicated_dme': 0, 'adjudicated_gradable': 1})
    pd.DataFrame(rows).to_csv(os.path.join(M, 'messidor_data.csv'), index=False)

    # EyePACS (labels csv lives in a folder called trainLabels.csv)
    E = os.path.join(D, 'EyePACS')
    rows = []
    for j in range(25):
        i = f'{j}_left'
        img(os.path.join(E, 'EyePACS', i + '.jpeg'), j % 5)
        rows.append({'image': i, 'level': j % 5})
    os.makedirs(os.path.join(E, 'trainLabels.csv'), exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(E, 'trainLabels.csv', 'trainLabels.csv'), index=False)
    os.makedirs(os.path.join(E, 'lesion_labels'), exist_ok=True)

    # DDR: grade 5 = ungradable must be dropped
    R = os.path.join(D, 'DDR-dataset', 'DR_grading')
    for split, n in (('train', 15), ('valid', 15), ('test', 15)):
        lines = []
        for j in range(n):
            g = j % 6
            name = f'{split}{j:03d}.jpg'
            img(os.path.join(R, split, name), min(g, 4))
            lines.append(f'{name} {g}')
        os.makedirs(R, exist_ok=True)
        open(os.path.join(R, f'{split}.txt'), 'w').write('\n'.join(lines) + '\n')
    # DDR lesion masks for 3 test images
    L = os.path.join(D, 'DDR-dataset', 'lesion_segmentation')
    for j in (1, 2, 3):
        name = f'test{j:03d}'
        src = cv2.imread(os.path.join(R, 'test', name + '.jpg'))
        os.makedirs(os.path.join(L, 'images', 'test'), exist_ok=True)
        cv2.imwrite(os.path.join(L, 'images', 'test', name + '.jpg'), src)
        for les in ('EX', 'HE', 'MA', 'SE'):
            m = np.zeros(src.shape[:2], np.uint8)
            if les in ('HE', 'MA'):
                cv2.circle(m, (30, 30), 6, 255, -1)
            os.makedirs(os.path.join(L, 'annotations', 'tet', les), exist_ok=True)
            cv2.imwrite(os.path.join(L, 'annotations', 'tet', les, name + '.tif'), m)

    # DeepDRiD regular fundus: 4 images per patient; one row with the label in the other eye column
    Q = os.path.join(D, 'DeepDRiD', 'regular_fundus_images')
    header = ['patient_id', 'image_id', 'image_path', 'Overall quality', 'left_eye_DR_Level',
              'right_eye_DR_Level', 'patient_DR_Level', 'Clarity', 'Field definition', 'Artifact']
    pid = 0
    for sub, csv in (('regular-fundus-training', 'regular-fundus-training.csv'),
                     ('regular-fundus-validation', 'regular-fundus-validation.csv')):
        rows = []
        for p in range(4):
            pid += 1
            gl, gr = pid % 5, (pid + 1) % 5
            for iid, (le, re_) in ((f'{pid}_l1', (gl, None)), (f'{pid}_l2', (None, gl) if p == 0 else (gl, None)),
                                   (f'{pid}_r1', (None, gr)), (f'{pid}_r2', (None, gr))):
                img(os.path.join(Q, sub, 'Images', str(pid), iid + '.jpg'), le if le is not None else re_)
                rows.append([pid, iid, f'\\{sub}\\{pid}\\{iid}.jpg', 1, le, re_, max(gl, gr), 8, 8, 0])
        pd.DataFrame(rows, columns=header).to_csv(os.path.join(Q, sub, csv), index=False)
    rows = []
    for p in range(4):
        pid += 1
        for iid, g in ((f'{pid}_l1', pid % 5), (f'{pid}_l2', pid % 5), (f'{pid}_r1', (pid + 2) % 5),
                       (f'{pid}_r2', (pid + 2) % 5)):
            img(os.path.join(Q, 'Online-Challenge1_2-Evaluation', 'Images', str(pid), iid + '.jpg'), g)
            rows.append([iid, g])
    write_xlsx(os.path.join(Q, 'Online-Challenge1_2-Evaluation', 'Challenge1_labels.xlsx'), ['image_id', 'DR_Levels'], rows)

    # IDRiD disease grading (train and test reuse the same file names)
    G = os.path.join(D, 'IDRiD', 'B. Disease Grading')
    for sub, csv, n in (('a. Training Set', 'a. IDRiD_Disease Grading_Training Labels.csv', 15),
                        ('b. Testing Set', 'b. IDRiD_Disease Grading_Testing Labels.csv', 10)):
        rows = []
        for j in range(n):
            name = f'IDRiD_{j + 1:03d}'
            img(os.path.join(G, '1. Original Images', sub, name + '.jpg'), j % 5)
            rows.append({'Image name': name, 'Retinopathy grade': j % 5, 'Risk of macular edema ': 0})
        os.makedirs(os.path.join(G, '2. Groundtruths'), exist_ok=True)
        pd.DataFrame(rows).to_csv(os.path.join(G, '2. Groundtruths', csv), index=False)
    S = os.path.join(D, 'IDRiD', 'A. Segmentation')
    for j in (55, 56):
        name = f'IDRiD_{j}'
        img(os.path.join(S, '1. Original Images', 'b. Testing Set', name + '.jpg'), 3)
        for les, folder in (('MA', '1. Microaneurysms'), ('HE', '2. Haemorrhages'), ('EX', '3. Hard Exudates'),
                            ('SE', '4. Soft Exudates')):
            m = np.zeros((64, 72), np.uint8)
            if les == 'EX':
                cv2.circle(m, (40, 34), 7, 1, -1)
            os.makedirs(os.path.join(S, '2. All Segmentation Groundtruths', 'b. Testing Set', folder), exist_ok=True)
            cv2.imwrite(os.path.join(S, '2. All Segmentation Groundtruths', 'b. Testing Set', folder,
                                     f'{name}_{les}.tif'), m)

    cfg = yaml.safe_load(open('configs/config.yaml'))
    P = cfg['paths']
    P['aptos'] = {'train_images': os.path.join(A, 'train_images'), 'val_images': os.path.join(A, 'val_images'),
                  'test_images': os.path.join(A, 'test_images'), 'train_csv': os.path.join(A, 'train_1.csv'),
                  'val_csv': os.path.join(A, 'valid.csv'), 'test_csv': os.path.join(A, 'test.csv')}
    P['messidor'] = {'images': os.path.join(M, 'images'), 'csv': os.path.join(M, 'messidor_data.csv')}
    P['eyepacs'] = {'images': os.path.join(E, 'EyePACS'),
                    'labels_csv': os.path.join(E, 'trainLabels.csv', 'trainLabels.csv'),
                    'lesion_labels': os.path.join(E, 'lesion_labels')}
    P['ddr']['root'] = R
    P['ddr']['lesion_root'] = L
    P['deepdrid']['root'] = Q
    P['idrid']['root'] = G
    P['idrid']['lesion_root'] = S
    cfg.update({'backbone': 'resnet18', 'pretrained': False, 'img_size': 64, 'epochs': 1, 'warmup_epochs': 0,
                'n_folds': 2, 'batch_size': 4, 'grad_accum': 1, 'num_workers': 0, 'mix_prob': 0.0,
                'ema_decay': 0.0, 'preproc': 'none', 'freeze_epochs': 0, 'bootstrap_ci': 50, 'fewshot_draws': 3,
                'fewshot_n': [5], 'eyepacs_max': 25, 'source_max': 25})
    return cfg


def install_config(cfg, path='configs/config.yaml'):
    """Write cfg as the project default config so CLI-style build_cfg calls use the synthetic paths."""
    with open(path, 'w') as f:
        yaml.safe_dump(cfg, f)
