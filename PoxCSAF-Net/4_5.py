#!/usr/bin/env python3
"""
external_validation_strict.py   (findings 04 and 05, inference only, no training)

WHAT IT DOES
Loads the five fold-wise federated checkpoints (heterogeneous setting, FL_Run2_Heterogeneous) that were used for the
external evaluation, averages their four-class softmax outputs on MPox-Vision, and reports three variants on the SAME
images and the SAME checkpoints:
  1. STRICT four-output : arg-max over all four classes, every Healthy prediction is an error (finding 04)
  2. RESTRICTED three-class, uncorrected : Healthy dropped, remaining probabilities renormalized (as in the paper)
  3. RESTRICTED + heuristic score adjustment : each retained probability divided by its focal alpha, renormalized (finding 05)
It also reports the share of external disease images predicted as Healthy, class-level precision / recall / F1, the
confusion matrices (rows = true class, columns = the four predicted classes), and exports the per-image predictions.
No external label or statistic is used to build any of the three variants. Labels are used only for scoring.
Note on the strict variant: precision, recall and F1 are computed over the three classes that exist in MPox-Vision. A Healthy
prediction is never a false positive for those classes, but it always lowers recall and accuracy, so for the strict variant
report accuracy and macro-recall as the headline numbers (the classes are balanced, so macro-recall equals accuracy).

Run from the folder that contains datasets/ and pipeline_v2_single_gpu/ .   Takes a few minutes on one GPU.
"""
import os, glob, json, math, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
warnings.filterwarnings('ignore')

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')

CLASSES          = ['Chickenpox', 'Healthy', 'Measles', 'Monkeypox']
NUM_CLASSES      = 4
EXTERNAL_CLASSES = ['Chickenpox', 'Measles', 'Monkeypox']
SCORED           = [CLASSES.index(c) for c in EXTERNAL_CLASSES]          # [0, 2, 3]
EXTERNAL_DIR     = 'datasets/external_mpox_vision/'
MAIN_OUT_DIR     = 'pipeline_v2_single_gpu'
RUN_NAME         = 'FL_Run2_Heterogeneous'
FOLDS            = [f'Fold_{i}' for i in range(1, 6)]
OUT_DIR          = 'outputs/external_validation_strict'
os.makedirs(OUT_DIR, exist_ok=True)

IMAGE_SIZE, BATCH_SIZE, NUM_WORKERS = 384, 16, 4
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
MSAF_DIM, GEM_P, DROP_PATH = 256, 3.0, 0.10

# same fixed class-weight vector used in training and in the paper
TEST_COUNTS = np.array([42.0, 94.0, 35.0, 116.0])
_inv = 1.0 / (TEST_COUNTS + 1e-6)
FOCAL_ALPHA = (_inv / _inv.sum()).tolist()

# values currently printed in the paper, used only for a reproduction check
PAPER = {'restricted_acc': 0.881, 'restricted_f1': 0.878, 'adjusted_acc': 0.888, 'adjusted_f1': 0.886}


# ═══ Model (identical to the training architecture, msaf_primary) ═══
class GeMPool(nn.Module):
    def __init__(self, p=GEM_P, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps
    def forward(self, x):
        return F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))).pow(1.0 / self.p).flatten(1)

class ECABlock(nn.Module):
    def __init__(self, channels, gamma=2, b=1, init_alpha=0.01):
        super().__init__()
        t = int(abs(math.log2(max(channels, 2)) / gamma + b / gamma))
        k = max(t if t % 2 else t + 1, 3)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.alpha = nn.Parameter(torch.tensor(float(init_alpha)))
    def forward(self, x):
        b, c, _, _ = x.shape
        w = self.sigmoid(self.conv(self.gap(x).view(b, 1, c))).view(b, c, 1, 1)
        return x + self.alpha * (x * w - x)

class StochasticDepth(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob
    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        return x * (torch.rand(shape, dtype=x.dtype, device=x.device) < keep).float() / (keep + 1e-8)

class CBAMBlock(nn.Module):
    def __init__(self, channels, reduction=16, spatial_kernel=7, init_alpha=0.01, drop_path=DROP_PATH):
        super().__init__()
        reduced = max(4, channels // reduction)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ch_fc1 = nn.Linear(channels, reduced, bias=False)
        self.ch_fc2 = nn.Linear(reduced, channels, bias=False)
        self.ch_sig = nn.Sigmoid()
        self.sp_conv = nn.Conv2d(2, 1, spatial_kernel, padding=spatial_kernel // 2, bias=False)
        self.sp_sig = nn.Sigmoid()
        self.alpha = nn.Parameter(torch.tensor(float(init_alpha)))
        self.drop = StochasticDepth(drop_path)
    def _ch(self, x):
        b, c, _, _ = x.shape
        mx, av = self.max_pool(x).view(b, c), self.avg_pool(x).view(b, c)
        gate = self.ch_sig(self.ch_fc2(F.relu(self.ch_fc1(mx), inplace=True)) +
                           self.ch_fc2(F.relu(self.ch_fc1(av), inplace=True))).view(b, c, 1, 1)
        return x * gate
    def _sp(self, x):
        sp = torch.cat([x.max(dim=1, keepdim=True)[0], x.mean(dim=1, keepdim=True)], dim=1)
        return x * self.sp_sig(self.sp_conv(sp))
    def forward(self, x):
        return x + self.alpha * self.drop(self._sp(self._ch(x)) - x)

class CrossScaleAttentionHead(nn.Module):
    def __init__(self, dims, d=MSAF_DIM, num_classes=NUM_CLASSES):
        super().__init__()
        self.gem = nn.ModuleList([GeMPool() for _ in dims])
        self.proj = nn.ModuleList([nn.Sequential(nn.Linear(c, d, bias=False), nn.LayerNorm(d)) for c in dims])
        self.q_lin = nn.Linear(d, d, bias=False)
        self.k_lin = nn.Linear(d, d, bias=False)
        self.v_lin = nn.Linear(d, d, bias=False)
        self.scale = d ** -0.5
        self.norm = nn.LayerNorm(d)
        self.temp = nn.Parameter(torch.ones(1))
        self.head = nn.Sequential(nn.Dropout(0.35), nn.Linear(d, d // 2), nn.GELU(),
                                  nn.Dropout(0.15), nn.Linear(d // 2, num_classes))
    def forward(self, feat_list):
        tokens = [self.proj[i](self.gem[i](f)) for i, f in enumerate(feat_list)]
        seq = torch.stack(tokens, dim=1)
        q, k, v = self.q_lin(seq[:, -1:, :]), self.k_lin(seq), self.v_lin(seq)
        attn = torch.softmax(q @ k.transpose(-2, -1) * self.scale, dim=-1)
        fused = self.norm((attn @ v).squeeze(1) + tokens[-1]) * self.temp
        return self.head(fused)

class ConvNeXtV2MSAFv5(nn.Module):
    DIMS = [96, 192, 384, 768]
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        try:
            import timm
            bb = timm.create_model('convnextv2_tiny.fcmae_ft_in22k_in1k', pretrained=False)
            self._backend = 'timm'
            self.stem = bb.stem
            self.stage0, self.stage1, self.stage2, self.stage3 = bb.stages[0], bb.stages[1], bb.stages[2], bb.stages[3]
        except Exception:
            bb = models.convnext_tiny(weights=None)
            self._backend = 'tv'
            self.stem = bb.features[0]
            self.ds1, self.ds2, self.ds3 = bb.features[2], bb.features[4], bb.features[6]
            self.stage0, self.stage1 = bb.features[1], bb.features[3]
            self.stage2, self.stage3 = bb.features[5], bb.features[7]
        dims = self.DIMS
        self.attn0, self.attn1 = ECABlock(dims[0]), ECABlock(dims[1])
        self.attn2 = CBAMBlock(dims[2], reduction=max(4, dims[2] // 16))
        self.attn3 = CBAMBlock(dims[3], reduction=max(4, dims[3] // 16))
        self.head = CrossScaleAttentionHead(dims=[dims[1], dims[2], dims[3]], d=MSAF_DIM, num_classes=num_classes)
    def _stages(self, x):
        if self._backend == 'timm':
            x = self.stem(x)
            s0 = self.attn0(self.stage0(x)); s1 = self.attn1(self.stage1(s0))
            s2 = self.attn2(self.stage2(s1)); s3 = self.attn3(self.stage3(s2))
        else:
            x = self.stem(x)
            s0 = self.attn0(self.stage0(x)); s1 = self.attn1(self.stage1(self.ds1(s0)))
            s2 = self.attn2(self.stage2(self.ds2(s1))); s3 = self.attn3(self.stage3(self.ds3(s2)))
        return s1, s2, s3
    def forward(self, x):
        return self.head(list(self._stages(x)))


# ═══ Pure analysis helpers (numpy only) ═══
def wilson_ci(k, n, z=1.96):
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return centre - half, centre + half

def summarize(y_true, y_pred):
    """Scores the three external classes. Rows of the confusion matrix = true class, columns = all four predicted classes."""
    n = len(y_true)
    k = int((y_true == y_pred).sum())
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=SCORED, average=None, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])[SCORED, :]
    lo, hi = wilson_ci(k, n)
    return {'n': n, 'accuracy': k / n, 'ci_low': lo, 'ci_high': hi,
            'macro_precision': float(p.mean()), 'macro_recall': float(r.mean()), 'macro_f1': float(f.mean()),
            'per_class': {CLASSES[c]: {'precision': float(p[i]), 'recall': float(r[i]), 'f1': float(f[i]),
                                       'support': int(s[i])} for i, c in enumerate(SCORED)},
            'confusion_matrix_rows_true_cols_pred': cm.tolist(),
            'confusion_matrix_columns': CLASSES,
            'healthy_prediction_rate': float((y_pred == 1).mean())}

def build_predictions(P4):
    """P4: (N,4) ensemble softmax. Returns strict, restricted and adjusted predictions in four-class index space."""
    strict = P4.argmax(axis=1)
    P3 = P4[:, SCORED]
    P3 = P3 / P3.sum(axis=1, keepdims=True)
    restricted = np.array(SCORED)[P3.argmax(axis=1)]
    alpha3 = np.array([FOCAL_ALPHA[c] for c in SCORED])
    P3a = P3 / alpha3
    P3a = P3a / P3a.sum(axis=1, keepdims=True)
    adjusted = np.array(SCORED)[P3a.argmax(axis=1)]
    return strict, restricted, adjusted


# ═══ Run ═══
if __name__ == '__main__':
    EVAL_TRANSFORM = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor(),
                                         transforms.Normalize(MEAN, STD)])
    ds = datasets.ImageFolder(EXTERNAL_DIR, transform=EVAL_TRANSFORM)
    if sorted(ds.classes) != sorted(EXTERNAL_CLASSES):
        raise SystemExit(f'External folders are {ds.classes}, expected {EXTERNAL_CLASSES}. Rename the folders or edit EXTERNAL_CLASSES.')
    ext_to4 = np.array([CLASSES.index(c) for c in ds.classes])          # external label -> four-class index
    y4 = ext_to4[np.array(ds.targets)]
    print(f'External images: {len(ds)}  per class: {dict(zip(ds.classes, np.bincount(np.array(ds.targets)).tolist()))}')
    loader = DataLoader(ds, BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    def find_checkpoint(fold):
        exact = os.path.join(MAIN_OUT_DIR, fold, RUN_NAME, 'fl', f'{fold}_{RUN_NAME}_fl_best.pt')
        if os.path.isfile(exact):
            return exact
        hits = glob.glob(os.path.join(MAIN_OUT_DIR, '**', f'{fold}_{RUN_NAME}_fl_best.pt'), recursive=True)
        return hits[0] if hits else None

    probs_per_fold, used = [], []
    for fold in FOLDS:
        ckpt = find_checkpoint(fold)
        if ckpt is None:
            print(f'[warning] no checkpoint found for {fold}, skipping it')
            continue
        model = ConvNeXtV2MSAFv5(NUM_CLASSES).to(DEVICE)
        sd = torch.load(ckpt, map_location=DEVICE)
        sd = {(k[7:] if k.startswith('module.') else k): v for k, v in sd.items()}
        res = model.load_state_dict(sd, strict=False)
        if res.missing_keys:
            raise RuntimeError(f'{fold}: checkpoint is missing weights {res.missing_keys[:5]} ...')
        if res.unexpected_keys:
            print(f'[warning] {fold}: unexpected keys ignored {res.unexpected_keys[:5]}')
        model.eval()
        chunks = []
        with torch.no_grad():
            for imgs, _ in loader:
                chunks.append(torch.softmax(model(imgs.to(DEVICE)), dim=1).cpu().numpy())
        probs_per_fold.append(np.concatenate(chunks))
        used.append(fold)
        print(f'{fold}: loaded {ckpt}')
        del model
        torch.cuda.empty_cache()
    if not probs_per_fold:
        raise SystemExit('No checkpoints found. Check MAIN_OUT_DIR and RUN_NAME.')
    if len(used) < 5:
        print(f'[warning] only {len(used)} of 5 fold checkpoints were available, numbers will not match the paper exactly')

    P4 = np.mean(probs_per_fold, axis=0)
    strict, restricted, adjusted = build_predictions(P4)
    results = {'checkpoints_used': used,
               'strict_four_output': summarize(y4, strict),
               'restricted_uncorrected': summarize(y4, restricted),
               'restricted_score_adjusted': summarize(y4, adjusted)}

    results['healthy_prediction_rate_by_true_class'] = {
        CLASSES[c]: float((strict[y4 == c] == 1).mean()) for c in SCORED}
    results['mean_healthy_probability_by_true_class'] = {
        CLASSES[c]: float(P4[y4 == c, 1].mean()) for c in SCORED}

    print('\n================ EXTERNAL VALIDATION, THREE VARIANTS (same images, same checkpoints) ================')
    print(f'{"Variant":<28}{"Acc":>8}{"95% CI":>20}{"MacroF1":>10}{"Healthy-pred":>14}')
    for key, name in [('strict_four_output', 'Strict four-output'),
                      ('restricted_uncorrected', 'Restricted, uncorrected'),
                      ('restricted_score_adjusted', 'Restricted, score-adjusted')]:
        r = results[key]
        print(f'{name:<28}{r["accuracy"]:>8.3f}   [{r["ci_low"]:.3f}, {r["ci_high"]:.3f}]{r["macro_f1"]:>10.3f}'
              f'{100 * r["healthy_prediction_rate"]:>13.1f}%')
    print('\nShare of external disease images predicted as Healthy (strict), by true class:')
    for c, v in results['healthy_prediction_rate_by_true_class'].items():
        print(f'   {c:<12} {100 * v:5.1f}%   (mean Healthy probability {results["mean_healthy_probability_by_true_class"][c]:.3f})')
    for key in ['strict_four_output', 'restricted_uncorrected', 'restricted_score_adjusted']:
        print(f'\n{key}: per-class precision / recall / F1 and confusion matrix (rows true, columns {CLASSES})')
        for c, v in results[key]['per_class'].items():
            print(f'   {c:<12} P={v["precision"]:.3f} R={v["recall"]:.3f} F1={v["f1"]:.3f} (n={v["support"]})')
        for row_name, row in zip(EXTERNAL_CLASSES, results[key]['confusion_matrix_rows_true_cols_pred']):
            print(f'   {row_name:<12} {row}')

    rr, ra = results['restricted_uncorrected'], results['restricted_score_adjusted']
    ok = (abs(rr['accuracy'] - PAPER['restricted_acc']) < 0.0015 and abs(rr['macro_f1'] - PAPER['restricted_f1']) < 0.0015 and
          abs(ra['accuracy'] - PAPER['adjusted_acc']) < 0.0015 and abs(ra['macro_f1'] - PAPER['adjusted_f1']) < 0.0015)
    print(f'\nReproduction check against the paper (88.1 / 0.878 and 88.8 / 0.886): {"MATCH" if ok else "DIFFERS, tell me the printed numbers before editing the paper"}')

    classes_arr = np.array(CLASSES)
    df = pd.DataFrame({'path': [s[0] for s in ds.samples], 'true_class': classes_arr[y4],
                       'p_Chickenpox': P4[:, 0], 'p_Healthy': P4[:, 1], 'p_Measles': P4[:, 2], 'p_Monkeypox': P4[:, 3],
                       'pred_strict': classes_arr[strict], 'pred_restricted': classes_arr[restricted],
                       'pred_adjusted': classes_arr[adjusted]})
    df.to_csv(os.path.join(OUT_DIR, 'external_predictions.csv'), index=False)
    with open(os.path.join(OUT_DIR, 'external_validation_strict_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved: {OUT_DIR}/external_validation_strict_results.json and external_predictions.csv')
