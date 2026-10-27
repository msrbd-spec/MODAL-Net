#!/usr/bin/env python3
"""
calibration_exp3.py
Calibration for Exp. 3 (Dirichlet label-skew), 5 folds. Inference only, no training.
Same procedure as the Exp. 1 and Exp. 2 calibration run:
  ECE and MCE with 15 bins, Brier score, NLL, post-hoc temperature scaling.
  Temperature: LBFGS (init 1.5, lr 0.01, max_iter 50) on the pooled validation logits.
  Metrics: on the pooled test set of all 5 clients.

BEFORE YOU RUN (mandatory)
1. Keep CHECK_PATHS_ONLY = True and run once. It loads no model. It only prints folder names, image counts,
   and the checkpoint it found for every fold.
2. Check these four things in the printout:
   a. Folder names inside Fold_1 include RUN_NAME exactly as written below.
   b. Valid and Test counts look right (test total around 287 per fold, all 4 classes present).
   c. Every fold shows exactly ONE checkpoint, ending with _best.pt (not _swa.pt).
   d. That checkpoint is the Exp. 3 FedProx model of the SAME fold, trained with the same pipeline as Exp. 1 and 2.
3. If a path is wrong or two checkpoints appear, fix RUN_NAME or CKPT_SEARCH_ROOTS, or fill CKPT_OVERRIDE.
4. Then set CHECK_PATHS_ONLY = False and run again.

Focal alpha is NOT used here (no training, no loss). For the record only:
  TRAIN counts [127, 276, 132, 333]  (used for the final paper models)
  TEST  counts [42, 94, 35, 116]     (old weighting)

Outputs: outputs/calibration_exp3/ (separate from every earlier output). Resumable.
Sized for RTX 4070 Super 12GB (inference only, batch 32).
"""
import os, sys, gc, glob, json, math, warnings, traceback
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets, transforms, models
warnings.filterwarnings('ignore')

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['TORCH_HOME'] = os.path.expanduser('~/.cache/torch')

# ── Configuration ──
NUM_CLIENTS = 5
CLASSES = ['Chickenpox', 'Healthy', 'Measles', 'Monkeypox']
NUM_CLASSES = 4
EXPECTED_CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
DATA_ROOT = 'datasets/final_5_fold_pruned/'
RUN_NAME = 'FL_Run3_LabelSkew_Dirichlet'              # taken from the Figure 4 file names, verify in the path check
FOLDS = [f'Fold_{i}' for i in range(1, 6)]
CKPT_SEARCH_ROOTS = ['pipeline_v2_single_gpu', 'outputs']   # folders searched recursively for *_best.pt
CKPT_OVERRIDE = {}      # manual paths if needed, e.g. {'Fold_1': 'pipeline_v2_single_gpu/Fold_1/.../x_best.pt'}
OUT_DIR = 'outputs/calibration_exp3'
CHECK_PATHS_ONLY = True
REF_TABLE7_CSV = 'pipeline_v2_single_gpu/table7_calibration.csv'   # Exp. 1 and 2 per-fold calibration table (optional)
STD_DDOF = 0            # 0 = np.std (population), 1 = sample std. The reference check below tells which one matches your table.

IMAGE_SIZE = 384
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
BATCH_SIZE = 32
NUM_WORKERS = 4
N_BINS = 15
GEM_P, MSAF_DIM, DROP_PATH = 3.0, 256, 0.10
IMG_EXT = ('.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tif', '.tiff')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

def save_json(obj, path):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2, default=float)

def load_json(path, default=None):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return default if default is not None else {}

# ═══ Model: PoxCSAF-Net (same layer names as the training script, so checkpoint keys match) ═══
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
        noise = torch.rand(shape, dtype=x.dtype, device=x.device) < keep
        return x * noise.float() / (keep + 1e-8)

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
        mx = self.max_pool(x).view(b, c)
        av = self.avg_pool(x).view(b, c)
        gate = self.ch_sig(self.ch_fc2(F.relu(self.ch_fc1(mx), inplace=True)) +
                           self.ch_fc2(F.relu(self.ch_fc1(av), inplace=True))).view(b, c, 1, 1)
        return x * gate
    def _sp(self, x):
        sp = torch.cat([x.max(dim=1, keepdim=True)[0], x.mean(dim=1, keepdim=True)], dim=1)
        return x * self.sp_sig(self.sp_conv(sp))
    def forward(self, x):
        x_attn = self._sp(self._ch(x))
        return x + self.alpha * self.drop(x_attn - x)

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
        tokens = []
        for i, feat in enumerate(feat_list):
            tokens.append(self.proj[i](self.gem[i](feat)))
        seq = torch.stack(tokens, dim=1)
        q = self.q_lin(seq[:, -1:, :])
        k = self.k_lin(seq)
        v = self.v_lin(seq)
        attn = torch.softmax(q @ k.transpose(-2, -1) * self.scale, dim=-1)
        fused = (attn @ v).squeeze(1)
        fused = self.norm(fused + tokens[-1]) * self.temp
        return self.head(fused)

class ConvNeXtV2MSAFv5(nn.Module):
    """Primary model (ECA + CBAM + CSAH, no aux heads). backend must match the checkpoint: 'timm' or 'tv'.
    Pretrained weights are NOT loaded here, because the whole state dict comes from the checkpoint."""
    DIMS = [96, 192, 384, 768]
    def __init__(self, backend, num_classes=NUM_CLASSES):
        super().__init__()
        self._backend = backend
        if backend == 'timm':
            import timm
            bb = timm.create_model('convnextv2_tiny.fcmae_ft_in22k_in1k', pretrained=False)
            self.stem = bb.stem
            self.stage0, self.stage1, self.stage2, self.stage3 = bb.stages[0], bb.stages[1], bb.stages[2], bb.stages[3]
        else:
            bb = models.convnext_tiny(weights=None)
            self.stem = bb.features[0]
            self.ds1, self.ds2, self.ds3 = bb.features[2], bb.features[4], bb.features[6]
            self.stage0, self.stage1 = bb.features[1], bb.features[3]
            self.stage2, self.stage3 = bb.features[5], bb.features[7]
        dims = self.DIMS
        self.attn0, self.attn1 = ECABlock(dims[0]), ECABlock(dims[1])
        self.attn2 = CBAMBlock(dims[2], reduction=max(4, dims[2] // 16), drop_path=DROP_PATH)
        self.attn3 = CBAMBlock(dims[3], reduction=max(4, dims[3] // 16), drop_path=DROP_PATH)
        self.head = CrossScaleAttentionHead(dims=[dims[1], dims[2], dims[3]], d=MSAF_DIM, num_classes=num_classes)

    def forward(self, x):
        if self._backend == 'timm':
            x = self.stem(x)
            s0 = self.attn0(self.stage0(x))
            s1 = self.attn1(self.stage1(s0))
            s2 = self.attn2(self.stage2(s1))
            s3 = self.attn3(self.stage3(s2))
        else:
            x = self.stem(x)
            s0 = self.attn0(self.stage0(x))
            s1 = self.attn1(self.stage1(self.ds1(s0)))
            s2 = self.attn2(self.stage2(self.ds2(s1)))
            s3 = self.attn3(self.stage3(self.ds3(s2)))
        return self.head([s1, s2, s3])

# ═══ Checkpoints ═══
def find_ckpts(fold):
    """All candidate best checkpoints of this fold and run (SWA files excluded)."""
    if fold in CKPT_OVERRIDE:
        return [CKPT_OVERRIDE[fold]]
    hits = []
    for root in CKPT_SEARCH_ROOTS:
        if not os.path.isdir(root):
            continue
        for p in glob.glob(os.path.join(root, '**', '*.pt'), recursive=True):
            norm = p.replace('\\', '/')
            name = os.path.basename(p).lower()
            if RUN_NAME in norm and fold in norm and name.endswith('_best.pt') and 'swa' not in name:
                hits.append(p)
    return sorted(set(hits))

def resolve_ckpt(fold):
    hits = find_ckpts(fold)
    if len(hits) == 0:
        raise FileNotFoundError(f'No checkpoint found for {fold} / {RUN_NAME}. Fix RUN_NAME, CKPT_SEARCH_ROOTS or CKPT_OVERRIDE.')
    if len(hits) > 1:
        raise RuntimeError(f'{len(hits)} checkpoints found for {fold}: {hits}. Put the right one in CKPT_OVERRIDE.')
    return hits[0]

def load_state(path):
    sd = torch.load(path, map_location='cpu')
    if isinstance(sd, dict) and 'state_dict' in sd:
        sd = sd['state_dict']
    if any(k.startswith('module.') for k in sd):
        sd = {k.replace('module.', '', 1): v for k, v in sd.items()}
    return sd

def build_model_from_ckpt(path):
    sd = load_state(path)
    backend = 'tv' if any(k.startswith('ds1.') for k in sd) else 'timm'
    model = ConvNeXtV2MSAFv5(backend)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(f'Checkpoint does not match the model. missing={list(missing)[:5]} '
                           f'unexpected={list(unexpected)[:5]} (backend guessed: {backend}).')
    return model.to(DEVICE).eval(), backend

# ═══ Data: pooled Valid and Test over the 5 clients (Train is never loaded) ═══
def make_loader(fold, split):
    sets = []
    for c in range(1, NUM_CLIENTS + 1):
        d = os.path.join(DATA_ROOT, fold, RUN_NAME, f'Client_{c}', split)
        ds = datasets.ImageFolder(d, transform=EVAL_TRANSFORM)
        if ds.class_to_idx != EXPECTED_CLASS_TO_IDX:
            raise RuntimeError(f'class_to_idx mismatch in {d}: {ds.class_to_idx}')
        sets.append(ds)
    pooled = ConcatDataset(sets)
    return DataLoader(pooled, BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True), len(pooled)

@torch.no_grad()
def get_logits(model, loader):
    model.eval()
    logits, labels = [], []
    for imgs, y in loader:
        out = model(imgs.to(DEVICE, non_blocking=True))
        if isinstance(out, tuple):
            out = out[0]
        logits.append(out.float().cpu())
        labels.append(y)
    return torch.cat(logits), torch.cat(labels)

# ═══ Calibration ═══
def fit_temperature(val_logits, val_labels):
    """Same procedure as Exp. 1 and 2: LBFGS on validation NLL, init 1.5, lr 0.01, max_iter 50."""
    temperature = nn.Parameter(torch.ones(1, device=DEVICE) * 1.5)
    opt = optim.LBFGS([temperature], lr=0.01, max_iter=50)
    logits, labels = val_logits.to(DEVICE), val_labels.to(DEVICE)
    nll = nn.CrossEntropyLoss()
    def closure():
        opt.zero_grad()
        loss = nll(logits / temperature, labels)
        loss.backward()
        return loss
    opt.step(closure)
    temperature.data.clamp_(min=0.001)
    return float(temperature.item())

def calib_metrics(logits, labels, T=1.0, n_bins=N_BINS):
    probs = torch.softmax(logits / T, dim=1).numpy()
    y = labels.numpy()
    N = len(y)
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    edges = np.linspace(0, 1, n_bins + 1)
    ece, mce, bins = 0.0, 0.0, []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (conf > lo) & (conf <= hi)
        cnt = int(m.sum())
        center = (lo + hi) / 2
        if cnt > 0:
            acc_b = float((pred[m] == y[m]).mean())
            conf_b = float(conf[m].mean())
            gap = abs(acc_b - conf_b)
            ece += (cnt / N) * gap
            mce = max(mce, gap)
            bins.append({'bin_center': float(center), 'accuracy': acc_b, 'confidence': conf_b, 'count': cnt, 'gap': float(gap)})
        else:
            bins.append({'bin_center': float(center), 'accuracy': None, 'confidence': None, 'count': 0, 'gap': 0.0})
    one_hot = np.zeros((N, NUM_CLASSES))
    one_hot[np.arange(N), y] = 1.0
    brier = float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))
    nll = float(-np.mean(np.log(probs[np.arange(N), y] + 1e-12)))
    return {'ece': float(ece), 'mce': float(mce), 'brier_score': brier, 'nll': nll,
            'accuracy': float((pred == y).mean()), 'n_samples': int(N), 'bin_data': bins}

def plot_reliability(bins, title, path):
    fig, ax = plt.subplots(figsize=(6, 6))
    centers = [b['bin_center'] for b in bins]
    accs = [b['accuracy'] if b['accuracy'] is not None else 0 for b in bins]
    gaps = [b['gap'] if b['gap'] is not None else 0 for b in bins]
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Perfect')
    ax.bar(centers, accs, width=1 / len(bins) * 0.8, alpha=0.7, color='#0066CC', label='Accuracy')
    ax.bar(centers, gaps, width=1 / len(bins) * 0.8, bottom=accs, alpha=0.3, color='#CC0000', label='Gap')
    ax.set_xlabel('Confidence')
    ax.set_ylabel('Accuracy')
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc='upper left')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

# ═══ Path check ═══
def count_images(d):
    out = {}
    for cls in CLASSES:
        p = os.path.join(d, cls)
        out[cls] = len([f for f in os.listdir(p) if f.lower().endswith(IMG_EXT)]) if os.path.isdir(p) else 0
    return out

def path_check():
    print('=' * 70)
    print('PATH CHECK (no model is loaded)')
    print('=' * 70)
    print(f'DATA_ROOT = {os.path.abspath(DATA_ROOT)}  exists={os.path.isdir(DATA_ROOT)}')
    f1 = os.path.join(DATA_ROOT, FOLDS[0])
    if os.path.isdir(f1):
        print(f'Folders inside {FOLDS[0]}: {sorted(os.listdir(f1))}')
    print(f'RUN_NAME  = {RUN_NAME}')
    print(f'Checkpoint search roots = {CKPT_SEARCH_ROOTS}')
    all_ok = True
    for fold in FOLDS:
        print(f'\n--- {fold} ---')
        run_dir = os.path.join(DATA_ROOT, fold, RUN_NAME)
        if not os.path.isdir(run_dir):
            print(f'  [MISSING] {run_dir}')
            all_ok = False
        else:
            for split in ['Valid', 'Test']:
                tot = {c: 0 for c in CLASSES}
                for c in range(1, NUM_CLIENTS + 1):
                    d = os.path.join(run_dir, f'Client_{c}', split)
                    if not os.path.isdir(d):
                        print(f'  [MISSING] {d}')
                        all_ok = False
                        continue
                    for cls, n in count_images(d).items():
                        tot[cls] += n
                print(f'  {split}: total={sum(tot.values())} per class={tot}')
                if any(v == 0 for v in tot.values()):
                    print(f'  [WARNING] a class has zero images in {split}')
                    all_ok = False
        hits = find_ckpts(fold)
        if len(hits) == 1:
            print(f'  [OK] checkpoint: {hits[0]}')
        elif len(hits) == 0:
            print('  [MISSING] no checkpoint found')
            all_ok = False
        else:
            print(f'  [AMBIGUOUS] {len(hits)} checkpoints:')
            for h in hits:
                print(f'      {h}')
            all_ok = False
    print('\n' + ('All checks passed. Verify a to d from the header, then set CHECK_PATHS_ONLY = False.'
                  if all_ok else 'Some checks failed. Fix the paths above and run the check again.'))

def reference_check():
    """Prints the std of the Exp. 1 ECA_uncal column both ways, so you can see which std your table used."""
    if not os.path.exists(REF_TABLE7_CSV):
        print(f'[reference check skipped] {REF_TABLE7_CSV} not found. Set STD_DDOF by hand if needed.')
        return
    ref = pd.read_csv(REF_TABLE7_CSV)
    for run in sorted(ref['Run'].unique()):
        v = pd.to_numeric(ref[ref['Run'] == run]['ECE_uncal'], errors='coerce').dropna().values
        if len(v) > 1:
            print(f'[reference] {run}: ECE_uncal mean={v.mean():.3f}  std(ddof=0)={v.std(ddof=0):.3f}  std(ddof=1)={v.std(ddof=1):.3f}')
    print(f'[reference] The paper table shows 0.048 +- 0.011 (Exp. 1) and 0.047 +- 0.016 (Exp. 2). '
          f'Use the ddof whose std matches, current STD_DDOF = {STD_DDOF}.')

# ═══ Main ═══
def main():
    if CHECK_PATHS_ONLY:
        path_check()
        return
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f'Device: {DEVICE}')
    master = os.path.join(OUT_DIR, 'calibration_results_exp3.json')
    results = load_json(master, default={})

    for fold in FOLDS:
        if fold in results and 'uncalibrated' in results[fold]:
            print(f'[skip, already done] {fold}')
            continue
        print(f'\n--- Calibration: {RUN_NAME} / {fold} ---')
        try:
            ckpt = resolve_ckpt(fold)
            print(f'  checkpoint: {ckpt}')
            model, backend = build_model_from_ckpt(ckpt)
            val_loader, n_val = make_loader(fold, 'Valid')
            test_loader, n_test = make_loader(fold, 'Test')
            print(f'  backend={backend}  valid={n_val}  test={n_test}')
            val_logits, val_labels = get_logits(model, val_loader)
            test_logits, test_labels = get_logits(model, test_loader)
            T = fit_temperature(val_logits, val_labels)
            uncal = calib_metrics(test_logits, test_labels, T=1.0)
            cal = calib_metrics(test_logits, test_labels, T=T)
            print(f'  T={T:.4f}')
            print(f'  Uncalibrated: ECE={uncal["ece"]:.4f} MCE={uncal["mce"]:.4f} Brier={uncal["brier_score"]:.4f} NLL={uncal["nll"]:.4f}')
            print(f'  Calibrated:   ECE={cal["ece"]:.4f} MCE={cal["mce"]:.4f} Brier={cal["brier_score"]:.4f} NLL={cal["nll"]:.4f}')
            plot_reliability(uncal['bin_data'], f'Reliability: {fold} {RUN_NAME} (uncalibrated)',
                             os.path.join(OUT_DIR, f'reliability_{fold}_{RUN_NAME}_uncal.png'))
            plot_reliability(cal['bin_data'], f'Reliability: {fold} {RUN_NAME} (calibrated)',
                             os.path.join(OUT_DIR, f'reliability_{fold}_{RUN_NAME}_cal.png'))
            results[fold] = {'checkpoint': ckpt, 'backend': backend, 'temperature': T,
                             'uncalibrated': {k: v for k, v in uncal.items() if k != 'bin_data'},
                             'calibrated': {k: v for k, v in cal.items() if k != 'bin_data'}}
            save_json(results, master)
            del model
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as e:
            print(f'ERROR on {fold}: {e}')
            traceback.print_exc()
            results[fold] = {'error': str(e)}
            save_json(results, master)

    # ── Per-fold table ──
    rows = []
    for fold in FOLDS:
        r = results.get(fold, {})
        if 'uncalibrated' not in r:
            continue
        u, c = r['uncalibrated'], r['calibrated']
        rows.append({'Run': RUN_NAME, 'Fold': fold,
                     'ECE_uncal': u['ece'], 'ECE_cal': c['ece'],
                     'MCE_uncal': u['mce'], 'MCE_cal': c['mce'],
                     'Brier_uncal': u['brier_score'], 'Brier_cal': c['brier_score'],
                     'NLL_uncal': u['nll'], 'NLL_cal': c['nll'],
                     'Temperature': r['temperature']})
    df = pd.DataFrame(rows)
    if len(df) == 0:
        print('\nNo fold finished. Nothing to aggregate.')
        return
    print('\nPer-fold calibration (Exp. 3):')
    print(df.to_string(index=False))
    df.to_csv(os.path.join(OUT_DIR, 'table7_calibration_exp3.csv'), index=False)

    # ── Aggregate: mean and std across folds, plus the Exp. 3 column for tab:calibration ──
    metric_cols = ['ECE_uncal', 'ECE_cal', 'MCE_uncal', 'MCE_cal', 'Brier_uncal', 'Brier_cal', 'NLL_uncal', 'NLL_cal', 'Temperature']
    agg = pd.DataFrame({'mean': df[metric_cols].mean(), 'std': df[metric_cols].std(ddof=STD_DDOF)})
    agg['n_folds'] = len(df)
    print(f'\nAggregate over {len(df)} fold(s), std ddof={STD_DDOF}:')
    print(agg.to_string())
    agg.to_csv(os.path.join(OUT_DIR, 'table7_calibration_exp3_aggregate.csv'))

    labels = {'ECE_uncal': r'ECE$_{\text{uncal}}$', 'ECE_cal': r'ECE$_{\text{cal}}$  ',
              'MCE_uncal': r'MCE$_{\text{uncal}}$', 'MCE_cal': r'MCE$_{\text{cal}}$  ',
              'Brier_uncal': r'Brier$_{\text{uncal}}$', 'Brier_cal': r'Brier$_{\text{cal}}$  ',
              'NLL_uncal': r'NLL$_{\text{uncal}}$', 'NLL_cal': r'NLL$_{\text{cal}}$  ',
              'Temperature': r'Temperature $T$'}
    lines = []
    for col in metric_cols:
        lines.append('{} & ${:.3f} \\pm {:.3f}$'.format(labels[col], agg.loc[col, 'mean'], agg.loc[col, 'std']))
    with open(os.path.join(OUT_DIR, 'exp3_column_for_latex.txt'), 'w') as f:
        f.write('\n'.join(lines))
    print('\nExp. 3 values in table order (also saved to exp3_column_for_latex.txt):')
    for ln in lines:
        print('  ' + ln)

    # ── ECE before and after, per fold ──
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(df))
    w = 0.35
    ax.bar(x - w / 2, df['ECE_uncal'], w, label='ECE (uncalibrated)', color='#CC0000', alpha=0.85)
    ax.bar(x + w / 2, df['ECE_cal'], w, label='ECE (calibrated)', color='#0066CC', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(df['Fold'], rotation=0)
    ax.set_ylabel('ECE (lower is better)')
    ax.set_title('Expected calibration error before and after temperature scaling (Exp. 3)')
    ax.legend()
    ax.grid(axis='y', alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'fig_reliability_exp3.png'), dpi=300)
    plt.close()

    reference_check()
    print(f'\nDone. Outputs in {OUT_DIR}')

if __name__ == '__main__':
    main()
