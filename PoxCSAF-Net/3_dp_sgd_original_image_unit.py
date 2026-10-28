#!/usr/bin/env python3
"""
dp_sgd_original_image_unit.py  (finding 03)

WHAT WAS WRONG
In the earlier DP runs every client trained on its Train folder as it sits on disk, which contains the raw photographs
plus offline augmented copies (aug_*.jpg). Opacus sampled all of these files as independent records, so the reported
epsilon protects one augmented record, not one original photograph. One photograph appears roughly 6 to 14 times
(offline multiplicity), so the guarantee for the photograph itself is far weaker (group privacy).

WHAT THIS SCRIPT DOES
1. Privacy unit = one original photograph. Each client's private training set contains ONLY its unique raw images
   (offline augmented copies are excluded by file name).
2. Augmentation is applied online inside the data loader. A sampled image is turned into ONE random view per step, so an
   original image contributes at most one clipped gradient per step and the standard subsampled Gaussian accounting
   applies with Poisson sampling over unique originals.
3. One persistent PrivacyEngine per client (cumulative RDP accounting over all rounds), noise multiplier calibrated once
   for the full 50 x 3 horizon, achieved epsilon read from the accountant. Per-client and worst-case (max) epsilon are saved.
4. The no-DP reference uses the identical data pipeline (unique originals plus the same online augmentation), so the
   privacy-utility comparison is fair.
Patient-level privacy is NOT claimed because the datasets contain no patient identifiers.

Run from the folder that contains datasets/.   pip install opacus
Sized for RTX 4070 Super 12GB: BATCH_SIZE=16, NUM_WORKERS=4. The script is resumable via dp_results_orig_unit.json.
"""
import os, sys, gc, copy, json, math, random, warnings, traceback
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets, transforms, models
import torchvision
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, roc_curve, auc,
                             f1_score, precision_score, recall_score)
from sklearn.preprocessing import label_binarize
warnings.filterwarnings('ignore')

try:
    from opacus import PrivacyEngine
    from opacus.validators import ModuleValidator
    from opacus.accountants.utils import get_noise_multiplier
except ImportError:
    raise SystemExit('Opacus is required. Run: pip install opacus')

_TV = tuple(int(x) for x in torchvision.__version__.split('+')[0].split('.')[:2])
if _TV < (0, 16):
    raise SystemExit(f'torchvision {torchvision.__version__} is too old for allow_empty. Run: pip install -U torchvision')

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['TORCH_HOME'] = os.path.expanduser('~/.cache/torch')

SEED = 42
def seed_everything(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
seed_everything(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')
if torch.cuda.is_available():
    print(f'GPU   : {torch.cuda.get_device_name(0)}')

# ── Configuration (same as the earlier DP experiment, only the data pipeline changes) ──
NUM_CLIENTS = 5
CLASSES     = ['Chickenpox', 'Healthy', 'Measles', 'Monkeypox']
NUM_CLASSES = 4
DATA_ROOT   = 'datasets/final_5_fold_pruned/'
DP_RUN_NAME = 'FL_Run2_Heterogeneous'
DP_FOLDS    = ['Fold_1', 'Fold_3', 'Fold_5']
DP_EPSILONS = [None, 12.0, 8.0]          # None = non-private reference
TARGET_DELTA  = 1e-5
MAX_GRAD_NORM = 2.0
ACCOUNTANT    = 'rdp'
EPS_TOLERANCE = 0.01

OUT_DIR = 'outputs/dp_sgd_original_image_unit'
os.makedirs(OUT_DIR, exist_ok=True)

IMAGE_SIZE  = 384
MEAN, STD   = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
BATCH_SIZE  = 16
NUM_WORKERS = 4
LR_BACKBONE, LR_ATTN, LR_HEAD = 1e-5, 3e-4, 1e-3
WEIGHT_DECAY = 2e-4
MSAF_DIM, GEM_P, DROP_PATH = 256, 3.0, 0.10
FOCAL_GAMMA = 2.0
TEST_COUNTS = np.array([42.0, 94.0, 35.0, 116.0])
_inv = 1.0 / (TEST_COUNTS + 1e-6)
FOCAL_ALPHA = (_inv / _inv.sum()).tolist()
FL_ROUNDS, LOCAL_EPOCHS, PATIENCE = 100, 3, 20
IMG_EXT = ('.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tif', '.tiff')

PRIVACY_UNIT = ('one original photograph (unique pre-augmentation image). Offline augmented copies are excluded, '
                'augmentation is online with one random view per sampled image per step, sampling is Poisson over '
                'unique originals. Not patient-level, since no patient identifiers exist.')

def save_json(obj, path):
    d = os.path.dirname(path)
    if d: os.makedirs(d, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2, default=float)

def load_json(path, default=None):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return default if default is not None else {}

matplotlib.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 7, 'xtick.labelsize': 6, 'ytick.labelsize': 6,
    'legend.fontsize': 6, 'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02, 'axes.linewidth': 0.5, 'lines.linewidth': 1.0, 'lines.markersize': 3,
})

# ═══ Model (Opacus-safe: GeM, projection and stacking happen in the parent forward) ═══
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
    """Receives one stacked (B,3,d) tensor. GeM pooling and projection run in the parent forward."""
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
    def forward(self, token_seq):
        q = self.q_lin(token_seq[:, -1:, :])
        k = self.k_lin(token_seq)
        v = self.v_lin(token_seq)
        attn = torch.softmax(q @ k.transpose(-2, -1) * self.scale, dim=-1)
        fused = (attn @ v).squeeze(1)
        fused = self.norm(fused + token_seq[:, -1, :]) * self.temp
        return self.head(fused)

class ConvNeXtV2MSAFv5(nn.Module):
    DIMS = [96, 192, 384, 768]
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        try:
            import timm
            bb = timm.create_model('convnextv2_tiny.fcmae_ft_in22k_in1k', pretrained=True)
            self._backend = 'timm'
            self.stem = bb.stem
            self.stage0, self.stage1, self.stage2, self.stage3 = bb.stages[0], bb.stages[1], bb.stages[2], bb.stages[3]
        except Exception:
            bb = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
            self._backend = 'tv'
            self.stem = bb.features[0]
            self.ds1, self.ds2, self.ds3 = bb.features[2], bb.features[4], bb.features[6]
            self.stage0, self.stage1 = bb.features[1], bb.features[3]
            self.stage2, self.stage3 = bb.features[5], bb.features[7]
        dims = self.DIMS
        self.attn0, self.attn1 = ECABlock(dims[0]), ECABlock(dims[1])
        self.attn2 = CBAMBlock(dims[2], reduction=max(4, dims[2] // 16), drop_path=DROP_PATH)
        self.attn3 = CBAMBlock(dims[3], reduction=max(4, dims[3] // 16), drop_path=DROP_PATH)
        self.head = CrossScaleAttentionHead(dims=[dims[1], dims[2], dims[3]], d=MSAF_DIM, num_classes=num_classes)

    def _stages(self, x):
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
        return s1, s2, s3

    def forward(self, x):
        s1, s2, s3 = self._stages(x)
        tokens = [self.head.proj[i](self.head.gem[i](f)) for i, f in enumerate((s1, s2, s3))]
        return self.head(torch.stack(tokens, dim=1))

    def get_param_groups(self):
        bb = {'stem', 'stage0', 'stage1', 'stage2', 'stage3', 'ds1', 'ds2', 'ds3'}
        atn = {'attn0', 'attn1', 'attn2', 'attn3'}
        bp, ap, hp = [], [], []
        for name, param in self.named_parameters():
            top = name.split('.')[0]
            if top in bb:    bp.append(param)
            elif top in atn: ap.append(param)
            else:            hp.append(param)
        return [{'params': bp, 'lr': LR_BACKBONE, 'name': 'backbone'},
                {'params': ap, 'lr': LR_ATTN, 'name': 'attention'},
                {'params': hp, 'lr': LR_HEAD, 'name': 'head'}]

def build_primary():
    return ConvNeXtV2MSAFv5(NUM_CLASSES)

_m = build_primary().cpu().eval()
with torch.no_grad():
    assert _m(torch.zeros(2, 3, IMAGE_SIZE, IMAGE_SIZE)).shape == (2, NUM_CLASSES)
print(f'Smoke test passed. Parameters: {sum(p.numel() for p in _m.parameters()) / 1e6:.2f}M')
del _m

# ═══ Loss, early stopping, metrics ═══
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=FOCAL_GAMMA, num_classes=NUM_CLASSES):
        super().__init__()
        self.gamma = gamma
        self.alpha = (torch.ones(num_classes) / num_classes) if alpha is None else torch.tensor(alpha, dtype=torch.float32)
    def forward(self, pred, target):
        alpha = self.alpha.to(pred.device)
        log_prob = F.log_softmax(pred, dim=1)
        prob = log_prob.exp()
        focal_w = (1 - prob) ** self.gamma
        idx = torch.arange(len(target), device=pred.device)
        return -(alpha[target] * focal_w[idx, target] * log_prob[idx, target]).mean()

class EarlyStopping:
    def __init__(self, patience=PATIENCE, checkpoint_path='best.pt', min_delta=5e-5):
        self.patience, self.checkpoint, self.min_delta = patience, checkpoint_path, min_delta
        self.counter, self.best, self.stop = 0, None, False
    def step(self, score, model):
        if self.best is None or score > self.best + self.min_delta:
            self.best, self.counter = score, 0
            torch.save(model.state_dict(), self.checkpoint)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True

def compute_epoch_metrics(model, loader, criterion):
    model.eval()
    ls, ap, al = 0.0, [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            out = model(imgs)
            ls += criterion(out, labels).item() * imgs.size(0)
            ap.extend(out.argmax(1).cpu().numpy())
            al.extend(labels.cpu().numpy())
    acc = float((np.array(ap) == np.array(al)).mean())
    prec = float(precision_score(al, ap, average='macro', zero_division=0))
    rec = float(recall_score(al, ap, average='macro', zero_division=0))
    f1 = float(f1_score(al, ap, average='macro', zero_division=0))
    return ls / len(al), acc, prec, rec, f1

# ═══ Data: unique originals for training, online augmentation ═══
DP_TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomApply([transforms.RandomAffine(degrees=15, translate=(0.05, 0.05), scale=(0.9, 1.1))], p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.3),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

def _is_raw_image(path):
    """True for original photographs, False for offline augmented copies (aug_*.jpg)."""
    name = os.path.basename(path)
    return name.lower().endswith(IMG_EXT) and not name.startswith('aug_')

def audit_client_train_sets(fold_dir, run_dir_name):
    """Counts raw originals and offline augmented copies per client and class (the offline multiplicity)."""
    rows = []
    for c in range(1, NUM_CLIENTS + 1):
        per_class, raw_tot, aug_tot = {}, 0, 0
        for cls in CLASSES:
            d = os.path.join(fold_dir, run_dir_name, f'Client_{c}', 'Train', cls)
            names = [f for f in os.listdir(d) if f.lower().endswith(IMG_EXT)] if os.path.isdir(d) else []
            n_aug = sum(f.startswith('aug_') for f in names)
            n_raw = len(names) - n_aug
            per_class[cls] = {'raw': n_raw, 'offline_aug': n_aug,
                              'multiplicity': round(len(names) / n_raw, 2) if n_raw else None}
            raw_tot += n_raw
            aug_tot += n_aug
        if raw_tot == 0:
            raise RuntimeError(f'Client_{c} has no raw images under {fold_dir}/{run_dir_name}. '
                               f'Check the aug_ naming convention.')
        rows.append({'client': c, 'raw_total': raw_tot, 'offline_aug_total': aug_tot,
                     'augmented_set_size': raw_tot + aug_tot,
                     'mean_multiplicity': round((raw_tot + aug_tot) / raw_tot, 2), 'per_class': per_class})
    return rows

def get_client_dataloaders(fold_dir, run_dir_name, client_id):
    base = os.path.join(fold_dir, run_dir_name, f'Client_{client_id}')
    tr_ds = datasets.ImageFolder(os.path.join(base, 'Train'), transform=DP_TRAIN_TRANSFORM,
                                 is_valid_file=_is_raw_image, allow_empty=True)
    vl_ds = datasets.ImageFolder(os.path.join(base, 'Valid'), transform=EVAL_TRANSFORM, allow_empty=True)
    te_ds = datasets.ImageFolder(os.path.join(base, 'Test'), transform=EVAL_TRANSFORM, allow_empty=True)
    expected = {c: i for i, c in enumerate(CLASSES)}
    for name, ds in [('Train', tr_ds), ('Valid', vl_ds), ('Test', te_ds)]:
        if ds.class_to_idx != expected:
            raise RuntimeError(f'class_to_idx mismatch in Client_{client_id}/{name}: {ds.class_to_idx}')
    return (DataLoader(tr_ds, BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True),
            DataLoader(vl_ds, BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True),
            DataLoader(te_ds, BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True))

def get_agg_test_loader(fold_dir, run_dir_name):
    te_list = [get_client_dataloaders(fold_dir, run_dir_name, c)[2].dataset for c in range(1, NUM_CLIENTS + 1)]
    return DataLoader(ConcatDataset(te_list), BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

# ═══ Federated training helpers ═══
def _fedavg(gm, lms, sizes):
    total = sum(sizes)
    w = [n / total for n in sizes]
    gd = gm.state_dict()
    for k in gd:
        gd[k] = sum(w[i] * lms[i][k].float() for i in range(len(lms)))
    gm.load_state_dict(gd)
    return gm

def _identity_forward(x):
    return x

def _disable_random_ops_for_dp(model):
    """Opacus per-sample gradients cannot trace batch-level random masks, so Dropout and stochastic depth are disabled in DP runs."""
    for _, module in model.named_modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.0
            module.forward = _identity_forward
        if type(module).__name__ == 'StochasticDepth':
            module.drop_prob = 0.0
            module.forward = _identity_forward
    return model

def calibrate_client_noise_multipliers(tr_loaders, target_epsilon):
    """One fixed sigma per client for the full R x E horizon, sample rate 1/len(loader) over UNIQUE originals."""
    sigmas, audit = [], []
    for ci, loader in enumerate(tr_loaders):
        n_k = len(loader.dataset)
        bpe = len(loader)
        q = 1.0 / bpe
        steps = int(FL_ROUNDS * LOCAL_EPOCHS * bpe)
        sigma = get_noise_multiplier(target_epsilon=target_epsilon, target_delta=TARGET_DELTA, sample_rate=q,
                                     steps=steps, accountant=ACCOUNTANT, epsilon_tolerance=EPS_TOLERANCE)
        sigmas.append(sigma)
        audit.append({'client': ci + 1, 'n_unique_images': n_k, 'batches_per_epoch': bpe, 'sample_rate': q,
                      'total_steps_budgeted': steps, 'sigma': sigma,
                      'target_epsilon': target_epsilon, 'target_delta': TARGET_DELTA})
        print(f'  [calib] Client_{ci + 1}: unique images={n_k} batches/ep={bpe} q={q:.4f} steps={steps} sigma={sigma:.3f}')
    return sigmas, audit

def _local_update_plain(global_model, client_loader, focal_loss):
    local_model = copy.deepcopy(global_model).to(DEVICE)
    opt = optim.AdamW(local_model.get_param_groups(), weight_decay=WEIGHT_DECAY)
    local_model.train()
    total, preds, labs = 0.0, [], []
    for _ in range(LOCAL_EPOCHS):
        ep = 0.0
        for imgs, labels in client_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            opt.zero_grad()
            out = local_model(imgs)
            loss = focal_loss(out, labels)
            loss.backward()
            opt.step()
            preds.extend(out.argmax(1).detach().cpu().numpy())
            labs.extend(labels.cpu().numpy())
            ep += loss.item()
        total += ep
    acc = float((np.array(preds) == np.array(labs)).mean()) if preds else 0.0
    return local_model.cpu(), total / LOCAL_EPOCHS, acc

def _dp_local_update_persistent(privacy_engine, global_model, client_loader, focal_loss, noise_multiplier):
    local_model = copy.deepcopy(global_model).to(DEVICE)
    errors = ModuleValidator.validate(local_model, strict=False)
    if errors:
        local_model = ModuleValidator.fix(local_model)
    local_model = _disable_random_ops_for_dp(local_model)
    opt = optim.AdamW(local_model.get_param_groups(), weight_decay=WEIGHT_DECAY)
    local_model.train()
    # client_loader must be the ORIGINAL plain loader. The same persisted engine composes privacy across rounds.
    dp_model, dp_opt, dp_loader = privacy_engine.make_private(
        module=local_model, optimizer=opt, data_loader=client_loader,
        noise_multiplier=noise_multiplier, max_grad_norm=MAX_GRAD_NORM, poisson_sampling=True)
    total, preds, labs = 0.0, [], []
    for _ in range(LOCAL_EPOCHS):
        ep = 0.0
        for imgs, labels in dp_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            dp_opt.zero_grad()
            out = dp_model(imgs)
            loss = focal_loss(out, labels)
            loss.backward()
            dp_opt.step()
            preds.extend(out.argmax(1).detach().cpu().numpy())
            labs.extend(labels.cpu().numpy())
            ep += loss.item()
        total += ep
    trained = dp_model._module if hasattr(dp_model, '_module') else dp_model
    acc = float((np.array(preds) == np.array(labs)).mean()) if preds else 0.0
    return trained.cpu(), total / LOCAL_EPOCHS, acc

def train_fedavg_dp(fold_dir, save_dir, run_name, target_epsilon):
    os.makedirs(save_dir, exist_ok=True)
    ckpt = os.path.join(save_dir, f'{run_name}_best.pt')
    focal_loss = FocalLoss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA)
    ce = nn.CrossEntropyLoss()
    tr_loaders, vl_loaders, sizes = [], [], []
    for c in range(1, NUM_CLIENTS + 1):
        tr, vl, _ = get_client_dataloaders(fold_dir, DP_RUN_NAME, c)
        tr_loaders.append(tr); vl_loaders.append(vl); sizes.append(len(tr.dataset))
    agg_vl = DataLoader(ConcatDataset([l.dataset for l in vl_loaders]), BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True)
    use_dp = target_epsilon is not None
    engines, sigmas, calib_audit = None, None, None
    if use_dp:
        sigmas, calib_audit = calibrate_client_noise_multipliers(tr_loaders, target_epsilon)
        engines = [PrivacyEngine(accountant=ACCOUNTANT) for _ in range(NUM_CLIENTS)]
    gm = build_primary().to(DEVICE)
    stopper = EarlyStopping(PATIENCE, ckpt)
    history = {k: [] for k in ['round', 'avg_local_loss', 'avg_local_acc', 'global_val_loss', 'global_val_acc',
                               'global_val_prec', 'global_val_rec', 'global_val_f1',
                               'epsilon_mean', 'epsilon_max', 'epsilon_per_client']}
    print(f'FedAvg | {"DP target eps=" + str(target_epsilon) if use_dp else "no DP"} | unique images per client {sizes}')
    for rnd in range(1, FL_ROUNDS + 1):
        lms, lls, las = [], [], []
        for ci in range(NUM_CLIENTS):
            if use_dp:
                lm, ll, la = _dp_local_update_persistent(engines[ci], gm, tr_loaders[ci], focal_loss, sigmas[ci])
            else:
                lm, ll, la = _local_update_plain(gm, tr_loaders[ci], focal_loss)
            lms.append({k: v.cpu() for k, v in lm.state_dict().items()})
            lls.append(ll); las.append(la)
            del lm
            gc.collect()
        gm = _fedavg(gm, lms, sizes).to(DEVICE)
        vl_loss, vl_acc, vl_prec, vl_rec, vl_f1 = compute_epoch_metrics(gm, agg_vl, ce)
        for k, v in zip(['round', 'avg_local_loss', 'avg_local_acc', 'global_val_loss', 'global_val_acc',
                         'global_val_prec', 'global_val_rec', 'global_val_f1'],
                        [rnd, float(np.mean(lls)), float(np.mean(las)), vl_loss, vl_acc, vl_prec, vl_rec, vl_f1]):
            history[k].append(v)
        eps_now = [engines[ci].get_epsilon(TARGET_DELTA) for ci in range(NUM_CLIENTS)] if use_dp else [0.0] * NUM_CLIENTS
        history['epsilon_per_client'].append(eps_now)
        history['epsilon_mean'].append(float(np.mean(eps_now)))
        history['epsilon_max'].append(float(np.max(eps_now)))
        stopper.step(vl_f1, gm)
        if rnd % 5 == 0 or stopper.stop:
            e = f'eps(mean/max)={history["epsilon_mean"][-1]:.3f}/{history["epsilon_max"][-1]:.3f}' if use_dp else 'eps=n/a'
            print(f'  Round {rnd:3d}/{FL_ROUNDS} | A={vl_acc:.4f} F1={vl_f1:.4f} | {e} | ES={stopper.counter}/{PATIENCE}')
        if stopper.stop:
            print(f'  Early stopping at round {rnd}.')
            break
        torch.cuda.empty_cache()
    gm.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    summary = None
    if use_dp:
        final = history['epsilon_per_client'][-1]
        summary = {'privacy_unit': PRIVACY_UNIT, 'target_epsilon': target_epsilon, 'target_delta': TARGET_DELTA,
                   'accountant': ACCOUNTANT, 'rounds_run': len(history['round']),
                   'rounds_budgeted_for_sigma': FL_ROUNDS,
                   'n_unique_images_per_client': sizes, 'offline_augmented_copies_used': 0,
                   'achieved_epsilon_per_client': final, 'achieved_epsilon_mean': float(np.mean(final)),
                   'achieved_epsilon_max': float(np.max(final)), 'sigma_per_client': sigmas, 'audit_trail': calib_audit}
        print(f'  [privacy] achieved eps per client {[round(e, 3) for e in final]} '
              f'(max={summary["achieved_epsilon_max"]:.3f}, target={target_epsilon})')
    return gm, history, summary

# ═══ Evaluation and plots ═══
def evaluate_model(model, loader, save_dir, label):
    model.eval()
    ys, ps, pr = [], [], []
    with torch.no_grad():
        for imgs, labels in loader:
            probs = torch.softmax(model(imgs.to(DEVICE)), 1)
            pr.extend(probs.cpu().numpy()); ps.extend(probs.argmax(1).cpu().numpy()); ys.extend(labels.numpy())
    y_true, y_pred, y_prob = np.array(ys), np.array(ps), np.array(pr)
    rep = classification_report(y_true, y_pred, target_names=CLASSES, output_dict=True, zero_division=0)
    print(classification_report(y_true, y_pred, target_names=CLASSES, zero_division=0))
    bins = label_binarize(y_true, classes=list(range(NUM_CLASSES)))
    auroc = float(roc_auc_score(bins, y_prob, average='macro', multi_class='ovr'))
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True'); ax.set_title(f'CM -- {label}')
    plt.tight_layout(); plt.savefig(os.path.join(save_dir, f'cm_{label}.png')); plt.close()
    fig, ax = plt.subplots(figsize=(7, 6))
    for i, cls in enumerate(CLASSES):
        fpr, tpr, _ = roc_curve(bins[:, i], y_prob[:, i])
        ax.plot(fpr, tpr, label=f'{cls} (AUC={auc(fpr, tpr):.3f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=0.8)
    ax.set_xlabel('FPR'); ax.set_ylabel('TPR'); ax.set_title(f'ROC -- {label}'); ax.legend(loc='lower right')
    plt.tight_layout(); plt.savefig(os.path.join(save_dir, f'roc_{label}.png')); plt.close()
    return {'accuracy': float((y_pred == y_true).mean()), 'macro_f1': float(rep['macro avg']['f1-score']),
            'macro_precision': float(rep['macro avg']['precision']), 'macro_recall': float(rep['macro avg']['recall']),
            'auroc_macro': auroc,
            'per_class': {c: {'precision': float(rep[c]['precision']), 'recall': float(rep[c]['recall']),
                              'f1': float(rep[c]['f1-score']), 'support': int(rep[c]['support'])} for c in CLASSES},
            'confusion_matrix': cm.tolist(), 'n_samples': int(len(y_true))}

def plot_privacy_trajectory(history, save_dir, label):
    if not history['epsilon_max'] or all(e == 0.0 for e in history['epsilon_max']):
        return
    per_client = np.array(history['epsilon_per_client'])
    fig, ax = plt.subplots(figsize=(7, 5))
    for ci in range(per_client.shape[1]):
        ax.plot(history['round'], per_client[:, ci], alpha=0.35, lw=0.8, color='#888888')
    ax.plot(history['round'], history['epsilon_mean'], color='#0066CC', lw=1.6, label='Mean over clients')
    ax.plot(history['round'], history['epsilon_max'], color='#CC0000', lw=1.6, label='Max over clients (worst case)')
    ax.set_xlabel('Communication round'); ax.set_ylabel('Cumulative achieved epsilon')
    ax.set_title(f'Composed privacy loss (unit = original photograph) -- {label}')
    ax.legend(); ax.grid(alpha=0.4)
    plt.tight_layout(); plt.savefig(os.path.join(save_dir, f'epsilon_trajectory_{label}.png')); plt.close()

# ═══ Main ═══
master = os.path.join(OUT_DIR, 'dp_results_orig_unit.json')
results = load_json(master, default={})
print(f'\nPrivacy unit: {PRIVACY_UNIT}\n')

for fold_idx, fold in enumerate(DP_FOLDS):
    fold_dir = os.path.join(DATA_ROOT, fold)
    results.setdefault(fold, {})
    audit_rows = audit_client_train_sets(fold_dir, DP_RUN_NAME)
    save_json(audit_rows, os.path.join(OUT_DIR, f'unit_audit_{fold}.json'))
    print(f'{fold}: offline multiplicity per client (augmented set size / raw originals)')
    for r in audit_rows:
        pc = ', '.join(f'{c[:3]} x{v["multiplicity"]}' for c, v in r['per_class'].items())
        print(f'   Client_{r["client"]}: raw={r["raw_total"]} augmented-set={r["augmented_set_size"]} '
              f'mean x{r["mean_multiplicity"]}  ({pc})')

    for eps_target in DP_EPSILONS:
        label_eps = f'eps{int(eps_target)}' if eps_target else 'noDP'
        if results[fold].get(label_eps, {}).get('standard', {}).get('accuracy'):
            print(f'[skip, already done] {fold} / {label_eps}')
            continue
        print(f'\n{"=" * 66}\n  {fold} | target epsilon = {eps_target}\n{"=" * 66}')
        save_dir = os.path.join(OUT_DIR, fold, label_eps)
        lbl = f'dpunit_{fold}_{label_eps}'
        seed_everything(SEED + fold_idx)
        try:
            gm, hist, summary = train_fedavg_dp(fold_dir, save_dir, lbl, eps_target)
            save_json(hist, os.path.join(save_dir, f'history_{lbl}.json'))
            plot_privacy_trajectory(hist, save_dir, lbl)
            metrics = evaluate_model(gm, get_agg_test_loader(fold_dir, DP_RUN_NAME), save_dir, lbl)
            save_json(metrics, os.path.join(save_dir, f'metrics_{lbl}.json'))
            if summary is not None:
                save_json(summary, os.path.join(save_dir, f'privacy_audit_{lbl}.json'))
            results[fold][label_eps] = {'standard': metrics, 'epsilon_target': eps_target, 'privacy_summary': summary}
            del gm
            gc.collect(); torch.cuda.empty_cache()
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                save_json(results, master)
                print('\nCUDA out of memory. Progress is saved. Restart the script to resume from the next unfinished run.')
                sys.exit(1)
            print(f'ERROR: {e}'); traceback.print_exc()
            results[fold][label_eps] = {'error': str(e), 'epsilon_target': eps_target}
        except Exception as e:
            print(f'ERROR: {e}'); traceback.print_exc()
            results[fold][label_eps] = {'error': str(e), 'epsilon_target': eps_target}
        save_json(results, master)

# ── Table 9 (privacy unit = original photograph) ──
rows = []
for fold in DP_FOLDS:
    for eps_target in DP_EPSILONS:
        label_eps = f'eps{int(eps_target)}' if eps_target else 'noDP'
        entry = results.get(fold, {}).get(label_eps, {})
        r, ps = entry.get('standard', {}), entry.get('privacy_summary') or {}
        if r:
            rows.append({'Fold': fold, 'Epsilon_Target': eps_target if eps_target else float('inf'),
                         'Epsilon_Achieved_Mean': ps.get('achieved_epsilon_mean'),
                         'Epsilon_Achieved_Max': ps.get('achieved_epsilon_max'),
                         'Rounds_Run': ps.get('rounds_run'),
                         'Accuracy': r['accuracy'], 'Macro_F1': r['macro_f1'], 'AUROC': r['auroc_macro']})
df = pd.DataFrame(rows)
print('\nTable 9 (privacy unit = original photograph):')
print(df.to_string(index=False))
df.to_csv(os.path.join(OUT_DIR, 'table9_dp_tradeoff_orig_unit.csv'), index=False)

if len(df) > 0:
    agg = df.groupby('Epsilon_Target').agg(
        acc_mean=('Accuracy', 'mean'), acc_std=('Accuracy', 'std'),
        f1_mean=('Macro_F1', 'mean'), f1_std=('Macro_F1', 'std'),
        auroc_mean=('AUROC', 'mean'), auroc_std=('AUROC', 'std'),
        eps_max_mean=('Epsilon_Achieved_Max', 'mean'), eps_max_worst=('Epsilon_Achieved_Max', 'max')).reset_index()
    print('\nAggregate (mean and std across folds):')
    print(agg.to_string(index=False))
    agg.to_csv(os.path.join(OUT_DIR, 'table9_dp_tradeoff_orig_unit_aggregate.csv'), index=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    xs = range(len(agg))
    labels = ['no DP' if np.isinf(e) else f'target {e:.0f}' for e in agg['Epsilon_Target']]
    ax.errorbar(xs, agg['f1_mean'], yerr=agg['f1_std'], marker='o', capsize=5, color='#0066CC', label='Macro F1')
    ax.errorbar(xs, agg['acc_mean'], yerr=agg['acc_std'], marker='s', capsize=5, color='#CC0000', label='Accuracy')
    ax.set_xticks(list(xs)); ax.set_xticklabels(labels)
    ax.set_xlabel('Privacy setting (unit = original photograph)'); ax.set_ylabel('Score')
    ax.set_title('DP-SGD privacy-utility trade-off, unique originals with online augmentation')
    ax.legend(); ax.grid(alpha=0.4)
    plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, 'fig_dp_tradeoff_orig_unit.png')); plt.close()

print(f'\nDone. Results: {master}')
