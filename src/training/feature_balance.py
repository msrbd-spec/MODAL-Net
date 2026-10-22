"""Feature balancing methods in latent penultimate feature space (S7n).

Methods:
  - none: Plain training without rebalancing.
  - reweight: Logit adjustment (Menon et al., 2021) adding tau * log(pi) during training,
              evaluated with plain logits at test time.
  - smote: Synthetic Minority Over-sampling Technique via feature interpolation.
  - gauss: Class-conditional Gaussian sampling with shrinkage covariance.
  - lfsd: Class-conditional 1-D Latent Feature Space Diffusion (DDPM) with MLP denoiser (T=200).

LFSD retention rule:
  Keep lfsd only if it beats the best cheaper method with a 95% bootstrap CI excluding 0.
"""

import math
import os
import json
import time
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.data.datasets import DRDataset
from src.data.splits import fold_frames, test_frame, messidor_frame
from src.models.gradenet import GradeNet
from src.evaluation.stats import paired_bootstrap
from sklearn.metrics import cohen_kappa_score
from src.evaluation.metrics import metrics5, compute_oof_thresholds, fit_temperature
from src.utils.logger import get_logger
from src.utils.config import ckpt_owner, find_exp

qwk_score = lambda y, yh: float(cohen_kappa_score(y, yh, weights='quadratic', labels=list(range(5))))


# ---------------------------------------------------------------------------
# Sinusoidal Timestep Embeddings for 1-D DDPM
# ---------------------------------------------------------------------------
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        device = timesteps.device
        half_dim = self.dim // 2
        exponent = -math.log(10000) * torch.arange(
            0, half_dim, dtype=torch.float32, device=device
        ) / half_dim
        emb = torch.exp(exponent)
        emb = timesteps.float()[:, None] * emb[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)


# ---------------------------------------------------------------------------
# Class-conditional 1-D DDPM (LFSD)
# ---------------------------------------------------------------------------
class LFSDDenoiser(nn.Module):
    """MLP Denoiser for 1-D Latent Feature Space Diffusion."""

    def __init__(
        self,
        in_dim: int,
        num_classes: int = 5,
        hidden_dim: int = 512,
        time_dim: int = 64,
        class_dim: int = 32,
    ):
        super().__init__()
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.GELU(),
        )
        self.class_embed = nn.Embedding(num_classes, class_dim)

        self.net = nn.Sequential(
            nn.Linear(in_dim + time_dim + class_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, in_dim),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embed(t)
        c_emb = self.class_embed(c)
        feat = torch.cat([x, t_emb, c_emb], dim=-1)
        return self.net(feat)


class LFSD(nn.Module):
    """Class-conditional 1-D Latent Feature Space Diffusion Model."""

    def __init__(
        self,
        in_dim: int,
        num_classes: int = 5,
        hidden_dim: int = 512,
        T: int = 200,
        beta_1: float = 1e-4,
        beta_T: float = 0.02,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.num_classes = num_classes
        self.T = T
        self.denoiser = LFSDDenoiser(in_dim, num_classes, hidden_dim)

        betas = torch.linspace(beta_1, beta_T, T, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ac = self.sqrt_alphas_cumprod[t].unsqueeze(-1)
        sqrt_om_ac = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1)
        return sqrt_ac * x0 + sqrt_om_ac * noise

    def loss(self, x0: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        t = torch.randint(0, self.T, (x0.shape[0],), device=x0.device)
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise)
        pred_noise = self.denoiser(xt, t, c)
        return F.mse_loss(pred_noise, noise)

    @torch.no_grad()
    def sample(self, num_samples: int, target_class: int, device: str = 'cpu') -> torch.Tensor:
        self.eval()
        x = torch.randn(num_samples, self.in_dim, device=device)
        c = torch.full((num_samples,), target_class, dtype=torch.long, device=device)

        for t_step in reversed(range(self.T)):
            t = torch.full((num_samples,), t_step, dtype=torch.long, device=device)
            pred_noise = self.denoiser(x, t, c)

            alpha = self.alphas[t_step]
            alpha_cumprod = self.alphas_cumprod[t_step]
            beta = self.betas[t_step]

            mean = (1.0 / torch.sqrt(alpha)) * (
                x - (beta / torch.sqrt(1.0 - alpha_cumprod)) * pred_noise
            )

            if t_step > 0:
                noise = torch.randn_like(x)
                var = torch.sqrt(self.posterior_variance[t_step])
                x = mean + var * noise
            else:
                x = mean
        return x


# ---------------------------------------------------------------------------
# Balanced Head Model
# ---------------------------------------------------------------------------
class BalancedHead(nn.Module):
    """Linear or MLP classification head trained on penultimate features."""

    def __init__(self, in_features: int, num_classes: int = 5, hidden_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


# ---------------------------------------------------------------------------
# Feature Extraction Routine
# ---------------------------------------------------------------------------
@torch.no_grad()
def extract_dataset_features(model: nn.Module, loader: DataLoader, device: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Extract penultimate features, labels, and ids from a dataloader."""
    model.eval()
    feats, targets, sample_ids = [], [], []

    for batch in loader:
        x = batch['x'].to(device)
        y = batch['y']
        ids = batch['id']

        if hasattr(model, 'forward_features'):
            f = model.forward_features(x)
        elif hasattr(model, 'backbone'):
            f = model.backbone(x)
        else:
            f = model(x)

        if f.dim() > 2:
            f = torch.flatten(f, 1)

        feats.append(f.cpu().numpy())
        targets.append(y.numpy())
        sample_ids.extend([str(i) for i in ids])

    all_feats = np.concatenate(feats, axis=0) if feats else np.empty((0, 0), dtype=np.float32)
    all_targets = np.concatenate(targets, axis=0) if targets else np.empty((0,), dtype=np.int64)
    return all_feats, all_targets, sample_ids


# ---------------------------------------------------------------------------
# Feature Rebalancing Implementations
# ---------------------------------------------------------------------------
def rebalance_smote(
    X: np.ndarray, y: np.ndarray, k_neighbors: int = 5, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """Synthetic Minority Over-sampling Technique (SMOTE) on feature space."""
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)
    max_count = int(np.max(counts))

    syn_X, syn_y = [X], [y]

    for c, count in zip(classes, counts):
        if count >= max_count:
            continue
        n_needed = max_count - count
        X_c = X[y == c]

        if len(X_c) < 2:
            # If only 1 sample, add small Gaussian perturbation
            noise = rng.normal(0, 0.05, size=(n_needed, X.shape[1])).astype(np.float32)
            gen = np.repeat(X_c, n_needed, axis=0) + noise
        else:
            k = min(k_neighbors, len(X_c) - 1)
            # Pairwise Euclidean distances
            dists = np.linalg.norm(X_c[:, None, :] - X_c[None, :, :], axis=-1)
            np.fill_diagonal(dists, np.inf)
            nn_idx = np.argsort(dists, axis=1)[:, :k]

            base_indices = rng.choice(len(X_c), size=n_needed, replace=True)
            chosen_nn = rng.choice(k, size=n_needed, replace=True)
            neighbor_indices = nn_idx[base_indices, chosen_nn]

            lambdas = rng.uniform(0.0, 1.0, size=(n_needed, 1)).astype(np.float32)
            gen = X_c[base_indices] + lambdas * (X_c[neighbor_indices] - X_c[base_indices])

        syn_X.append(gen)
        syn_y.append(np.full((n_needed,), c, dtype=y.dtype))

    return np.concatenate(syn_X, axis=0), np.concatenate(syn_y, axis=0)


def rebalance_gauss(
    X: np.ndarray, y: np.ndarray, shrinkage: float = 0.1, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """Class-conditional Gaussian sampling with shrinkage covariance."""
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)
    max_count = int(np.max(counts))
    D = X.shape[1]
    syn_X, syn_y = [X], [y]

    for c, count in zip(classes, counts):
        if count >= max_count:
            continue
        n_needed = max_count - count
        X_c = torch.from_numpy(X[y == c])

        mu = X_c.mean(dim=0)
        cov = torch.cov(X_c.T) if len(X_c) > 1 else torch.eye(D) * 0.1
        nu = torch.trace(cov) / max(1, D)
        cov = (1.0 - shrinkage) * cov + shrinkage * nu * torch.eye(D) + torch.eye(D) * 1e-4

        eigvals, eigvecs = torch.linalg.eigh(cov)
        eigvals = torch.clamp(eigvals, min=1e-4)

        z = torch.randn(n_needed, D)
        gen = mu + torch.matmul(z * torch.sqrt(eigvals), eigvecs.T)
        syn_X.append(gen.numpy().astype(np.float32))
        syn_y.append(np.full((n_needed,), c, dtype=y.dtype))

    return np.concatenate(syn_X, axis=0), np.concatenate(syn_y, axis=0)


def rebalance_lfsd(
    X: np.ndarray,
    y: np.ndarray,
    epochs: int = 60,
    batch_size: int = 64,
    device: str = 'cpu',
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Train class-conditional 1-D DDPM (LFSD) and sample synthetic minority features."""
    torch.manual_seed(seed)
    classes, counts = np.unique(y, return_counts=True)
    max_count = int(np.max(counts))
    D = X.shape[1]

    model = LFSD(in_dim=D, num_classes=5, hidden_dim=512, T=200).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    dataset = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).long())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    model.train()
    for _ in range(epochs):
        for bx, by in loader:
            bx = bx.to(device)
            by = by.to(device)
            optimizer.zero_grad()
            loss = model.loss(bx, by)
            loss.backward()
            optimizer.step()

    syn_X, syn_y = [X], [y]
    model.eval()

    for c, count in zip(classes, counts):
        if count >= max_count:
            continue
        n_needed = max_count - count
        samples = model.sample(n_needed, target_class=int(c), device=device)
        syn_X.append(samples.cpu().numpy())
        syn_y.append(np.full((n_needed,), c, dtype=y.dtype))

    return np.concatenate(syn_X, axis=0), np.concatenate(syn_y, axis=0)


# ---------------------------------------------------------------------------
# Head Training Function
# ---------------------------------------------------------------------------
def train_balanced_head(
    X_train: np.ndarray,
    y_train: np.ndarray,
    method: str = 'none',
    tau: float = 1.0,
    epochs: int = 50,
    batch_size: int = 64,
    device: str = 'cpu',
    seed: int = 42,
) -> Tuple[BalancedHead, Optional[np.ndarray]]:
    """Train BalancedHead on penultimate features with the chosen balancing method."""
    torch.manual_seed(seed)
    D = X_train.shape[1]
    head = BalancedHead(in_features=D, num_classes=5).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    log_prior = None
    if method == 'reweight':
        # Logit adjustment: log prior
        classes, counts = np.unique(y_train, return_counts=True)
        prior = np.zeros(5, dtype=np.float32)
        total = float(len(y_train))
        for c, cnt in zip(classes, counts):
            prior[c] = cnt / total
        prior = np.clip(prior, 1e-6, 1.0)
        log_prior = torch.from_numpy(np.log(prior) * tau).to(device)

    dataset = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    head.train()
    for _ in range(epochs):
        for bx, by in loader:
            bx = bx.to(device)
            by = by.to(device)
            optimizer.zero_grad()
            logits = head(bx)
            if method == 'reweight' and log_prior is not None:
                logits = logits + log_prior
            loss = F.cross_entropy(logits, by)
            loss.backward()
            optimizer.step()
        scheduler.step()

    head.eval()
    return head, log_prior


@torch.no_grad()
def predict_with_head(head: BalancedHead, X: np.ndarray, device: str = 'cpu') -> np.ndarray:
    """Predict calibrated probabilities using BalancedHead (plain logits at test time)."""
    head.eval()
    bx = torch.from_numpy(X).float().to(device)
    logits = head(bx)
    probs = F.softmax(logits, dim=-1).cpu().numpy()
    return probs


# ---------------------------------------------------------------------------
# Main Feature Balance Experiment Runner (S7n)
# ---------------------------------------------------------------------------
def run_feature_balance_experiment(cfg: dict) -> Dict[str, Any]:
    """Execute feature balancing experiment (none | reweight | smote | gauss | lfsd)."""
    t0 = time.time()
    tag = cfg.get('tag')
    phase = cfg.get('phase', 'phase7_feature_balance')
    method = cfg.get('method', 'none').lower()
    valid_methods = ['none', 'reweight', 'smote', 'gauss', 'lfsd']
    if method not in valid_methods:
        raise ValueError(f"Invalid method '{method}'. Must be one of {valid_methods}")

    base_tag = cfg.get('base')
    if not base_tag:
        raise ValueError("--base tag is required for feature_balance mode.")

    owner = ckpt_owner(base_tag)
    base_exp_dir = find_exp(owner)
    logger = get_logger(tag, 'feature_balance')
    logger.info(f"Starting feature_balance experiment: method={method}, base={base_tag}, owner={owner}")

    exp_dir = os.path.join('results', phase, tag)
    ckpt_dir = os.path.join('checkpoints', tag)
    curves_dir = os.path.join(exp_dir, 'curves')
    figures_dir = os.path.join(exp_dir, 'figures')
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(curves_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    # Save snapshot with ckpt_from
    snap = dict(cfg)
    snap['ckpt_from'] = owner
    import yaml
    with open(os.path.join(ckpt_dir, 'config_snapshot.yaml'), 'w') as f_snap:
        yaml.safe_dump(snap, f_snap)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = int(cfg.get('batch_size', 16))
    num_workers = int(cfg.get('num_workers', 4))

    # Feature extraction cache directory
    feat_cache_dir = os.path.join('checkpoints', owner, 'extracted_features')
    os.makedirs(feat_cache_dir, exist_ok=True)

    all_cached = all(
        os.path.isfile(os.path.join(feat_cache_dir, f'fold{k}_features.npz'))
        for k in range(5)
    )

    test_loader, dev_loader, mtest_loader = None, None, None
    if not all_cached:
        # Prepare datasets for feature extraction
        aptos_paths = cfg['paths']['aptos']
        aptos_dirs = [aptos_paths['train_images'], aptos_paths['val_images']]
        test_df = test_frame(cfg)
        dev_df = messidor_frame(cfg, 'dev')
        mtest_df = messidor_frame(cfg, 'test')

        test_ds = DRDataset(test_df, aptos_dirs, cfg, train=False, domain=0)
        dev_ds = DRDataset(dev_df, [cfg['paths']['messidor2']['images']], cfg, train=False, domain=2)
        mtest_ds = DRDataset(mtest_df, [cfg['paths']['messidor2']['images']], cfg, train=False, domain=2)

        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        dev_loader = DataLoader(dev_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        mtest_loader = DataLoader(mtest_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    oof_feats_all = []
    oof_y_all = []
    oof_ids_all = []
    fold_indices_map = []  # tracks slice of OOF belonging to each fold

    test_feats_folds = []
    dev_feats_folds = []
    mtest_feats_folds = []

    test_ids, test_y = None, None
    dev_ids, dev_y = None, None
    mtest_ids, mtest_y = None, None

    logger.info("Extracting penultimate features across 5 folds...")
    for k in range(5):
        ckpt_path = os.path.join('checkpoints', owner, f'fold{k}.pth')
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Missing fold checkpoint: {ckpt_path}")

        cache_k_path = os.path.join(feat_cache_dir, f'fold{k}_features.npz')
        if os.path.isfile(cache_k_path):
            logger.info(f"Loading cached features for fold {k} from {cache_k_path}")
            f_k = np.load(cache_k_path, allow_pickle=True)
            k_oof_feat = f_k['oof_feat']
            k_oof_y = f_k['oof_y']
            k_oof_ids = [str(x) for x in f_k['oof_ids']]

            k_test_feat = f_k['test_feat']
            k_dev_feat = f_k['dev_feat']
            k_mtest_feat = f_k['mtest_feat']
            if test_ids is None:
                test_ids = [str(x) for x in f_k['test_ids']]
                test_y = f_k['test_y']
                dev_ids = [str(x) for x in f_k['dev_ids']]
                dev_y = f_k['dev_y']
                mtest_ids = [str(x) for x in f_k['mtest_ids']]
                mtest_y = f_k['mtest_y']
        else:
            model = GradeNet(cfg).to(device)
            sd = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(sd, strict=False)

            _, va_df = fold_frames(cfg, k)
            val_ds = DRDataset(va_df, aptos_dirs, cfg, train=False, domain=0)
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

            k_oof_feat, k_oof_y, k_oof_ids = extract_dataset_features(model, val_loader, device)
            k_test_feat, k_test_y, k_test_ids = extract_dataset_features(model, test_loader, device)
            k_dev_feat, k_dev_y, k_dev_ids = extract_dataset_features(model, dev_loader, device)
            k_mtest_feat, k_mtest_y, k_mtest_ids = extract_dataset_features(model, mtest_loader, device)

            if test_ids is None:
                test_ids, test_y = k_test_ids, k_test_y
                dev_ids, dev_y = k_dev_ids, k_dev_y
                mtest_ids, mtest_y = k_mtest_ids, k_mtest_y

            np.savez(
                cache_k_path,
                oof_feat=k_oof_feat,
                oof_y=k_oof_y,
                oof_ids=np.array(k_oof_ids),
                test_feat=k_test_feat,
                test_y=k_test_y,
                test_ids=np.array(k_test_ids),
                dev_feat=k_dev_feat,
                dev_y=k_dev_y,
                dev_ids=np.array(k_dev_ids),
                mtest_feat=k_mtest_feat,
                mtest_y=k_mtest_y,
                mtest_ids=np.array(k_mtest_ids),
            )

        start_idx = len(oof_y_all)
        oof_feats_all.append(k_oof_feat)
        oof_y_all.extend(k_oof_y)
        oof_ids_all.extend(k_oof_ids)
        end_idx = len(oof_y_all)
        fold_indices_map.append((start_idx, end_idx))

        test_feats_folds.append(k_test_feat)
        dev_feats_folds.append(k_dev_feat)
        mtest_feats_folds.append(k_mtest_feat)

    oof_feats_all = np.concatenate(oof_feats_all, axis=0)
    oof_y_all = np.array(oof_y_all, dtype=np.int64)

    # 5-fold cross validation for balanced heads
    logger.info(f"Training balanced heads using method '{method}'...")
    oof_p_all = np.zeros((len(oof_y_all), 5), dtype=np.float32)
    test_p_folds = []
    dev_p_folds = []
    mtest_p_folds = []
    fold_metric_records = []

    seed = int(cfg.get('seed', 42))

    for k in range(5):
        st, ed = fold_indices_map[k]
        val_mask = np.zeros(len(oof_y_all), dtype=bool)
        val_mask[st:ed] = True
        train_mask = ~val_mask

        X_tr = oof_feats_all[train_mask]
        y_tr = oof_y_all[train_mask]
        X_va = oof_feats_all[val_mask]
        y_va = oof_y_all[val_mask]

        # Apply rebalancing to training features
        if method == 'smote':
            X_bal, y_bal = rebalance_smote(X_tr, y_tr, seed=seed + k)
        elif method == 'gauss':
            X_bal, y_bal = rebalance_gauss(X_tr, y_tr, seed=seed + k)
        elif method == 'lfsd':
            X_bal, y_bal = rebalance_lfsd(X_tr, y_tr, device=device, seed=seed + k)
        else:
            X_bal, y_bal = X_tr, y_tr

        # Train BalancedHead
        head, _ = train_balanced_head(
            X_bal,
            y_bal,
            method=method,
            tau=float(cfg.get('tau', 1.0)),
            device=device,
            seed=seed + k,
        )

        # Save head weights
        head_save_path = os.path.join(ckpt_dir, f'fold{k}_head.pth')
        torch.save(head.state_dict(), head_save_path)

        # Predict OOF for fold k
        va_p = predict_with_head(head, X_va, device=device)
        oof_p_all[st:ed] = va_p

        # Predict test, dev, mtest for fold k
        test_p_folds.append(predict_with_head(head, test_feats_folds[k], device=device))
        dev_p_folds.append(predict_with_head(head, dev_feats_folds[k], device=device))
        mtest_p_folds.append(predict_with_head(head, mtest_feats_folds[k], device=device))

        # Fold validation metrics
        va_m = metrics5(y_va, va_p)
        fold_record = {
            'fold': k,
            'epoch': 50,
            'val_acc': va_m['acc'],
            'val_qwk': va_m['qwk'],
            'val_f1_macro': va_m['f1_macro'],
            'val_bin_auroc': va_m.get('bin_auroc', 0.0),
        }
        fold_metric_records.append(fold_record)
        with open(os.path.join(ckpt_dir, f'fold{k}_val_metrics.json'), 'w') as f_vm:
            json.dump(fold_record, f_vm, indent=2)

    # Ensemble test, dev, mtest across the 5 fold heads
    t_p_arr = np.mean(test_p_folds, axis=0)
    dev_p_arr = np.mean(dev_p_folds, axis=0)
    mtest_p_arr = np.mean(mtest_p_folds, axis=0)

    # Compute thresholds & temperature scaling on pooled OOF
    thresholds = compute_oof_thresholds(oof_y_all, oof_p_all)
    fitted_T = fit_temperature(np.log(np.clip(oof_p_all, 1e-15, 1.0)), oof_y_all)
    thresholds['temperature'] = float(fitted_T)
    with open(os.path.join(ckpt_dir, 'thresholds.json'), 'w') as f_t:
        json.dump(thresholds, f_t, indent=2)
    with open(os.path.join(exp_dir, 'thresholds.json'), 'w') as f_t:
        json.dump(thresholds, f_t, indent=2)

    # LFSD retention comparison rule
    if method == 'lfsd':
        cheaper_methods = ['none', 'reweight', 'smote', 'gauss']
        logger.info(f"Evaluating LFSD against cheaper methods: {cheaper_methods}")
        # Look for cheaper methods' results
        best_cheaper_qwk = -1.0
        best_cheaper_tag = None
        best_cheaper_oof_p = None

        for cm in cheaper_methods:
            cm_tag = f"{base_tag}_{cm}"
            cm_dir = os.path.join('results', phase, cm_tag)
            oof_file = os.path.join(cm_dir, 'oof.npz')
            if os.path.isfile(oof_file):
                f_cm = np.load(oof_file, allow_pickle=True)
                cm_p = f_cm['P'] if 'P' in f_cm else f_cm['y_prob']
                cm_qwk = qwk_score(oof_y_all, cm_p.argmax(axis=1))
                if cm_qwk > best_cheaper_qwk:
                    best_cheaper_qwk = cm_qwk
                    best_cheaper_tag = cm_tag
                    best_cheaper_oof_p = cm_p

        lfsd_qwk = qwk_score(oof_y_all, oof_p_all.argmax(axis=1))
        if best_cheaper_oof_p is not None:
            # Paired bootstrap CI between LFSD and best cheaper method
            diff, ci, p_val = paired_bootstrap(
                oof_y_all,
                oof_p_all.argmax(axis=1),
                best_cheaper_oof_p.argmax(axis=1),
                fn=lambda y, yh: qwk_score(y, yh),
                B=1000,
                seed=seed,
            )
            ci_lo, ci_hi = ci[0], ci[1]
            logger.info(
                f"LFSD QWK={lfsd_qwk:.4f} vs {best_cheaper_tag} QWK={best_cheaper_qwk:.4f} | "
                f"Diff={diff:.4f}, 95% CI=[{ci_lo:.4f}, {ci_hi:.4f}]"
            )
            # Rule: Keep LFSD only if CI excludes 0 (ci_lo > 0)
            keep_lfsd = bool(ci_lo > 0.0)
            logger.info(f"Keep LFSD decision: {keep_lfsd}")
            with open(os.path.join(exp_dir, 'lfsd_retention.json'), 'w') as f_ret:
                json.dump({
                    'lfsd_qwk': lfsd_qwk,
                    'best_cheaper_tag': best_cheaper_tag,
                    'best_cheaper_qwk': best_cheaper_qwk,
                    'delta_qwk': diff,
                    'ci_lo': ci_lo,
                    'ci_hi': ci_hi,
                    'keep_lfsd': keep_lfsd,
                }, f_ret, indent=2)

    # Save artifacts using unified evaluator
    from src.evaluation.experiment import _evaluate_and_save_artifacts

    summary = _evaluate_and_save_artifacts(
        cfg=cfg,
        tag=tag,
        phase=phase,
        exp_dir=exp_dir,
        ckpt_dir=ckpt_dir,
        t0=t0,
        n_params_M=0.0,
        fold_metric_records=fold_metric_records,
        oof_ids=oof_ids_all,
        oof_y=oof_y_all,
        oof_p=oof_p_all,
        thresholds=thresholds,
        fitted_T=thresholds.get('temperature', 1.0),
        t_ids=test_ids,
        t_y=test_y,
        t_p=t_p_arr,
        t_p_single=test_p_folds,
        dev_ids=dev_ids,
        dev_y=dev_y,
        dev_p=dev_p_arr,
        dev_p_single=dev_p_folds,
        mtest_ids=mtest_ids,
        mtest_y=mtest_y,
        mtest_p=mtest_p_arr,
        mtest_p_single=mtest_p_folds,
        protocol='5fold_feature_balance',
        logger=logger,
        winner_eligible=True,
        kind='feature_balance',
    )

    return summary
