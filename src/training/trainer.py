"""Training loop: fold training, Mixup/CutMix, LLRD, EMA, early stopping, and OOF recording."""

import os
import math
import json
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset

from src.data.splits import fold_frames, eyepacs_frame
from src.data.datasets import DRDataset
from src.data.samplers import JointBatchSampler
from src.losses.losses import total_loss
from src.models.gradenet import GradeNet, llrd_groups, probs_from_logits
from src.training.ema import ModelEMA
from src.evaluation.metrics import metrics5, metrics_bin


def mix_batch(x: torch.Tensor, q: torch.Tensor, cfg: dict):
    """Mixup or CutMix on a batch with probability cfg['mix_prob']."""
    mix_prob = float(cfg.get('mix_prob', 0.5))
    if mix_prob <= 0.0 or torch.rand(1).item() > mix_prob:
        return x, q

    perm = torch.randperm(x.size(0), device=x.device)
    if torch.rand(1).item() < 0.5:
        # Mixup
        alpha = float(cfg.get('mixup_alpha', 0.2))
        lam = float(np.random.beta(alpha, alpha))
        x_mixed = lam * x + (1.0 - lam) * x[perm]
        q_mixed = lam * q + (1.0 - lam) * q[perm]
        return x_mixed, q_mixed
    else:
        # CutMix
        alpha = float(cfg.get('cutmix_alpha', 1.0))
        lam = float(np.random.beta(alpha, alpha))
        H, W = x.shape[-2:]
        r = math.sqrt(1.0 - lam)
        ch, cw = int(H * r), int(W * r)
        cy, cx = int(np.random.randint(H)), int(np.random.randint(W))
        y1, y2 = max(cy - ch // 2, 0), min(cy + ch // 2, H)
        x1, x2 = max(cx - cw // 2, 0), min(cx + cw // 2, W)

        x_mixed = x.clone()
        x_mixed[:, :, y1:y2, x1:x2] = x[perm][:, :, y1:y2, x1:x2]
        effective_lam = 1.0 - float((y2 - y1) * (x2 - x1)) / float(H * W)
        q_mixed = effective_lam * q + (1.0 - effective_lam) * q[perm]
        return x_mixed, q_mixed


def build_style_bank(cfg: dict, tr_df, logger=None):
    """Build list of image paths for Fourier amplitude mixing style bank."""
    if not cfg.get('amp_mix'):
        return None

    pool = cfg.get('uda_style_pool', 'none')
    bank_size = int(cfg.get('style_bank_size', 2000))

    if pool == 'messidor_dev':
        from src.data.splits import messidor_frame
        try:
            m_dev = messidor_frame(cfg, 'dev')
            mess_dir = cfg['paths']['messidor']['images']
            paths = []
            for _, row in m_dev.iterrows():
                code = str(row['id_code'])
                for ext in ('', '.jpg', '.png', '.jpeg'):
                    cand = os.path.join(mess_dir, code + ext)
                    if os.path.isfile(cand):
                        paths.append(cand)
                        break
            if logger:
                logger.info(f"Loaded UDA style bank: {len(paths)} images from Messidor-DEV")
            return paths
        except Exception as e:
            if logger:
                logger.warning(f"Could not load Messidor-DEV style bank ({e}); falling back to empty bank")
            return None
    else:
        paths = []
        if cfg.get('joint_eyepacs'):
            from src.data.splits import eyepacs_frame
            try:
                eye_df = eyepacs_frame(cfg)
                eye_dir = cfg['paths']['eyepacs']['images']
                for _, row in eye_df.head(bank_size).iterrows():
                    code = str(row['id_code'])
                    for ext in ('.jpeg', '.jpg', '.png', ''):
                        cand = os.path.join(eye_dir, code + ext)
                        if os.path.isfile(cand):
                            paths.append(cand)
                            break
            except Exception:
                pass
        if not paths:
            aptos_paths = cfg['paths']['aptos']
            dirs = [aptos_paths['train_images'], aptos_paths['val_images']]
            for _, row in tr_df.iterrows():
                code = str(row['id_code'])
                for d in dirs:
                    for ext in ('.png', '.jpg', '.jpeg', ''):
                        cand = os.path.join(d, code + ext)
                        if os.path.isfile(cand):
                            paths.append(cand)
                            break
        if logger:
            logger.info(f"Loaded style bank: {len(paths)} images from source domain")
        return paths[:bank_size]


def _freeze_stem_stage0(model: nn.Module, freeze: bool):
    """Freeze or unfreeze stem and stage 0 of GradeNet's backbone."""
    for name, param in model.backbone.named_parameters():
        if name.startswith('stem') or name.startswith('stages.0'):
            param.requires_grad = not freeze


def build_cosine_schedule_with_warmup(optimizer, warmup_steps: int, total_steps: int, min_lr_ratio: float = 0.01):
    """Linear warmup followed by cosine decay down to min_lr_ratio * base_lr."""
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return max(1e-6, float(step) / float(max(1, warmup_steps)))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def evaluate_loader(model: nn.Module, loader: DataLoader, cfg: dict, device: str):
    """Evaluate model on a DataLoader without TTA. Returns (metrics, ids, y, P)."""
    model.eval()
    head_type = cfg.get('head_type', 'softmax')
    all_ids = []
    all_y = []
    all_p = []

    with torch.no_grad():
        for batch in loader:
            x = batch['x'].to(device)
            y = batch['y'].cpu().numpy()
            b_ids = batch['id']
            z = model(x)
            p = probs_from_logits(z, head_type).cpu().numpy()

            all_ids.extend([str(i) for i in b_ids])
            all_y.extend(list(y))
            all_p.append(p)

    all_y = np.array(all_y, dtype=int)
    all_p = np.concatenate(all_p, axis=0) if all_p else np.empty((0, int(cfg.get('num_classes', 5))))

    m5 = metrics5(all_y, all_p)
    m_bin = metrics_bin(all_y, all_p)
    combined = {**m5, **m_bin}
    return combined, all_ids, all_y, all_p


def train_fold(cfg: dict, k: int, tag: str, logger, train_df=None, val_df=None, out=None):
    phase = cfg.get('phase', 'default_phase')
    curves_dir = os.path.join('results', phase, tag, 'curves')
    res_dir = os.path.join('results', phase, tag)

    if out is not None and str(out).endswith('.pth'):
        ckpt_path = out
        ckpt_dir = os.path.dirname(out)
        base_stem = os.path.splitext(os.path.basename(out))[0]
        val_metrics_path = os.path.join(ckpt_dir, f'{base_stem}_val_metrics.json')
        oof_fold_path = os.path.join(res_dir, f'{base_stem}_oof.npz')
        curve_csv_path = os.path.join(curves_dir, f'{base_stem}.csv')
    else:
        ckpt_dir = out if out is not None else os.path.join('checkpoints', tag)
        ckpt_path = os.path.join(ckpt_dir, f'fold{k}.pth')
        val_metrics_path = os.path.join(ckpt_dir, f'fold{k}_val_metrics.json')
        oof_fold_path = os.path.join(res_dir, f'oof_fold{k}.npz')
        curve_csv_path = os.path.join(curves_dir, f'fold{k}.csv')
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(curves_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)

    # Resume check
    if os.path.isfile(ckpt_path) and os.path.isfile(val_metrics_path) and os.path.isfile(oof_fold_path):
        logger.info(f"Fold {k} already completed. Resuming without retraining.")
        return json.load(open(val_metrics_path))

    # 1. Prepare dataframes
    if train_df is None or val_df is None:
        tr_df, va_df = fold_frames(cfg, k)
    else:
        tr_df, va_df = train_df, val_df

    aptos_paths = cfg['paths']['aptos']
    aptos_dirs = [aptos_paths['train_images'], aptos_paths['val_images']]

    # Load soft labels if configured
    soft_dict = None
    if cfg.get('soft_labels') == 'aasl' and cfg.get('soft_label_run'):
        sl_path = os.path.join('softlabels', cfg['soft_label_run'], f'outer{k}.npz')
        if os.path.isfile(sl_path):
            with np.load(sl_path, allow_pickle=True) as f_sl:
                sl_ids = [str(x) for x in f_sl['ids']]
                sl_q = f_sl['Q']
                soft_dict = {i: q for i, q in zip(sl_ids, sl_q)}
            logger.info(f"Loaded {len(soft_dict)} AASL soft labels for fold {k} from {sl_path}")

    batch_size = int(cfg.get('batch_size', 16))
    num_workers = int(cfg.get('num_workers', 4))

    # Datasets
    style_bank = build_style_bank(cfg, tr_df, logger)
    aptos_train_ds = DRDataset(tr_df, aptos_dirs, cfg, train=True, soft=soft_dict, domain=0, style_bank=style_bank)
    aptos_val_ds = DRDataset(va_df, aptos_dirs, cfg, train=False, domain=0)

    # Multi-source joint dataset if joint_eyepacs is True
    is_joint = bool(cfg.get('joint_eyepacs', False))
    if is_joint:
        P_eye = cfg['paths']['eyepacs']
        eye_df = eyepacs_frame(cfg)
        max_eye = int(cfg.get('eyepacs_max', 20000))
        if len(eye_df) > max_eye:
            eye_df = eye_df.sample(n=max_eye, random_state=int(cfg.get('seed', 0))).reset_index(drop=True)
        eyepacs_train_ds = DRDataset(eye_df, [P_eye['images']], cfg, train=True, domain=1, style_bank=style_bank)
        concat_train = ConcatDataset([aptos_train_ds, eyepacs_train_ds])
        joint_sampler = JointBatchSampler(len(aptos_train_ds), eye_df['diagnosis'].values, batch_size, seed=int(cfg.get('seed', 0)) + k)
        train_loader = DataLoader(
            concat_train,
            batch_sampler=joint_sampler,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
    else:
        train_loader = DataLoader(
            aptos_train_ds,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    val_loader = DataLoader(
        aptos_val_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # Diagnostic 600-image train subset
    n_diag = min(600, len(tr_df))
    diag_indices = np.random.default_rng(42 + k).choice(len(tr_df), size=n_diag, replace=False)
    diag_df = tr_df.iloc[diag_indices].reset_index(drop=True)
    diag_ds = DRDataset(diag_df, aptos_dirs, cfg, train=False, domain=0)
    diag_loader = DataLoader(
        diag_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # 2. Model setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = GradeNet(cfg).to(device)

    decay = float(cfg.get('ema_decay', 0.999))
    ema = ModelEMA(model, decay) if decay > 0.0 else None

    # Optimizer with LLRD
    lr = float(cfg.get('lr', 1.0e-4))
    llrd = float(cfg.get('llrd', 0.8))
    wd = float(cfg.get('weight_decay', 0.05))
    head_mult = float(cfg.get('head_mult', 10.0))
    param_groups = llrd_groups(model, lr, llrd, wd, head_mult)
    use_sam = bool(cfg.get('use_sam', False))
    if use_sam:
        from src.training.sam import SAM
        sam_rho = float(cfg.get('sam_rho', 0.05))
        optimizer = SAM(param_groups, torch.optim.AdamW, rho=sam_rho)
    else:
        optimizer = torch.optim.AdamW(param_groups)

    epochs = int(cfg.get('epochs', 30))
    warmup_epochs = int(cfg.get('warmup_epochs', 2))
    patience = int(cfg.get('patience', 8))
    grad_accum = 1 if use_sam else int(cfg.get('grad_accum', 1))
    grad_clip = float(cfg.get('grad_clip', 1.0))
    freeze_epochs = int(cfg.get('freeze_epochs', 1))
    joint_w = float(cfg.get('joint_w', 0.5))

    total_steps = len(train_loader) * epochs
    warmup_steps = len(train_loader) * warmup_epochs
    scheduler = build_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best_val_qwk = -999.0
    best_metrics = {}
    patience_counter = 0
    curve_rows = []

    use_autocast = (device == 'cuda')

    logger.info(f"Starting training Fold {k}: {epochs} epochs, batch_size={batch_size}, grad_accum={grad_accum}, use_sam={use_sam}, device={device}")

    for epoch in range(epochs):
        # Freeze stem + stage 0 for the first freeze_epochs
        if epoch < freeze_epochs:
            _freeze_stem_stage0(model, freeze=True)
        elif epoch == freeze_epochs:
            _freeze_stem_stage0(model, freeze=False)

        model.train()
        train_loss_accum = 0.0
        train_batches = 0
        optimizer.zero_grad()

        for step_idx, batch in enumerate(train_loader):
            x = batch['x'].to(device)
            q = batch['q'].to(device)
            y = batch['y'].to(device)

            x, q = mix_batch(x, q, cfg)

            w = None
            if is_joint and 'domain' in batch:
                dom = batch['domain'].to(device)
                w = torch.where(dom == 1, torch.tensor(joint_w, device=device), torch.tensor(1.0, device=device))

            if use_sam:
                # First pass
                if use_autocast:
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        z = model(x)
                        loss = total_loss(z.float(), q, y, cfg, w=w)
                else:
                    z = model(x)
                    loss = total_loss(z.float(), q, y, cfg, w=w)
                loss.backward()
                optimizer.first_step(zero_grad=True)

                # Second pass
                if use_autocast:
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        z2 = model(x)
                        loss2 = total_loss(z2.float(), q, y, cfg, w=w)
                else:
                    z2 = model(x)
                    loss2 = total_loss(z2.float(), q, y, cfg, w=w)
                loss2.backward()
                if grad_clip > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.second_step(zero_grad=True)
                scheduler.step()
                train_loss_accum += loss.item()
                train_batches += 1
                if ema is not None:
                    ema.update(model)
            else:
                if use_autocast:
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        z = model(x)
                        loss = total_loss(z.float(), q, y, cfg, w=w) / float(grad_accum)
                else:
                    z = model(x)
                    loss = total_loss(z.float(), q, y, cfg, w=w) / float(grad_accum)

                loss.backward()
                train_loss_accum += loss.item() * float(grad_accum)
                train_batches += 1

                if (step_idx + 1) % grad_accum == 0 or (step_idx + 1) == len(train_loader):
                    if grad_clip > 0.0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    if ema is not None:
                        ema.update(model)

        mean_train_loss = train_loss_accum / max(1, train_batches)
        current_lr = scheduler.get_last_lr()[0]

        # Validation evaluation with EMA model
        eval_model = ema.m if ema is not None else model
        val_res, val_ids, val_y, val_p = evaluate_loader(eval_model, val_loader, cfg, device)
        diag_res, _, _, _ = evaluate_loader(eval_model, diag_loader, cfg, device)

        val_qwk = val_res['qwk']
        val_acc = val_res['acc']
        val_bin_auroc = val_res.get('bin_auroc', float('nan'))
        train_acc = diag_res['acc']

        curve_rows.append({
            'epoch': epoch,
            'lr': current_lr,
            'train_loss': mean_train_loss,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'val_qwk': val_qwk,
            'val_bin_auroc': val_bin_auroc,
        })
        pd.DataFrame(curve_rows).to_csv(curve_csv_path, index=False)

        logger.info(
            f"Fold {k} Ep {epoch+1:02d}/{epochs:02d} | LR {current_lr:.2e} | Loss {mean_train_loss:.4f} | "
            f"Train Acc {train_acc:.2f} | Val Acc {val_acc:.2f} | Val QWK {val_qwk:.4f} (best {best_val_qwk:.4f})"
        )

        # Early stopping check on val_qwk
        if val_qwk > best_val_qwk + 1e-4:
            best_val_qwk = val_qwk
            patience_counter = 0
            best_metrics = {
                'epoch': int(epoch),
                'val_acc': float(val_acc),
                'val_qwk': float(val_qwk),
                'val_bin_auroc': float(val_bin_auroc) if not np.isnan(val_bin_auroc) else 0.0,
            }

            # Save best checkpoint
            torch.save(eval_model.state_dict(), ckpt_path)
            with open(val_metrics_path, 'w') as f_vm:
                json.dump(best_metrics, f_vm, indent=2)

            # Save OOF arrays
            np.savez(oof_fold_path, ids=np.array(val_ids), y=val_y, P=val_p)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Fold {k} early stopping triggered at epoch {epoch+1} (patience={patience}).")
                break

    return best_metrics
