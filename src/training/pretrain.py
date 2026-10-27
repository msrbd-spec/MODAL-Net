"""Pretraining: EyePACS supervised pretraining and Lesion-Aware SSL pretraining.

Supports three pretraining modes (S7m):
  - eyepacs_sup: EyePACS frame (dedup-excluded), 2% held out, GradeNet from ImageNet,
                 sqrt-inverse class sampler, soft_ce loss with ordinal smoothing,
                 LLRD, EMA, pre_epochs -> saves checkpoints/pretrain/eyepacs_sup.pth.
  - ssl: Lesion-aware multi-task SSL on EyePACS using SSLModel and SSLTrainer
         -> saves checkpoints/pretrain/ssl_maxvit/ssl_pretrained_backbone.pth.
  - ssl_sup: Same as eyepacs_sup, but initialized with ssl_init (the SSL backbone)
             -> saves checkpoints/pretrain/ssl_sup.pth.
"""

import math
import os
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import yaml

try:
    from torch.amp import GradScaler, autocast
    _NEW_AMP = True
except ImportError:
    from torch.cuda.amp import GradScaler, autocast
    _NEW_AMP = False

from src.data.datasets import DRDataset
from src.data.samplers import SqrtInverseClassSampler
from src.data.splits import eyepacs_frame
from src.evaluation.metrics import qwk_score
from src.losses.losses import soft_ce
from src.models.gradenet import GradeNet, llrd_groups, probs_from_logits
from src.models.ssl_model import SSLModel
from src.training.ema import EMA
from src.training.ssl_trainer import SSLTrainer
from src.utils.logger import get_logger


def get_eyepacs_exclude_ids():
    """Retrieve EyePACS image IDs identified as duplicates during dedup audit."""
    dup_csv = os.path.join('results', 'phase0_protocol', 'audit', 'duplicates.csv')
    exclude = set()
    if os.path.isfile(dup_csv):
        df = pd.read_csv(dup_csv)
        for _, row in df.iterrows():
            if row.get('dataset_a') == 'EyePACS' and row.get('dataset_b') != 'EyePACS':
                exclude.add(str(row.get('id_a')))
            if row.get('dataset_b') == 'EyePACS' and row.get('dataset_a') != 'EyePACS':
                exclude.add(str(row.get('id_b')))
    return exclude


def run_pretrain(cfg: dict, kind: str):
    """Execute pretraining workflow for the specified kind."""
    tag = cfg.get('tag') or f'pre_{kind}'
    logger = get_logger(tag, 'pretrain')
    logger.info(f"Starting pretraining: kind={kind}, tag={tag}")

    # Snapshot config
    ckpt_dir = os.path.join('checkpoints', tag)
    os.makedirs(ckpt_dir, exist_ok=True)
    snap_path = os.path.join(ckpt_dir, 'config_snapshot.yaml')
    with open(snap_path, 'w') as f_snap:
        yaml.dump(cfg, f_snap)
    logger.info(f"Config snapshot saved to {snap_path}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_type = device.type if hasattr(device, 'type') else ('cuda' if 'cuda' in str(device) else 'cpu')
    is_cuda = (device_type == 'cuda')

    # ------------------------------------------------------------------
    # Kind: ssl
    # ------------------------------------------------------------------
    if kind == 'ssl':
        from src.data.eyepacs_dataset import EyePACSDataset

        image_dirs = cfg.get('ssl_image_dirs') or [cfg['paths']['eyepacs']['images']]
        lesion_dir = cfg.get('ssl_lesion_label_dir') or cfg['paths']['eyepacs']['lesion_labels']
        img_size = int(cfg.get('ssl_img_size') or cfg.get('img_size', 512))
        mask_size = int(cfg.get('ssl_mask_size', 128))

        logger.info(f"Initializing EyePACSDataset from {image_dirs}")
        dataset = EyePACSDataset(
            image_dirs=image_dirs,
            lesion_label_dir=lesion_dir,
            img_size=img_size,
            mask_size=mask_size
        )

        batch_size = int(cfg.get('ssl_batch_size', 8))
        num_workers = int(cfg.get('ssl_num_workers', cfg.get('num_workers', 4)))
        train_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=is_cuda,
            drop_last=True
        )

        backbone_name = str(cfg.get('ssl_backbone') or cfg.get('backbone', 'maxvit_large_tf_512.in1k'))
        model = SSLModel(
            backbone_name=backbone_name,
            projection_dim=int(cfg.get('ssl_projection_dim', 128)),
            use_contrastive=bool(cfg.get('ssl_use_contrastive', True)),
            use_multitask=bool(cfg.get('ssl_use_multitask', True)),
            grad_checkpointing=bool(cfg.get('grad_checkpointing', False)),
            pretrained_fallback=str(cfg.get('pretrained_fallback', 'strict')),
        )
        if not getattr(model, 'pretrained_ok', True):
            logger.warning("BACKBONE RANDOMLY INITIALISED - pretrained weights were not loaded")

        trainer = SSLTrainer(
            model=model,
            train_loader=train_loader,
            config=cfg,
            device=device,
            logger=logger
        )
        trainer.train()
        return

    # ------------------------------------------------------------------
    # Kind: eyepacs_sup or ssl_sup
    # ------------------------------------------------------------------
    if kind not in ('eyepacs_sup', 'ssl_sup'):
        raise ValueError(f"Unknown pretrain kind: {kind!r}. Expected eyepacs_sup, ssl, or ssl_sup.")

    exclude_ids = get_eyepacs_exclude_ids()
    logger.info(f"EyePACS deduplication excluded {len(exclude_ids)} candidate duplicate images.")
    df = eyepacs_frame(cfg, exclude_ids=exclude_ids)
    if cfg.get('eyepacs_max') and len(df) > int(cfg['eyepacs_max']):
        logger.info(f"Limiting EyePACS dataset from {len(df)} to eyepacs_max={cfg['eyepacs_max']}")
        df = df.iloc[:int(cfg['eyepacs_max'])].reset_index(drop=True)

    # 2% held-out monitoring split
    train_df, val_df = train_test_split(
        df,
        test_size=0.02,
        random_state=int(cfg.get('seed', 0)),
        stratify=df['diagnosis']
    )
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    logger.info(f"EyePACS pretraining split: {len(train_df)} train, {len(val_df)} val (2% holdout)")

    eyepacs_img_dir = cfg['paths']['eyepacs']['images']
    train_ds = DRDataset(train_df, dirs=eyepacs_img_dir, cfg=cfg, train=True)
    val_ds = DRDataset(val_df, dirs=eyepacs_img_dir, cfg=cfg, train=False)

    sampler = SqrtInverseClassSampler(train_df['diagnosis'].values, seed=int(cfg.get('seed', 0)))
    batch_size = int(cfg.get('batch_size', 16))
    num_workers = int(cfg.get('num_workers', 4))

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=is_cuda,
        drop_last=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=is_cuda
    )

    # Build model
    sup_cfg = dict(cfg)
    if kind == 'eyepacs_sup':
        sup_cfg['pretrain_path'] = None
    elif kind == 'ssl_sup':
        ssl_init = cfg.get('ssl_init')
        if not ssl_init:
            raise ValueError("kind='ssl_sup' requires --set ssl_init=<path_to_ssl_backbone>")
        if not os.path.isfile(ssl_init):
            raise FileNotFoundError(f"ssl_init file not found: {ssl_init}")
        sup_cfg['pretrain_path'] = ssl_init

    model = GradeNet(sup_cfg).to(device)

    # Optimizer with LLRD
    lr = float(cfg.get('lr', 1.0e-4))
    decay = float(cfg.get('llrd', 0.8))
    wd = float(cfg.get('weight_decay', 0.05))
    head_mult = float(cfg.get('head_mult', 10.0))
    param_groups = llrd_groups(model, lr=lr, decay=decay, wd=wd, head_mult=head_mult)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=wd)

    # Cosine scheduler with warmup
    epochs = int(cfg.get('pre_epochs', 15))
    warmup_epochs = int(cfg.get('warmup_epochs', 2))

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / max(1, warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, (epochs - warmup_epochs))
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # EMA
    ema = EMA(model, decay=float(cfg.get('ema_decay', 0.999)))

    # AMP
    amp_enabled = is_cuda
    if _NEW_AMP:
        scaler = GradScaler(device_type, enabled=amp_enabled)
        autocast_ctx = autocast(device_type, enabled=amp_enabled)
    else:
        scaler = GradScaler(enabled=amp_enabled)
        autocast_ctx = autocast(enabled=amp_enabled)

    grad_clip = float(cfg.get('grad_clip', 1.0))
    grad_accum = int(cfg.get('grad_accum', 1))

    best_qwk = -1.0
    best_backbone_state = {k: v.cpu().clone() for k, v in model.backbone.state_dict().items()}

    logger.info("=" * 60)
    logger.info(f"Starting Supervised Pretraining ({kind}) for {epochs} epochs")
    logger.info("=" * 60)

    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        n_batches = 0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            imgs = batch['img'].to(device, non_blocking=True)
            targets = batch['target'].to(device, non_blocking=True)

            with autocast_ctx:
                logits = model(imgs)
                loss = soft_ce(logits, targets) / grad_accum

            scaler.scale(loss).backward()

            if (step + 1) % grad_accum == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                ema.update(model)

            train_loss += loss.item() * grad_accum
            n_batches += 1

        scheduler.step()
        train_loss /= max(1, n_batches)

        # Validation on held-out 2%
        val_eval_model = ema.eval_model(model)
        val_eval_model.eval()
        val_loss = 0.0
        val_batches = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in val_loader:
                imgs = batch['img'].to(device, non_blocking=True)
                targets = batch['target'].to(device, non_blocking=True)
                y_true = batch['y'].numpy()

                with autocast_ctx:
                    logits = val_eval_model(imgs)
                    loss = soft_ce(logits, targets)

                probs = probs_from_logits(logits, head_type=cfg.get('head_type', 'softmax')).cpu().numpy()
                preds = np.argmax(probs, axis=1)

                val_loss += loss.item()
                val_batches += 1
                all_preds.extend(preds)
                all_targets.extend(y_true)

        val_loss /= max(1, val_batches)
        val_qwk = qwk_score(np.array(all_targets), np.array(all_preds))
        elapsed = time.time() - t0

        logger.info(
            f"Pretrain Epoch {epoch+1:02d}/{epochs:02d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val QWK: {val_qwk:.4f} | "
            f"Time: {elapsed:.1f}s"
        )

        if val_qwk > best_qwk:
            best_qwk = val_qwk
            best_backbone_state = {k: v.cpu().clone() for k, v in val_eval_model.backbone.state_dict().items()}

    # Save final backbone weights
    save_path = 'checkpoints/pretrain/eyepacs_sup.pth' if kind == 'eyepacs_sup' else 'checkpoints/pretrain/ssl_sup.pth'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(best_backbone_state, save_path)
    logger.info(f"Pretraining complete. Saved {kind} backbone to {save_path} (best val QWK: {best_qwk:.4f})")
