"""Rebuild deleted prediction artefacts from surviving checkpoints (inference only).

A phase folder under results/ can lose its artefacts (oof_foldK.npz, oof.npz,
predictions_*.npz, summary.csv, WINNER.txt) while checkpoints/<tag>/ survives.
train_fold() only skips training when foldK.pth AND foldK_val_metrics.json AND
oof_foldK.npz all exist (trainer.py:185), so re-running 'experiment' after such a
loss silently retrains every fold from scratch. This tool instead regenerates
oof_foldK.npz from the surviving foldK.pth using the exact code path the trainer
used: evaluate_loader() with no TTA, same val loader settings.

That is faithful, not approximate: foldK.pth is saved together with oof_foldK.npz
from the same eval_model object (trainer.py:434-439), so the checkpoint holds the
very weights (EMA weights when ema_decay > 0) whose validation predictions were
recorded. Once the per-fold files exist, run_experiment() resumes normally: it
skips all five folds and re-derives thresholds, the fitted temperature, the 5-model
ensemble test/DEV predictions, per_fold.csv, summary.csv and config.yaml. Per-fold
metrics (foldK_val_metrics.json) and the training curves are never touched.

Usage (on a GPU node):
  python tools/rebuild_oof.py --tag p2a_reg --phase phase2a_recipe
  python tools/rebuild_oof.py --tag p2a_reg --phase phase2a_recipe --folds 0,1 --no-experiment
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import yaml
from sklearn.metrics import cohen_kappa_score
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.datasets import DRDataset
from src.data.splits import fold_frames
from src.models.gradenet import GradeNet
from src.training.trainer import evaluate_loader
from src.utils.config import validate_cfg
from src.utils.seed import seed_everything


def rebuild_fold(cfg, tag, phase, k, logger):
    """Regenerate results/<phase>/<tag>/oof_fold<k>.npz from checkpoints/<tag>/fold<k>.pth."""
    res_dir = os.path.join('results', phase, tag)
    oof_path = os.path.join(res_dir, f'oof_fold{k}.npz')
    ckpt_dir = os.path.join('checkpoints', tag)
    pth = os.path.join(ckpt_dir, f'fold{k}.pth')
    metrics_json = os.path.join(ckpt_dir, f'fold{k}_val_metrics.json')

    if os.path.isfile(oof_path):
        logger.info(f'rebuild fold {k}: artefact already present, nothing to do')
        return
    if not os.path.isfile(pth):
        raise FileNotFoundError(f'{tag} fold {k}: missing checkpoint {pth}')
    if not os.path.isfile(metrics_json):
        raise FileNotFoundError(f'{tag} fold {k}: missing {metrics_json} (per-fold metrics cannot be rebuilt)')

    _tr_df, va_df = fold_frames(cfg, k)
    aptos = cfg['paths']['aptos']
    aptos_dirs = [aptos['train_images'], aptos['val_images']]
    # Mirrors trainer.py:244-251 exactly so the probabilities match the originals.
    val_loader = DataLoader(
        DRDataset(va_df, aptos_dirs, cfg, train=False, domain=0),
        batch_size=int(cfg.get('batch_size', 16)),
        shuffle=False,
        drop_last=False,
        num_workers=int(cfg.get('num_workers', 4)),
        pin_memory=torch.cuda.is_available(),
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = GradeNet(cfg)
    model.load_state_dict(torch.load(pth, map_location='cpu'))
    model.to(device)
    _res, ids, y, P = evaluate_loader(model, val_loader, cfg, device)
    del model
    if device == 'cuda':
        torch.cuda.empty_cache()

    os.makedirs(res_dir, exist_ok=True)
    np.savez(oof_path, ids=np.array(ids), y=y, P=P)

    rebuilt_qwk = float(cohen_kappa_score(y, P.argmax(1), weights='quadratic'))
    expected_qwk = float(json.load(open(metrics_json))['val_qwk'])
    flag = 'OK' if abs(rebuilt_qwk - expected_qwk) < 5e-3 else 'MISMATCH'
    logger.info(
        f'rebuild fold {k}: wrote {oof_path} (n={len(ids)}) '
        f'QWK rebuilt={rebuilt_qwk:.4f} recorded={expected_qwk:.4f} -> {flag}'
    )


def main():
    ap = argparse.ArgumentParser(description='Rebuild results/ artefacts from checkpoints (no retraining).')
    ap.add_argument('--tag', required=True)
    ap.add_argument('--phase', required=True)
    ap.add_argument('--folds', default=None, help='Comma-separated fold ids (default: all)')
    ap.add_argument('--no-experiment', action='store_true',
                    help='Only rebuild the per-fold OOF files; skip run_experiment().')
    args = ap.parse_args()

    snap = os.path.join('checkpoints', args.tag, 'config_snapshot.yaml')
    if not os.path.isfile(snap):
        raise FileNotFoundError(f'No config snapshot for tag {args.tag}: {snap}')
    cfg = yaml.safe_load(open(snap))
    # The snapshot is the resolved config the folds were actually trained with.
    cfg['tag'] = args.tag
    cfg['phase'] = args.phase
    validate_cfg(cfg)
    seed_everything(int(cfg.get('seed', 0)))

    from src.utils.logger import get_logger
    logger = get_logger(args.tag, 'rebuild')
    logger.info(f'=== Rebuilding {args.tag} in {args.phase} from checkpoints (preproc={cfg.get("preproc")}) ===')

    folds = ([int(x) for x in args.folds.split(',')] if args.folds
             else list(range(int(cfg.get('n_folds', 5)))))
    for k in folds:
        rebuild_fold(cfg, args.tag, args.phase, k, logger)

    if args.no_experiment:
        logger.info('Per-fold OOF files rebuilt; --no-experiment given, stopping.')
        return

    from src.evaluation.experiment import run_experiment
    run_experiment(cfg)


if __name__ == '__main__':
    main()
