#!/usr/bin/env python3
"""MODAL-Net CLI entry point.

Usage:
  python main.py --mode MODE [--phase P] [--tag T] [--inherit T|@phase]
                 [--set k=v ...] [--fold N] [--base T|@phase]
                 [--members a,b,c] [--kind K] [--method M]
                 [--reference T|@phase] [--design parallel] [--csv path]

See MODAL_Part1.md S7b for details.
"""

import argparse
import os
import sys

# Offline cluster: never download model weights
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

ALL_MODES = [
    'make_splits', 'dedup_audit', 'detect_lesions', 'import_zoo',
    'pretrain', 'experiment', 'train', 'test', 'external_validation',
    'make_softlabels', 'inference_experiment', 'ensemble_experiment',
    'feature_balance', 'calibrate_conformal', 'phase_select',
    'seed_summary', 'master_table', 'report', 'xai',
]


def main():
    parser = argparse.ArgumentParser(
        description='MODAL-Net: Diabetic Retinopathy Grading')
    parser.add_argument('--mode', required=True, choices=ALL_MODES,
                        help='Operation mode')
    parser.add_argument('--phase', default=None, help='Phase folder name')
    parser.add_argument('--tag', default=None, help='Experiment tag')
    parser.add_argument('--inherit', default=None,
                        help='Inherit config from tag or @phase')
    parser.add_argument('--set', nargs='*', default=[],
                        help='Config overrides as k=v pairs')
    parser.add_argument('--fold', type=int, default=None,
                        help='Single fold to train (default: all)')
    parser.add_argument('--base', default=None,
                        help='Base tag for inference experiments')
    parser.add_argument('--members', default=None,
                        help='Comma-separated tags for ensemble')
    parser.add_argument('--kind', default=None,
                        help='Pretrain kind: eyepacs_sup | ssl | ssl_sup')
    parser.add_argument('--method', default=None,
                        help='Feature balance method')
    parser.add_argument('--reference', default=None,
                        help='Reference tag for phase_select')
    parser.add_argument('--design', default=None,
                        help='Experiment design (parallel)')
    parser.add_argument('--csv', default=None,
                        help='Path to CSV file (for import_zoo)')

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Modes that need only the base config (no tag / phase / inheritance)
    # ------------------------------------------------------------------
    if args.mode == 'make_splits':
        import yaml
        cfg = yaml.safe_load(open('configs/config.yaml'))
        from src.data.splits import make_splits
        make_splits(cfg)
        return

    if args.mode == 'dedup_audit':
        import yaml
        cfg = yaml.safe_load(open('configs/config.yaml'))
        from tools.dedup_audit import run_dedup_audit
        run_dedup_audit(cfg)
        return

    if args.mode == 'import_zoo':
        assert args.csv is not None, '--csv is required for import_zoo'
        assert args.phase is not None, '--phase is required for import_zoo'
        from src.utils.io_utils import import_zoo
        import_zoo(args.csv, args.phase)
        return

    # ------------------------------------------------------------------
    # Modes that need full config resolution
    # ------------------------------------------------------------------
    from src.utils.config import build_cfg
    from src.utils.seed import seed_everything
    from src.utils.logger import get_logger

    cfg = build_cfg(args)
    seed_everything(cfg['seed'])

    tag = cfg.get('tag') or 'default'
    logger = get_logger(tag, args.mode)
    logger.info(f'Mode: {args.mode}  Tag: {tag}  Phase: {cfg.get("phase")}')

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------
    if args.mode == 'detect_lesions':
        from src.preprocessing.lesion_detection import detect_lesions_batch
        P = cfg['paths']['eyepacs']
        detect_lesions_batch([P['images']], P['lesion_labels'], 128)

    elif args.mode == 'experiment':
        from src.evaluation.experiment import run_experiment
        run_experiment(cfg)

    elif args.mode == 'train':
        from src.training.trainer import train_fold
        fold = args.fold if args.fold is not None else 0
        train_fold(cfg, fold, tag, logger)

    elif args.mode == 'test':
        from src.evaluation.experiment import run_experiment
        run_experiment(cfg)

    elif args.mode == 'external_validation':
        from src.evaluation.experiment import run_experiment
        run_experiment(cfg)

    elif args.mode == 'pretrain':
        from src.training.pretrain import run_pretrain
        kind = args.kind or cfg.get('kind', 'eyepacs_sup')
        run_pretrain(cfg, kind)

    elif args.mode == 'make_softlabels':
        from src.training.softlabels import make_softlabels
        make_softlabels(cfg)

    elif args.mode == 'inference_experiment':
        from src.evaluation.experiment import run_inference_experiment
        run_inference_experiment(cfg)

    elif args.mode == 'ensemble_experiment':
        from src.evaluation.experiment import run_ensemble_experiment
        run_ensemble_experiment(cfg)

    elif args.mode == 'feature_balance':
        from src.training.feature_balance import run_feature_balance_experiment
        run_feature_balance_experiment(cfg)

    elif args.mode == 'calibrate_conformal':
        from src.evaluation.experiment import run_calibrate_conformal
        run_calibrate_conformal(cfg)

    elif args.mode == 'phase_select':
        from src.evaluation.tables import phase_select
        assert args.phase is not None, '--phase is required for phase_select'
        assert args.reference is not None, '--reference is required for phase_select'
        phase_select(args.phase, args.reference, design=getattr(args, 'design', 'parallel'))

    elif args.mode == 'seed_summary':
        from src.evaluation.tables import seed_summary
        phase = args.phase or 'phase9_final'
        seed_summary(phase)

    elif args.mode == 'master_table':
        from src.evaluation.tables import master_table
        master_table()

    elif args.mode == 'report':
        from src.evaluation.tables import master_table, build_paper_tables
        from src.evaluation.figures import build_paper_figures
        master_table()
        build_paper_tables()
        build_paper_figures()

    elif args.mode == 'xai':
        from src.xai.grid import run_xai_experiment
        run_xai_experiment(cfg)

    else:
        raise ValueError(f'Unknown mode: {args.mode}')


if __name__ == '__main__':
    main()
