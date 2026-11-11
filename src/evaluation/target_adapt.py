"""Target-site adaptation baselines on a finished run (no labels, no retraining of the run itself).

Standard "adapt the network to a new site" comparators for the post-hoc cut-point recalibration. Protocol
(identical for every method, fixed a priori; nothing is tuned on TEST):

  * only the held-out external datasets of the run are used (datasets that are not training sources,
    src.evaluation.evaluate.heldout_datasets);
  * adaptation sees ONLY the unlabelled images of the target site's DEV split (labels are never read
    by the adaptation code: the loaders return them, the adaptation loops ignore them);
  * the adapted fold models then predict DEV and TEST of that site; TEST is only predicted and scored;
  * every fold model is adapted separately from its own checkpoint (the checkpoint file is never written),
    the ensemble is the mean over folds and the run's own OOF temperature is applied exactly as
    src.evaluation.evaluate does for the normal prediction files.

Methods (--mode target_adapt --tag X --set target_adapt_method=bn|tent):

  bn    BatchNorm statistics adaptation (Li et al. 2016 "AdaBN"; Schneider et al. 2020). The running mean /
        variance of every BatchNorm layer (that tracks running statistics) is reset and recomputed from the
        target DEV images in ONE pass (every DEV image once, batches in a fixed random order, seed cfg
        seed + 1) with a cumulative average weighted by batch size (adapt_bn); only the BatchNorm modules
        are in train mode, everything else (dropout, stochastic depth, MixStyle) stays in eval mode; no
        gradients. LayerNorm layers have no statistics and are unchanged.
        All parameters are unchanged; only BatchNorm buffers change.

  tent  TENT (Wang et al., ICLR 2021). Starts from the bn-adapted model; optimises only the affine
        parameters (weight / bias) of the normalisation layers (BatchNorm and LayerNorm, incl. timm
        LayerNorm2d) by minimising the mean Shannon entropy of the predicted class probabilities on the
        unlabelled target DEV images. Adam, lr = cfg tent_lr (default 1e-4), cfg tent_epochs passes
        (default 1), batch = cfg batch_size, DEV images shuffled with a fixed seed (cfg seed), last
        incomplete batch dropped (batch statistics of a 1-image batch are degenerate). As in TENT,
        BatchNorm layers normalise with the statistics of the current batch during adaptation (train
        mode); their running buffers are frozen during the optimisation (momentum 0) and, because the
        upstream affine parameters moved, re-estimated once more with the bn procedure on the same DEV
        images after the last step, so that the eval-mode model is consistent with its final parameters.
        Probabilities are produced exactly as src.evaluation.predict does (probs_from_logits; for
        coral / coral_src heads these are the 5 class probabilities from the cumulative logits, and
        coral_src receives the image's domain id, which for a held-out site selects the shared cut-points).
        On CUDA the adaptation forward/backward uses bf16 autocast like training; prediction is fp32 like
        evaluate. cfg grad_checkpointing=true (GradeNet -> timm set_grad_checkpointing) reduces memory.

Outputs (results/<phase>/<tag>/target_adapt/<method>/; the run's checkpoints and normal result files are
never modified):
  predictions_<d>_dev.npz, predictions_<d>_test.npz   same keys as the normal prediction files
                       (ids, y_true, y_pred, y_prob [temperature-scaled], y_prob_raw, y_prob_folds, y, P,
                       P_per_model), same image order as the frozen split files
  summary.csv          one row per set: argmax grade metrics (cutpoints.grade_metrics), referable AUROC,
                       and the same metrics of the unadapted run (from its normal prediction files)
  adapt_log.csv        one row per (dataset, fold): BatchNorm layers re-estimated, adapted parameters,
                       mean DEV prediction entropy before / after, seconds
  config.yaml          configuration used

--mode extract_features --tag X (CE / softmax heads only) writes results/<phase>/<tag>/features/:
  <d>_dev.npz, <d>_test.npz   ids, y_true, feats (n_folds, N, D) float16 = pooled feature entering the
                              final nn.Linear of each fold model (eval mode, no TTA)
  head.npz                    W (n_folds, K, D) float32, b (n_folds, K) float32, T (OOF temperature),
                              use_T, heldout datasets
The normal prediction of the run is reproduced exactly by
  y_prob_folds[k] = softmax(feats[k] @ W[k].T + b[k]),   y_prob = apply_temperature(mean_k y_prob_folds[k], T)
(evaluate.py applies the temperature to the fold-averaged probabilities); this is verified in code against
the saved normal prediction files whenever they were computed without TTA.
"""

import gc
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from src.data.datasets import DRDataset
from src.data.external import eval_frame
from src.evaluation.cutpoints import grade_metrics
from src.evaluation.metrics import apply_temperature
from src.models.gradenet import ORDINAL_HEADS, probs_from_logits

METHODS = ('bn', 'tent')
SUMMARY_GRADE_METRICS = ('qwk', 'acc', 'f1_macro', 'mild_recall', 'bin_sens', 'bin_spec', 'pred_nodr_share')


# ---------------------------------------------------------------------------
# Normalisation layers
# ---------------------------------------------------------------------------

def bn_layers(model):
    """BatchNorm modules (incl. timm BatchNormAct2d) that track running statistics."""
    return [m for m in model.modules()
            if isinstance(m, nn.modules.batchnorm._BatchNorm) and m.track_running_stats]


def _is_norm(m):
    return isinstance(m, (nn.modules.batchnorm._BatchNorm, nn.LayerNorm))


def norm_affine_params(model):
    """[(qualified name, parameter)] of the affine weight / bias of every BatchNorm and LayerNorm layer
    (timm LayerNorm2d and BatchNormAct2d subclass these)."""
    out = []
    for mname, m in model.named_modules():
        if not _is_norm(m):
            continue
        for pname, p in m.named_parameters(recurse=False):
            if pname in ('weight', 'bias') and p is not None:
                out.append((f'{mname}.{pname}' if mname else pname, p))
    return out


def _forward_probs(model, x, dom, head_type):
    """Class probabilities exactly as src.evaluation.predict.predict_probs computes them."""
    z = model(x, domain=dom) if head_type == 'coral_src' else model(x)
    return probs_from_logits(z.float(), head_type)


def _entropy(P):
    return -(P * torch.log(P.clamp_min(1e-12))).sum(1)


def _batch(batch, device):
    x = batch['x'].to(device)
    dom = batch['domain'].to(device) if 'domain' in batch else None
    return x, dom


@torch.no_grad()
def adapt_bn(model, loader, device, head_type='softmax'):
    """Reset and recompute the running statistics of every BatchNorm layer from the images of `loader`
    (one pass, cumulative average weighted by batch size: before a batch of b images, after s images seen
    in total including it, momentum = b / s, so running_mean is exactly the image-weighted mean of the
    batch means and running_var the image-weighted mean of the (unbiased) within-batch variances; a
    small last batch is not over-weighted). Only the BatchNorm modules themselves are put in train mode
    (not their children, e.g. the dropout of timm BatchNormAct2d). Batches should be random samples of
    the images (run_target_adapt passes a shuffled loader with a fixed seed): batches of consecutive
    split-file rows hold the images of one patient / exam together and under-estimate the variance.
    Returns the number of BatchNorm layers."""
    bns = bn_layers(model)
    model.eval()
    if not bns:
        return 0
    moms = [m.momentum for m in bns]
    for m in bns:
        m.reset_running_stats()
        m.momentum = None
        m.training = True                       # this module only (no recursion)
    seen = 0
    try:
        for batch in loader:
            x, dom = _batch(batch, device)
            seen += int(x.shape[0])
            for m in bns:
                m.momentum = float(x.shape[0]) / seen   # first batch: 1.0 (running stats = its statistics)
            if head_type == 'coral_src':
                model(x, domain=dom)
            else:
                model(x)
    finally:
        for m, mom in zip(bns, moms):
            m.momentum = mom
        model.eval()
    return len(bns)


@torch.no_grad()
def mean_entropy(model, loader, device, head_type='softmax'):
    """Mean entropy of the eval-mode class probabilities over the images of `loader` (no TTA)."""
    model.eval()
    tot, n = 0.0, 0
    for batch in loader:
        x, dom = _batch(batch, device)
        P = _forward_probs(model, x, dom, head_type)
        tot += float(_entropy(P).sum())
        n += int(P.shape[0])
    return tot / max(n, 1)


def adapt_tent(model, loader, device, head_type='softmax', lr=1e-4, epochs=1, stats_loader=None):
    """TENT on a (bn-adapted) model: Adam on the normalisation affine parameters only, mean prediction
    entropy on the images of `loader`; BatchNorm in batch-statistics mode with frozen running buffers;
    afterwards the BatchNorm statistics are re-estimated on `stats_loader` (default: `loader`).
    Returns {'n_params', 'n_tensors', 'steps', 'entropy_first', 'entropy_last'}."""
    named = norm_affine_params(model)
    params = [p for _, p in named]
    req = {id(p): p.requires_grad for p in model.parameters()}
    for p in model.parameters():
        p.requires_grad_(False)
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.Adam(params, lr=float(lr))
    bns = bn_layers(model)
    moms = [m.momentum for m in bns]
    model.eval()
    for m in bns:
        m.momentum = 0.0                        # batch statistics for normalisation, buffers untouched
        m.training = True
    use_autocast = str(device).startswith('cuda')
    steps, ent_first, ent_last = 0, float('nan'), float('nan')
    try:
        for _ in range(int(epochs)):
            for batch in loader:
                x, dom = _batch(batch, device)
                if use_autocast:
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        z = model(x, domain=dom) if head_type == 'coral_src' else model(x)
                else:
                    z = model(x, domain=dom) if head_type == 'coral_src' else model(x)
                loss = _entropy(probs_from_logits(z.float(), head_type)).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                if steps == 0:
                    ent_first = float(loss.detach())
                ent_last = float(loss.detach())
                steps += 1
    finally:
        for m, mom in zip(bns, moms):
            m.momentum = mom
        for p in model.parameters():
            p.requires_grad_(req.get(id(p), True))
        opt.zero_grad(set_to_none=True)
        del opt
        model.eval()
    adapt_bn(model, stats_loader if stats_loader is not None else loader, device, head_type)
    return {'n_params': int(sum(p.numel() for p in params)), 'n_tensors': len(params), 'steps': steps,
            'entropy_first': ent_first, 'entropy_last': ent_last}


# ---------------------------------------------------------------------------
# Run context (snapshot config, checkpoints, temperature) shared by both modes
# ---------------------------------------------------------------------------

def _loader(cfg, set_name, shuffle=False, seed=0, drop_last=False):
    """Eval-transform loader of an evaluation set (same preprocessing as evaluate.make_loader)."""
    df, dirs, domain = eval_frame(cfg, set_name)
    ds = DRDataset(df, dirs, cfg, train=False, domain=domain)
    bs = int(cfg.get('batch_size', 16))
    g = None
    if shuffle:
        g = torch.Generator()
        g.manual_seed(int(seed))
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, generator=g,
                      drop_last=bool(drop_last and len(ds) > bs), num_workers=int(cfg.get('num_workers', 4)))


def stats_loader_for(cfg, set_name):
    """Loader for BatchNorm statistics: every image of the set once, in a fixed random order (seed cfg
    seed + 1), so that batches are random samples of the set and not runs of one patient's / exam's
    consecutive split-file rows (which would bias the BatchNorm statistics)."""
    return _loader(cfg, set_name, shuffle=True, seed=int(cfg.get('seed', 0)) + 1)


def run_context(cfg):
    """tag, phase, exp_dir, owner of the checkpoints, n_folds, OOF temperature, held-out datasets."""
    from src.evaluation.evaluate import heldout_datasets
    from src.utils.config import ckpt_owner, find_exp
    tag = cfg['tag']
    if not tag:
        raise ValueError('--tag is required')
    exp_dir = find_exp(tag)
    phase = os.path.basename(os.path.dirname(exp_dir))
    snap_path = os.path.join('checkpoints', tag, 'config_snapshot.yaml')
    snap = (yaml.safe_load(open(snap_path)) or {}) if os.path.isfile(snap_path) else {}
    if snap.get('members'):
        raise ValueError(f'{tag} is an ensemble of runs ({snap["members"]}); adapt its member runs instead')
    owner = ckpt_owner(tag)
    n_folds = int(cfg.get('n_folds', 5))
    thr_path = None
    for cand in (os.path.join('checkpoints', tag, 'thresholds.json'),
                 os.path.join('checkpoints', owner, 'thresholds.json')):
        if os.path.isfile(cand):
            thr_path = cand
            break
    if thr_path is None:
        raise FileNotFoundError(f'{tag}: no thresholds.json in checkpoints/{tag} or checkpoints/{owner}')
    fitted_T = float(json.load(open(thr_path)).get('temperature', 1.0))
    use_T = bool(cfg.get('temperature_scaling', True)) and fitted_T is not None and fitted_T > 0
    for k in range(n_folds):
        p = os.path.join('checkpoints', owner, f'fold{k}.pth')
        if not os.path.isfile(p):
            raise FileNotFoundError(f'missing checkpoint {p}')
    return {'tag': tag, 'phase': phase, 'exp_dir': exp_dir, 'owner': owner, 'n_folds': n_folds,
            'T': fitted_T, 'use_T': use_T, 'heldout': heldout_datasets(cfg)}


def load_fold_model(cfg, owner, k):
    """Fresh fold model from checkpoints/<owner>/fold<k>.pth (read only), eval mode, on CPU."""
    from src.models.gradenet import GradeNet
    m = GradeNet(dict(cfg, pretrained=False, pretrain_path=None))
    m.load_state_dict(torch.load(os.path.join('checkpoints', owner, f'fold{k}.pth'), map_location='cpu'))
    m.eval()
    return m


def _cleanup():
    """Release memory after the caller dropped its last reference to a fold model."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _saved_normal(exp_dir, set_name):
    """Normal prediction file of the run for a set (dict of arrays) or None."""
    from src.evaluation.evaluate import LEGACY_FILES
    for name in (f'predictions_{set_name}.npz', LEGACY_FILES.get(set_name)):
        if name and os.path.isfile(os.path.join(exp_dir, name)):
            with np.load(os.path.join(exp_dir, name), allow_pickle=True) as f:
                return {k: f[k] for k in f.files}
    return None


def _plain_tta(cfg):
    """True when predictions are a single eval-mode view (no flips / rotations / rescaling)."""
    scales = cfg.get('tta_scales') or [1.0]
    if not isinstance(scales, (list, tuple)):
        scales = [scales]
    return str(cfg.get('tta') or 'none') == 'none' and all(abs(float(x) - 1.0) < 1e-5 for x in scales)


def _set_metrics(y, P):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y, dtype=int)
    out = {m: v for m, v in grade_metrics(y, np.asarray(P).argmax(1)).items() if m in SUMMARY_GRADE_METRICS}
    yb = (y >= 2).astype(int)
    out['bin_auroc'] = float(roc_auc_score(yb, np.asarray(P)[:, 2:].sum(1))) if 0 < yb.sum() < len(yb) \
        else float('nan')
    return out


# ---------------------------------------------------------------------------
# --mode target_adapt
# ---------------------------------------------------------------------------

def run_target_adapt(cfg):
    """BN / TENT adaptation of every fold model to every held-out site of run cfg['tag'] (see module doc)."""
    from src.evaluation.evaluate import _save_npz
    from src.evaluation.predict import predict_probs
    from src.utils.logger import get_logger
    method = cfg.get('target_adapt_method', 'bn')
    if method not in METHODS:
        raise ValueError(f'target_adapt_method must be one of {METHODS}, got {method!r}')
    ctx = run_context(cfg)
    tag, exp_dir = ctx['tag'], ctx['exp_dir']
    logger = get_logger(tag, 'target_adapt')
    head_type = cfg.get('head_type', 'softmax')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    out_dir = os.path.join(exp_dir, 'target_adapt', method)
    os.makedirs(out_dir, exist_ok=True)
    lr, epochs, seed = float(cfg.get('tent_lr', 1e-4)), int(cfg.get('tent_epochs', 1)), int(cfg.get('seed', 0))
    logger.info(f'=== target_adapt {tag} method={method} (checkpoints of {ctx["owner"]}, {ctx["n_folds"]} folds, '
                f'head={head_type}, T={ctx["T"]:.4f} applied={ctx["use_T"]}, tta={cfg.get("tta", "none")}, '
                f'held-out sites={ctx["heldout"]}, device={device}) ===')
    if method == 'tent':
        logger.info(f'  TENT: Adam lr={lr:g}, epochs={epochs}, batch={cfg.get("batch_size")}, '
                    f'grad_checkpointing={bool(cfg.get("grad_checkpointing"))}')
    if cfg.get('recalibrate', 'none') != 'none':
        logger.info(f'  note: recalibrate={cfg.get("recalibrate")} of the run is NOT applied to adapted predictions')
    if not ctx['heldout']:
        logger.info('  the run has no held-out external dataset: nothing to adapt')

    plain_tta = _plain_tta(cfg)
    sum_rows, log_rows = [], []
    for d in ctx['heldout']:
        dv, ts = f'{d}_dev', f'{d}_test'
        eval_loaders = {dv: _loader(cfg, dv), ts: _loader(cfg, ts)}
        stats_loader = stats_loader_for(cfg, dv)
        tent_loader = _loader(cfg, dv, shuffle=True, seed=seed, drop_last=True) if method == 'tent' else None
        per_fold = {dv: [], ts: []}
        ids, ys = {}, {}
        # entropy of the unadapted fold models: from the run's saved single-view DEV predictions when possible
        dev_ids = [str(i) for i in stats_loader.dataset.df['id_code']]
        orig_dev = _saved_normal(exp_dir, dv)
        orig_folds = None
        if plain_tta and orig_dev is not None and [str(i) for i in orig_dev['ids']] == dev_ids and \
                np.asarray(orig_dev.get('y_prob_folds', np.empty(0))).shape[:2] == (ctx['n_folds'], len(dev_ids)):
            orig_folds = np.asarray(orig_dev['y_prob_folds'], dtype=np.float64)
        for k in range(ctx['n_folds']):
            t0 = time.time()
            model = load_fold_model(cfg, ctx['owner'], k).to(device)
            if orig_folds is not None:
                ent0 = float(_entropy(torch.as_tensor(orig_folds[k])).mean())
            else:
                ent0 = mean_entropy(model, stats_loader, device, head_type)
            n_bn = adapt_bn(model, stats_loader, device, head_type)
            info = {'n_params': 0, 'n_tensors': 0, 'steps': 0, 'entropy_first': np.nan, 'entropy_last': np.nan}
            ent_bn = np.nan
            if method == 'tent':
                ent_bn = mean_entropy(model, stats_loader, device, head_type)
                info = adapt_tent(model, tent_loader, device, head_type, lr=lr, epochs=epochs,
                                  stats_loader=stats_loader)
            for s, ld in eval_loaders.items():
                i_s, y_s, _, P_s = predict_probs([model], ld, cfg, tta=cfg.get('tta'), scales=cfg.get('tta_scales'),
                                                 device=device)
                per_fold[s].append(np.asarray(P_s[0]))
                if s in ids and ids[s] != [str(i) for i in i_s]:
                    raise RuntimeError(f'{s}: image order differs between folds')
                ids[s], ys[s] = [str(i) for i in i_s], np.asarray(y_s, dtype=int)
            if plain_tta:      # the DEV prediction is exactly one eval-mode pass over every DEV image
                Pd = torch.as_tensor(per_fold[dv][-1], dtype=torch.float64)
                ent1 = float(_entropy(Pd).mean())
            else:
                ent1 = mean_entropy(model, stats_loader, device, head_type)
            row = {'dataset': d, 'fold': k, 'method': method, 'n_dev_images': len(stats_loader.dataset),
                   'n_bn_layers': n_bn, 'n_adapted_params': info['n_params'],
                   'n_adapted_tensors': info['n_tensors'], 'tent_steps': info['steps'],
                   'dev_entropy_orig': ent0, 'dev_entropy_bn': ent_bn if method == 'tent' else ent1,
                   'dev_entropy_final': ent1, 'tent_batch_entropy_first': info['entropy_first'],
                   'tent_batch_entropy_last': info['entropy_last'], 'seconds': round(time.time() - t0, 1)}
            log_rows.append(row)
            logger.info(f'  {d} fold {k}: {n_bn} BatchNorm layers re-estimated on {row["n_dev_images"]} DEV images'
                        + (f', TENT adapted {info["n_params"]} parameters in {info["n_tensors"]} norm tensors '
                           f'({info["steps"]} steps)' if method == 'tent' else ', 0 parameters changed')
                        + f'; DEV entropy {ent0:.4f} -> {ent1:.4f} ({row["seconds"]} s)')
            if n_bn == 0:
                logger.info(f'  {d} fold {k}: WARNING the model has no BatchNorm layer (bn adaptation is a no-op)')
            del model
            _cleanup()
        for s in (dv, ts):
            P_folds = np.asarray(per_fold[s])
            P_raw = P_folds.mean(axis=0)
            P = apply_temperature(P_raw, ctx['T']) if ctx['use_T'] else P_raw
            _save_npz(os.path.join(out_dir, f'predictions_{s}.npz'), ids[s], ys[s], P, P_raw, P_folds)
            met = _set_metrics(ys[s], P)
            r = {'tag': tag, 'method': method, 'dataset': d, 'set': s, 'split': s.rsplit('_', 1)[1],
                 'n': len(ys[s]), 'n_folds': ctx['n_folds'], 'temperature': ctx['T'] if ctx['use_T'] else 1.0,
                 'tta': cfg.get('tta', 'none'), **met}
            orig = _saved_normal(exp_dir, s)
            if orig is not None and [str(i) for i in orig['ids']] == ids[s]:
                om = _set_metrics(ys[s], np.asarray(orig['y_prob'], dtype=float))
                r.update({f'orig_{m}': v for m, v in om.items()})
            r.update({'n_bn_layers': int(np.mean([x['n_bn_layers'] for x in log_rows if x['dataset'] == d])),
                      'n_adapted_params': int(np.mean([x['n_adapted_params'] for x in log_rows
                                                       if x['dataset'] == d]))})
            sum_rows.append(r)
            logger.info(f'  {s}: QWK={met["qwk"]:.4f} acc={met["acc"]:.2f} bin_auroc={met["bin_auroc"]:.4f}'
                        + (f' (unadapted QWK={r["orig_qwk"]:.4f})' if 'orig_qwk' in r else ''))
    pd.DataFrame(sum_rows).to_csv(os.path.join(out_dir, 'summary.csv'), index=False)
    pd.DataFrame(log_rows).to_csv(os.path.join(out_dir, 'adapt_log.csv'), index=False)
    with open(os.path.join(out_dir, 'config.yaml'), 'w') as f:
        yaml.safe_dump(cfg, f)
    logger.info(f'target_adapt {tag} {method}: written {out_dir}')
    return out_dir


# ---------------------------------------------------------------------------
# --mode extract_features
# ---------------------------------------------------------------------------

def final_linear(model):
    """The final nn.Linear of a softmax (CE) GradeNet head: model.head = Sequential(Dropout, Linear)."""
    head = model.head
    lin = head[-1] if isinstance(head, nn.Sequential) and len(head) else head
    if not isinstance(lin, nn.Linear):
        raise ValueError(f'final head layer is {type(lin).__name__}, expected nn.Linear')
    for m in (head if isinstance(head, nn.Sequential) else [head]):
        if not isinstance(m, (nn.Linear, nn.Dropout, nn.Identity)):
            raise ValueError(f'unexpected head module {type(m).__name__}: features would not enter the final '
                             f'linear layer directly')
    return lin


@torch.no_grad()
def _features(model, loader, device):
    """Eval-mode pooled features (input of the final linear layer), model logits, ids, labels."""
    model.eval()
    F, Z, ids, ys = [], [], [], []
    for batch in loader:
        x = batch['x'].to(device)
        f = model.forward_features(x)
        F.append(f.float().cpu().numpy())
        Z.append(model.head(f).float().cpu().numpy())
        ids.extend([str(i) for i in batch['id']])
        ys.extend(list(batch['y'].cpu().numpy()))
    return np.concatenate(F), np.concatenate(Z), ids, np.asarray(ys, dtype=int)


def _softmax(Z):
    Z = Z - Z.max(1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(1, keepdims=True)


def head_probs(feats, W, b):
    """(n_folds, N, K) raw per-fold probabilities softmax(feats[k] @ W[k].T + b[k])."""
    feats = np.asarray(feats, dtype=np.float64)
    return np.stack([_softmax(feats[k] @ np.asarray(W[k], np.float64).T + np.asarray(b[k], np.float64))
                     for k in range(len(W))])


def run_extract_features(cfg, atol=1e-3):
    """Pooled features + final linear layer of every fold model on DEV / TEST of every held-out site."""
    from src.utils.logger import get_logger
    head_type = cfg.get('head_type', 'softmax')
    if head_type in ORDINAL_HEADS:
        raise ValueError(f'extract_features supports CE (softmax) heads only; {cfg.get("tag")} has head_type='
                         f'{head_type} (a single severity score with cut-points, not a K-way linear layer)')
    ctx = run_context(cfg)
    tag, exp_dir = ctx['tag'], ctx['exp_dir']
    logger = get_logger(tag, 'extract_features')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    out_dir = os.path.join(exp_dir, 'features')
    os.makedirs(out_dir, exist_ok=True)
    sets = [f'{d}_{s}' for d in ctx['heldout'] for s in ('dev', 'test')]
    logger.info(f'=== extract_features {tag} (checkpoints of {ctx["owner"]}, {ctx["n_folds"]} folds, sets={sets}) ===')
    feats = {s: [] for s in sets}
    logits = {s: [] for s in sets}
    meta = {}
    Ws, bs = [], []
    for k in range(ctx['n_folds']):
        model = load_fold_model(cfg, ctx['owner'], k).to(device)
        lin = final_linear(model)
        Ws.append(lin.weight.detach().float().cpu().numpy())
        bs.append(lin.bias.detach().float().cpu().numpy() if lin.bias is not None
                  else np.zeros(lin.out_features, np.float32))
        for s in sets:
            F, Z, ids, y = _features(model, _loader(cfg, s), device)
            feats[s].append(F)
            logits[s].append(Z)
            meta[s] = (ids, y)
        logger.info(f'  fold {k}: final linear {tuple(Ws[-1].shape)} on pooled features of dim {Ws[-1].shape[1]}')
        del model, lin
        _cleanup()
    W, b = np.stack(Ws).astype(np.float32), np.stack(bs).astype(np.float32)
    np.savez(os.path.join(out_dir, 'head.npz'), W=W, b=b, T=np.float64(ctx['T']), use_T=bool(ctx['use_T']),
             heldout=np.array(ctx['heldout']), tag=tag, head_type=head_type, n_folds=ctx['n_folds'])
    checks = []
    plain_tta = _plain_tta(cfg)
    for s in sets:
        ids, y = meta[s]
        F32 = np.stack(feats[s])
        F16 = F32.astype(np.float16)
        np.savez(os.path.join(out_dir, f'{s}.npz'), ids=np.array(ids), y_true=y, feats=F16)
        # 1) the linear layer applied to the features is the model's own output
        P_model = np.stack([_softmax(z.astype(np.float64)) for z in logits[s]])
        P32 = head_probs(F32, W, b)
        d_lin = float(np.abs(P32 - P_model).max()) if P32.size else 0.0
        if d_lin > atol:
            raise RuntimeError(f'{s}: softmax(feats @ W.T + b) differs from the model output by {d_lin:.2e}')
        d16 = float(np.abs(head_probs(F16, W, b) - P_model).max()) if P32.size else 0.0
        # 2) reproduces the run's saved normal predictions (only comparable when they were made without TTA)
        rec = {'set': s, 'n': len(ids), 'max_abs_diff_model': d_lin, 'max_abs_diff_float16': d16,
               'checked_vs_saved': False, 'max_abs_diff_saved_folds': np.nan, 'max_abs_diff_saved_y_prob': np.nan}
        orig = _saved_normal(exp_dir, s)
        if not plain_tta:
            logger.info(f'  {s}: run used TTA ({cfg.get("tta")}, scales {cfg.get("tta_scales")}): saved predictions '
                        f'not comparable to single-view features, check skipped')
        elif orig is None or [str(i) for i in orig['ids']] != ids:
            logger.info(f'  {s}: no saved normal predictions with the same images, check skipped')
        else:
            P_raw = P32.mean(0)
            P_sc = apply_temperature(P_raw, ctx['T']) if ctx['use_T'] else P_raw
            df = np.asarray(orig['y_prob_folds'], dtype=float) if 'y_prob_folds' in orig else np.empty((0,))
            d_f = float(np.abs(P32 - df).max()) if df.shape == P32.shape else np.nan
            d_p = float(np.abs(P_sc - np.asarray(orig['y_prob'], dtype=float)).max())
            rec.update({'checked_vs_saved': True, 'max_abs_diff_saved_folds': d_f, 'max_abs_diff_saved_y_prob': d_p})
            if d_p > atol or (not np.isnan(d_f) and d_f > atol):
                raise RuntimeError(f'{s}: features + head do not reproduce the saved predictions '
                                   f'(max |diff| folds={d_f:.2e}, y_prob={d_p:.2e}); were they computed with '
                                   f'another preprocessing / checkpoint?')
        checks.append(rec)
        logger.info(f'  {s}: {len(ids)} images, feats {F16.shape} float16; max |dP| model={d_lin:.1e} '
                    f'float16={d16:.1e} saved={rec["max_abs_diff_saved_y_prob"]:.1e}')
    pd.DataFrame(checks).to_csv(os.path.join(out_dir, 'reproduction_check.csv'), index=False)
    logger.info(f'extract_features {tag}: written {out_dir}')
    return out_dir
