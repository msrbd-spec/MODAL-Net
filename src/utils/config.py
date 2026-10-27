"""Config resolution: build_cfg, validate_cfg, resolve, find_exp, ckpt_owner, parse_set."""

import os
import glob
import json
import yaml


def parse_set(pairs):
    """Parse --set k=v pairs into a dict.

    Values are parsed by yaml.safe_load so that true/false/null/numbers/lists
    are converted to their Python equivalents.
    """
    if not pairs:
        return {}
    out = {}
    for p in pairs:
        k, v = p.split('=', 1)
        out[k] = yaml.safe_load(v)
    return out


def resolve(x):
    """Resolve '@phase2a_recipe' -> tag from results/phase2a_recipe/WINNER.txt.

    Returns the tag string, or x unchanged if it does not start with '@',
    or None if x is None.
    """
    if not x:
        return x
    if x.startswith('@'):
        winner_path = os.path.join('results', x[1:], 'WINNER.txt')
        if not os.path.isfile(winner_path):
            raise FileNotFoundError(
                f'WINNER.txt not found for phase reference {x}: {winner_path}')
        return open(winner_path).read().split()[0]
    return x


def find_exp(tag):
    """Return the results folder of an experiment (cross-phase lookup).

    Searches results/phase*/<tag> and returns the first match (sorted).
    Raises FileNotFoundError if no match.
    """
    hits = sorted(glob.glob(os.path.join('results', 'phase*', tag)))
    if not hits:
        raise FileNotFoundError(
            f'No experiment folder found for tag {tag!r} under results/phase*/')
    return hits[0]


def ckpt_owner(cfg_or_tag):
    """Return the tag that actually owns the 5 fold checkpoints.

    Follows ckpt_from links until reaching a tag that has checkpoints/<tag>/fold0.pth.
    """
    if isinstance(cfg_or_tag, dict):
        cur = cfg_or_tag.get('ckpt_from') or cfg_or_tag.get('base') or cfg_or_tag.get('tag')
    else:
        cur = str(cfg_or_tag)

    visited = set()
    while cur and cur not in visited:
        visited.add(cur)
        fold0 = os.path.join('checkpoints', cur, 'fold0.pth')
        if os.path.isfile(fold0):
            return cur
        # Check config_snapshot.yaml in checkpoints
        snap = os.path.join('checkpoints', cur, 'config_snapshot.yaml')
        found_parent = False
        if os.path.isfile(snap):
            try:
                data = yaml.safe_load(open(snap)) or {}
                parent = data.get('ckpt_from') or data.get('base')
                if parent and parent != cur:
                    cur = parent
                    found_parent = True
            except Exception:
                pass
        if not found_parent:
            # Check results/phase*/<cur>/config.yaml
            try:
                exp_dir = find_exp(cur)
                exp_cfg_path = os.path.join(exp_dir, 'config.yaml')
                if os.path.isfile(exp_cfg_path):
                    data = yaml.safe_load(open(exp_cfg_path)) or {}
                    parent = data.get('ckpt_from') or data.get('base')
                    if parent and parent != cur:
                        cur = parent
                        found_parent = True
            except Exception:
                pass
        if not found_parent:
            break

    return cur


def validate_cfg(c):
    """Reject invalid option combinations at start-up."""
    assert not (c.get('amp_mix') and c.get('preproc') == 'legacy'), \
        'amp_mix needs preproc != legacy (equal-size, aspect-preserving images)'
    assert not (c.get('mixstyle') and 'maxvit' not in str(c.get('backbone', ''))), \
        'mixstyle manual stage loop is implemented for timm MaxViT only'
    assert not (c.get('head_type') == 'coral' and c.get('soft_labels', 'none') != 'none'), \
        'coral is incompatible with soft labels'
    assert c.get('uda_style_pool', 'none') in ('none', 'messidor_dev'), \
        f"uda_style_pool must be 'none' or 'messidor_dev', got {c.get('uda_style_pool')}"
    assert c.get('tta', 'none') in ('none', 'flip', 'flip_rot'), \
        f"tta must be 'none', 'flip', or 'flip_rot', got {c.get('tta')}"
    assert c.get('recalibrate', 'none') in ('none', 'devcut'), \
        f"recalibrate must be 'none' or 'devcut', got {c.get('recalibrate')}"
    assert str(c.get('pretrained_fallback', 'strict')).lower() in ('strict', 'warn'), \
        f"pretrained_fallback must be 'strict' or 'warn', got {c.get('pretrained_fallback')}"

    # XAI validations
    if 'xai_methods' in c and c['xai_methods'] is not None:
        allowed_xai = {'gradcam', 'gradcampp', 'layercam', 'eigencam'}
        methods = c['xai_methods']
        if isinstance(methods, str):
            methods = [m.strip() for m in methods.strip('[]').split(',') if m.strip()]
        for m in methods:
            assert m in allowed_xai, f"Invalid xai_method '{m}'. Allowed: {allowed_xai}"
    if 'xai_split' in c and c['xai_split'] is not None:
        assert c['xai_split'] in ('test', 'external_test'), \
            f"xai_split must be 'test' or 'external_test', got {c['xai_split']}"


def build_cfg(args):
    """Build the full configuration dict from CLI args.

    Resolution order:
    1. Load configs/config.yaml
    2. If --inherit or --base points to a saved snapshot, overlay it
    3. Apply --set overrides
    4. If bundle_from is set, merge ADOPT/TENTATIVE overrides from that phase
    5. Re-apply --set overrides (so they always win)
    6. Set tag, phase, set_overrides, inherit metadata
    7. validate_cfg
    """
    cfg = yaml.safe_load(open('configs/config.yaml'))

    # If --mode pretrain and --kind ssl, merge configs/config_ssl.yaml
    if getattr(args, 'mode', None) == 'pretrain' and getattr(args, 'kind', None) == 'ssl':
        ssl_path = 'configs/config_ssl.yaml'
        if os.path.isfile(ssl_path):
            ssl_yaml = yaml.safe_load(open(ssl_path))
            if ssl_yaml:
                cfg.update(ssl_yaml)

    # Inheritance: resolve @phase -> tag, then load snapshot
    inherit_resolved = resolve(getattr(args, 'inherit', None))
    base_resolved = resolve(getattr(args, 'base', None))
    base = inherit_resolved or base_resolved
    if base is None and getattr(args, 'mode', None) in \
            ('test', 'external_validation', 'calibrate_conformal'):
        base = getattr(args, 'tag', None)

    if base:
        snap = os.path.join('checkpoints', base, 'config_snapshot.yaml')
        if os.path.isfile(snap):
            saved = yaml.safe_load(open(snap))
            if saved:
                cfg.update(saved)
        else:
            try:
                exp_dir = find_exp(base)
                cfg_path = os.path.join(exp_dir, 'config.yaml')
                if os.path.isfile(cfg_path):
                    saved = yaml.safe_load(open(cfg_path))
                    if saved:
                        cfg.update(saved)
            except Exception:
                pass

    # CLI overrides
    ov = parse_set(getattr(args, 'set', None) or [])
    cfg.update(ov)

    # Bundle: merge overrides from all ADOPT/TENTATIVE rows of a phase
    if cfg.get('bundle_from'):
        import pandas as pd
        pt_path = os.path.join('results', cfg['bundle_from'], 'phase_table.csv')
        t = pd.read_csv(pt_path)
        for _, r in t[t['decision'].isin(['ADOPT', 'TENTATIVE'])].iterrows():
            loaded = yaml.safe_load(r['set_overrides'])
            if loaded:
                cfg.update(loaded)
        # Re-apply CLI overrides so they always win
        cfg.update(ov)

    # Metadata
    cfg['tag'] = getattr(args, 'tag', None)
    cfg['phase'] = getattr(args, 'phase', None)
    cfg['kind'] = getattr(args, 'kind', None)
    cfg['method'] = getattr(args, 'method', None) or cfg.get('method')
    cfg['base'] = base_resolved
    cfg['members'] = getattr(args, 'members', None)
    cfg['set_overrides'] = json.dumps(ov)
    cfg['inherit'] = str(getattr(args, 'inherit', None))

    validate_cfg(cfg)
    return cfg
