"""GradeNet: MaxViT backbone with optional MixStyle and three heads (softmax, coral, coral_src)."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import warnings

from src.models.mixstyle import MixStyle

ORDINAL_HEADS = ('coral', 'coral_src')


def probs_from_logits(z: torch.Tensor, head_type: str = 'softmax') -> torch.Tensor:
    """Convert raw model logits to predicted class probabilities.

    For the ordinal heads ('coral', 'coral_src') the K-1 outputs are cumulative logits
    logit P(y > k); they are converted via sigmoid differences and normalised to sum to 1.
    For the softmax head, applies softmax over the class dimension.
    """
    if head_type not in ORDINAL_HEADS:
        return F.softmax(z, dim=1)

    c = torch.sigmoid(z)
    c = torch.cat([torch.ones_like(c[:, :1]), c, torch.zeros_like(c[:, :1])], dim=1)
    diff = (c[:, :-1] - c[:, 1:]).clamp_min(1e-8)
    return diff / diff.sum(dim=1, keepdim=True)


def _split_decay(ps):
    decay_p = [p for k, p in ps if p.ndim > 1 and 'bias' not in k and 'norm' not in k and 'bn' not in k]
    no_decay_p = [p for k, p in ps if p.ndim <= 1 or 'bias' in k or 'norm' in k or 'bn' in k]
    return decay_p, no_decay_p


def llrd_groups(model: nn.Module, lr: float, decay: float, wd: float, head_mult: float):
    """Layer-wise learning rate decay parameter groups with bias/norm zero-decay separation.

    MaxViT / ConvNeXt / EfficientNet (timm 'stem' + 'stages.i'): 5 depth levels, as before.
    ViT (timm 'blocks.i', e.g. RETFound): embeddings = level 0, block i = level i + 1, final norm = top
    level; lr(level) = lr * decay ** (top - level), the BEiT / MAE / RETFound layer-decay schedule.
    """
    groups = []
    used = set()
    named = [(k, p) for k, p in model.backbone.named_parameters() if p.requires_grad]
    is_vit = any(k.startswith('patch_embed') for k, _ in named)   # EfficientNet also has 'blocks.i', ViT has patch_embed
    n_blocks = 1 + max([int(k.split('.')[1]) for k, _ in named if k.startswith('blocks.')] or [-1]) if is_vit else 0

    if n_blocks > 0:
        top = n_blocks + 1
        levels = {}
        for k, p in named:
            if k.startswith('blocks.'):
                lvl = int(k.split('.')[1]) + 1
            elif k.startswith(('cls_token', 'pos_embed', 'patch_embed', 'reg_token', 'dist_token')):
                lvl = 0
            else:
                lvl = top
            levels.setdefault(lvl, []).append((k, p))
        for lvl in sorted(levels):
            group_lr = lr * (decay ** (top - lvl)) if decay != 1.0 else lr
            decay_p, no_decay_p = _split_decay(levels[lvl])
            if decay_p:
                groups.append({'params': decay_p, 'lr': group_lr, 'weight_decay': wd})
            if no_decay_p:
                groups.append({'params': no_decay_p, 'lr': group_lr, 'weight_decay': 0.0})
        used = {k for k, _ in named}

    order = ['stem', 'stages.0', 'stages.1', 'stages.2', 'stages.3']
    for i, n in enumerate(order):
        if n_blocks > 0:
            break
        ps = [(k, p) for k, p in named if k.startswith(n)]
        used |= {k for k, _ in ps}
        if ps:
            group_lr = lr * (decay ** (len(order) - 1 - i)) if decay != 1.0 else lr
            decay_p, no_decay_p = _split_decay(ps)
            if decay_p:
                groups.append({'params': decay_p, 'lr': group_lr, 'weight_decay': wd})
            if no_decay_p:
                groups.append({'params': no_decay_p, 'lr': group_lr, 'weight_decay': 0.0})

    remaining = [(k, p) for k, p in named if k not in used]
    if remaining:
        decay_p, no_decay_p = _split_decay(remaining)
        if decay_p:
            groups.append({'params': decay_p, 'lr': lr, 'weight_decay': wd})
        if no_decay_p:
            groups.append({'params': no_decay_p, 'lr': lr, 'weight_decay': 0.0})

    # Head parameters
    # Cut-point parameters of the coral_src head (raw, delta) are regularised by their own L2 term,
    # not by weight decay.
    def _no_wd(k, p):
        return p.ndim <= 1 or 'bias' in k or k.endswith('raw') or k.endswith('delta')
    head_decay = [p for k, p in model.head.named_parameters() if p.requires_grad and not _no_wd(k, p)]
    head_no_decay = [p for k, p in model.head.named_parameters() if p.requires_grad and _no_wd(k, p)]
    if head_decay:
        groups.append({'params': head_decay, 'lr': lr * head_mult, 'weight_decay': wd})
    if head_no_decay:
        groups.append({'params': head_no_decay, 'lr': lr * head_mult, 'weight_decay': 0.0})

    return [g for g in groups if g['params']]


class SourceOrdinalHead(nn.Module):
    """One severity score per image plus ordered cut-points, optionally one cut-point set per source.

    logit P(y > k | x, source d) = s(x) - b_k^(d),   s(x) = w^T f(x)
    b^(d) = cumulative(raw + delta_d): b_0 = r_0, b_k = b_{k-1} + softplus(r_k)  -> always ordered.
    delta_d is the offset of training source d (zero-initialised, L2-tied to the shared cut-points).
    Images whose source is not a training source (every evaluation image) use the shared cut-points
    (delta = 0), i.e. the zero-shot prediction for an unseen site.
    """

    def __init__(self, in_dim: int, K: int, n_sources: int, dropout: float, use_offsets: bool):
        super().__init__()
        self.K = K
        self.drop = nn.Dropout(dropout)
        self.score = nn.Linear(in_dim, 1, bias=False)
        init_b = torch.linspace(-1.5, 1.5, K - 1)
        gaps = init_b[1:] - init_b[:-1]
        raw = torch.cat([init_b[:1], torch.log(torch.expm1(gaps))])
        self.raw = nn.Parameter(raw)
        self.use_offsets = bool(use_offsets) and n_sources > 0
        self.delta = nn.Parameter(torch.zeros(max(1, n_sources), K - 1)) if self.use_offsets else None

    @staticmethod
    def _ordered(raw: torch.Tensor) -> torch.Tensor:
        first = raw[..., :1]
        return torch.cat([first, first + torch.cumsum(F.softplus(raw[..., 1:]), dim=-1)], dim=-1)

    def thresholds(self, src_idx: torch.Tensor = None) -> torch.Tensor:
        """(B, K-1) cut-points for source rows src_idx (-1 = shared), or (1, K-1) shared cut-points."""
        raw = self.raw.unsqueeze(0)
        if src_idx is None or self.delta is None:
            return self._ordered(raw)
        pad = torch.cat([self.delta, torch.zeros_like(self.delta[:1])], dim=0)
        idx = torch.where(src_idx < 0, torch.full_like(src_idx, pad.size(0) - 1), src_idx)
        return self._ordered(raw + pad[idx])

    def offset_penalty(self) -> torch.Tensor:
        if self.delta is None:
            return self.raw.new_zeros(())
        return (self.delta ** 2).sum()

    def forward(self, f: torch.Tensor, src_idx: torch.Tensor = None) -> torch.Tensor:
        s = self.score(self.drop(f))
        return s - self.thresholds(src_idx)


class GradeNet(nn.Module):
    """MODAL-Net backbone model."""

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        backbone_name = str(cfg.get('backbone', 'maxvit_large_tf_512.in1k'))
        drop_path_rate = float(cfg.get('drop_path', 0.3))
        pretrained = cfg.get('pretrained', True)

        kw = dict(pretrained=pretrained, num_classes=0, drop_path_rate=drop_path_rate)
        if cfg.get('img_size_arg'):
            kw['img_size'] = cfg['img_size_arg']
        if cfg.get('backbone_global_pool'):
            kw['global_pool'] = cfg['backbone_global_pool']   # e.g. 'avg' for RETFound (ViT): adds fc_norm

        self.pretrained_ok = bool(pretrained)
        try:
            self.backbone = timm.create_model(backbone_name, **kw)
        except Exception as exc:
            if not pretrained:
                raise
            msg = (f"Pretrained weights for '{backbone_name}' could not be loaded "
                   f"({type(exc).__name__}: {exc}). Continuing would silently train "
                   f"a randomly initialised backbone. Fix the offline HF cache, or set "
                   f"pretrained_fallback=warn to allow it deliberately.")
            if str(cfg.get('pretrained_fallback', 'strict')).lower() == 'warn':
                warnings.warn(msg)
                self.pretrained_ok = False
                kw['pretrained'] = False
                self.backbone = timm.create_model(backbone_name, **kw)
            else:
                raise RuntimeError(msg) from exc

        if cfg.get('grad_checkpointing') and hasattr(self.backbone, 'set_grad_checkpointing'):
            self.backbone.set_grad_checkpointing(True)

        if cfg.get('mixstyle'):
            self.ms_idx = [int(s) for s in cfg.get('ms_stages', [0, 1])]
            self.ms = nn.ModuleList([
                MixStyle(float(cfg.get('ms_p', 0.5)), float(cfg.get('ms_alpha', 0.1)))
                for _ in self.ms_idx
            ])
        else:
            self.ms_idx = []
            self.ms = nn.ModuleList()

        K = int(cfg.get('num_classes', 5))
        num_features = getattr(self.backbone, 'num_features', 1024)
        self.head_type = cfg.get('head_type', 'softmax')

        # domain id -> row of the per-source offset table (-1 = not a training source)
        from src.utils.config import train_sources
        from src.data.external import DOMAIN_IDS
        self.sources = train_sources(cfg)
        lut = torch.full((max(DOMAIN_IDS.values()) + 1,), -1, dtype=torch.long)
        for row, name in enumerate(self.sources):
            lut[DOMAIN_IDS[name]] = row
        self.register_buffer('src_lut', lut, persistent=False)

        if self.head_type == 'coral_src':
            self.head = SourceOrdinalHead(num_features, K, len(self.sources),
                                          float(cfg.get('head_dropout', 0.3)), bool(cfg.get('src_offsets', True)))
        else:
            out_dim = K - 1 if self.head_type == 'coral' else K
            self.head = nn.Sequential(
                nn.Dropout(float(cfg.get('head_dropout', 0.3))),
                nn.Linear(num_features, out_dim),
            )

        if cfg.get('pretrain_path'):
            sd = torch.load(cfg['pretrain_path'], map_location='cpu', weights_only=False)
            # backbone state dicts of this project, {'state_dict': ...} or MAE/RETFound {'model': ...}
            sd = sd.get('state_dict', sd.get('model', sd)) if isinstance(sd, dict) else sd
            sd = {k[len('module.'):] if k.startswith('module.') else k: v for k, v in sd.items()}
            own = self.backbone.state_dict()
            sd = {k: v for k, v in sd.items() if not k.startswith('head.')}
            shape_bad = [k for k, v in sd.items() if k in own and tuple(own[k].shape) != tuple(v.shape)]
            if shape_bad:
                raise ValueError(f'pretrain load: shape mismatch for {shape_bad[:5]} (img_size / patch size of the '
                                 f'checkpoint differ from the configured backbone)')
            mis, unx = self.backbone.load_state_dict(sd, strict=False)
            assert len(mis) < 0.05 * len(own), (
                f'pretrain load: {len(mis)} missing, {len(unx)} unexpected'
            )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        b = self.backbone

        if not self.ms_idx:
            f = b(x)
        else:
            # Manual MaxViT stage loop for internal MixStyle
            f = b.stem(x)
            for i, st in enumerate(b.stages):
                f = st(f)
                if i in self.ms_idx:
                    f = self.ms[self.ms_idx.index(i)](f)
            f = b.forward_head(b.norm(f))

        return f

    def forward(self, x: torch.Tensor, domain: torch.Tensor = None) -> torch.Tensor:
        """Logits. `domain` (B,) selects per-source cut-points for coral_src during training;
        without it (all inference paths) the shared cut-points are used."""
        f = self.forward_features(x)
        if self.head_type == 'coral_src':
            src_idx = None if domain is None else self.src_lut[domain.long().to(self.src_lut.device)]
            return self.head(f, src_idx)
        return self.head(f)

    def source_thresholds(self) -> dict:
        """coral_src only: {'shared': [...], '<source>': [...]} learned cut-points on the score axis."""
        assert self.head_type == 'coral_src', 'source_thresholds needs head_type=coral_src'
        out = {'shared': self.head.thresholds().detach().cpu().numpy()[0].tolist()}
        if self.head.delta is not None:
            for row, name in enumerate(self.sources):
                idx = torch.tensor([row], device=self.head.raw.device)
                out[name] = self.head.thresholds(idx).detach().cpu().numpy()[0].tolist()
        return out
