"""GradeNet: Multi-stage MaxViT architecture with MixStyle and WaveletGate."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import warnings

from src.models.wavelet_gate import WaveletGate
from src.models.mixstyle import MixStyle


def probs_from_logits(z: torch.Tensor, head_type: str = 'softmax') -> torch.Tensor:
    """Convert raw model logits to predicted class probabilities.

    For 'coral', converts cumulative ordinal logits via sigmoid differences
    and normalizes to ensure sum == 1.0.
    For standard heads, applies softmax over class dimension.
    """
    if head_type != 'coral':
        return F.softmax(z, dim=1)

    c = torch.sigmoid(z)
    c = torch.cat([torch.ones_like(c[:, :1]), c, torch.zeros_like(c[:, :1])], dim=1)
    diff = (c[:, :-1] - c[:, 1:]).clamp_min(1e-8)
    return diff / diff.sum(dim=1, keepdim=True)


def llrd_groups(model: nn.Module, lr: float, decay: float, wd: float, head_mult: float):
    """Layer-wise learning rate decay parameter groups with bias/norm zero-decay separation."""
    order = ['stem', 'stages.0', 'stages.1', 'stages.2', 'stages.3']
    groups = []
    used = set()

    for i, n in enumerate(order):
        ps = [(k, p) for k, p in model.backbone.named_parameters() if k.startswith(n) and p.requires_grad]
        used |= {k for k, _ in ps}
        if ps:
            group_lr = lr * (decay ** (len(order) - 1 - i)) if decay != 1.0 else lr
            decay_p = [p for k, p in ps if p.ndim > 1 and 'bias' not in k and 'norm' not in k and 'bn' not in k]
            no_decay_p = [p for k, p in ps if p.ndim <= 1 or 'bias' in k or 'norm' in k or 'bn' in k]
            if decay_p:
                groups.append({'params': decay_p, 'lr': group_lr, 'weight_decay': wd})
            if no_decay_p:
                groups.append({'params': no_decay_p, 'lr': group_lr, 'weight_decay': 0.0})

    remaining = [(k, p) for k, p in model.backbone.named_parameters() if k not in used and p.requires_grad]
    if remaining:
        decay_p = [p for k, p in remaining if p.ndim > 1 and 'bias' not in k and 'norm' not in k and 'bn' not in k]
        no_decay_p = [p for k, p in remaining if p.ndim <= 1 or 'bias' in k or 'norm' in k or 'bn' in k]
        if decay_p:
            groups.append({'params': decay_p, 'lr': lr, 'weight_decay': wd})
        if no_decay_p:
            groups.append({'params': no_decay_p, 'lr': lr, 'weight_decay': 0.0})

    # Head parameters
    head_decay = [p for k, p in model.head.named_parameters() if p.requires_grad and p.ndim > 1 and 'bias' not in k]
    head_no_decay = [p for k, p in model.head.named_parameters() if p.requires_grad and (p.ndim <= 1 or 'bias' in k)]
    if head_decay:
        groups.append({'params': head_decay, 'lr': lr * head_mult, 'weight_decay': wd})
    if head_no_decay:
        groups.append({'params': head_no_decay, 'lr': lr * head_mult, 'weight_decay': 0.0})

    return [g for g in groups if g['params']]


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

        self.wghr = WaveletGate() if cfg.get('use_wghr') else nn.Identity()

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
        out_dim = K - 1 if cfg.get('head_type') == 'coral' else K
        num_features = getattr(self.backbone, 'num_features', 1024)

        self.head = nn.Sequential(
            nn.Dropout(float(cfg.get('head_dropout', 0.3))),
            nn.Linear(num_features, out_dim),
        )

        if cfg.get('pretrain_path'):
            sd = torch.load(cfg['pretrain_path'], map_location='cpu')
            sd = sd.get('state_dict', sd)
            mis, unx = self.backbone.load_state_dict(sd, strict=False)
            assert len(mis) < 0.05 * len(self.backbone.state_dict()), (
                f'pretrain load: {len(mis)} missing, {len(unx)} unexpected'
            )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.wghr(x)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.forward_features(x)
        return self.head(f)
