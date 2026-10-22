"""Tests for WP3: Pretraining (SSLModel, SSLCombinedLoss, Checkpoint compatibility, SqrtInverseClassSampler, LesionDetection)."""

import os
import tempfile
import cv2
import numpy as np
import pytest
import torch
import yaml

from src.models.ssl_model import SSLModel, SSLCombinedLoss, info_nce_loss
from src.models.gradenet import GradeNet
from src.data.samplers import SqrtInverseClassSampler
from src.preprocessing.lesion_detection import detect_lesions_batch, LesionDetector
from src.utils.config import build_cfg


class DummyArgs:
    def __init__(self, **kwargs):
        self.mode = kwargs.get('mode', 'pretrain')
        self.kind = kwargs.get('kind', 'ssl')
        self.tag = kwargs.get('tag', 'test_tag')
        self.phase = kwargs.get('phase', None)
        self.inherit = kwargs.get('inherit', None)
        self.set = kwargs.get('set', [])


def test_ssl_model_forward():
    """Verify MaxViT SSLModel outputs and shapes."""
    model = SSLModel(
        backbone_name='maxvit_large_tf_512.in21k_ft_in1k',
        projection_dim=128,
        use_contrastive=True,
        use_multitask=True,
        pretrained=False,
        grad_checkpointing=False
    )
    model.eval()

    v1 = torch.randn(2, 3, 512, 512)
    v2 = torch.randn(2, 3, 512, 512)

    with torch.no_grad():
        out = model(v1, v2)

    assert out['proj1'] is not None and out['proj1'].shape == (2, 128)
    assert out['proj2'] is not None and out['proj2'].shape == (2, 128)

    # Unit L2 norm for projections
    norm1 = torch.norm(out['proj1'], p=2, dim=1)
    assert torch.allclose(norm1, torch.ones_like(norm1), atol=1e-5)

    assert out['lesion_type1'] is not None and out['lesion_type1'].shape == (2, 3)
    assert out['lesion_count1'] is not None and out['lesion_count1'].shape == (2, 3)
    assert out['lesion_mask1'] is not None and out['lesion_mask1'].shape == (2, 3, 128, 128)

    # Sigmoid count head outputs [0, 1]
    assert (out['lesion_count1'] >= 0.0).all() and (out['lesion_count1'] <= 1.0).all()


def test_ssl_loss():
    """Verify SSLCombinedLoss computes correct loss components."""
    criterion = SSLCombinedLoss(
        temperature=0.07,
        w_contrastive=1.0,
        w_lesion_type=0.5,
        w_lesion_count=0.3,
        w_reconstruction=0.2,
        use_contrastive=True,
        use_multitask=True
    )

    outputs = {
        'proj1': F_norm(torch.randn(2, 128)),
        'proj2': F_norm(torch.randn(2, 128)),
        'lesion_type1': torch.randn(2, 3),
        'lesion_type2': torch.randn(2, 3),
        'lesion_count1': torch.rand(2, 3),
        'lesion_count2': torch.rand(2, 3),
        'lesion_mask1': torch.rand(2, 3, 128, 128),
        'lesion_mask2': torch.rand(2, 3, 128, 128),
    }

    gt_mask = torch.rand(2, 3, 128, 128)
    gt_count = torch.tensor([[10.0, 5.0, 2.0], [0.0, 1.0, 4.0]])
    gt_presence = torch.tensor([[1.0, 1.0, 1.0], [0.0, 1.0, 1.0]])

    loss, ldict = criterion(outputs, gt_mask, gt_count, gt_presence)
    assert loss.ndim == 0
    assert loss.item() > 0.0
    for key in ['total', 'contrastive', 'lesion_type', 'lesion_count', 'reconstruction']:
        assert key in ldict
        assert ldict[key] >= 0.0


def F_norm(x):
    return torch.nn.functional.normalize(x, dim=1)


def test_checkpoint_compatibility():
    """Verify saved SSL backbone weights load cleanly into GradeNet with 0 missing/unexpected keys."""
    ssl_model = SSLModel(
        backbone_name='maxvit_large_tf_512.in21k_ft_in1k',
        pretrained=False,
        grad_checkpointing=False
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = os.path.join(tmp_dir, 'ssl_backbone.pth')
        ssl_model.save_backbone(ckpt_path)
        assert os.path.isfile(ckpt_path)

        cfg = {
            'backbone': 'maxvit_large_tf_512.in21k_ft_in1k',
            'drop_path': 0.1,
            'pretrained': False,
            'pretrain_path': ckpt_path,
            'num_classes': 5,
            'head_type': 'softmax',
            'head_dropout': 0.3,
        }

        # GradeNet initialization loads pretrain_path and verifies missing/unexpected keys
        gradenet = GradeNet(cfg)
        sd = torch.load(ckpt_path, map_location='cpu')
        mis, unx = gradenet.backbone.load_state_dict(sd, strict=True)
        assert len(mis) == 0, f"Missing keys: {mis}"
        assert len(unx) == 0, f"Unexpected keys: {unx}"


def test_sqrt_inverse_class_sampler():
    """Verify SqrtInverseClassSampler weights inversely by class frequency."""
    # 5 classes with heavily skewed distribution
    y = np.array([0] * 1000 + [1] * 200 + [2] * 50 + [3] * 20 + [4] * 10)
    sampler = SqrtInverseClassSampler(y, num_samples=500, seed=42)
    assert len(sampler) == 500

    sampled = list(iter(sampler))
    assert len(sampled) == 500

    # Rare classes should be sampled more frequently than their raw distribution
    counts = np.bincount(y[sampled], minlength=5)
    raw_props = np.bincount(y, minlength=5) / len(y)
    sampled_props = counts / len(sampled)

    # Class 4 raw is 10 / 1280 ~ 0.78%, sampled proportion should be substantially higher
    assert sampled_props[4] > raw_props[4] * 2


def test_detect_lesions_batch():
    """Verify detect_lesions_batch processes fundus images and generates valid .npz artifacts."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        img_dir = os.path.join(tmp_dir, 'images')
        out_dir = os.path.join(tmp_dir, 'labels')
        os.makedirs(img_dir, exist_ok=True)

        # Create synthetic fundus image (circle on dark background)
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.circle(img, (100, 100), 80, (20, 100, 180), -1)  # BGR
        # Add a bright spot (exudate) and a dark spot (hemorrhage)
        cv2.circle(img, (120, 120), 5, (255, 255, 255), -1)
        cv2.circle(img, (80, 80), 5, (0, 0, 50), -1)

        cv2.imwrite(os.path.join(img_dir, 'sample01.jpeg'), img)

        detect_lesions_batch([img_dir], out_dir, mask_size=128)

        npz_path = os.path.join(out_dir, 'sample01.npz')
        assert os.path.isfile(npz_path)

        data = np.load(npz_path)
        assert data['mask'].shape == (128, 128, 3)
        assert data['count'].shape == (3,)
        assert data['presence'].shape == (3,)
        assert data['heatmap'].shape == (128, 128)


def test_config_ssl_merging():
    """Verify build_cfg merges config_ssl.yaml on --mode pretrain --kind ssl."""
    args = DummyArgs(
        mode='pretrain',
        kind='ssl',
        set=['ssl_epochs=10', 'ssl_batch_size=4']
    )
    cfg = build_cfg(args)

    assert cfg['kind'] == 'ssl'
    assert 'ssl_backbone' in cfg
    assert cfg['ssl_epochs'] == 10  # CLI override wins
    assert cfg['ssl_batch_size'] == 4  # CLI override wins
    assert cfg['ssl_save_dir'] == 'checkpoints/pretrain/ssl_maxvit'
