"""
SSL Pretraining Model: Lesion-Aware Contrastive + Multi-Task.

Architecture (per MODAL-Net S7m):
  MaxViT-L backbone (shared, timm num_classes=0)
  ├── Projection head (MLP 1024→512→128) → Contrastive loss (InfoNCE)
  ├── Lesion type head (Multi-label BCE, 3 outputs) → from stage-3 features (512ch)
  ├── Lesion count head (MSE regression, 3 outputs) → from stage-4 features (1024ch)
  └── Lightweight decoder (ConvTranspose/Upsample → 128×128) → Reconstruction loss

Features are extracted directly via backbone.stem and backbone.stages loop (NCHW format).
save_backbone() saves backbone.state_dict() with num_classes=0 key names for GradeNet.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import warnings


class ProjectionHead(nn.Module):
    """
    SimCLR-style projection head for contrastive learning.
    Maps backbone features to a 128-dim normalized embedding space.

    Input: (B, in_channels) — pooled stage-4 features
    Output: (B, 128) — L2-normalized projection
    """

    def __init__(self, in_channels=1024, hidden_dim=512, out_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        return F.normalize(x, dim=1)


class LesionTypeHead(nn.Module):
    """
    Multi-label classification head for lesion presence prediction.
    Predicts whether each lesion type (microaneurysm, hemorrhage, exudate)
    is present in the image.

    Input: (B, C, H, W) — stage-3 features (512ch for MaxViT-L)
    Output: (B, 3) — logits for 3 lesion types
    """

    def __init__(self, in_channels=512, num_lesion_types=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(in_channels, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, num_lesion_types)
        )

    def forward(self, x):
        pooled = self.avg_pool(x).flatten(1)
        return self.fc(pooled)


class LesionCountHead(nn.Module):
    """
    Regression head for lesion count prediction.
    Predicts the count of each lesion type (normalized to [0, 1]).

    Input: (B, C, H, W) — stage-4 features (1024ch for MaxViT-L)
    Output: (B, 3) — predicted normalized counts
    """

    def __init__(self, in_channels=1024, num_lesion_types=3, max_count=100):
        super().__init__()
        self.max_count = max_count
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(in_channels, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, num_lesion_types)
        )

    def forward(self, x):
        pooled = self.avg_pool(x).flatten(1)
        counts = self.fc(pooled)
        return torch.sigmoid(counts)


class LesionDecoder(nn.Module):
    """
    Lightweight decoder for lesion mask reconstruction.
    Takes stage-3 and stage-4 features, fuses and upsamples to produce
    a 3-channel lesion mask at 128×128 resolution.

    Input:
        stage3: (B, stage3_channels, 32, 32)
        stage4: (B, stage4_channels, 16, 16)
    Output:
        mask: (B, 3, 128, 128) — reconstructed lesion masks
    """

    def __init__(self, stage3_channels=512, stage4_channels=1024, out_channels=3):
        super().__init__()

        # Project stage4 (16×16) to stage3 resolution (32×32)
        self.stage4_upsample = nn.Sequential(
            nn.Conv2d(stage4_channels, stage3_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(stage3_channels),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        )

        # Fuse stage3 + upsampled stage4
        self.fuse = nn.Sequential(
            nn.Conv2d(stage3_channels + stage3_channels, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # Upsample to 128×128 (32 → 64 → 128)
        self.upsample1 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 32→64
        )

        self.upsample2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 64→128
        )

        # Final 1×1 conv to produce 3-channel mask
        self.output_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, stage3, stage4):
        stage4_up = self.stage4_upsample(stage4)  # (B, stage3_ch, 32, 32)
        fused = torch.cat([stage3, stage4_up], dim=1)  # (B, 2*stage3_ch, 32, 32)
        fused = self.fuse(fused)  # (B, 256, 32, 32)
        x = self.upsample1(fused)    # (B, 128, 64, 64)
        x = self.upsample2(x)        # (B, 64, 128, 128)
        mask = self.output_conv(x)   # (B, 3, 128, 128)
        return mask


class SSLModel(nn.Module):
    """
    Full SSL pretraining model combining contrastive learning + multi-task.

    Args:
        backbone_name: timm model name (default: maxvit_large_tf_512.in1k)
        stage_channels: Channel dims for stages (default: determined dynamically)
        projection_dim: Output dim for contrastive projection
        use_contrastive: Whether to include contrastive branch
        use_multitask: Whether to include multi-task branch
        pretrained: Whether to load ImageNet pretrained weights (aborts when weights cannot be loaded unless pretrained_fallback='warn')
        drop_path_rate: Stochastic depth rate
        grad_checkpointing: Whether to enable gradient checkpointing
    """

    def __init__(self,
                 backbone_name='maxvit_large_tf_512.in1k',
                 stage_channels=None,
                 projection_dim=128,
                 use_contrastive=True,
                 use_multitask=True,
                 pretrained=True,
                 pretrained_fallback='strict',
                 drop_path_rate=0.1,
                 grad_checkpointing=True):
        super().__init__()
        self.use_contrastive = use_contrastive
        self.use_multitask = use_multitask

        self.pretrained_ok = bool(pretrained)
        try:
            self.backbone = timm.create_model(
                backbone_name,
                pretrained=pretrained,
                num_classes=0,
                drop_path_rate=drop_path_rate,
            )
        except Exception as exc:
            if not pretrained:
                raise
            msg = (f"Pretrained weights for '{backbone_name}' could not be loaded "
                   f"({type(exc).__name__}: {exc}). Continuing would silently pretrain "
                   f"from a randomly initialised backbone. Fix the offline HF cache, or "
                   f"pass pretrained_fallback='warn' to allow it deliberately.")
            if str(pretrained_fallback).lower() == 'warn':
                warnings.warn(msg)
                self.pretrained_ok = False
                self.backbone = timm.create_model(
                    backbone_name,
                    pretrained=False,
                    num_classes=0,
                    drop_path_rate=drop_path_rate,
                )
            else:
                raise RuntimeError(msg) from exc

        if grad_checkpointing and hasattr(self.backbone, 'set_grad_checkpointing'):
            self.backbone.set_grad_checkpointing(True)

        if stage_channels is not None:
            self.stage_channels = list(stage_channels)
            in_ch_stage3 = self.stage_channels[2]
            in_ch_stage4 = self.stage_channels[3]
        else:
            in_ch_stage3, in_ch_stage4 = None, None
            if hasattr(self.backbone, 'feature_info') and self.backbone.feature_info:
                chs = {
                    info.get('module'): info.get('num_chs')
                    for info in self.backbone.feature_info
                    if isinstance(info, dict)
                }
                if 'stages.2' in chs and 'stages.3' in chs:
                    in_ch_stage3 = chs['stages.2']
                    in_ch_stage4 = chs['stages.3']
            if in_ch_stage3 is None or in_ch_stage4 is None:
                with torch.no_grad():
                    dummy = torch.zeros(1, 3, 512, 512)
                    dummy = self.backbone.stem(dummy)
                    for i, s in enumerate(self.backbone.stages):
                        dummy = s(dummy)
                        if i == 2:
                            in_ch_stage3 = dummy.shape[1]
                        elif i == 3:
                            in_ch_stage4 = dummy.shape[1]
            in_ch_stage3 = in_ch_stage3 or 512
            in_ch_stage4 = in_ch_stage4 or 1024
            self.stage_channels = [128, 256, in_ch_stage3, in_ch_stage4]

        # Contrastive branch
        if self.use_contrastive:
            self.projection_head = ProjectionHead(
                in_channels=in_ch_stage4,
                hidden_dim=512,
                out_dim=projection_dim
            )

        # Multi-task branches
        if self.use_multitask:
            self.lesion_type_head = LesionTypeHead(
                in_channels=in_ch_stage3,
                num_lesion_types=3
            )
            self.lesion_count_head = LesionCountHead(
                in_channels=in_ch_stage4,
                num_lesion_types=3
            )
            self.lesion_decoder = LesionDecoder(
                stage3_channels=in_ch_stage3,
                stage4_channels=in_ch_stage4,
                out_channels=3
            )

    def _extract_features(self, x):
        """Extract stage-3 and stage-4 features via manual stem + stages loop."""
        x = self.backbone.stem(x)
        stage3 = None
        stage4 = None
        for i, s in enumerate(self.backbone.stages):
            x = s(x)
            if i == 2:
                stage3 = x
            elif i == 3:
                stage4 = x
        return stage3, stage4

    def forward(self, view1, view2):
        """Forward pass for SSL pretraining."""
        results = {
            'proj1': None, 'proj2': None,
            'lesion_type1': None, 'lesion_type2': None,
            'lesion_count1': None, 'lesion_count2': None,
            'lesion_mask1': None, 'lesion_mask2': None
        }

        # Process view1
        stage3_1, stage4_1 = self._extract_features(view1)
        if self.use_contrastive:
            pooled1 = F.adaptive_avg_pool2d(stage4_1, (1, 1)).flatten(1)
            results['proj1'] = self.projection_head(pooled1)

        if self.use_multitask:
            results['lesion_type1'] = self.lesion_type_head(stage3_1)
            results['lesion_count1'] = self.lesion_count_head(stage4_1)
            results['lesion_mask1'] = self.lesion_decoder(stage3_1, stage4_1)

        # Process view2
        stage3_2, stage4_2 = self._extract_features(view2)
        if self.use_contrastive:
            pooled2 = F.adaptive_avg_pool2d(stage4_2, (1, 1)).flatten(1)
            results['proj2'] = self.projection_head(pooled2)

        if self.use_multitask:
            results['lesion_type2'] = self.lesion_type_head(stage3_2)
            results['lesion_count2'] = self.lesion_count_head(stage4_2)
            results['lesion_mask2'] = self.lesion_decoder(stage3_2, stage4_2)

        return results

    def get_backbone_state_dict(self):
        """Extract backbone state_dict for loading into GradeNet."""
        return self.backbone.state_dict()

    def save_backbone(self, path):
        """Save backbone weights for use in GradeNet fine-tuning."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(self.get_backbone_state_dict(), path)


def info_nce_loss(proj1, proj2, temperature=0.07):
    """InfoNCE (NT-Xent) contrastive loss for SimCLR-style training."""
    batch_size = proj1.shape[0]

    # Concatenate both views: (2B, D)
    projections = torch.cat([proj1, proj2], dim=0)

    # Cosine similarity matrix: (2B, 2B)
    sim = torch.mm(projections, projections.t()) / temperature

    # Mask out self-similarity
    mask = torch.eye(2 * batch_size, dtype=torch.bool, device=proj1.device)
    sim.masked_fill_(mask, -1e9)

    # Positive targets: view1[i] pairs with view2[i] (index i + B)
    labels = torch.cat([
        torch.arange(batch_size, 2 * batch_size, device=proj1.device),
        torch.arange(0, batch_size, device=proj1.device)
    ])

    return F.cross_entropy(sim, labels)


class SSLCombinedLoss(nn.Module):
    """
    Combined SSL loss: Contrastive (InfoNCE) + Multi-Task (Lesion).

    L_pretrain = w_contrastive * L_contrastive
               + w_type * L_lesion_type (BCE)
               + w_count * L_lesion_count (MSE)
               + w_recon * L_reconstruction (L1)
    """

    def __init__(self,
                 temperature=0.07,
                 w_contrastive=1.0,
                 w_lesion_type=0.5,
                 w_lesion_count=0.3,
                 w_reconstruction=0.2,
                 max_lesion_count=100.0,
                 use_contrastive=True,
                 use_multitask=True):
        super().__init__()
        self.temperature = temperature
        self.w_contrastive = w_contrastive if use_contrastive else 0.0
        self.w_lesion_type = w_lesion_type if use_multitask else 0.0
        self.w_lesion_count = w_lesion_count if use_multitask else 0.0
        self.w_reconstruction = w_reconstruction if use_multitask else 0.0
        self.max_lesion_count = max_lesion_count
        self.use_contrastive = use_contrastive
        self.use_multitask = use_multitask

        self.bce_loss = nn.BCEWithLogitsLoss()
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

    def forward(self, outputs, lesion_mask, lesion_count, lesion_presence):
        total_loss = torch.tensor(0.0, device=lesion_mask.device)
        loss_dict = {}

        if self.use_contrastive and outputs['proj1'] is not None:
            contrastive_loss = info_nce_loss(
                outputs['proj1'], outputs['proj2'], self.temperature
            )
            total_loss = total_loss + self.w_contrastive * contrastive_loss
            loss_dict['contrastive'] = contrastive_loss.item()
        else:
            loss_dict['contrastive'] = 0.0

        if self.use_multitask:
            if outputs['lesion_type1'] is not None:
                type_loss = self.bce_loss(outputs['lesion_type1'], lesion_presence)
                type_loss += self.bce_loss(outputs['lesion_type2'], lesion_presence)
                type_loss = type_loss / 2
                total_loss = total_loss + self.w_lesion_type * type_loss
                loss_dict['lesion_type'] = type_loss.item()
            else:
                loss_dict['lesion_type'] = 0.0

            if outputs['lesion_count1'] is not None:
                count_norm = lesion_count / self.max_lesion_count
                count_loss = self.mse_loss(outputs['lesion_count1'], count_norm)
                count_loss += self.mse_loss(outputs['lesion_count2'], count_norm)
                count_loss = count_loss / 2
                total_loss = total_loss + self.w_lesion_count * count_loss
                loss_dict['lesion_count'] = count_loss.item()
            else:
                loss_dict['lesion_count'] = 0.0

            if outputs['lesion_mask1'] is not None:
                recon_loss = self.l1_loss(outputs['lesion_mask1'], lesion_mask)
                recon_loss += self.l1_loss(outputs['lesion_mask2'], lesion_mask)
                recon_loss = recon_loss / 2
                total_loss = total_loss + self.w_reconstruction * recon_loss
                loss_dict['reconstruction'] = recon_loss.item()
            else:
                loss_dict['reconstruction'] = 0.0
        else:
            loss_dict['lesion_type'] = 0.0
            loss_dict['lesion_count'] = 0.0
            loss_dict['reconstruction'] = 0.0

        loss_dict['total'] = total_loss.item()
        return total_loss, loss_dict
