"""Augmentations: basic (zoo-like) and medical."""

import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(cfg, train: bool):
    """Build Albumentations transform pipeline for train or evaluation.

    Args:
        cfg: Dict with 'img_size', 'aug' ('basic' or 'medical').
        train: If True, returns training augmentation pipeline; else val/test resize + norm.
    """
    s = int(cfg['img_size'])

    if not train:
        return A.Compose([
            A.Resize(height=s, width=s),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])

    aug_mode = cfg.get('aug', 'basic')

    if aug_mode == 'basic':
        return A.Compose([
            A.RandomResizedCrop(
                size=(s, s),
                scale=(0.9, 1.0),
                ratio=(0.9, 1.1),
                interpolation=cv2.INTER_CUBIC,
            ),
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02, p=0.5
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.2, contrast_limit=0.2, p=0.5
            ),
            A.CLAHE(clip_limit=2.0, p=0.3),
            A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.3),
            A.Affine(translate_percent=0.1, scale=0.9, rotate=45, p=0.3),
            A.CoarseDropout(
                num_holes_range=(1, 3),
                hole_height_range=(8, 16),
                hole_width_range=(8, 16),
                fill=0,
                p=0.15,
            ),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])

    elif aug_mode == 'medical':
        return A.Compose([
            A.Resize(height=s, width=s),
            A.Affine(
                scale=(0.9, 1.1),
                rotate=(-180, 180),
                translate_percent=(-0.05, 0.05),
                p=0.8,
            ),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.2, contrast_limit=0.2, p=0.5
            ),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            A.GaussianBlur(blur_limit=(3, 3), p=0.2),
            A.ImageCompression(quality_range=(70, 100), p=0.3),
            A.CoarseDropout(
                num_holes_range=(1, 3),
                hole_height_range=(8, 16),
                hole_width_range=(8, 16),
                fill=0,
                p=0.15,
            ),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])

    else:
        raise ValueError(f"Unknown aug mode: {aug_mode}")
