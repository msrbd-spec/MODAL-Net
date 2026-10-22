"""Test T2: Preprocessing pipeline modes, aspect preservation, output shapes and dtypes."""

import cv2
import numpy as np
import pytest

from src.data.preprocess import preprocess, crop_fundus, pad_square_resize, ben_graham, clahe


def _make_dummy_fundus(h=600, w=800):
    """Create a synthetic fundus-like image: bright circular disc on black background."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    center = (w // 2, h // 2)
    radius = min(h, w) // 2 - 20
    # Orange/reddish fundus background
    cv2.circle(img, center, radius, (30, 80, 180), -1)
    # Bright optic disc
    cv2.circle(img, (center[0] + radius // 2, center[1]), radius // 6, (120, 200, 240), -1)
    return img


def test_preprocessing_shapes_and_dtypes():
    """Verify output shapes and uint8 dtypes for all 4 modes."""
    raw = _make_dummy_fundus(h=600, w=800)
    target_size = 512

    for mode in ['scaled', 'clahe_only', 'none']:
        out = preprocess(raw, mode=mode, size=target_size)
        assert out.dtype == np.uint8
        assert out.shape == (target_size, target_size, 3)

    # Legacy mode returns native resolution of the cropped bounding box (not resized)
    out_legacy = preprocess(raw, mode='legacy', size=target_size)
    assert out_legacy.dtype == np.uint8
    # Cropped bounding box dimensions will be close to the circle diameter (approx 2*radius)
    assert out_legacy.shape != (target_size, target_size, 3)
    assert out_legacy.ndim == 3


def test_scaled_mode_aspect_preserving():
    """Scaled mode must preserve aspect ratio with black border padding."""
    raw = _make_dummy_fundus(h=400, w=800)
    target_size = 512
    out = preprocess(raw, mode='scaled', size=target_size)

    assert out.shape == (target_size, target_size, 3)
    assert out.dtype == np.uint8

    # The top and bottom rows should be black padding
    assert (out[0, :, :] == 0).all()
    assert (out[-1, :, :] == 0).all()


def test_pad_square_resize():
    """pad_square_resize should center the non-square image and pad to square."""
    img = np.ones((200, 400, 3), dtype=np.uint8) * 120
    padded = pad_square_resize(img, 512)
    assert padded.shape == (512, 512, 3)
    # Top and bottom margins should be black (0)
    assert (padded[0, :, :] == 0).all()
    # Center should be non-zero
    assert padded[256, 256, 0] > 0
