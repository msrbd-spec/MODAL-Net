"""Preprocessing routines: legacy, scaled, clahe_only, none."""

import cv2
import numpy as np
from src.data.amplitude_mix import amplitude_mix


def crop_fundus(img, thr=10, margin=5):
    """Crop fundus bounding box from black background."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, t = cv2.threshold(g, thr, 255, cv2.THRESH_BINARY)
    cs, _ = cv2.findContours(t, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cs:
        x, y, w, h = cv2.boundingRect(max(cs, key=cv2.contourArea))
        x, y = max(0, x - margin), max(0, y - margin)
        w, h = min(img.shape[1] - x, w + 2 * margin), min(img.shape[0] - y, h + 2 * margin)
        img = img[y:y + h, x:x + w]
    return img


def clahe(img):
    """Apply CLAHE to L channel in LAB color space."""
    l, a, b = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2LAB))
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def ben_graham(img, sigma):
    """Ben Graham local contrast enhancement."""
    return cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), sigma), -4, 128)


def pad_square_resize(img, size):
    """Aspect-preserving square padding with black borders, then resize to size x size."""
    h, w = img.shape[:2]
    s = max(h, w)
    c = np.zeros((s, s, 3), dtype=img.dtype)
    y0, x0 = (s - h) // 2, (s - w) // 2
    c[y0:y0 + h, x0:x0 + w] = img
    return cv2.resize(c, (size, size), interpolation=cv2.INTER_AREA)


def preprocess(bgr, mode, size, style_rgb=None, amp=None):
    """Preprocess a fundus image (BGR uint8 input -> BGR uint8 output).

    Modes:
      - 'legacy': crop -> CLAHE -> Ben Graham(sigma=10 at native resolution).
      - 'scaled': crop -> pad_square_resize(size) -> CLAHE -> Ben Graham(sigma=size/50).
      - 'clahe_only': crop -> pad_square_resize(size) -> CLAHE.
      - 'none': crop -> pad_square_resize(size).

    Optional amplitude mixing:
      - amp={'stage': 'pre'|'post', 'beta': ..., 'lam': ...}
      - style_rgb: (H, W, 3) uint8 RGB image from the style bank.
    """
    img = crop_fundus(bgr)

    if mode == 'legacy':
        # Native-resolution sigma 10; resized later in albumentations (zoo behaviour)
        return ben_graham(clahe(img), 10)

    # All non-legacy modes are aspect-preserving padded to square at target size
    img = pad_square_resize(img, size)
    border_mask = (img.sum(axis=2, keepdims=True) > 15)

    if amp and amp.get('stage') == 'pre' and style_rgb is not None:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mixed_rgb = amplitude_mix(rgb, style_rgb, amp['beta'], amp['lam'])
        img = cv2.cvtColor(mixed_rgb, cv2.COLOR_RGB2BGR)

    if mode == 'scaled':
        img = ben_graham(clahe(img), size / 50)
        img = (img * border_mask).astype(np.uint8)
    elif mode == 'clahe_only':
        img = clahe(img)
        img = (img * border_mask).astype(np.uint8)
    elif mode == 'none':
        pass
    else:
        raise ValueError(f"Unknown preproc mode: {mode}")

    if amp and amp.get('stage') == 'post' and style_rgb is not None:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mixed_rgb = amplitude_mix(rgb, style_rgb, amp['beta'], amp['lam'])
        img = cv2.cvtColor(mixed_rgb, cv2.COLOR_RGB2BGR)

    return img
