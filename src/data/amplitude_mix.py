"""Fourier amplitude mixing (FDA / FACT) for source-only domain generalisation."""

import cv2
import numpy as np


def amplitude_mix(src, trg, beta, lam):
    """Mix Fourier amplitude spectrum of src with that of trg.

    Args:
        src: (H, W, 3) uint8 RGB image.
        trg: (H_t, W_t, 3) uint8 RGB image. Resized to src's shape.
        beta: Window ratio in [0, 1] around the DC frequency.
        lam: Interpolation weight in [0, 1].

    Returns:
        (H, W, 3) uint8 RGB image with black border preserved.
    """
    trg = cv2.resize(trg, (src.shape[1], src.shape[0]))
    s, t = src.astype(np.float32), trg.astype(np.float32)

    Fs = np.fft.fftshift(np.fft.fft2(s, axes=(0, 1)), axes=(0, 1))
    Ft = np.fft.fftshift(np.fft.fft2(t, axes=(0, 1)), axes=(0, 1))

    As, Ps, At = np.abs(Fs), np.angle(Fs), np.abs(Ft)
    h, w = s.shape[:2]
    b = max(1, int(min(h, w) * beta))
    cy, cx = h // 2, w // 2

    # Interpolate low-frequency amplitude box
    As[cy - b:cy + b, cx - b:cx + b] = (
        (1 - lam) * As[cy - b:cy + b, cx - b:cx + b]
        + lam * At[cy - b:cy + b, cx - b:cx + b]
    )

    out = np.fft.ifft2(np.fft.ifftshift(As * np.exp(1j * Ps), axes=(0, 1)), axes=(0, 1)).real

    # Preserve black border: mask out background where source sum <= 15
    mask = (s.sum(2, keepdims=True) > 15)
    return np.clip(out * mask, 0, 255).astype(np.uint8)
