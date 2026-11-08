"""Photometric training augmentation ("train with the variation you expect").

Used only for the robust variants of the methods, to test whether showing the model lighting, focus and noise
changes during training makes it tolerant to them. Random parameters are drawn from continuous ranges, so the
model never sees the exact test levels.
"""
from __future__ import annotations

import cv2
import numpy as np


def random_photometric(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    f = img.astype(np.float32)
    # brightness and contrast
    m = f.mean()
    f = m + (f - m) * rng.uniform(0.35, 1.2)
    f = f * rng.uniform(0.35, 2.0)
    # uneven illumination
    if rng.random() < 0.4:
        a = rng.uniform(0, 0.6)
        ramp = np.linspace(1.0, 1.0 - a, f.shape[1], dtype=np.float32)
        f = f * (ramp[None, :] if rng.random() < 0.5 else ramp[::-1][None, :])
    out = np.clip(f, 0, 255).astype(np.uint8)
    # focus / motion blur
    r = rng.random()
    if r < 0.3:
        out = cv2.GaussianBlur(out, (0, 0), rng.uniform(0.5, 3.0))
    elif r < 0.5:
        k = int(rng.integers(3, 12)) | 1
        kernel = np.zeros((k, k), np.float32); kernel[k // 2, :] = 1.0 / k
        out = cv2.filter2D(out, -1, kernel, borderType=cv2.BORDER_REFLECT)
    # sensor noise
    if rng.random() < 0.5:
        out = np.clip(out.astype(np.float32) + rng.normal(0, rng.uniform(3, 25), out.shape), 0, 255).astype(np.uint8)
    return out


def augment_set(x: np.ndarray, y: np.ndarray, copies: int = 1, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Original images plus `copies` randomly degraded copies of each."""
    rng = np.random.default_rng(seed)
    xs, ys = [x], [y]
    for _ in range(copies):
        xs.append(np.stack([random_photometric(im, rng) for im in x]))
        ys.append(y)
    return np.concatenate(xs), np.concatenate(ys)
