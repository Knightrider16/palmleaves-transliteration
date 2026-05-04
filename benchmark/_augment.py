"""
Shared augmentation transforms for benchmark training.

Two variants:
    augment_word(img)  - variable-width word crops (ICFHR D Balinese)
    augment_char(img)  - square character crops (AMADI charrec / lineocr)

The character variant skips horizontal stretch (would distort 1:1 aspect)
and reduces erasing rectangle size since char crops are tiny.
"""
from __future__ import annotations

import cv2
import numpy as np


def _affine(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    h, w = img.shape
    angle = rng.uniform(-3, 3)
    shear = rng.uniform(-0.05, 0.05)
    scale = rng.uniform(0.95, 1.05)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    M[0, 1] += shear
    return cv2.warpAffine(img, M, (w, h),
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _elastic(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    h, w = img.shape
    alpha = rng.uniform(2.0, 6.0)
    sigma = rng.uniform(2.0, 4.0)
    dx = cv2.GaussianBlur(
        rng.uniform(-1, 1, (h, w)).astype(np.float32),
        (0, 0), sigma) * alpha
    dy = cv2.GaussianBlur(
        rng.uniform(-1, 1, (h, w)).astype(np.float32),
        (0, 0), sigma) * alpha
    xx, yy = np.meshgrid(np.arange(w, dtype=np.float32),
                         np.arange(h, dtype=np.float32))
    return cv2.remap(img, (xx + dx).astype(np.float32),
                     (yy + dy).astype(np.float32),
                     cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _stroke_jitter(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    op = rng.choice(["dilate", "erode"])
    kernel = np.ones((2, 2), np.uint8)
    if op == "dilate":
        return cv2.dilate(img, kernel, iterations=1)
    return cv2.erode(img, kernel, iterations=1)


def _horizontal_stretch(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    h, w = img.shape
    scale_w = rng.uniform(0.85, 1.15)
    new_w = max(8, int(w * scale_w))
    return cv2.resize(img, (new_w, h), interpolation=cv2.INTER_AREA)


def _erase(img: np.ndarray, rng: np.random.Generator,
           h_div: int, w_div: int) -> np.ndarray:
    h, w = img.shape
    rh = int(rng.integers(2, max(3, h // h_div)))
    rw = int(rng.integers(2, max(3, w // w_div)))
    ry = int(rng.integers(0, max(1, h - rh)))
    rx = int(rng.integers(0, max(1, w - rw)))
    img = img.copy()
    img[ry:ry + rh, rx:rx + rw] = 0
    return img


def augment_word(img: np.ndarray,
                 rng: np.random.Generator | None = None,
                 strength: float = 1.0) -> np.ndarray:
    """
    Aggressive augmentation for variable-width word images.

    `strength` scales each transform's probability (1.0 = full, 0.5 = half).
    Used to reduce aug intensity for unstable archs (e.g. crnn_ctc BiLSTM).
    """
    rng = rng or np.random.default_rng()

    if rng.random() < 0.6 * strength:
        img = _affine(img, rng)
    if rng.random() < 0.4 * strength:
        img = _elastic(img, rng)
    if rng.random() < 0.4 * strength:
        img = _stroke_jitter(img, rng)
    if rng.random() < 0.3 * strength:
        img = _horizontal_stretch(img, rng)
    if rng.random() < 0.2 * strength:
        img = _erase(img, rng, h_div=8, w_div=12)
    return img


def augment_char(img: np.ndarray,
                 rng: np.random.Generator | None = None,
                 strength: float = 1.0) -> np.ndarray:
    """
    Augmentation for square character crops.
    Drops horizontal stretch (preserves 1:1 aspect).
    Tighter erase box since char crops are small.
    """
    rng = rng or np.random.default_rng()

    if rng.random() < 0.6 * strength:
        img = _affine(img, rng)
    if rng.random() < 0.4 * strength:
        img = _elastic(img, rng)
    if rng.random() < 0.4 * strength:
        img = _stroke_jitter(img, rng)
    if rng.random() < 0.2 * strength:
        img = _erase(img, rng, h_div=6, w_div=6)
    return img
