"""
Palm-leaf manuscript degradation pipeline.

Each function takes a (uint8) binary or grayscale image and returns a
degraded image of the same shape.  All randomness uses an injectable
np.random.Generator so generation is reproducible.

Default ranges are tuned to roughly match what we see in
data/masks_clean_upscaled/.
"""
from __future__ import annotations
import cv2
import numpy as np


# ─────────────────────────────────────────────────────────────────────
# Geometric
# ─────────────────────────────────────────────────────────────────────

def random_affine(img: np.ndarray, rng: np.random.Generator,
                  rot_deg: float = 4.0,
                  shear: float = 0.05,
                  scale_jitter: float = 0.07) -> np.ndarray:
    h, w = img.shape[:2]
    cx, cy = w / 2, h / 2
    angle = rng.uniform(-rot_deg, rot_deg)
    sx = rng.uniform(-shear, shear)
    sy = rng.uniform(-shear, shear)
    s  = 1.0 + rng.uniform(-scale_jitter, scale_jitter)

    M = cv2.getRotationMatrix2D((cx, cy), angle, s)
    # Add shear
    M[0, 1] += sx
    M[1, 0] += sy
    return cv2.warpAffine(img, M, (w, h),
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def elastic_warp(img: np.ndarray, rng: np.random.Generator,
                 alpha: float = 8.0, sigma: float = 4.0) -> np.ndarray:
    h, w = img.shape[:2]
    dx = cv2.GaussianBlur(
        rng.uniform(-1, 1, (h, w)).astype(np.float32),
        (0, 0), sigma) * alpha
    dy = cv2.GaussianBlur(
        rng.uniform(-1, 1, (h, w)).astype(np.float32),
        (0, 0), sigma) * alpha
    xx, yy = np.meshgrid(np.arange(w, dtype=np.float32),
                         np.arange(h, dtype=np.float32))
    map_x = (xx + dx).astype(np.float32)
    map_y = (yy + dy).astype(np.float32)
    return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)


# ─────────────────────────────────────────────────────────────────────
# Photometric / structural
# ─────────────────────────────────────────────────────────────────────

def random_dilate_erode(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Simulate ink-bleed (dilate) or thin strokes (erode)."""
    op = rng.choice(["none", "dilate", "erode"])
    if op == "none":
        return img
    k = rng.choice([(2, 2), (3, 3), (2, 3), (3, 2)])
    kernel = np.ones(k, np.uint8)
    if op == "dilate":
        return cv2.dilate(img, kernel, iterations=1)
    return cv2.erode(img, kernel, iterations=1)


def random_blur(img: np.ndarray, rng: np.random.Generator,
                p: float = 0.5) -> np.ndarray:
    if rng.random() > p:
        return img
    sigma = rng.uniform(0.4, 1.4)
    return cv2.GaussianBlur(img, (0, 0), sigma)


def gaussian_noise(img: np.ndarray, rng: np.random.Generator,
                   p: float = 0.7) -> np.ndarray:
    if rng.random() > p:
        return img
    sigma = rng.uniform(8, 25)
    noise = rng.normal(0, sigma, img.shape).astype(np.float32)
    out = img.astype(np.float32) + noise
    return np.clip(out, 0, 255).astype(np.uint8)


def fibre_streaks(img: np.ndarray, rng: np.random.Generator,
                  n: int = 3) -> np.ndarray:
    """Add thin horizontal lines to simulate palm-fibre noise."""
    h, w = img.shape[:2]
    out = img.copy()
    for _ in range(rng.integers(0, n + 1)):
        y0 = rng.integers(0, h)
        x0 = rng.integers(0, w)
        L  = rng.integers(w // 4, w)
        thickness = int(rng.integers(1, 3))
        intensity = int(rng.integers(120, 220))
        cv2.line(out, (x0, y0), (min(w, x0 + L), y0 + rng.integers(-2, 3)),
                 intensity, thickness)
    return out


def random_erase(img: np.ndarray, rng: np.random.Generator,
                 n: int = 3, p: float = 0.6) -> np.ndarray:
    """Erase small random rectangles (simulates ink fade)."""
    if rng.random() > p:
        return img
    h, w = img.shape[:2]
    out = img.copy()
    for _ in range(rng.integers(1, n + 1)):
        rh = rng.integers(2, max(3, h // 6))
        rw = rng.integers(2, max(3, w // 6))
        ry = rng.integers(0, h - rh)
        rx = rng.integers(0, w - rw)
        out[ry:ry + rh, rx:rx + rw] = 0
    return out


def punch_hole(img: np.ndarray, rng: np.random.Generator,
               p: float = 0.10) -> np.ndarray:
    """Add a circular punch hole.  Lower probability for char crops."""
    if rng.random() > p:
        return img
    h, w = img.shape[:2]
    out = img.copy()
    radius = int(rng.integers(min(h, w) // 5, max(2, min(h, w) // 3)))
    cx = int(rng.integers(radius, max(radius + 1, w - radius)))
    cy = int(rng.integers(radius, max(radius + 1, h - radius)))
    cv2.circle(out, (cx, cy), radius, 0, -1)
    return out


# ─────────────────────────────────────────────────────────────────────
# Pipelines
# ─────────────────────────────────────────────────────────────────────

def degrade_char(img: np.ndarray,
                 rng: np.random.Generator | None = None) -> np.ndarray:
    """Full degradation pipeline for character crops."""
    rng = rng or np.random.default_rng()
    out = img
    out = random_affine(out, rng)
    out = elastic_warp(out, rng, alpha=4.0, sigma=3.0)
    out = random_dilate_erode(out, rng)
    out = random_blur(out, rng, p=0.4)
    out = gaussian_noise(out, rng, p=0.5)
    out = random_erase(out, rng, n=2, p=0.5)
    return out


def degrade_line(img: np.ndarray,
                 rng: np.random.Generator | None = None) -> np.ndarray:
    """Full degradation pipeline for line strips."""
    rng = rng or np.random.default_rng()
    out = img
    out = random_affine(out, rng, rot_deg=2.0, shear=0.03, scale_jitter=0.05)
    out = elastic_warp(out, rng, alpha=6.0, sigma=4.0)
    out = random_dilate_erode(out, rng)
    out = random_blur(out, rng, p=0.5)
    out = fibre_streaks(out, rng, n=4)
    out = gaussian_noise(out, rng, p=0.7)
    out = random_erase(out, rng, n=4, p=0.6)
    out = punch_hole(out, rng, p=0.05)
    return out


if __name__ == "__main__":
    from .glyph_renderer import GlyphRenderer, render_line
    rng = np.random.default_rng(42)

    r = GlyphRenderer()
    base = r.render("കാ", jitter=False)
    sheet = [base]
    for _ in range(7):
        sheet.append(degrade_char(base.copy(), rng))
    sheet = np.hstack(sheet)
    cv2.imwrite("synthetic/_test_degrade_char.png", sheet)
    print(f"Wrote synthetic/_test_degrade_char.png ({sheet.shape})")

    line = render_line("ശ്രീ ഭാരത രാഷ്ട്രം ജയ", height=64, font_size=48)
    sheets = [line]
    for _ in range(3):
        sheets.append(degrade_line(line.copy(), rng))
    out = np.vstack([s if s.shape[1] == line.shape[1]
                     else cv2.copyMakeBorder(s, 0, 0, 0,
                                              line.shape[1] - s.shape[1],
                                              cv2.BORDER_CONSTANT, value=0)
                     for s in sheets])
    cv2.imwrite("synthetic/_test_degrade_line.png", out)
    print(f"Wrote synthetic/_test_degrade_line.png ({out.shape})")
