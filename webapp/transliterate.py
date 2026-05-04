"""
Glue between the Flask app and the project's `crnn` inference code.

Models are discovered by scanning `models/<arch>/best.pth`. They're loaded
lazily on first use and cached. For an uploaded image we run a lightweight
single-image preprocessing pass (CLAHE + adaptive-threshold mask cleanup)
to mirror the offline `preprocess_pipeline.py` without the slow Real-ESRGAN
upscale step.
"""
from __future__ import annotations
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch

from crnn.extract_lines import split_lines_by_peaks
from crnn.infer import _line_to_tensor, _load_model

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

# Architectures we expose in the UI. Order is the dropdown order.
PREFERRED_ORDER = ["cnn_ctc", "crnn_ctc", "conformer", "vit_ctc", "crnn_attn", "trocr"]

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_MODEL_CACHE: dict[str, tuple] = {}


def available_models() -> list[str]:
    found = []
    for arch in PREFERRED_ORDER:
        ckpt = MODELS_DIR / arch / "best.pth"
        if ckpt.is_file():
            found.append(arch)
    return found


def _get_model(arch: str):
    if arch in _MODEL_CACHE:
        return _MODEL_CACHE[arch]
    ckpt_path = MODELS_DIR / arch / "best.pth"
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"No checkpoint at {ckpt_path}")
    model, vocab = _load_model(str(ckpt_path), _DEVICE, arch=arch)
    _MODEL_CACHE[arch] = (model, vocab)
    return model, vocab


# ---- single-image preprocessing -------------------------------------------

def _to_clean_mask(img_bgr_or_gray: np.ndarray) -> np.ndarray:
    """CLAHE + adaptive-threshold mask, mirroring batch_mask_clean.py."""
    if img_bgr_or_gray.ndim == 3:
        gray = cv2.cvtColor(img_bgr_or_gray, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_bgr_or_gray

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    blur = cv2.GaussianBlur(enhanced, (3, 3), 0)
    th = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        41, 7,
    )

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        th, connectivity=8
    )
    H, W = th.shape
    clean = np.zeros_like(th)
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area < 40 or h < 3 or w < 3 or area > 0.25 * H * W:
            continue
        clean[labels == i] = 255

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
    clean = cv2.dilate(clean, kernel, iterations=1)
    return clean


def _looks_like_mask(img: np.ndarray) -> bool:
    """If the image is already binary (white text on black), skip preprocess."""
    if img.ndim != 2:
        return False
    h = np.histogram(img, bins=8)[0]
    # Heavy weight at the two extremes => likely a mask
    edges = h[0] + h[-1]
    return edges > 0.9 * img.size


def _split_lines(mask: np.ndarray) -> list[np.ndarray]:
    binary = (mask > 127).astype(np.uint8) * 255
    bounds = split_lines_by_peaks(binary)
    crops = []
    for (y0, y1) in bounds:
        crop = binary[y0:y1, :]
        col_sums = np.sum(crop > 0, axis=0)
        nz = np.where(col_sums > 0)[0]
        if len(nz) >= 2:
            crop = crop[:, nz[0]:nz[-1] + 1]
        if crop.shape[0] >= 8 and crop.shape[1] >= 8:
            crops.append(crop)
    return crops


# ---- public entry point ---------------------------------------------------

@torch.no_grad()
def run(arch: str, image_path: str) -> list[list[str]]:
    """
    Read `image_path`, preprocess if necessary, segment into lines, and
    transliterate each line. Returns a list of per-line token lists.
    """
    raw = cv2.imread(image_path)
    if raw is None:
        # Try grayscale fallback (some PNGs)
        raw_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if raw_gray is None:
            raise ValueError(f"Could not read image: {image_path}")
        raw = raw_gray

    if raw.ndim == 2 and _looks_like_mask(raw):
        mask = raw
    else:
        mask = _to_clean_mask(raw)

    crops = _split_lines(mask)
    if not crops:
        return []

    model, vocab = _get_model(arch)
    out: list[list[str]] = []
    for crop in crops:
        x = _line_to_tensor(crop).to(_DEVICE)
        pred = model(x)
        tokens = model.decode(pred)[0]
        out.append(tokens)
    return out
