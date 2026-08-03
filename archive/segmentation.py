"""
Script A (v4) — Palm Leaf Segmentation for Horizontal Strips
=============================================================
Handles wide horizontal manuscript strips (e.g. 7904x468px).
Detects lines by row projection across the full image width,
with multi-scale smoothing to handle paragraph column gaps.

Key improvements:
  - No column splitting — treats full strip as one unit
  - Multi-scale row projection finds lines across all columns
  - Much stricter punch hole detection (3 tighter conditions)
  - Characters saved as: {image_id}_line{N:02d}_char{N:03d}.png
  - character_index.csv for Script B

CSV format this produces index for:
    image, line, confidence, transcript
    1A1_pre_x2_mask, 1, high, ka/la/...

Usage:
    python script_A_segmentation.py

Inputs:  data/masks_clean_upscaled/
Outputs: data/characters_named/
         data/characters_named/character_index.csv
"""

import cv2
import numpy as np
import os
import csv

INPUT_FOLDER  = "data/masks_clean_upscaled"
OUTPUT_FOLDER = "data/characters_named"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ─────────────────────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────────────────────

def sauvola_threshold(gray, window_size=25, k=0.5, R=128):
    gray_f  = gray.astype(np.float64)
    mean    = cv2.boxFilter(gray_f,    -1, (window_size, window_size))
    mean_sq = cv2.boxFilter(gray_f**2, -1, (window_size, window_size))
    std     = np.sqrt(np.abs(mean_sq - mean**2))
    return (gray_f < mean * (1 + k * (std / R - 1))).astype(np.uint8) * 255


def remove_horizontal_lines(binary, w_img):
    """Two-pass: removes long fibres then medium fragments."""
    for h_len in [max(80, w_img // 10), max(40, w_img // 20)]:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (h_len, 1))
        binary = cv2.subtract(binary,
                    cv2.morphologyEx(binary, cv2.MORPH_OPEN, k))
    return binary


def remove_specks(binary, min_area=60):
    """Delete connected components smaller than min_area."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    clean = np.zeros_like(binary)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            clean[labels == i] = 255
    return clean


def remove_punch_holes(binary, num_holes=2):
    """
    Strict three-condition punch hole removal.
    All three must be satisfied simultaneously:

    1. Large area  — must be bigger than any realistic character
       (uses adaptive threshold: 1.5% of image area)
    2. High circularity  — threshold raised to 0.78
       (characters rarely exceed 0.65 even when circular)
    3. Strong hollow interior — inner void > 35% of parent area
       (characters have strokes, not empty rings)

    Also: dilates removal mask by 9px to erase halo.
    """
    h_img, w_img = binary.shape[:2]
    # Punch holes must be at least 1.5% of image area
    min_punch_area = max(800, int(h_img * w_img * 0.015))

    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    mask        = np.zeros_like(binary)
    holes_found = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)

        # Condition 1 — large enough
        if area < min_punch_area:
            break   # sorted by area, no point continuing

        # Condition 2 — roughly square bounding box
        _, _, w, h = cv2.boundingRect(cnt)
        aspect = w / float(h) if h > 0 else 0
        if not (0.65 < aspect < 1.35):
            continue

        # Condition 2b — high circularity
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = (4 * np.pi * area) / (perimeter ** 2)
        if circularity < 0.78:           # raised from 0.60
            continue

        # Condition 3 — hollow interior (the actual hole)
        c_all, hier = cv2.findContours(
            binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        has_strong_hole = False
        if hier is not None:
            for i, he in enumerate(hier[0]):
                if he[3] >= 0:           # has a parent = inner contour
                    hole_area   = cv2.contourArea(c_all[i])
                    parent_area = cv2.contourArea(c_all[he[3]])
                    if parent_area > 0 and hole_area / parent_area > 0.35:
                        has_strong_hole = True
                        break
        if not has_strong_hole:
            continue

        cv2.drawContours(mask, [cnt], -1, 255, -1)
        holes_found += 1
        if holes_found >= num_holes:
            break

    # Dilate mask to erase ink halo around each hole
    kernel  = np.ones((9, 9), np.uint8)
    dilated = cv2.dilate(mask, kernel, iterations=1)
    return cv2.bitwise_and(binary, cv2.bitwise_not(dilated))


def preprocess(img):
    h_img, w_img = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if np.mean(gray) < 127:
        gray = cv2.bitwise_not(gray)
    denoised = cv2.fastNlMeansDenoising(
        gray, h=12, templateWindowSize=7, searchWindowSize=21)
    denoised = cv2.medianBlur(denoised, 3)
    binary   = sauvola_threshold(denoised)
    binary   = remove_punch_holes(binary)
    binary   = remove_horizontal_lines(binary, w_img)
    binary   = remove_specks(binary, min_area=60)
    binary   = cv2.morphologyEx(
        binary, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
    return binary


# ─────────────────────────────────────────────────────────────
# LINE SEGMENTATION FOR WIDE HORIZONTAL STRIPS
#
# Problem: a 7904×468 strip has text lines stacked vertically,
# but they span across multiple paragraph columns horizontally.
# Standard row projection works but needs careful smoothing.
#
# Solution: use two-scale smoothing —
#   - fine scale  (5px)  to find individual line peaks
#   - coarse scale (25px) to handle column gaps without
#     treating inter-column whitespace as line breaks
# ─────────────────────────────────────────────────────────────

def segment_lines_horizontal_strip(binary, min_line_height=12,
                                    gap_thresh=0.03):
    """
    Row projection profile segmentation tuned for wide strips.
    Uses coarse smoothing to bridge inter-column gaps vertically.
    """
    row_sums = np.sum(binary == 255, axis=1).astype(float)

    # Coarse smoothing — bridges small gaps within a line
    kernel_coarse = np.ones(25) / 25
    smoothed = np.convolve(row_sums, kernel_coarse, mode='same')

    peak = smoothed.max()
    if peak == 0:
        return [(0, binary.shape[0])]

    threshold = peak * gap_thresh
    in_line   = False
    lines     = []
    y_start   = 0

    for y, val in enumerate(smoothed):
        if not in_line and val > threshold:
            in_line = True
            y_start = y
        elif in_line and val <= threshold:
            in_line = False
            if y - y_start >= min_line_height:
                lines.append((max(0, y_start - 3),
                               min(binary.shape[0], y + 3)))

    if in_line and binary.shape[0] - y_start >= min_line_height:
        lines.append((max(0, y_start - 3), binary.shape[0]))

    return lines if lines else [(0, binary.shape[0])]


# ─────────────────────────────────────────────────────────────
# CHARACTER VALIDATION
# ─────────────────────────────────────────────────────────────

def solidity(char_img):
    cnts, _ = cv2.findContours(
        char_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 0
    cnt  = max(cnts, key=cv2.contourArea)
    hull = cv2.contourArea(cv2.convexHull(cnt))
    return cv2.contourArea(cnt) / hull if hull > 0 else 0


def is_valid_character(char_img, w, h):
    fill   = np.sum(char_img == 255) / float(w * h) if w * h > 0 else 0
    if fill < 0.06:                         return False
    aspect = w / float(h) if h > 0 else 0
    if aspect > 5.0 or aspect < 0.15:      return False
    if solidity(char_img) < 0.15:          return False
    return True


# ─────────────────────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────────────────────

def _run_main():
    index_rows  = []
    total_chars = 0

    for img_name in sorted(os.listdir(INPUT_FOLDER)):
        if not img_name.lower().endswith(
                ('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
            continue

        img = cv2.imread(os.path.join(INPUT_FOLDER, img_name))
        if img is None:
            print(f"  [skip] {img_name}")
            continue

        image_id     = os.path.splitext(img_name)[0]
        h_img, w_img = img.shape[:2]
        print(f"\nProcessing: {image_id}  ({w_img}×{h_img})")

        binary = preprocess(img)

        # Line segmentation
        lines = segment_lines_horizontal_strip(binary)
        print(f"  Lines detected: {len(lines)}")

        # Dynamic size thresholds based on image size
        img_area = h_img * w_img
        min_area = max(100, img_area // 60000)
        max_dim  = max(100, h_img // 2)

        for line_idx, (y_start, y_end) in enumerate(lines):
            line_num   = line_idx + 1
            line_strip = binary[y_start:y_end, :]

            if line_strip.size == 0:
                continue

            n, _labels, stats, _ = cv2.connectedComponentsWithStats(line_strip)
            chars_this_line = []

            for i in range(1, n):
                x    = stats[i, cv2.CC_STAT_LEFT]
                y    = stats[i, cv2.CC_STAT_TOP]
                w    = stats[i, cv2.CC_STAT_WIDTH]
                h    = stats[i, cv2.CC_STAT_HEIGHT]
                area = stats[i, cv2.CC_STAT_AREA]

                if area < min_area:             continue
                if w < 8 or h < 8:             continue
                if w > max_dim or h > max_dim: continue

                char_img = line_strip[y:y+h, x:x+w]
                if not is_valid_character(char_img, w, h):
                    continue

                chars_this_line.append((x, y, w, h, char_img))

            chars_this_line.sort(key=lambda c: c[0])   # left to right

            for char_idx, (x, y, w, h, char_img) in enumerate(chars_this_line):
                char_num = char_idx + 1
                size     = max(w, h)
                padded   = np.zeros((size, size), dtype=np.uint8)
                padded[(size-h)//2:(size-h)//2+h,
                       (size-w)//2:(size-w)//2+w] = char_img

                filename = f"{image_id}_line{line_num:02d}_char{char_num:03d}.png"
                cv2.imwrite(os.path.join(OUTPUT_FOLDER, filename), padded)

                index_rows.append({
                    "filename": filename,
                    "image_id": image_id,
                    "line_num": line_num,
                    "char_num": char_num,
                    "x": x,
                    "y": y + y_start,
                    "w": w,
                    "h": h,
                })
                total_chars += 1

        print(f"  Characters this image: "
              f"{sum(1 for r in index_rows if r['image_id'] == image_id)}")
        print(f"  Running total: {total_chars}")

    # Save index
    index_path = os.path.join(OUTPUT_FOLDER, "character_index.csv")
    with open(index_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["filename","image_id","line_num","char_num",
                            "x","y","w","h"])
        writer.writeheader()
        writer.writerows(index_rows)

    print("\n" + "="*50)
    print(f"Done. Total characters : {total_chars}")
    print(f"Index saved to         : {index_path}")


if __name__ == "__main__":
    _run_main()