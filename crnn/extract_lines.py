"""
Slice each clean mask in `data/masks_clean_upscaled/` into horizontal line
strips using row-projection segmentation, and join with `data/labels/labels.csv`
to produce a fine-tuning dataset.

The labels CSV indexes lines per image starting at 1.  Row projection
must produce the same ordered set of lines for the join to work.  We
print a per-image diagnostic so the user can verify the count matches.

Outputs:
    data/real_lines/<image_id>_line<N:02d>.png
    data/real_lines/index.csv   (filename, image_id, line, confidence, transcript)
    data/real_lines/_line_count_check.csv  (image_id, csv_lines, detected_lines)
"""
from __future__ import annotations
import os
import csv
import argparse
from collections import defaultdict
import cv2
import numpy as np

try:
    from scipy.signal import find_peaks
except ImportError:
    find_peaks = None


def split_lines_by_peaks(binary: np.ndarray,
                         target_n: int | None = None,
                         smooth_k: int = 7) -> list[tuple[int, int]]:
    """
    Detect text-line bands by finding peaks in the smoothed row sum.

    Palm-leaf strips have lines packed so tightly that the inter-line
    gaps never reach zero, so a simple threshold-based segmenter merges
    them.  This function finds local maxima instead, then slices between
    adjacent peaks at the local minimum.

    If `target_n` is given, the top-N peaks by prominence are kept.
    """
    if find_peaks is None:
        raise RuntimeError("scipy.signal.find_peaks unavailable")

    h = binary.shape[0]
    row = np.sum(binary == 255, axis=1).astype(np.float32)
    smoothed = np.convolve(row, np.ones(smooth_k) / smooth_k, mode="same")

    # Estimate min peak distance from target_n if known, else 12px
    distance = max(8, h // (2 * (target_n or 25)))
    peaks, props = find_peaks(smoothed, distance=distance, prominence=20)

    if target_n is not None and len(peaks) > target_n:
        # Keep the top-N most prominent peaks
        order = np.argsort(props["prominences"])[::-1][:target_n]
        peaks = np.sort(peaks[order])

    if len(peaks) == 0:
        return [(0, h)]

    # Find a valley between every pair of adjacent peaks
    bounds = []
    for i, p in enumerate(peaks):
        if i == 0:
            y0 = 0
        else:
            seg = smoothed[peaks[i - 1]:p + 1]
            y0 = peaks[i - 1] + int(np.argmin(seg))
        if i == len(peaks) - 1:
            y1 = h
        else:
            seg = smoothed[p:peaks[i + 1] + 1]
            y1 = p + int(np.argmin(seg))
        bounds.append((y0, y1))
    return bounds


def extract_image_lines(mask_path: str,
                        target_n: int | None = None) -> list[np.ndarray]:
    img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return []
    binary = (img > 127).astype(np.uint8) * 255

    bounds = split_lines_by_peaks(binary, target_n=target_n)
    crops = []
    for (y0, y1) in bounds:
        crop = binary[y0:y1, :]
        # Trim leading/trailing blank columns
        col_sums = np.sum(crop > 0, axis=0)
        nz = np.where(col_sums > 0)[0]
        if len(nz) >= 2:
            crop = crop[:, nz[0]:nz[-1] + 1]
        crops.append(crop)
    return crops


def run(mask_dir: str, labels_csv: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    # image_id (without _pre_x2_mask suffix? — labels CSV uses *with* it)
    transcripts: dict[str, dict[int, dict]] = defaultdict(dict)
    with open(labels_csv, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            transcripts[r["image"].strip()][int(r["line"])] = r

    rows_out = []
    count_check = []

    for image_id, line_map in transcripts.items():
        candidates = [
            os.path.join(mask_dir, image_id + ".png"),
            os.path.join(mask_dir, image_id + ".jpg"),
        ]
        mask_path = next((p for p in candidates if os.path.isfile(p)), None)
        if mask_path is None:
            print(f"  [skip] no mask file for {image_id}")
            continue

        n_csv = len(line_map)
        crops = extract_image_lines(mask_path, target_n=n_csv)
        n_det = len(crops)
        count_check.append({"image_id":  image_id,
                            "csv_lines": n_csv,
                            "detected":  n_det})
        print(f"  {image_id}: csv lines = {n_csv}, detected = {n_det}")

        # Match by index (1-based to detection order)
        n_use = min(n_csv, n_det)
        for i in range(n_use):
            line_no = sorted(line_map.keys())[i]
            crop = crops[i]
            fname = f"{image_id}_line{line_no:02d}.png"
            cv2.imwrite(os.path.join(out_dir, fname), crop)
            rec = line_map[line_no]
            rows_out.append({
                "filename":   fname,
                "image_id":   image_id,
                "line":       line_no,
                "confidence": rec["confidence"],
                "transcript": rec["transcript"],
            })

    idx_path = os.path.join(out_dir, "index.csv")
    with open(idx_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "filename", "image_id", "line", "confidence", "transcript"])
        w.writeheader()
        w.writerows(rows_out)

    chk_path = os.path.join(out_dir, "_line_count_check.csv")
    with open(chk_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image_id", "csv_lines", "detected"])
        w.writeheader()
        w.writerows(count_check)

    print(f"\nWrote {len(rows_out)} line crops")
    print(f"Index : {idx_path}")
    print(f"Check : {chk_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mask-dir",   default="data/masks_clean_upscaled")
    ap.add_argument("--labels-csv", default="data/labels/labels.csv")
    ap.add_argument("--out-dir",    default="data/real_lines")
    args = ap.parse_args()
    run(args.mask_dir, args.labels_csv, args.out_dir)
