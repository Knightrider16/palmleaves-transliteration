import os
import cv2
import numpy as np

# INPUT: your enhanced images
input_dir = "data/upscaled"

# OUTPUT: masks only
output_dir = "data/masks_upscaled"
os.makedirs(output_dir, exist_ok=True)

for fname in os.listdir(input_dir):
    if fname.lower().endswith(".png"):

        inp = os.path.join(input_dir, fname)
        out = os.path.join(
            output_dir,
            fname.replace(".png", "_mask.png")
        )

        img = cv2.imread(inp, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        # ---------------------------
        # slight denoise
        # ---------------------------
        blur = cv2.GaussianBlur(img, (3, 3), 0)

        # ---------------------------
        # adaptive threshold (best for palm leaves)
        # ---------------------------
        th = cv2.adaptiveThreshold(
            blur,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            41,   # block size (local region)
            7     # constant subtraction
        )

        # ---------------------------
        # remove tiny noise
        # ---------------------------
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        mask = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)

        # ---------------------------
        # reconnect broken strokes
        # ---------------------------
        kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
        mask = cv2.dilate(mask, kernel2, iterations=1)

        cv2.imwrite(out, mask)

print("Foreground masks created in data/masks/")
