import os
import cv2
import numpy as np

input_dir = "data/upscaled"
output_dir = "data/masks_clean_upscaled"
os.makedirs(output_dir, exist_ok=True)

for fname in os.listdir(input_dir):
    if not fname.lower().endswith(".png"):
        continue

    inp = os.path.join(input_dir, fname)
    out = os.path.join(output_dir, fname.replace(".png", "_mask.png"))

    img = cv2.imread(inp, 0)

    # --------------------------------
    # Step 1 — adaptive threshold
    # --------------------------------
    blur = cv2.GaussianBlur(img, (3,3), 0)

    th = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        41, 7
    )

    # --------------------------------
    # Step 2 — connected components
    # --------------------------------
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(th, connectivity=8)

    clean_mask = np.zeros_like(th)

    H, W = th.shape

    for i in range(1, num_labels):  # skip background
        x, y, w, h, area = stats[i]

        # ---------- FILTER RULES ----------
        # remove tiny dust
        if area < 40:
            continue

        # remove very thin cracks
        if h < 3 or w < 3:
            continue

        # remove huge non-text blobs
        if area > 0.25 * H * W:
            continue

        # keep this component
        clean_mask[labels == i] = 255

    # --------------------------------
    # Step 3 — reconnect strokes slightly
    # --------------------------------
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,1))
    clean_mask = cv2.dilate(clean_mask, kernel, iterations=1)

    cv2.imwrite(out, clean_mask)

print("Clean masks saved in data/masks_clean/")
