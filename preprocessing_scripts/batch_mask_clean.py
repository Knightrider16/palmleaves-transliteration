def run_batch_mask_clean():

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

        out = os.path.join(
            output_dir,
            fname.replace(".png", "_mask.png")
        )

        # Skip if mask exists
        if os.path.exists(out):
            print(f"Skipping {fname} (mask exists)")
            continue

        print(f"Creating mask for {fname}")

        img = cv2.imread(inp, 0)

        blur = cv2.GaussianBlur(img, (3, 3), 0)

        th = cv2.adaptiveThreshold(
            blur,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            41,
            7
        )

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            th,
            connectivity=8
        )

        clean_mask = np.zeros_like(th)

        H, W = th.shape

        for i in range(1, num_labels):

            x, y, w, h, area = stats[i]

            if area < 40:
                continue

            if h < 3 or w < 3:
                continue

            if area > 0.25 * H * W:
                continue

            clean_mask[labels == i] = 255

        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (2, 1)
        )

        clean_mask = cv2.dilate(clean_mask, kernel, iterations=1)

        cv2.imwrite(out, clean_mask)

    print("Clean masks saved in data/masks_clean_upscaled/")


if __name__ == "__main__":
    run_batch_mask_clean()