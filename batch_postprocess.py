import os
import cv2
import numpy as np

input_dir = "data/upscaled"
output_dir = "data/final"
os.makedirs(output_dir, exist_ok=True)

for fname in os.listdir(input_dir):
    if fname.lower().endswith(".png"):
        inp = os.path.join(input_dir, fname)
        out = os.path.join(
            output_dir, fname.replace(".png", "_final.png")
        )

        img = cv2.imread(inp, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        # Local contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        contrast = clahe.apply(img)

        # Gentle sharpening (text-safe)
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        sharpened = cv2.filter2D(contrast, -1, kernel)

        cv2.imwrite(out, sharpened)

print("Post-processing completed")
