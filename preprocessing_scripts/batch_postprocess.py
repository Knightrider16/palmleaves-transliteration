def run_batch_postprocess():

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
                output_dir,
                fname.replace(".png", "_final.png")
            )

            # Skip already processed
            if os.path.exists(out):
                print(f"Skipping {fname} (already postprocessed)")
                continue

            print(f"Postprocessing {fname}")

            img = cv2.imread(inp, cv2.IMREAD_GRAYSCALE)

            if img is None:
                continue

            clahe = cv2.createCLAHE(
                clipLimit=2.0,
                tileGridSize=(8, 8)
            )

            contrast = clahe.apply(img)

            kernel = np.array([
                [0, -1, 0],
                [-1, 5, -1],
                [0, -1, 0]
            ])

            sharpened = cv2.filter2D(contrast, -1, kernel)

            cv2.imwrite(out, sharpened)

    print("Post-processing completed")


if __name__ == "__main__":
    run_batch_postprocess()