def run_preprocess():

    import cv2
    import os

    input_dir = "data/original"
    output_dir = "data/preprocessed"

    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):

        if filename.lower().endswith((".png", ".jpg", ".jpeg")):

            input_path = os.path.join(input_dir, filename)

            output_path = os.path.join(
                output_dir,
                filename.rsplit(".", 1)[0] + "_pre.png"
            )

            # Skip already processed
            if os.path.exists(output_path):
                print(f"Skipping {filename} (already preprocessed)")
                continue

            print(f"Processing {filename}")

            img = cv2.imread(input_path)

            if img is None:
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            clahe = cv2.createCLAHE(
                clipLimit=2.0,
                tileGridSize=(8, 8)
            )

            enhanced = clahe.apply(gray)

            cv2.imwrite(output_path, enhanced)

    print("Batch preprocessing completed!")


if __name__ == "__main__":
    run_preprocess()