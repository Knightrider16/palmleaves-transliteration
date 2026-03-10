def run_batch_upscale():

    import os
    import torch
    import cv2
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer

    input_dir = "data/preprocessed"
    output_dir = "data/upscaled"

    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=4
    )

    upsampler = RealESRGANer(
        scale=4,
        model_path="weights/realesrgan_x4plus.pth",
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=torch.cuda.is_available(),
        device=device,
    )

    for fname in os.listdir(input_dir):

        if fname.lower().endswith((".png", ".jpg", ".jpeg")):

            inp = os.path.join(input_dir, fname)

            out = os.path.join(
                output_dir,
                fname.rsplit(".", 1)[0] + "_x2.png"
            )

            # Skip already upscaled
            if os.path.exists(out):
                print(f"Skipping {fname} (already upscaled)")
                continue

            print(f"Upscaling {fname}")

            img = cv2.imread(inp, cv2.IMREAD_GRAYSCALE)

            if img is None:
                continue

            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            output, _ = upsampler.enhance(img, outscale=2)

            output_gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)

            cv2.imwrite(out, output_gray)

    print("Batch upscaling completed successfully")


if __name__ == "__main__":
    run_batch_upscale()