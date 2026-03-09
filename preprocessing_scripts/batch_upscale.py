import os
import torch
import cv2
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

# -------- Paths --------
input_dir = "data/preprocessed"
output_dir = "data/upscaled"
os.makedirs(output_dir, exist_ok=True)

# -------- Device --------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------- Model definition (OFFICIAL x4plus architecture) --------
model = RRDBNet(
    num_in_ch=3,    # MUST be 3 for x4plus
    num_out_ch=3,
    num_feat=64,
    num_block=23,
    num_grow_ch=32,
    scale=4
)

# -------- Upsampler --------
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

# -------- Batch processing --------
for fname in os.listdir(input_dir):
    if fname.lower().endswith((".png", ".jpg", ".jpeg")):
        inp = os.path.join(input_dir, fname)
        out = os.path.join(
            output_dir, fname.rsplit(".", 1)[0] + "_x2.png"
        )

        img = cv2.imread(inp, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        # Convert grayscale → 3-channel (REQUIRED)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # Use outscale=2 to avoid hallucination
        output, _ = upsampler.enhance(img, outscale=2)

        # Convert back to grayscale
        output_gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(out, output_gray)

print("Batch upscaling completed successfully")
