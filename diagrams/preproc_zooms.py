"""
Generate zoomed crops of the same region across all five preprocessing
stages so the differences are visible. Stage 0 (original) gets a
composite figure with the full leaf at top and the zoomed crop below,
plus a red box on the full leaf marking where the zoom came from.

Source images live in `data/`:
    data/original/<id>.jpg                 (3952x234, 1x scale)
    data/preprocessed/<id>_pre.png         (3952x234, 1x scale)
    data/upscaled/<id>_pre_x2.png          (7904x468, 2x after Real-ESRGAN)
    data/final/<id>_pre_x2_final.png       (7904x468, 2x)
    data/masks_clean_upscaled/<id>_pre_x2_mask.png  (7904x468, 2x)

We pick a 1500x420 crop in 2x coordinates (i.e. a 750x210 region in 1x
coordinates) from a section that has clear text. Stages 0/1 are
upscaled 2x with Lanczos so all five zoom outputs land at the same
1500x420 px so they can be displayed side-by-side at identical size.
"""
from __future__ import annotations
from pathlib import Path
from PIL import Image, ImageDraw

ID = "1A1"

# Crop region in 2x (Real-ESRGAN-doubled) coordinates.
# (x0, y0, x1, y1)
ZOOM_2X = (3500, 60, 5000, 460)  # 1500 wide x 400 tall

# 1x equivalents
ZOOM_1X = tuple(c // 2 for c in ZOOM_2X)

ZOOM_W = ZOOM_2X[2] - ZOOM_2X[0]   # 1500 px (display width)
ZOOM_H = ZOOM_2X[3] - ZOOM_2X[1]   # 400 px

FIG = Path("report/DUK SoCSE Thesis Template/figures")

SOURCES = [
    # (stage_index, file_path, scale_factor)
    (0, Path(f"data/original/{ID}.jpg"),                              1),
    (1, Path(f"data/preprocessed/{ID}_pre.png"),                       1),
    (2, Path(f"data/upscaled/{ID}_pre_x2.png"),                        2),
    (3, Path(f"data/final/{ID}_pre_x2_final.png"),                     2),
    (4, Path(f"data/masks_clean_upscaled/{ID}_pre_x2_mask.png"),       2),
]

LABELS = {
    0: "Stage 0: original colour scan",
    1: "Stage 1: after CLAHE contrast enhancement",
    2: "Stage 2: after Real-ESRGAN 2x super-resolution",
    3: "Stage 3: after adaptive thresholding + sharpening",
    4: "Stage 4: cleaned binary mask after CC filtering",
}


def crop_zoom(img: Image.Image, scale: int) -> Image.Image:
    """Crop the fixed zoom region from `img` (whose own scale is `scale`),
    then upscale to ZOOM_W x ZOOM_H so all five outputs are the same size."""
    box = ZOOM_1X if scale == 1 else ZOOM_2X
    crop = img.crop(box)
    if crop.size != (ZOOM_W, ZOOM_H):
        crop = crop.resize((ZOOM_W, ZOOM_H), Image.LANCZOS)
    return crop


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)

    # Stages 1-4: just save the zoomed crop.
    for stage, src, scale in SOURCES[1:]:
        img = Image.open(src)
        zoom = crop_zoom(img, scale)
        out = FIG / f"preproc_zoom_stage{stage}.png"
        # Stage 4 (binary mask) stays PNG; stages 1-3 are photographic
        # so JPEG is fine and keeps the file small.
        if stage == 4:
            zoom.save(out, "PNG", optimize=True)
        else:
            jpg = out.with_suffix(".jpg")
            zoom.convert("RGB").save(jpg, "JPEG", quality=88, optimize=True)
            out = jpg
        print(f"  stage {stage}: {src.name} -> {out.name}  "
              f"({out.stat().st_size//1024} KB)")

    # Stage 0: composite (full leaf at top + zoomed crop at bottom +
    # red box on the full leaf showing where the zoom came from).
    stage0_src = SOURCES[0][1]
    full = Image.open(stage0_src).convert("RGB")
    fw, fh = full.size                      # 3952 x 234 in 1x coords
    box1x = ZOOM_1X                         # bbox in 1x coords

    # Render the full leaf at the same horizontal width as the zoomed
    # crop (1500 px) so the two stack neatly above each other.
    target_w = ZOOM_W
    target_h = int(fh * target_w / fw)
    full_disp = full.resize((target_w, target_h), Image.LANCZOS)

    # Map the zoom bbox from 1x coordinates onto the display image.
    sx = target_w / fw
    sy = target_h / fh
    box_disp = (int(box1x[0] * sx), int(box1x[1] * sy),
                int(box1x[2] * sx), int(box1x[3] * sy))

    # Add the red rectangle on the full-leaf strip.
    full_marked = full_disp.copy()
    draw = ImageDraw.Draw(full_marked)
    draw.rectangle(box_disp, outline="red", width=3)

    # Build the zoomed crop the same way as the others.
    zoom = crop_zoom(full, scale=1)

    # Vertical composite: full leaf at top, gap, zoom below.
    GAP = 18
    composite_h = full_marked.height + GAP + zoom.height
    composite = Image.new("RGB", (ZOOM_W, composite_h), "white")
    composite.paste(full_marked, (0, 0))
    composite.paste(zoom, (0, full_marked.height + GAP))
    out = FIG / "preproc_zoom_stage0.jpg"
    composite.save(out, "JPEG", quality=88, optimize=True)
    print(f"  stage 0 composite: {out.name}  "
          f"({out.stat().st_size//1024} KB)")

    print("done")


if __name__ == "__main__":
    main()
