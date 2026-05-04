"""
Render single Malayalam glyphs as binary character images.

Each glyph is drawn with PIL onto a square canvas, then converted to a
white-on-black binary image at the size used by the rest of the pipeline.
"""
from __future__ import annotations
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

DEFAULT_FONT = os.path.join(
    os.path.dirname(__file__), "fonts", "NotoSansMalayalam-Regular.ttf")


class GlyphRenderer:
    def __init__(self,
                 font_path: str = DEFAULT_FONT,
                 canvas: int = 96,
                 out_size: int = 64,
                 base_font_size: int = 64):
        if not os.path.isfile(font_path):
            raise FileNotFoundError(font_path)
        self.font_path     = font_path
        self.canvas        = canvas
        self.out_size      = out_size
        self.base_font_size = base_font_size

    def _font(self, size: int) -> ImageFont.FreeTypeFont:
        return ImageFont.truetype(self.font_path, size)

    def render(self, text: str,
               jitter: bool = True,
               rng: np.random.Generator | None = None) -> np.ndarray:
        """
        Render `text` to a binary uint8 image (white glyph on black bg).
        Output shape: (out_size, out_size).
        """
        rng = rng or np.random.default_rng()

        # Random font size jitter for variety
        if jitter:
            size = int(self.base_font_size * rng.uniform(0.85, 1.05))
        else:
            size = self.base_font_size
        font = self._font(size)

        # Render at high res, then downsample
        big = self.canvas * 2
        img = Image.new("L", (big, big), 0)
        draw = ImageDraw.Draw(img)

        # Center the text
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x = (big - tw) // 2 - bbox[0]
        y = (big - th) // 2 - bbox[1]

        if jitter:
            x += int(rng.normal(0, big * 0.01))
            y += int(rng.normal(0, big * 0.01))

        draw.text((x, y), text, fill=255, font=font)

        arr = np.array(img, dtype=np.uint8)

        # Threshold (font anti-aliasing → binary)
        arr = (arr > 64).astype(np.uint8) * 255

        # Tight crop to glyph bounding box, then square-pad
        ys, xs = np.where(arr > 0)
        if len(xs) == 0:
            return np.zeros((self.out_size, self.out_size), dtype=np.uint8)
        x0, x1 = xs.min(), xs.max() + 1
        y0, y1 = ys.min(), ys.max() + 1
        glyph = arr[y0:y1, x0:x1]

        # Pad to square
        h, w = glyph.shape
        side = max(h, w)
        pad = np.zeros((side, side), dtype=np.uint8)
        pad[(side - h) // 2:(side - h) // 2 + h,
            (side - w) // 2:(side - w) // 2 + w] = glyph

        # Resize to output size
        out = Image.fromarray(pad).resize(
            (self.out_size, self.out_size), Image.LANCZOS)
        out = np.array(out, dtype=np.uint8)
        out = (out > 64).astype(np.uint8) * 255
        return out


def render_line(text: str,
                font_path: str = DEFAULT_FONT,
                height: int = 64,
                font_size: int = 48,
                padding: int = 16,
                rng: np.random.Generator | None = None,
                jitter: bool = True) -> np.ndarray:
    """
    Render a multi-token line of text as a single binary strip.
    Output shape: (height, variable_width), uint8 in {0, 255}.
    """
    rng = rng or np.random.default_rng()
    if jitter:
        font_size = int(font_size * rng.uniform(0.9, 1.1))
    font = ImageFont.truetype(font_path, font_size)

    # Render at 2x then downscale
    H = height * 2
    big_size = font_size * 2
    big_font = ImageFont.truetype(font_path, big_size)

    # Measure
    img_tmp = Image.new("L", (10, 10), 0)
    draw_tmp = ImageDraw.Draw(img_tmp)
    bbox = draw_tmp.textbbox((0, 0), text, font=big_font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    W = tw + padding * 4

    img = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(img)
    x = (W - tw) // 2 - bbox[0]
    y = (H - th) // 2 - bbox[1]
    draw.text((x, y), text, fill=255, font=big_font)

    arr = np.array(img, dtype=np.uint8)
    arr = (arr > 64).astype(np.uint8) * 255

    # Downscale to target height
    pil = Image.fromarray(arr).resize(
        (max(1, W // 2), height), Image.LANCZOS)
    out = np.array(pil, dtype=np.uint8)
    out = (out > 64).astype(np.uint8) * 255
    return out


if __name__ == "__main__":
    import cv2
    r = GlyphRenderer()
    test_glyphs = ["ക", "കാ", "കി", "ശ്രീ", "ഴാ", "ന്ദ്ര"]
    sheet = []
    for g in test_glyphs:
        img = r.render(g, jitter=False)
        sheet.append(img)
    sheet = np.hstack(sheet)
    cv2.imwrite("synthetic/_test_glyphs.png", sheet)
    print(f"Wrote synthetic/_test_glyphs.png ({sheet.shape})")

    line = render_line("ശ്രീ ഭാ ര ത ം ", height=64, font_size=48)
    cv2.imwrite("synthetic/_test_line.png", line)
    print(f"Wrote synthetic/_test_line.png ({line.shape})")
