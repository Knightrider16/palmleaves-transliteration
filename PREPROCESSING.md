# Preprocessing for Palm Leaves Transliteration

This document describes the preprocessing functionality available in the web application deployment.

## Overview

The webapp includes a configurable preprocessing pipeline that enhances palm leaf manuscript images before transliteration. The preprocessing is designed to work within the memory constraints of free hosting tiers while providing significant quality improvements.

## Preprocessing Levels

The application offers four preprocessing levels that can be selected in the UI:

### None (Raw Image)
- No preprocessing applied
- Uses the original uploaded image directly
- **Use when:** Image is already clean or preprocessed externally

### Light (CLAHE Only)
- Applies Contrast Limited Adaptive Histogram Equalization (CLAHE)
- Enhances contrast without creating binary mask
- **Use when:** Image has good lighting but needs contrast boost
- **Processing time:** Fast (~0.1-0.3s)

### Standard (Recommended) ⭐
- CLAHE contrast enhancement
- Adaptive thresholding to create binary mask
- Light noise removal (removes specks < 40 pixels)
- Morphological dilation to connect characters
- **Use when:** Most palm leaf manuscripts (default choice)
- **Processing time:** Medium (~0.3-0.8s)

### Heavy (Aggressive)
- CLAHE contrast enhancement
- Adaptive thresholding to create binary mask
- Aggressive noise removal (removes specks < 60 pixels)
- Morphological closing to fill gaps in characters
- Morphological dilation to strengthen strokes
- **Use when:** Very degraded or noisy manuscripts
- **Processing time:** Slower (~0.5-1.2s)

## Upscaling Option

An optional lightweight upscaling feature is available:

- **Method:** Bicubic interpolation (2x)
- **Effect:** Doubles image resolution before preprocessing
- **Use when:** Image resolution is very low (< 1000px width)
- **Trade-off:** Improves quality but increases processing time by 2-3x
- **Note:** This is NOT Real-ESRGAN (which is too resource-intensive for web deployment)

## Technical Details

### Preprocessing Pipeline

The preprocessing module (`webapp/preprocess.py`) implements the following operations:

1. **Contrast Enhancement (CLAHE)**
   - Clip limit: 2.0
   - Tile grid size: 8x8
   - Handles uneven lighting conditions

2. **Binary Mask Creation**
   - Adaptive thresholding with Gaussian weighting
   - Block size: 41 pixels
   - Constant offset: 7
   - Optional Gaussian blur (3x3 kernel) for noise reduction

3. **Noise Removal**
   - Connected component analysis
   - Filters by area, height, width
   - Removes both tiny specks and large blobs

4. **Morphological Operations**
   - Dilation: Strengthens character strokes (2x1 kernel)
   - Closing: Fills small gaps in characters (3x2 kernel, heavy mode only)
   - Opening: Removes small bright spots (not used by default)

### Memory Footprint

Preprocessing is designed for low-memory environments:

- **Light mode:** < 5 MB additional RAM per image
- **Standard mode:** < 10 MB additional RAM per image
- **Heavy mode:** < 15 MB additional RAM per image
- **With upscaling:** ~2-4x memory usage

### Performance

Typical processing times on free tier hosting (512 MB RAM):

| Level | No Upscale | With 2x Upscale |
|-------|-----------|-----------------|
| None | < 0.1s | 0.1-0.2s |
| Light | 0.1-0.3s | 0.3-0.6s |
| Standard | 0.3-0.8s | 0.8-1.5s |
| Heavy | 0.5-1.2s | 1.2-2.5s |

*Note: Times vary based on image size and server load*

## API Integration

The preprocessing options are exposed through the `/api/transliterate` endpoint:

### Request Parameters

```javascript
FormData:
  - model: string (required) - Model architecture name
  - image: file (optional) - Uploaded image file
  - sample: string (optional) - Sample image filename
  - preprocess: string (optional) - Preprocessing level
    Values: "none", "light", "standard", "heavy"
    Default: "standard"
  - upscale: string (optional) - Enable upscaling
    Values: "true", "false"
    Default: "false"
```

### Response

```json
{
  "image_url": "/static/uploads/manuscript.jpg",
  "model": "cnn_ctc",
  "lines": [["character", "tokens"], ...],
  "text": "transcribed text",
  "preprocess": "standard",
  "upscale": false
}
```

## Usage Examples

### JavaScript (Frontend)

```javascript
const fd = new FormData();
fd.append('model', 'cnn_ctc');
fd.append('image', imageFile);
fd.append('preprocess', 'standard');  // or "none", "light", "heavy"
fd.append('upscale', 'false');        // or "true"

const response = await fetch('/api/transliterate', {
  method: 'POST',
  body: fd
});

const result = await response.json();
console.log(result.text);
```

### Python (Backend Testing)

```python
from webapp.preprocess import preprocess_from_path

# Preprocess an image
processed = preprocess_from_path(
    "path/to/manuscript.jpg",
    level="standard",
    upscale=False,
    save_path="path/to/output.png"
)

# Use in transliteration
from webapp.transliterate import run

lines = run(
    arch="cnn_ctc",
    image_path="path/to/manuscript.jpg",
    preprocess_level="standard",
    use_upscale=False
)
```

## Deployment Considerations

### Free Tier (Render, Railway, etc.)

- **Recommended settings:**
  - Default to "standard" preprocessing
  - Disable upscaling by default (let users opt-in)
  - Consider adding timeout warnings for heavy preprocessing

- **Memory management:**
  - Preprocessing adds minimal memory overhead
  - Safe for 512 MB free tier with 1-2 models

- **Performance:**
  - Standard preprocessing adds ~0.5s to total request time
  - Acceptable for web deployment (< 3s total with small models)

### Paid Tier (1+ GB RAM)

- All preprocessing levels work smoothly
- Can enable upscaling by default for better quality
- Support larger images (4000+ px width)

## Comparison with Offline Pipeline

The web app preprocessing differs from the offline `preprocess_pipeline.py`:

| Feature | Offline Pipeline | Web App |
|---------|-----------------|---------|
| CLAHE | ✅ Yes | ✅ Yes |
| Adaptive Threshold | ✅ Yes | ✅ Yes |
| Noise Removal | ✅ Yes | ✅ Yes |
| Real-ESRGAN 4x | ✅ Yes (slow) | ❌ No (too heavy) |
| Lightweight Upscale | ❌ No | ✅ Yes (2x bicubic) |
| Batch Processing | ✅ Yes | ❌ No (single images) |
| Processing Time | ~30-60s per image | ~0.5-2s per image |
| Memory Usage | ~2-4 GB | ~50-100 MB |

The web app prioritizes speed and low resource usage over maximum quality.

## Troubleshooting

### Issue: Preprocessing too slow

**Solution:** 
- Use "light" preprocessing level
- Disable upscaling
- Ensure image is not excessively large (> 3000px)

### Issue: Characters look broken/disconnected

**Solution:**
- Switch to "heavy" preprocessing level
- Enable upscaling to increase resolution
- Check if original image quality is very poor

### Issue: Too much noise in output

**Solution:**
- Use "heavy" preprocessing level (more aggressive filtering)
- Try pre-cleaning the image externally
- Ensure image is well-lit and in focus

### Issue: Text appears too thin/faded

**Solution:**
- Use "light" preprocessing (avoid aggressive binarization)
- Adjust original image brightness before uploading
- Try "standard" with upscaling enabled

## Future Enhancements

Potential improvements for future versions:

1. **Custom parameter tuning**: Allow users to adjust CLAHE clip limit, block size, etc.
2. **Region-based preprocessing**: Apply different settings to different image regions
3. **Learning-based enhancement**: Use lightweight neural networks for preprocessing
4. **Batch upload**: Process multiple images with same settings
5. **Preview mode**: Show preprocessed image before running transliteration

## References

- OpenCV Adaptive Thresholding: https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html
- CLAHE: https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html
- Connected Components: https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html
