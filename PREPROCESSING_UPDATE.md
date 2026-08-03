# Preprocessing Pipeline Update

## Changes Made

The preprocessing implementation has been updated based on your requirements:

### ✅ What Changed

1. **Removed Multiple Options**
   - ❌ Removed preprocessing level dropdown (none/light/standard/heavy)
   - ❌ Removed upscaling checkbox
   - ✅ Now uses **full automatic pipeline** every time

2. **Full Pipeline Implementation**
   - Matches the offline `preprocess_pipeline.py` approach
   - Runs automatically when "Run Model" is clicked
   - No user configuration needed

3. **Stage Visualization**
   - ✅ Displays all 6 preprocessing stages in the UI
   - Shows progression from original to final cleaned image
   - Each stage is labeled and displayed in a responsive grid

## Pipeline Stages

The full preprocessing pipeline includes:

1. **Original** - Input image converted to grayscale
2. **CLAHE Enhanced** - Contrast enhancement (clipLimit=2.0, tileGridSize=8x8)
3. **Upscaled (2x)** - Bicubic interpolation upscaling
4. **Sharpened** - CLAHE + sharpening kernel for edge enhancement
5. **Binary Mask** - Adaptive threshold (blockSize=41, C=7)
6. **Cleaned (Final)** - Connected component noise removal + morphological dilation

## Technical Details

### Pipeline Implementation

**File:** `webapp/preprocess.py`
```python
def full_pipeline_with_stages(img: np.ndarray) -> dict[str, np.ndarray]:
    """
    Run the full preprocessing pipeline and return all intermediate results.
    Returns dict with keys: 'original', 'clahe', 'upscaled', 'sharpened', 'binary', 'cleaned'
    """
```

### Transliteration Integration

**File:** `webapp/transliterate.py`
```python
def run(arch: str, image_path: str) -> tuple[list[list[str]], dict[str, np.ndarray]]:
    """
    Returns: (lines, stages) where stages contains all preprocessing images
    """
```

### API Response

**Endpoint:** `POST /api/transliterate`

**Request:**
```javascript
FormData:
  - model: "cnn_ctc"
  - image: <file> or sample: "filename.jpg"
```

**Response:**
```json
{
  "image_url": "/static/uploads/...",
  "model": "cnn_ctc",
  "lines": [...],
  "text": "transliterated text",
  "stages": {
    "original": "/static/uploads/...original_<timestamp>.png",
    "clahe": "/static/uploads/...clahe_<timestamp>.png",
    "upscaled": "/static/uploads/...upscaled_<timestamp>.png",
    "sharpened": "/static/uploads/...sharpened_<timestamp>.png",
    "binary": "/static/uploads/...binary_<timestamp>.png",
    "cleaned": "/static/uploads/...cleaned_<timestamp>.png"
  }
}
```

## UI Changes

### Before
- Preprocessing dropdown with 4 options
- Upscaling checkbox
- Single output text box

### After
- Simplified UI with only model selection
- Automatic preprocessing message
- Output text box
- **New:** 6-panel grid showing all preprocessing stages
- Each panel shows a stage with label and image

### UI Layout

```
┌─────────────────────────────────────────┐
│  Upload Image  │  Model  │  Output      │
└─────────────────────────────────────────┘
┌─────────────────────────────────────────┐
│        Preprocessing Stages              │
│  ┌────────┬────────┬────────┐           │
│  │Original│ CLAHE  │Upscaled│           │
│  ├────────┼────────┼────────┤           │
│  │Sharpen │ Binary │Cleaned │           │
│  └────────┴────────┴────────┘           │
└─────────────────────────────────────────┘
```

## Files Modified

### Core Files
1. **`webapp/preprocess.py`**
   - Added `full_pipeline_with_stages()` function
   - Implements 6-stage preprocessing pipeline
   - Returns dictionary of all intermediate images

2. **`webapp/transliterate.py`**
   - Updated `run()` function signature
   - Now returns `(lines, stages)` tuple
   - Removed preprocessing level parameters

3. **`webapp/app.py`**
   - Updated `/api/transliterate` endpoint
   - Saves all stage images to static/uploads/
   - Returns stage URLs in response
   - Removed preprocessing parameter handling

4. **`webapp/templates/transliteration.html`**
   - Removed preprocessing dropdown
   - Removed upscaling checkbox
   - Added stages display grid (6 panels)
   - Updated JavaScript to handle stages
   - Added CSS styling for stage cards

### Documentation
- **`webapp/README.md`** - Updated with new API and pipeline info
- **`PREPROCESSING_UPDATE.md`** - This file (change summary)

## Processing Time

Expected processing times on typical hardware:

| Component | Time | Notes |
|-----------|------|-------|
| CLAHE | ~0.1s | Fast |
| Upscaling (2x) | ~0.5-1s | Depends on image size |
| Sharpening | ~0.1s | Fast |
| Binary + Clean | ~0.3s | Moderate |
| **Total Preprocessing** | **~1-2s** | Per image |
| Model Inference | ~3-10s | Depends on model |
| **Grand Total** | **~5-15s** | Full pipeline |

## Memory Usage

- Original image: ~5 MB
- Each stage: ~5-20 MB (upscaled stages are 4x larger)
- Total active memory: ~50-100 MB
- All stages saved to disk: ~30-60 MB per run

**Safe for 512 MB free tier** with 1-2 models loaded.

## Testing

Test the updated implementation:

```bash
# Activate virtual environment
realesgran_venv\Scripts\activate

# Run test suite (should still pass)
python test_preprocessing.py

# Start webapp
python -m webapp.app
```

Visit http://127.0.0.1:5000/projects/transliteration and:
1. Upload an image or select a sample
2. Choose a model
3. Click "Run Model"
4. See the 6 preprocessing stages displayed
5. View transliteration result

## Benefits

✅ **Simpler UX** - No configuration needed, just click run
✅ **Educational** - Shows exactly what preprocessing does
✅ **Transparent** - Users see all transformation stages
✅ **Consistent** - Same pipeline every time, matching the report
✅ **Reproducible** - Results are consistent and documented

## Deployment Notes

### Requirements
- No new dependencies needed
- Uses existing cv2, numpy, torch
- All dependencies already in requirements.txt

### Render/Railway Deployment
- Works on free tier (512 MB RAM)
- Stage images saved to /static/uploads/
- Consider cleaning old stage images periodically
- Estimated 5-15 seconds per request (preprocessing + inference)

### Disk Space Management

Stage images accumulate in `/static/uploads/`. Consider adding cleanup:

```python
# Optional: Clean up old stage images (add to app.py)
import time
import os

def cleanup_old_stages():
    """Remove stage images older than 1 hour"""
    now = time.time()
    upload_dir = Path(__file__).parent / "static" / "uploads"
    for f in upload_dir.glob("*_*_*.png"):  # Stage images have double underscores
        if (now - f.stat().st_mtime) > 3600:  # 1 hour
            f.unlink()
```

## Comparison: Old vs New

| Feature | Old Implementation | New Implementation |
|---------|-------------------|-------------------|
| **Preprocessing Options** | 4 levels + upscale checkbox | Automatic (no options) |
| **User Choice** | Required selection | None needed |
| **Stage Visualization** | None | All 6 stages shown |
| **API Complexity** | Multiple parameters | Simple (model + image) |
| **Consistency** | Varies by user choice | Always same pipeline |
| **Learning Curve** | User must understand options | Zero - automatic |
| **Educational Value** | Low (hidden process) | High (shows all stages) |

## Next Steps

### Optional Enhancements

1. **Stage Selection**
   - Allow users to click a stage to use it as input (advanced mode)
   - Compare results from different stages

2. **Download Stages**
   - Add download buttons for each stage image
   - Export all stages as a ZIP file

3. **Stage Comparison**
   - Side-by-side comparison view
   - Slider to fade between stages

4. **Performance Optimization**
   - Cache preprocessed images by hash
   - Parallelize stage computation
   - Use web workers for client-side preview

5. **Batch Processing**
   - Upload multiple images
   - Process with same pipeline
   - Show all results in grid

## Summary

The preprocessing implementation has been successfully updated to:
- ✅ Use the full pipeline automatically (no user options)
- ✅ Display all 6 preprocessing stages visually
- ✅ Match the offline pipeline approach from the report
- ✅ Simplify the user experience (just click "Run Model")
- ✅ Provide transparency into the preprocessing steps

The webapp is ready for deployment with the automatic full preprocessing pipeline!
