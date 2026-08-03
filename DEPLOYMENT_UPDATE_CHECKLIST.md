# Deployment Repository Update Checklist

## Files to Copy/Update from Main Repo to Palmleaves-Deployment

### ✅ NEW FILES TO ADD

1. **`webapp/preprocess.py`** (NEW - Core preprocessing module)
   - Source: `E:\S4\Palmleaves-Transliteration\webapp\preprocess.py`
   - This is the complete new preprocessing module with `full_pipeline_with_stages()`

### 🔄 FILES TO UPDATE (Replace existing)

2. **`webapp/transliterate.py`** (MODIFIED)
   - Current: Old version without preprocessing stages
   - Update with: `E:\S4\Palmleaves-Transliteration\webapp\transliterate.py`
   - Changes: 
     - Imports new `preprocess` module
     - `run()` function now returns `(lines, stages)` tuple
     - Calls `preprocess.full_pipeline_with_stages()`

3. **`webapp/app.py`** (MODIFIED)
   - Current: Old `/api/transliterate` endpoint
   - Update with: `E:\S4\Palmleaves-Transliteration\webapp\app.py`
   - Changes:
     - Removed preprocessing level/upscale parameters
     - Saves all 6 stage images to static/uploads
     - Returns `stages` URLs in API response

4. **`webapp/templates/transliteration.html`** (MAJOR UPDATE)
   - Current: Old UI with no stages display
   - Update with: `E:\S4\Palmleaves-Transliteration\webapp\templates\transliteration.html`
   - Changes:
     - Removed preprocessing dropdown and upscaling checkbox
     - Added 6-panel stages display grid
     - Added CSS for stage cards
     - Updated JavaScript to handle stages
     - Displays all preprocessing stages after running model

5. **`webapp/README.md`** (MODIFIED)
   - Update with: `E:\S4\Palmleaves-Transliteration\webapp\README.md`
   - Changes: Updated preprocessing documentation

### 📄 OPTIONAL DOCUMENTATION (Nice to have)

6. **`PREPROCESSING.md`** (NEW - Optional)
   - Detailed preprocessing documentation
   - Not required for deployment but good for reference

7. **`PREPROCESSING_UPDATE.md`** (NEW - Optional)
   - Change summary document
   - Not required for deployment

8. **`IMPLEMENTATION_COMPLETE.md`** (NEW - Optional)
   - Implementation summary
   - Not required for deployment

### ⚠️ FILES TO CHECK (May need adjustment)

9. **`requirements.txt`** (VERIFY)
   - Current deployment has all needed dependencies:
     - ✅ opencv-python-headless
     - ✅ numpy
     - ✅ torch
   - No new dependencies needed!

10. **`Dockerfile`** (KEEP AS IS)
    - Current Dockerfile should work fine
    - No changes needed

11. **`Procfile`** (KEEP AS IS)
    - Current: `gunicorn webapp.app:app --bind 0.0.0.0:$PORT --timeout 120 --workers 1`
    - Should work fine, but consider increasing timeout to 180 for preprocessing:
    - Recommended: `gunicorn webapp.app:app --bind 0.0.0.0:$PORT --timeout 180 --workers 1`

12. **`render.yaml`** (CHECK IF EXISTS)
    - If it exists in deployment repo, keep as is
    - Preprocessing works within current resource limits

---

## Step-by-Step Update Process

### Option 1: Manual Copy (Recommended for clarity)

```bash
# From your main development folder (E:\S4\Palmleaves-Transliteration)

# 1. Clone or navigate to your deployment repo
cd /path/to/Palmleaves-Deployment

# 2. Copy the new preprocessing module
cp E:\S4\Palmleaves-Transliteration\webapp\preprocess.py webapp\preprocess.py

# 3. Update webapp files
cp E:\S4\Palmleaves-Transliteration\webapp\transliterate.py webapp\transliterate.py
cp E:\S4\Palmleaves-Transliteration\webapp\app.py webapp\app.py

# 4. Update template
cp E:\S4\Palmleaves-Transliteration\webapp\templates\transliteration.html webapp\templates\transliteration.html

# 5. Update README (optional but recommended)
cp E:\S4\Palmleaves-Transliteration\webapp\README.md webapp\README.md

# 6. Update Procfile timeout (optional but recommended)
# Edit Procfile and change --timeout 120 to --timeout 180
```

### Option 2: Git Approach

If both repos are git repositories:

```bash
# In deployment repo
git remote add main-repo /path/to/Palmleaves-Transliteration
git fetch main-repo
git checkout main-repo/main -- webapp/preprocess.py
git checkout main-repo/main -- webapp/transliterate.py
git checkout main-repo/main -- webapp/app.py
git checkout main-repo/main -- webapp/templates/transliteration.html
git checkout main-repo/main -- webapp/README.md
```

---

## After Copying Files

### 1. Test Locally

```bash
# In deployment repo
python -m webapp.app

# Visit http://127.0.0.1:5000/projects/transliteration
# Upload an image and verify:
# - Model runs successfully
# - 6 preprocessing stages are displayed
# - Text result appears
```

### 2. Verify Changes

Check that these features work:
- ✅ Image upload or sample selection
- ✅ Model selection (no preprocessing options - automatic now)
- ✅ Click "Run Model" - should take 5-15 seconds
- ✅ See 6 preprocessing stages displayed in grid:
  1. Original
  2. CLAHE Enhanced
  3. Upscaled (2x)
  4. Sharpened
  5. Binary Mask
  6. Cleaned (Final)
- ✅ See transliteration text result

### 3. Commit and Push

```bash
git add .
git commit -m "Add full preprocessing pipeline with stage visualization

- Added webapp/preprocess.py with full_pipeline_with_stages()
- Updated transliterate.py to return preprocessing stages
- Updated app.py to save and serve stage images
- Updated transliteration.html with 6-stage display grid
- Removed preprocessing options (now automatic)
- Updated documentation"

git push origin main
```

### 4. Deploy

If using Render.com:
- Push will trigger automatic redeployment
- Wait 3-5 minutes for build
- Test on live URL

---

## Detailed File Changes Summary

### 1. `webapp/preprocess.py` (NEW - 267 lines)
**Purpose:** Core preprocessing module

**Key functions:**
- `enhance_contrast()` - CLAHE enhancement
- `create_binary_mask()` - Adaptive thresholding
- `remove_noise()` - Connected component filtering
- `apply_morphology()` - Dilation/erosion
- `lightweight_upscale()` - 2x bicubic upscaling
- `full_pipeline_with_stages()` - **Main function** that returns all 6 stages
- `preprocess_image()` - Configurable preprocessing (still used for testing)

### 2. `webapp/transliterate.py` (MODIFIED)
**Changes:**
```python
# OLD:
def run(arch: str, image_path: str) -> list[list[str]]:
    # Returns only lines
    
# NEW:
def run(arch: str, image_path: str) -> tuple[list[list[str]], dict[str, np.ndarray]]:
    # Returns (lines, stages)
    # stages = {'original', 'clahe', 'upscaled', 'sharpened', 'binary', 'cleaned'}
```

### 3. `webapp/app.py` (MODIFIED - Line ~195-240)
**Changes in `/api/transliterate` endpoint:**

**OLD:**
```python
lines = transliterate.run(model_name, str(src_path))
return jsonify({
    "image_url": image_url,
    "model": model_name,
    "lines": lines,
    "text": text,
})
```

**NEW:**
```python
lines, stages = transliterate.run(model_name, str(src_path))

# Save all stage images
stage_urls = {}
for stage_name, stage_img in stages.items():
    stage_filename = f"{base_name}_{stage_name}_{timestamp}.png"
    cv2.imwrite(str(stage_path), stage_img)
    stage_urls[stage_name] = url_for("static", filename=f"uploads/{stage_filename}")

return jsonify({
    "image_url": image_url,
    "model": model_name,
    "lines": lines,
    "text": text,
    "stages": stage_urls,  # NEW
})
```

### 4. `webapp/templates/transliteration.html` (MAJOR UPDATE)

**Removed:**
- Preprocessing level dropdown (4 options)
- Upscaling checkbox

**Added:**
- CSS for stage cards (in `{% block head %}`)
- 6-panel stages display grid (after output panel)
- Updated JavaScript to display stages
- Stage images: `stage-original`, `stage-clahe`, `stage-upscaled`, `stage-sharpened`, `stage-binary`, `stage-cleaned`

**JavaScript changes:**
```javascript
// OLD:
const res = await fetch(...);
const data = await res.json();
output.textContent = data.text;

// NEW:
const res = await fetch(...);
const data = await res.json();
output.textContent = data.text;

// Display preprocessing stages
if (data.stages) {
  $('stage-original').src = data.stages.original;
  $('stage-clahe').src = data.stages.clahe;
  $('stage-upscaled').src = data.stages.upscaled;
  $('stage-sharpened').src = data.stages.sharpened;
  $('stage-binary').src = data.stages.binary;
  $('stage-cleaned').src = data.stages.cleaned;
  stagesContainer.style.display = 'block';
}
```

---

## Testing Checklist

After updating, verify:

### Basic Functionality
- [ ] App starts without errors
- [ ] Dashboard loads
- [ ] Can navigate to transliteration tool
- [ ] Can upload image
- [ ] Can select sample
- [ ] Can select model

### Preprocessing & Display
- [ ] Click "Run Model" works
- [ ] Processing takes 5-15 seconds
- [ ] Text result appears
- [ ] 6 preprocessing stages appear below result
- [ ] Each stage image loads correctly
- [ ] Stage labels are correct

### Images Quality
- [ ] Original shows grayscale input
- [ ] CLAHE shows enhanced contrast
- [ ] Upscaled is 2x larger (visible detail increase)
- [ ] Sharpened shows edge enhancement
- [ ] Binary shows white text on black
- [ ] Cleaned shows final processed image

### API
- [ ] `/api/transliterate` returns `stages` object
- [ ] Stage URLs are valid
- [ ] Stage images can be accessed

---

## File Size Comparison

| File | Current (Deployment) | New (Main) | Change |
|------|---------------------|------------|--------|
| `webapp/preprocess.py` | ❌ Not exists | 267 lines | **NEW** |
| `webapp/transliterate.py` | ~120 lines | ~150 lines | +30 lines |
| `webapp/app.py` | ~240 lines | ~260 lines | +20 lines |
| `webapp/templates/transliteration.html` | ~210 lines | ~270 lines | +60 lines |

**Total changes:** ~400 new lines of code

---

## Rollback Plan

If issues occur after deployment:

```bash
# Revert to previous version
git log  # Find previous commit hash
git revert <commit-hash>
git push origin main

# OR reset to previous state
git reset --hard <previous-commit-hash>
git push --force origin main  # Use with caution
```

---

## Performance Notes

### Expected Processing Times
- **Free tier (512 MB):** 8-15 seconds
- **Paid tier (1+ GB):** 5-10 seconds

### Memory Usage
- Preprocessing stages: ~50-100 MB active
- Stage images on disk: ~30-60 MB per run
- Total memory with 2 models: ~400-500 MB

### Disk Space
Stage images accumulate in `webapp/static/uploads/`. Consider adding cleanup:
- Option 1: Delete stage images older than 1 hour
- Option 2: Limit to last 100 stage sets
- Option 3: Store only final cleaned image

---

## Summary

**Minimal files to update for full functionality:**
1. ✅ `webapp/preprocess.py` (add new)
2. ✅ `webapp/transliterate.py` (replace)
3. ✅ `webapp/app.py` (replace)
4. ✅ `webapp/templates/transliteration.html` (replace)

**Optional but recommended:**
5. ⭐ `webapp/README.md` (update docs)
6. ⭐ Update `Procfile` timeout to 180 seconds

**Total effort:** 10-15 minutes to copy and test

**Result:** Full preprocessing pipeline with visual stage display! 🎉
