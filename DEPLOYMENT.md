# Deployment Guide - Palm Leaves Transliteration Web App

## ⚠️ Important Considerations

**Model Size Issue**: Your project has 6 ML models totaling ~500MB. This is challenging for free hosting:
- Most free tiers have storage/memory limits
- Model loading requires significant RAM (1-2GB recommended)

## ✨ New Features

### Preprocessing Pipeline
The webapp now includes configurable preprocessing for better image quality:
- **Preprocessing Levels**: None, Light, Standard (default), Heavy
- **Optional Upscaling**: 2x bicubic interpolation (lightweight alternative to Real-ESRGAN)
- **UI Controls**: Users can select preprocessing level and upscaling in the web interface
- **Memory Efficient**: Designed to work on free tier (512MB RAM) hosting

See [PREPROCESSING.md](PREPROCESSING.md) for detailed documentation.

## Recommended Free Hosting Options

### Option 1: Render (Best for this project)

**Pros:**
- Easy Git-based deployment
- 750 hours/month free
- Supports large applications

**Cons:**
- Free tier has 512MB RAM (might need to use only 1-2 models)
- Sleeps after 15 min inactivity (30-60s cold start)

**Steps:**

1. **Push to GitHub:**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin <your-github-repo-url>
   git push -u origin main
   ```

2. **Deploy on Render:**
   - Go to https://render.com
   - Sign up with GitHub
   - Click "New +" → "Web Service"
   - Connect your repository
   - Settings will auto-detect from `render.yaml`
   - Click "Create Web Service"

3. **Reduce Model Count (IMPORTANT for free tier):**
   - Keep only 1-2 models (recommend `cnn_ctc` and `vit_ctc` - smallest ones)
   - Delete others to save memory
   - Update `webapp/transliterate.py` PREFERRED_ORDER

### Option 2: Railway

**Pros:**
- $5 free credit/month
- Better specs than Render free tier
- No sleep on inactivity

**Cons:**
- Credit runs out (project stops until next month)

**Steps:**

1. Push to GitHub (same as above)

2. **Deploy on Railway:**
   - Go to https://railway.app
   - Sign up with GitHub
   - Click "New Project" → "Deploy from GitHub repo"
   - Select your repo
   - Add environment variables:
     - `PORT` = 5000
     - `ARCHIVES_SECRET` = (generate a random string)
   - Railway will auto-detect Python and install dependencies

### Option 3: PythonAnywhere (Most reliable for Python)

**Pros:**
- No sleep
- Python-specific hosting
- Good documentation

**Cons:**
- Manual file upload
- 512MB disk limit (too small for all models)
- Need to manually configure WSGI

**Steps:**

1. **Sign up:** https://www.pythonanywhere.com

2. **Upload files:**
   - Use "Files" tab to upload your project
   - Or clone from GitHub

3. **Install dependencies:**
   ```bash
   pip3.11 install --user -r requirements.txt
   ```

4. **Configure WSGI:**
   - Web tab → Add new web app → Flask → Python 3.11
   - Edit WSGI file:
   ```python
   import sys
   path = '/home/yourusername/Palmleaves-Transliteration'
   if path not in sys.path:
       sys.path.append(path)
   
   from webapp.app import app as application
   ```

## Essential Files Included

✅ `requirements.txt` - Python dependencies
✅ `runtime.txt` - Python version
✅ `Procfile` - Start command for Heroku/Render
✅ `render.yaml` - Render-specific configuration
✅ `.gitignore` - Files to exclude from Git

## Required Directory Structure for Deployment

```
Palmleaves-Transliteration/
├── webapp/
│   ├── __init__.py
│   ├── app.py
│   ├── store.py
│   ├── transliterate.py
│   ├── embeddings.py
│   ├── static/
│   │   ├── css/
│   │   ├── tiles/
│   │   └── samples/
│   └── templates/
├── crnn/
│   ├── __init__.py
│   ├── models/
│   └── ...
├── models/
│   ├── cnn_ctc/best.pth  (keep only 1-2 for free tier!)
│   └── vit_ctc/best.pth
├── requirements.txt
├── runtime.txt
├── Procfile
└── render.yaml
```

## Memory Optimization Tips

To fit in free tier limits:

1. **Use only 1-2 smallest models:**
   - Keep: `vit_ctc` (42MB) and `cnn_ctc` (76MB)
   - Delete others or store separately

2. **Lazy loading** (already implemented):
   - Models load on first use, not at startup

3. **Reduce workers:**
   - Procfile uses `--workers 1` to minimize memory

## Testing Locally Before Deploy

```bash
# Activate environment
.\realesgran_venv\Scripts\Activate.ps1

# Install gunicorn (production server)
pip install gunicorn

# Test the production command
gunicorn webapp.app:app --bind 0.0.0.0:5000 --timeout 120 --workers 1
```

## Environment Variables Needed

- `PORT` - Auto-set by hosting platform
- `ARCHIVES_SECRET` - Generate a random string for session security
- `HOST` - Usually 0.0.0.0 (set by platform)

## Post-Deployment Setup

1. Create directories (if needed):
   - `webapp/static/uploads/`
   - `webapp/static/tiles/`

2. Upload your images:
   - Background video: `webapp/static/welcome-video.mp4`
   - Tile images: `webapp/static/tiles/*.jpg`

3. Initialize database:
   - First visit will auto-create `webapp/data/archives.db`

## Cost-Free Recommendation

**Best Choice: Render + GitHub**
- Use only 2 models (vit_ctc + cnn_ctc)
- Total size: ~120MB (manageable)
- Accept 30-60s cold start after inactivity
- Free forever within usage limits

**Alternative: Railway** if you don't mind monthly credit limits.

Need help with specific deployment issues? Let me know!
