# Render Deployment - Step by Step

## ✅ Prerequisites
- GitHub account
- Git installed on your computer

---

## 📦 Step 1: Reduce Models (CRITICAL for Free Tier)

Keep only 2 smallest models to fit in 512MB RAM:

**Option A: Keep vit_ctc + cnn_ctc (Recommended - 118MB total)**
```powershell
# Delete other model folders (keep vit_ctc and cnn_ctc)
Remove-Item "models\amadi_balinese" -Recurse -Force
Remove-Item "models\conformer" -Recurse -Force
Remove-Item "models\crnn_attn" -Recurse -Force
Remove-Item "models\crnn_ctc" -Recurse -Force
Remove-Item "models\trocr" -Recurse -Force
```

OR

**Option B: Keep all models in Git LFS (Advanced)**
- Requires Git LFS setup
- Models download on-demand
- More complex

---

## 🔧 Step 2: Create GitHub Repository

1. **Go to GitHub:** https://github.com/new

2. **Create new repository:**
   - Repository name: `palmleaves-transliteration` (or your choice)
   - Description: "Palm Leaf Manuscript Transliteration Web App"
   - Visibility: Public or Private (your choice)
   - ❌ **DO NOT** initialize with README (we already have files)
   - Click "Create repository"

3. **Copy the repository URL** (shown on next page)
   - Format: `https://github.com/yourusername/palmleaves-transliteration.git`

---

## 📤 Step 3: Push Code to GitHub

Open PowerShell in your project folder and run:

```powershell
# Navigate to project
cd E:\S4\Palmleaves-Transliteration

# Initialize Git (if not already done)
git init

# Add all files
git add .

# Commit
git commit -m "Initial commit for Render deployment"

# Add GitHub remote (replace with YOUR repo URL)
git remote add origin https://github.com/YOUR-USERNAME/palmleaves-transliteration.git

# Push to GitHub
git push -u origin main
```

**If you get "main" branch error, try:**
```powershell
git branch -M main
git push -u origin main
```

---

## 🌐 Step 4: Deploy on Render

### 4.1 Sign Up

1. Go to: https://render.com
2. Click "Get Started for Free"
3. Sign up with GitHub (recommended) or email

### 4.2 Create Web Service

1. Click "New +" button (top right)
2. Select "Web Service"
3. Click "Connect account" if prompted (authorize GitHub access)

### 4.3 Select Repository

1. Find and select `palmleaves-transliteration` (or your repo name)
2. Click "Connect"

### 4.4 Configure Service

Render should auto-detect settings from `render.yaml`, but verify:

- **Name:** `palmleaves-transliteration` (or your choice)
- **Environment:** `Python 3`
- **Region:** Choose closest to you
- **Branch:** `main`
- **Build Command:** `pip install -r requirements.txt`
- **Start Command:** `gunicorn webapp.app:app --bind 0.0.0.0:$PORT --timeout 120 --workers 1`

### 4.5 Set Environment Variables

Add these in "Environment" section:

| Key | Value |
|-----|-------|
| `PYTHON_VERSION` | `3.11.0` |
| `ARCHIVES_SECRET` | Click "Generate" or use any random string |

### 4.6 Select Plan

- Choose "Free" plan
- Accept the sleep behavior

### 4.7 Deploy!

1. Click "Create Web Service"
2. Wait 5-10 minutes for initial build
3. Watch the logs for any errors

---

## 🎉 Step 5: Access Your App

Once deployed (logs show "Starting Gunicorn"):

1. Your URL will be: `https://palmleaves-transliteration.onrender.com` (or your service name)
2. First visit may take 30-60 seconds (cold start)
3. Click the URL in Render dashboard

---

## 📁 Step 6: Add Your Media Files

After deployment, you need to upload your images:

### Option A: Add to Git and redeploy
```powershell
# Add your video and images
git add webapp/static/welcome-video.mp4
git add webapp/static/tiles/*.jpg
git add webapp/static/samples/*.png
git commit -m "Add media files"
git push
```
Render will auto-redeploy.

### Option B: Use external storage (Recommended for large files)
- Host video on YouTube/Vimeo
- Host images on Cloudinary/ImgBB (free CDN)
- Update URLs in templates

---

## 🐛 Troubleshooting

### Build Fails
- Check logs in Render dashboard
- Common issue: Missing dependencies in requirements.txt

### App Crashes
- Check "Logs" tab in Render
- Usually: Out of memory → reduce to 1 model only

### Slow Loading
- Expected on free tier after sleep
- First visit: 30-60s
- Subsequent visits: Fast

### Models Not Found
- Ensure models/ folder is in Git
- Check `.gitignore` doesn't exclude *.pth files
- Verify models pushed to GitHub

---

## 🔄 Updating Your App

After initial deployment, updates are easy:

```powershell
# Make your changes
# Then:
git add .
git commit -m "Description of changes"
git push
```

Render automatically redeploys on every push! 🎉

---

## ⚡ Performance Tips

1. **Use only 1 model** if you hit memory limits
2. **Add loading screen** for cold starts
3. **Enable caching** in Flask
4. **Compress images** before uploading
5. **Use CDN** for static files

---

## 💰 Upgrade Options

If free tier isn't enough:

- **Starter ($7/month):** No sleep, 512MB RAM
- **Standard ($25/month):** 2GB RAM, better for ML models

---

## 📞 Need Help?

- Render Docs: https://render.com/docs
- Community: https://community.render.com
- Check deployment logs for specific errors
