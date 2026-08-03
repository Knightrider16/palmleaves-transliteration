# Kerala Archives Research Portal

A login-protected Flask web app that wraps this repo's palm-leaf
transliteration pipeline. Implements the layout proposal: login →
dashboard, archive search, projects (with the Malayanma
transliteration tool), gallery, about, contact.

## Run

The repo's existing `realesgran_venv` already has torch / cv2 / scipy.
Flask was added on top:

```bash
realesgran_venv/Scripts/python.exe -m pip install flask
realesgran_venv/Scripts/python.exe -m webapp.app
```

Open <http://127.0.0.1:5000>.

Default seeded credentials:

| Username    | Password       | Role        |
|-------------|----------------|-------------|
| researcher  | archives2026   | researcher  |
| admin       | admin          | admin       |

## What's wired up

| Page         | Route                          | Source                          |
|--------------|--------------------------------|---------------------------------|
| Login        | `/`                            | `templates/login.html`          |
| Dashboard    | `/dashboard`                   | recent rows from `archives` table |
| Search       | `/search?q=...`                | LIKE search over title/desc/tags |
| Projects     | `/projects`                    | landing card → transliteration tool |
| Tool         | `/projects/transliteration`    | upload + model picker + output  |
| Gallery      | `/gallery`                     | files in `static/gallery/`      |
| About        | `/about`                       | static                          |
| Contact      | `/contact`                     | form posts → `contacts` table   |

## How transliteration works

- `webapp/transliterate.py` discovers any `models/<arch>/best.pth`
  checkpoint and exposes its name in the dropdown.
- On submit, the uploaded image is read with OpenCV and preprocessed
  using the new configurable preprocessing pipeline:
  - **Preprocessing Levels** (selectable in UI):
    - `none`: No preprocessing (raw image)
    - `light`: CLAHE contrast enhancement only
    - `standard`: CLAHE + adaptive threshold + light noise removal ⭐ (default)
    - `heavy`: CLAHE + adaptive threshold + aggressive noise removal
  - **Real-ESRGAN Upscaling**: 2x super-resolution using Real-ESRGAN (matches offline pipeline)
  - Implementation in `webapp/preprocess.py`
- Lines are split using the project's
  `crnn.extract_lines.split_lines_by_peaks`.
- Each line is transliterated by the chosen architecture
  (re-using `crnn.infer._load_model` and `_line_to_tensor`).
- Greedy CTC decoding is used (matches the `cnn_ctc` benchmark; beam
  search regresses without an LM — see `benchmark/REPORT.md`).

The first run of a model takes a few seconds (checkpoint load).
Subsequent calls reuse the cached `nn.Module`.

### Preprocessing Module

The `webapp/preprocess.py` module provides a **full automatic preprocessing pipeline** that runs on every transliteration:

**Pipeline Stages:**
1. **Original** - Input grayscale image
2. **CLAHE Enhanced** - Contrast Limited Adaptive Histogram Equalization
3. **Upscaled** - 2x Real-ESRGAN super-resolution (same as offline batch_upscale.py)
4. **Sharpened** - CLAHE + sharpening kernel for edge enhancement
5. **Binary Mask** - Adaptive threshold to create white-on-black text
6. **Cleaned** - Noise removal + morphological operations (final result)

The pipeline automatically runs when you click "Run Model" and displays all intermediate stages in the UI.

- **Memory-efficient:** Works on 512MB free hosting tiers
- **Processing time:** ~5-15 seconds total (preprocessing + transliteration)
- **Automatic:** No user configuration needed - full pipeline always applied

### API Usage

The `/api/transliterate` endpoint automatically runs the full preprocessing pipeline:

```javascript
POST /api/transliterate
FormData:
  - model: "cnn_ctc"    // required
  - image: <file>       // required (or sample)
  - sample: "name.jpg"  // optional: sample filename

Response:
{
  "image_url": "/static/uploads/...",
  "model": "cnn_ctc",
  "lines": [...],
  "text": "transliterated text",
  "stages": {
    "original": "/static/uploads/...original.png",
    "clahe": "/static/uploads/...clahe.png",
    "upscaled": "/static/uploads/...upscaled.png",
    "sharpened": "/static/uploads/...sharpened.png",
    "binary": "/static/uploads/...binary.png",
    "cleaned": "/static/uploads/...cleaned.png"
  }
}
```

## Adding samples / gallery items

Drop image files into:

```
webapp/static/samples/   # appears under the "Samples" tab in the tool
webapp/static/gallery/   # appears in the Gallery page grid
```

The first 6 samples are pre-seeded from `data/original/`.

## Storage

A small SQLite DB is created on first launch at
`webapp/data/archives.db` with three tables: `users`, `archives`,
`contacts`. Default users + 6 archive records are seeded automatically
when the tables are empty. Delete the file to reseed.

## Notes / known limits

- This is a demo/research portal, not production-hardened. The
  password hash is a salted SHA-256, not a slow KDF — fine for a
  research lab, not for the open internet.
- `cnn_ctc` is the benchmark winner on ICFHR-D Balinese (~24% CER) and
  is the recommended choice in the dropdown. The Malayanma project
  itself has very few labeled lines, so accuracy on real palm-leaf
  pages will be poor; see `benchmark/REPORT.md` §"What this means for
  your Malayanma project".
- The Gallery page currently just lists all files in
  `webapp/static/gallery/`. Adding pagination / metadata is a
  straightforward extension of `app.gallery()`.
