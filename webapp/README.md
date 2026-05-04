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
- On submit, the uploaded image is read with OpenCV. If it isn't
  already a binary mask we run a single-image preprocess (CLAHE +
  adaptive threshold + connected-components cleanup) — same as
  `preprocessing_scripts/batch_mask_clean.py` but per-image, without
  the slow Real-ESRGAN upscale step.
- Lines are split using the project's
  `crnn.extract_lines.split_lines_by_peaks`.
- Each line is transliterated by the chosen architecture
  (re-using `crnn.infer._load_model` and `_line_to_tensor`).
- Greedy CTC decoding is used (matches the `cnn_ctc` benchmark; beam
  search regresses without an LM — see `benchmark/REPORT.md`).

The first run of a model takes a few seconds (checkpoint load).
Subsequent calls reuse the cached `nn.Module`.

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
