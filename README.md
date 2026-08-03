# Ancient Palm-Leaf Manuscripts Transliteration using AI

An end-to-end deep learning pipeline for transliterating Mathilakam palm-leaf
manuscripts written in **Malayanma** — an early Kerala script that fell out
of common use in the late nineteenth century and has no Unicode block. The
project pairs a hand-curated Mathilakam labels dataset with two public
palm-leaf benchmarks (ICFHR 2016 AMADI, ICFHR 2018 Challenge D Balinese) to
separate "does the framework work?" from "is there enough labeled Malayanma
data yet?", and ships the result through a Flask research portal.

Full write-up: `GAUTHAM_KRISHNA_J_L___PROJECT_DRAFT_REPORT.pdf` (M.Sc.
Computer Science with Specialization in Machine Intelligence, Digital
University Kerala, May 2026). This README is the practical "how do I
actually run this" companion to that report.

## What's in here

A raw colour scan of a palm leaf goes through eight stages to become a
line-by-line romanized transcript:

1. **Preprocessing** — CLAHE contrast enhancement → Real-ESRGAN 2×
   super-resolution → adaptive Gaussian thresholding → connected-component
   cleanup (strips punch holes and fibre noise).
2. **Segmentation** — multi-scale row/column projection splits the cleaned
   mask into line strips and character crops.
3. **Pseudo-labeling** — ~25k unlabeled character crops are embedded with a
   CNN, clustered with HDBSCAN, and labeled by CSV-confidence-weighted
   majority vote; a separate CNN classifier filters crops by softmax
   confidence for stitching into synthetic-but-real-glyph training lines.
4. **Synthetic data** — training lines rendered from a Malayalam font through
   a programmatic palm-leaf degradation pipeline (parchment cast, ink-bleed
   blur, fibre noise, warp).
5. **Recognition** — six interchangeable line-recognizer architectures
   (`cnn_ctc`, `crnn_ctc`, `vit_ctc`, `crnn_attn`, `trocr`, `conformer`)
   share one training harness, so the architecture is a config choice, not a
   code change.
6. **LM rescoring** — a 5-gram syllable language model re-scores CTC beam
   search output for the Mathilakam deployment path.
7. **Benchmarking** — the same code paths run on two public palm-leaf
   datasets as viability checks.
8. **Web portal** — a Flask app (`webapp/`) exposing the trained model for
   interactive use, plus a hybrid lexical + semantic search over an archive
   catalogue.

## Headline results

| Benchmark | Metric | Result |
|---|---|---|
| ICFHR 2016 AMADI (char classification, 133 classes) | Top-1 accuracy | **89.44%** |
| ICFHR 2018 Challenge D Balinese (word transliteration) | CER / word acc | **23.99% / 50.72%** (`cnn_ctc`) — between the top participant and median of the original leaderboard |
| Mathilakam (in-house, 15-line real holdout) | Token error rate | **82.47%** (`cnn_ctc`, real-dominated mix + 5-gram LM rescoring), down from a 91.5% real-only baseline |

Full tables, ablations, and per-architecture comparisons are in
[`benchmark/REPORT.md`](benchmark/REPORT.md) and Chapter 4 of the project
report.

Three findings carried across every experiment:

- **Aggressive augmentation is the single most decisive lever.** Under weak
  augmentation, nearly every architecture mode-collapses to predicting the
  most frequent class; the shared recipe (affine + elastic + stroke jitter +
  horizontal stretch + random erasing) broke that collapse everywhere it
  showed up.
- **Architecture rankings are task-dependent, not universal.** `cnn_ctc`
  wins real multi-character transliteration (ICFHR D, Mathilakam);
  `crnn_ctc` / `crnn_attn` win the degenerate length-1 AMADI line task.
  Rankings don't transfer between the two settings.
- **The recipe transfers cleanly** across Mathilakam and Balinese without
  per-task tuning — the same preprocessing, augmentation, optimizer, and
  schedule are used everywhere.

## Repository layout

```
preprocessing_scripts/   Stage 1: CLAHE + Real-ESRGAN + adaptive threshold + CC cleanup
crnn/                     Segmentation, dataset loading, the 6 recognizer models, training, LM beam rescoring
pseudolabel/              CNN embedding, HDBSCAN clustering, cluster-vote labeling, stitched-line generation
synthetic/                Font-rendered character/line generation + palm-leaf degradation pipeline
benchmark/                Training/eval harness for AMADI, ICFHR D, and the Mathilakam phase study; REPORT.md
diagrams/                 Scripts that generate the architecture/pipeline figures used in the report
webapp/                   Flask research portal (login, dashboard, archive search, transliteration tool)
models/                   Checkpoints and vocab for each architecture (checkpoints themselves are gitignored)
tests/                    Sanity tests for the webapp preprocessing module
archive/                  Superseded early-iteration scripts, kept for reference (see below)
run_all.py                Orchestrator: resumable end-to-end training run driver (writes RUN_STATE.json)
Procfile, render.yaml,
runtime.txt, requirements.txt   Render deployment config for webapp/
```

`archive/` holds the first-draft versions of the preprocessing/segmentation/
clustering scripts (the "three attempts" at segmentation and the two
clustering-feature iterations discussed in Chapter 4 of the report). They
are **not used by the production pipeline** — the modular versions in
`crnn/`, `pseudolabel/`, and `preprocessing_scripts/` are — but are kept
around since the report documents them as part of the experimental record.

## Setup

Core dependencies: PyTorch 2.0.1 + CUDA 11.8, OpenCV, Pillow, umap-learn,
hdbscan, editdistance, scipy, tqdm, Flask, sentence-transformers. Training
was run on a single 6 GB laptop GPU (RTX 4050); batch sizes in the training
scripts are tuned for that budget. `requirements.txt` covers what the
**webapp** needs for deployment; for the full offline pipeline (training,
benchmarking, synthetic data generation) also install `umap-learn`,
`hdbscan`, and `editdistance` on top of that.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   |   macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
pip install umap-learn hdbscan editdistance scipy tqdm
```

You'll also need the Noto Sans Malayalam font, already vendored at
`synthetic/fonts/NotoSansMalayalam-Regular.ttf`, and (for Real-ESRGAN
super-resolution) the `realesrgan_x4plus.pth` weights placed at
`weights/realesrgan_x4plus.pth`.

### Expected local data layout (gitignored, not in this repo)

The scripts assume the following directories exist locally. None of these
are tracked in git (see `.gitignore`) — they're either large, regenerable,
or contain material (the Kerala Archives scans, the thesis PDF/source) that
isn't meant for the code repo.

```
data/
├── original/                    raw palm-leaf scans
├── preprocessed/                after CLAHE            (preprocessing_scripts/preprocess.py)
├── upscaled/                    after Real-ESRGAN 2x    (preprocessing_scripts/batch_upscale.py)
├── final/, masks_clean_upscaled/  after threshold + CC cleanup (batch_postprocess.py, batch_mask_clean.py)
├── labels/
│   ├── labels.csv                the hand-transcribed Mathilakam labels dataset (image, line, confidence, transcript)
│   └── mathilakam_5gram.pkl       trained syllable LM (crnn/build_lm.py)
├── characters_named/              25,040 extracted+named character crops
├── embeddings/                    CNN embeddings of the above (pseudolabel/embed_chars.py)
├── clusters/, pseudo_labeled*/    HDBSCAN cluster + pseudo-label outputs
├── synthetic/
│   ├── chars/                     font-rendered character classification set
│   ├── lines/                     font-rendered line set
│   └── real_stitched_lines/       real-glyph stitched lines (pseudolabel/stitch_lines.py)
└── real_lines/                    extracted real Mathilakam line crops + index.csv/vocab.txt (crnn/extract_lines.py)

data2/            intermediate segmentation output used by archive/pipeline.py only
weights/          realesrgan_x4plus.pth
results/          benchmark/eval output (comparison.csv, comparison.md, transliterations.csv)
benchmark/amadi/       ICFHR 2016 AMADI LontarSet (obtain separately, see benchmark/REPORT.md)
benchmark/icfhr2018/   ICFHR 2018 Challenge D Balinese (obtain separately)
```

## Running the pipeline

All commands are run from the repo root. `$PY` is your Python interpreter
(the environment set up above).

### 1. Preprocess raw scans

```bash
$PY -m preprocessing_scripts.preprocess         # CLAHE            -> data/preprocessed/
$PY -m preprocessing_scripts.batch_upscale       # Real-ESRGAN 2x   -> data/upscaled/
$PY -m preprocessing_scripts.batch_postprocess    # threshold + sharpen -> data/final/
$PY -m preprocessing_scripts.batch_mask_clean     # CC cleanup       -> data/masks_clean_upscaled/
```

### 2. Generate synthetic data

```bash
$PY -m synthetic.gen_chars --per-token 30   # ~14k char crops -> data/synthetic/chars/
$PY -m synthetic.gen_lines --n-lines 3000   # 3k font-rendered lines -> data/synthetic/lines/
```

### 3. Pretrain the CNN char classifier

```bash
$PY -m crnn.pretrain_cnn --epochs 12 --batch 128 --lr 0.05
```
Saves `models/cnn_backbone.pth` (CNN only, used to init CTC line models) and
`models/glyph_classifier.pth` (full classifier).

### 4. Pseudo-label the 25k real character crops

```bash
$PY -m pseudolabel.embed_chars                              # CNN embeddings
$PY -m pseudolabel.recluster                                 # HDBSCAN over CNN features
$PY -m pseudolabel.cnn_predict --full-ckpt models/glyph_classifier.pth
$PY -m pseudolabel.build_labels                               # cluster-vote pseudo-labels from data/labels/labels.csv
```

### 5. Build real-glyph stitched lines

```bash
$PY -m pseudolabel.stitch_lines --n-lines 3000 --min-conf 0.8
```
3,000 lines assembled from real high-confidence character crops — bridges
the synthetic → real domain gap.

### 6. Extract real labeled Mathilakam lines

```bash
$PY -m crnn.extract_lines
```
Slices the cleaned masks into per-line crops aligned against
`data/labels/labels.csv`, writing `data/real_lines/index.csv` +
`vocab.txt`. Also build the training-mix assets for later phases:

```bash
$PY -m crnn.build_phase2_assets     # synth + real
$PY -m crnn.build_phase3_assets     # synth + stitched + 50x-upsampled real (final training mix)
$PY -m crnn.build_lm                # 5-gram syllable LM -> data/labels/mathilakam_5gram.pkl
```

### 7. Train all six architectures — resumable

```bash
$PY run_all.py                             # train all six
$PY run_all.py --models crnn_ctc,cnn_ctc   # subset
$PY run_all.py --epochs 8                  # override per-model epochs
$PY run_all.py --status                    # see what's done
$PY run_all.py --reset                     # forget state, restart
```

Each run writes `models/<arch>/last.pth` (resume point), `models/<arch>/best.pth`
(best val TER), and `models/<arch>/log.csv`. `RUN_STATE.json` tracks
progress — you can Ctrl+C at any point and re-run `python run_all.py` to
resume from the last completed epoch of each model.

### 8. Compare trained models

```bash
$PY -m crnn.compare
```
Writes `results/comparison.csv` / `results/comparison.md`.

### 9. Public benchmarks (AMADI, ICFHR D) and the Mathilakam phase study

```bash
$PY -m benchmark.amadi_charrec
$PY -m benchmark.amadi_lineocr
$PY -m benchmark.icfhr_d_balinese
$PY -m benchmark.icfhr_d_eval_beam        # the beam-search-without-LM negative result
$PY -m benchmark.run_mathilakam           # Phases 1-3 of the Mathilakam study
$PY -m benchmark.eval_mathilakam_lm       # Phase 4: LM-rescored beam search
```
See [`benchmark/REPORT.md`](benchmark/REPORT.md) for the full numbers this
produces, and how to obtain the AMADI / ICFHR D datasets.

### 10. Inference on a single strip

```bash
$PY -m crnn.infer \
    --model models/cnn_ctc/best.pth \
    --inputs "data/masks_clean_upscaled/*.png" \
    --labels data/labels/labels.csv \
    --out results/transliterations.csv
```

### 11. Run the web portal locally

```bash
pip install flask
$PY -m webapp.app
```
Open <http://127.0.0.1:5000>. See [`webapp/README.md`](webapp/README.md)
for routes, seeded credentials, and how the transliteration tool wires into
the trained models.

## The six architectures

| Name | Encoder | Decoder | Loss | Params | Notes |
|---|---|---|---|---|---|
| `cnn_ctc` | 8-block CNN | Linear | CTC | 9.3M | De-facto choice — most stable, wins ICFHR D and the Mathilakam study |
| `crnn_ctc` | 8-block CNN | BiLSTM ×2 | CTC | 12.4M | Classical CRNN baseline; regresses under strong augmentation on real multi-char data |
| `vit_ctc` | ViT-Tiny | Linear | CTC | 5.1–5.2M | Pure transformer encoder, trained from scratch |
| `crnn_attn` | 8-block CNN | LSTM + Bahdanau attention | CE | 13.7M | Auto-regressive, non-monotonic; mode-collapses on real multi-char sequences |
| `trocr` | ViT (6-layer) | Transformer decoder (4-layer) | CE | 9.4M | TrOCR-style end-to-end transformer |
| `conformer` | Conv + self-attention hybrid | Linear | CTC | 9.3M | Strong on synthetic-side metrics; plateaus on real Balinese/Mathilakam data |

CTC models load `models/cnn_backbone.pth` automatically via `--cnn-init`.
Pure-transformer models (`vit_ctc`, `trocr`) train from scratch — see
`benchmark/REPORT.md` §2.2.10 for why they underperform on this data scale.

## Deployment (Render)

The webapp is configured for [Render](https://render.com) via `render.yaml`,
`Procfile`, `runtime.txt`, and `requirements.txt` at the repo root.

1. Push the repo to GitHub.
2. On Render: **New → Web Service → connect this repo.** Render
   auto-detects the build/start commands from `render.yaml`.
3. Set the `ARCHIVES_SECRET` env var (Render can auto-generate it).
4. **Free tier is 512MB RAM**, which won't comfortably hold all six model
   checkpoints (~500MB total). Trim `models/` to 1–2 of the smallest
   checkpoints (`cnn_ctc` + `vit_ctc` is a reasonable pair, ~120MB) and
   update `PREFERRED_ORDER` in `webapp/transliterate.py` to match.
5. First request after idle will cold-start (30–60s on the free tier);
   subsequent requests are fast since models are cached after first load.

Static assets (sample images, gallery, any hero video) need to be committed
under `webapp/static/` (or hosted externally and linked) since they aren't
generated at build time.

## Data

- The in-house Mathilakam labels dataset (romanized akshara transcriptions
  of handwritten archive annotations) was built by the author from scans
  provided by the Kerala Archives Department and is not redistributed in
  this repo.
- Public benchmarks: [ICFHR 2016 AMADI LontarSet](https://ieeexplore.ieee.org/document/7814088)
  and [ICFHR 2018 Challenge D](https://ieeexplore.ieee.org/document/8563250)
  (Balinese palm-leaf manuscripts) — see the respective competition pages
  for access, and `benchmark/REPORT.md` for the exact directory layout each
  benchmark script expects.

## Status / limitations

The framework is validated on two public benchmarks with plentiful ground
truth; the in-house Malayanma corpus (112 transcribed lines, 83 usable after
alignment filtering, 68 train / 15 val) is still too small to close the gap
to those benchmark numbers on its own. The current bottleneck is transcribed
data, not the modeling approach. Line segmentation currently recovers 14 of
19 labeled lines on the sample leaf (tightly-stacked lines merge); the web
portal is a research demonstrator, not production-hardened (salted SHA-256
login, not a slow KDF). See Chapter 5 of the report for the proposed
active-learning loop over cluster centroids and other next steps.

## Author

Gautham Krishna J L, M.Sc. Computer Science (Machine Intelligence),
Kerala University of Digital Sciences, Innovation and Technology, 2026.
Supervised by Dr. Malu G.
