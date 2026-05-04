# Palm Leaf Transliteration — Run Guide

End-to-end pipeline for transliterating ancient palm-leaf manuscripts
(Malayanma / Malayalam) using a multi-model line-recognizer framework.

## What's in the project

```
synthetic/             # synthetic data generation (font-based)
├── tokens.py          # romanization → Malayalam Unicode
├── glyph_renderer.py  # PIL rendering
├── degradation.py     # palm-leaf degradation pipeline
├── gen_chars.py       # synthetic char dataset
├── gen_lines.py       # synthetic line dataset
└── fonts/             # Noto Sans Malayalam

pseudolabel/           # building pseudo-labels for the unlabeled chars
├── embed_chars.py     # embed all 25k chars with the trained CNN
├── recluster.py       # cluster the CNN embeddings (HDBSCAN)
├── build_labels.py    # cluster-vote pseudo-labels
├── cnn_predict.py     # CNN-classifier direct predictions
└── stitch_lines.py    # stitch real char crops into "real-glyph synthetic" lines

crnn/                  # the line-recognizer framework
├── vocab.py           # token <-> id, CTC blank handling
├── dataset.py         # variable-width line dataset
├── extract_lines.py   # split mask strips into per-line crops
├── pretrain_cnn.py    # CNN char-classifier pretraining
├── train_v2.py        # unified resumable training
├── infer.py           # transliterate any mask
├── compare.py         # build the multi-model comparison table
└── models/
    ├── _base.py       # LineRecognizer + REGISTRY
    ├── _backbones.py  # shared CNN, ViT, patch embed
    ├── crnn_ctc.py    # CNN + BiLSTM + CTC
    ├── cnn_ctc.py     # CNN + Linear + CTC (no RNN)
    ├── crnn_attn.py   # CNN + BiLSTM + Attention decoder (AR)
    ├── vit_ctc.py     # ViT encoder + CTC
    ├── trocr.py       # ViT encoder + Transformer decoder (AR)
    └── conformer.py   # Conformer encoder + CTC

run_all.py             # multi-model orchestrator with state tracking
RUN_STATE.json         # persistent run state (created at runtime)
```

## Prerequisites

- Python 3.10 in `realesgran_venv/`
- PyTorch 2.0.1 + CUDA 11.8 (RTX 4050 detected)
- NumPy 1.26 (downgraded from 2.x for compatibility)
- `editdistance`, `scipy`, `Pillow`, `opencv-python`, `tqdm`, `umap-learn`, `hdbscan`
- Noto Sans Malayalam font at [synthetic/fonts/NotoSansMalayalam-Regular.ttf](synthetic/fonts/NotoSansMalayalam-Regular.ttf)

```bash
PY="e:/S4/Palmleaves-Transliteration/realesgran_venv/Scripts/python.exe"
export PYTHONIOENCODING=utf-8
```

## Step-by-step (full pipeline from scratch)

### 1. Generate synthetic data

```bash
$PY -m synthetic.gen_chars --per-token 30                  # ~14k char crops
$PY -m synthetic.gen_lines --n-lines 3000                  # 3k font-rendered lines
```

### 2. Pretrain the CNN char classifier (98% val acc on synthetic)

```bash
$PY -m crnn.pretrain_cnn --epochs 12 --batch 128 --lr 0.05
```
Saves `models/cnn_backbone.pth` (CNN only) and `models/glyph_classifier.pth` (full model).

### 3. Build pseudo-labels for the 25k real char crops

```bash
$PY -m pseudolabel.embed_chars                             # CNN embeddings
$PY -m pseudolabel.recluster                                # CNN-feature clusters
$PY -m pseudolabel.cnn_predict --full-ckpt models/glyph_classifier.pth
```
Outputs:
- `data/embeddings/char_embeddings.npy`
- `cluster_assignments_v2.csv`
- `data/pseudo_labeled_v3/cnn_predictions.csv`  (per-char predictions with confidence)

### 4. Build "real-glyph synthetic" line strips

```bash
$PY -m pseudolabel.stitch_lines --n-lines 3000 --min-conf 0.8
```
3000 lines stitched from high-confidence real char crops. These bridge
the synth → real domain gap.

### 5. Extract real labeled lines (eval set)

```bash
$PY -m crnn.extract_lines
```
Slices `data/masks_clean_upscaled/1A1_pre_x2_mask.png` into 14 per-line
crops aligned to the labeled CSV (line detection currently recovers
14 of the 19 transcribed lines).

### 6. Train all 6 models — **resumable**

```bash
$PY run_all.py                                # train all
$PY run_all.py --models crnn_ctc,cnn_ctc      # subset
$PY run_all.py --epochs 8                     # override per-model epochs
$PY run_all.py --status                       # see what's done
$PY run_all.py --reset                        # forget state, restart
```

Each training writes per-epoch checkpoints:
- `models/<arch>/last.pth`  (resume point)
- `models/<arch>/best.pth`  (best val TER)
- `models/<arch>/log.csv`   (loss / TER history)

`RUN_STATE.json` tracks which models are pending / in-progress / done.

**You can ctrl+C the training any time, shut down, restart, and run
`python run_all.py` again to continue.** Each model resumes from its
last completed epoch.

### 7. Compare all trained models

```bash
$PY -m crnn.compare
```
Outputs `results/comparison.csv` and `results/comparison.md`, plus a
table on stdout:

```
| Model     | Params | best val TER | synth TER | real TER | final loss |
|-----------|--------|--------------|-----------|----------|------------|
| cnn_ctc   |  9.5M  |  ...         |  ...      |  ...     |  ...       |
| crnn_ctc  | 12.6M  |  ...         |  ...      |  ...     |  ...       |
| vit_ctc   |  5.3M  |  ...         |  ...      |  ...     |  ...       |
| ...       |        |              |           |          |            |
```

### 8. Inference on a single strip

```bash
$PY -m crnn.infer \
    --model models/crnn_ctc/best.pth \
    --inputs "data/masks_clean_upscaled/*.png" \
    --labels data/labels/labels.csv \
    --out results/transliterations.csv
```

`--arch` is auto-detected from the parent dir name; pass it explicitly
if you stash checkpoints elsewhere.

## The 6 architectures

| Name       | Encoder        | Decoder    | Loss | Params  | Notes |
|------------|----------------|------------|------|---------|-------|
| `cnn_ctc`  | 8-block CNN    | Linear     | CTC  |  9.5 M  | Most stable on tiny datasets |
| `crnn_ctc` | 8-block CNN    | BiLSTM x2  | CTC  | 12.6 M  | Industry-standard OCR baseline |
| `vit_ctc`  | ViT-Tiny       | Linear     | CTC  |  5.3 M  | Pure transformer encoder |
| `crnn_attn`| 8-block CNN    | LSTM+Attn  | CE   | 14.1 M  | Auto-regressive, non-monotonic |
| `trocr`    | ViT            | TransformerDec | CE | 9.6 M | TrOCR-style end-to-end transformer |
| `conformer`| Conv+Attn hybrid | Linear   | CTC  |  9.4 M  | Best of both worlds |

CTC models load `models/cnn_backbone.pth` automatically (`--cnn-init`).
Pure-transformer models (`vit_ctc`, `trocr`) train from scratch.

## Known issues / next steps

- **Line-detection on `1A1`** recovers 14 of 19 labeled lines — the
  remaining 5 are merged because lines are stacked tightly. To fix,
  manually adjust line bounds, or label more images.
- **Synthetic glyphs are modern Malayalam**, the real palm-leaf script
  is Malayanma. The visual gap is partially closed by the
  "real-glyph stitched lines", but not fully — real palm leaves use
  Malayanma forms not present in our synthetic vocab.
- **Self-supervised pretraining** (DINO/SimCLR on the 25k unlabeled
  chars) is not yet implemented; should give a measurable win once
  more labeled data arrives.
- **Active learning loop** — pick the model's least-confident clusters,
  hand-label centroids, retrain. Not implemented.
- **Language-model rescoring** — Malayalam token bigram LM. Not implemented.

## Common issues

- *NumPy 2.x error*: `$PY -m pip install "numpy<2"`.
- *Unicode print errors on Windows*: `set PYTHONIOENCODING=utf-8`.
- *CTC blank collapse* (loss flat, val_TER 100%): all models register
  an anti-blank bias init; if you build a new model, set
  `head.bias[blank_idx] = -8.0` in `__init__`.
- *Crash mid-training*: re-run `python run_all.py` — it resumes.
