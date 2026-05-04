# AMADI Challenge 3 — Balinese Character Recognition

Benchmark of the project's `GlyphClassifier` architecture on the official ICFHR 2016 isolated-character recognition dataset.

## Setup

- Architecture: `crnn.pretrain_cnn.GlyphClassifier` (8-block CNN + GlobalAvgPool + Linear, ~9.5 M params)
- Train images: **11,710**
- Validation split: 585
- Test images:  **7,673**
- Classes:      **133** Balinese characters
- Optimizer: SGD lr=0.05, momentum=0.9, MultiStep schedule
- Augmentation: shared `benchmark._augment.augment_char` (affine, elastic, stroke jitter, erasing)
- Epochs: 50, batch 128

## Headline numbers

- **Test top-1 accuracy: 89.44%**  (6,863 / 7,673)
- **Test top-5 accuracy: 97.69%**  (7,496 / 7,673)
- Best val accuracy during training: 90.60%

## Comparison with Malayanma project's same architecture

Same `GlyphClassifier` architecture trained on synthetic Malayalam glyphs reaches ~98% val accuracy on its own synthetic test split (see `crnn/pretrain_cnn.py`).  This benchmark uses the *same architecture and recipe* but on real palm-leaf data with a different script — so the gap between the two numbers reflects the synthetic→real domain gap and the script difference.

## Reproducibility

```bash
python -m benchmark.amadi_charrec --epochs 12 --batch 128 --lr 0.05
```

Outputs:
- `models\amadi_balinese\glyph_classifier.pth` (best model on val)
- `benchmark\results\amadi_charrec\log.csv` per-epoch loss/val-accuracy
- `benchmark\results\amadi_charrec\predictions.csv` per-test-image predictions
