# Phase 4 — char/syllable n-gram LM rescoring

Re-decoded the cnn_ctc Phase 3 best.pth on the Mathilakam
val holdout (15 lines) under three decoding regimes.

| Regime | alpha | beta | beam | val TER |
|---|---|---|---|---|
| greedy | 0.0 | 0.0 | 1 | 87.29% |
| beam20 | 0.0 | 0.0 | 20 | 86.25% |
| beam_lm | 0.1 | 1.0 | 20 | 85.57% |
| beam_lm | 0.3 | 1.0 | 20 | 82.47% |
| beam_lm | 0.5 | 1.0 | 20 | 83.51% |
| beam_lm | 0.7 | 1.0 | 20 | 83.51% |
| beam_lm | 1.0 | 1.0 | 20 | 84.54% |
| beam_lm | 1.5 | 1.0 | 20 | 85.22% |
| beam_lm | 2.0 | 1.0 | 20 | 86.25% |

**Best regime:** `beam_lm` alpha=0.3 beta=1.0 -> **82.47% TER**

## Sample predictions (alpha=0.5)

- **1A1_pre_x2_mask_line02.png** (edit distance 18)
  - gold[:10]: `['zha', 'na', 'na', 'ne', 'me', 'la', 'thi', 'vi', 'la', 'ki']`
  - pred[:10]: `['na', 'ma', 'ka', 'ra', 'na', 'ka', 'ka', 'na', 'ka', 'ku']`
- **1A1_pre_x2_mask_line11.png** (edit distance 19)
  - gold[:10]: `['mu', 'da', 'da', 'vi', 'la', 'thaa', 'mo', 'thi', 'ra', 'na']`
  - pred[:10]: `['mu', 'na', 'ma', 'pi', 'la']`
- **1A1_pre_x2_mask_line10.png** (edit distance 17)
  - gold[:10]: `['la', 'thi', 'ru', 'vaa', 'na', 'na', 'tha', 'pu', 'ra', 'thhu']`
  - pred[:10]: `['na', 'ma', 'ka', 'ra', 'na', 'ka', 'na', 'na', 'na']`
- **1A2_pre_x2_mask_line02.png** (edit distance 26)
  - gold[:10]: `['va', 'zhi', 'vi', 'la', 'u', 'la', 'laa', 'da', 'du', 'ka']`
  - pred[:10]: `['a', 'tha', 'tha', 'na', 'na', 'maa', 'ka', 'tha', 'thi', 'vi']`
- **1A2_pre_x2_mask_line06.png** (edit distance 14)
  - gold[:10]: `['mu', 'na', 'naa', 'la', 'ma', 'laa', 'maa', 'na', 'du', 'tha']`
  - pred[:10]: `['ka', 'la', 'pa', 'pa', 'na', 'ma', 'la', 'na', 'ra', 'tha']`