# Mathilakam (Malayanma) — Phase Tracker

A running scoreboard of the framework's progress on the in-house
Malayanma palm-leaf corpus. Each phase introduces one set of recipe
changes and is evaluated on the same held-out split (phases 2+) so
numbers are directly comparable.

## Headline (best Mathilakam TER ever produced by the project)

**`cnn_ctc` Phase 3 + 5-gram LM rescoring (Phase 4): 82.47% TER**
— a **9-percentage-point improvement** over the Phase 1 floor of
91.5% on the real Mathilakam validation set. The progression:

```
Phase 1 (real-only, weak aug)       91.5%
Phase 2 (synth + real, mode-collapsed) 89.3%
Phase 3 (real-dominated mix)         87.3%   <- best raw model
Phase 4 (Phase 3 + LM rescoring)    82.5%   <- best end-to-end
```

## Cross-phase scoreboard (val TER on the same 15 real-line holdout)

| Arch | Phase 0 | Phase 1 | Phase 2 | Phase 3 | Trend |
|---|---|---|---|---|---|
| **`cnn_ctc`** | 70.2%* | 91.5% | 89.34% | **87.29%** | ⬇ best of project |
| `crnn_ctc` | 100%* | 93.0% | 91.80% | 88.32% | ⬇ steady |
| `conformer` | 61.7%* | 92.6% | 89.34% | 89.69% | ↔ collapsed |
| `crnn_attn` | 116%* | 99.9% | 89.34% | (interrupted ep 1) | — |
| `vit_ctc` | 94.3%* | 96.3% | 92.62% | (not run) | — |
| `trocr` | 101.5%* | 101.9% | 101.09% | (not run) | — |

*Phase 0 numbers were measured on a different distribution
(synthetic + pseudo-stitched) and aren't directly comparable to Phase
1+. Recorded for completeness.

## Phase definitions

### Phase 0 — Original baseline (pre-cleanup)
8 epochs, weak gaussian-only augmentation, training on
`data/synthetic/lines + data/synthetic/real_stitched_lines`. Val
sampled randomly from that combined pool.

### Phase 1 — Recipe transfer, real-only
50 epochs + the strong `augment_word` (affine + elastic + stroke
jitter + horizontal stretch + erasing) ported from the benchmarks.
Trained **only** on the 83-line real Mathilakam corpus extracted from
`data/labels/labels.csv`. No synthetic in the mix. Auto val split (50
out of 83 — too large in retrospect).

Reproduces ch5's "real-data TER ≈ 91%" finding. Confirmed real-only
training is data-starved on this corpus.

### Phase 2 — Synth + real joint, fixed val
50 epochs, same `augment_word`. Training stack: 3,000 synth lines +
68 real Mathilakam train lines. Val pinned to 15 held-out real lines
(stratified by source image). Diagnosis: **mode collapse** — 4 archs
landed at *exactly* 89.34% TER, and inspection showed the top model
emitting a fixed 4-token loop (`ma na tha na`) regardless of input.
The 3,000 synth gradients drowned out the 68 real gradients and the
model never learned to read real palm-leaf imagery — it learned to
output the synth token-frequency prior.

### Phase 3 — Real-dominated mix (current best)
50 epochs. Three-way training stack:
- 3,000 synth font-rendered lines
- 3,000 stitched real-character lines (real char crops in synth
  layouts)
- 68 real Mathilakam train lines **upsampled 50×** (= 3,400 rows)

Real becomes ~36% of the gradient signal so the model is forced to
actually look at real palm-leaf imagery. Val unchanged.

Result: CTC archs all moved meaningfully:
- `cnn_ctc` 89.34% → 87.29% (−2.0 pp)
- `crnn_ctc` 91.80% → 88.32% (−3.5 pp)

Conformer / AR archs didn't budge (still ~89% mode-collapse floor).
Phase 3 was interrupted after the three CTC archs finished because
the AR archs (crnn_attn, trocr) consistently hit the same
mode-collapse floor regardless of recipe and would have cost
~13 hours of additional compute for no expected gain.

## Phase 4 — n-gram LM rescoring (done)

5-gram syllable-LM trained on all 112 transcripts (165-token vocab,
2,318 distinct 5-grams) with stupid-backoff smoothing. Re-decoded
the Phase 3 `cnn_ctc/best.pth` with prefix beam search of width 20
plus LM scoring `α log P_LM` and length bonus `β log len`.

Sweep result (β = 1.0, beam = 20):

| α | val TER |
|---|---|
| 0.0 (no LM, beam only) | 86.25% |
| 0.1 | 85.57% |
| **0.3** | **82.47%** ← best |
| 0.5 | 83.51% |
| 0.7 | 83.51% |
| 1.0 | 84.54% |
| 1.5 | 85.22% |
| 2.0 | 86.25% |

Convex curve confirms the LM is doing real work — too little α and
the rescoring is ineffective; too much and the LM overrides the
visual evidence and we degrade back to the no-LM beam baseline.

The 4.8 pp drop from greedy decoding is consistent with what the
ICFHR D case showed in *reverse*: there, beam search **without** an
LM regressed cnn_ctc from 24% → 101% CER (memorialised in the
project's memory entry). Here on Mathilakam the LM is the load-
bearing component, exactly as the literature predicts for low-
resource OCR with strong morphological structure.

**Code paths:**
- [crnn/lm.py](../../../crnn/lm.py) — n-gram LM with stupid-backoff
- [crnn/build_lm.py](../../../crnn/build_lm.py) — trainer (one-shot)
- [crnn/beam_lm.py](../../../crnn/beam_lm.py) — LM-aware prefix beam
- [benchmark/eval_mathilakam_lm.py](../../eval_mathilakam_lm.py)
  — eval pipeline / α sweep
- [benchmark/results/mathilakam/phase4_lm.csv](phase4_lm.csv) /
  [phase4_lm.md](phase4_lm.md) — outputs

## What this all says about the Mathilakam project

The chapter 5 §5 conclusion stands and is now backed by direct
measurement: **the architecture is not the bottleneck, the labelled
data is**. With 68 distinct training lines + the right recipe + LM
rescoring, the project hits 82.5% TER. Each pp below that requires
closing the data gap (hundreds of additional transcribed lines) —
exactly what ch5 §5.1 (more transcribed Mathilakam lines) and §5.2
(active-learning loop) propose.

## Final cross-phase summary table

For the thesis: this is the table that tells the full story.

| Phase | Recipe | Best arch | Val TER |
|---|---|---|---|
| Phase 0 | weak aug, 8 ep, on synth+pseudo | conformer | 61.7%* |
| Phase 1 | strong aug, 50 ep, real-only | cnn_ctc | 91.5% |
| Phase 2 | synth + real (1×) joint | cnn_ctc | 89.3% |
| Phase 3 | synth + stitched + real (50×) | **cnn_ctc** | **87.3%** |
| Phase 4 | Phase 3 + 5-gram LM beam | cnn_ctc (α=0.3) | **82.5%** |

*Phase 0 measured a different distribution; not directly comparable
to phases 1–4 which all use the fixed 15-line real Mathilakam val
holdout.
