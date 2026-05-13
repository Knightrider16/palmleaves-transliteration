"""
Phase 4 eval: re-decode the cnn_ctc Phase 3 best model on the
Mathilakam val holdout under three regimes:

    1. Greedy (baseline; matches Phase 3's reported 87.29 % TER)
    2. Plain beam search (no LM, beam_width=20)
    3. LM-rescored beam (5-gram LM, alpha sweep)

Writes:
    benchmark/results/mathilakam/phase4_lm.csv      regime/alpha/beta/TER
    benchmark/results/mathilakam/phase4_lm.md       human-readable summary

Run:
    realesgran_venv/Scripts/python.exe -m benchmark.eval_mathilakam_lm
"""
from __future__ import annotations
import csv
from pathlib import Path

import editdistance
import numpy as np
import torch
from torch.utils.data import DataLoader

from crnn.dataset  import LineDataset, collate
from crnn.vocab    import Vocab
from crnn.models   import build
from crnn.lm       import NGramLM
from crnn.beam_lm  import beam_search_lm
from crnn.models._base import _ctc_beam_search


CKPT      = Path("benchmark/results/mathilakam/phase3/cnn_ctc/best.pth")
VOCAB     = Path("data/real_lines/vocab_v3.txt")
VAL_CSV   = Path("data/real_lines/index_val.csv")
VAL_DIR   = Path("data/real_lines")
LM_PATH   = Path("data/labels/mathilakam_5gram.pkl")

OUT_CSV   = Path("benchmark/results/mathilakam/phase4_lm.csv")
OUT_MD    = Path("benchmark/results/mathilakam/phase4_lm.md")


def _load_val():
    vocab = Vocab.load(str(VOCAB))
    ds = LineDataset(str(VAL_CSV), str(VAL_DIR), vocab=vocab, augment=False)
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate)
    return vocab, ds, loader


def _build_model(vocab):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build("cnn_ctc", vocab=vocab).to(device)
    ckpt = torch.load(str(CKPT), map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, device


def _run(model, device, loader, vocab, decode_fn) -> tuple[float, list[tuple]]:
    rows: list[tuple] = []
    total_ed   = 0
    total_gold = 0
    with torch.no_grad():
        for imgs, targets, in_lens, tgt_lens, names in loader:
            imgs = imgs.to(device)
            out  = model(imgs)
            logits = out["logits"]            # (T, B, C)
            logp = logits.log_softmax(-1).cpu().numpy()
            T, B, C = logp.shape
            offset = 0
            for b in range(B):
                gold_ids = targets[offset:offset + int(tgt_lens[b])].tolist()
                offset += int(tgt_lens[b])
                gold = [vocab.itos[i] for i in gold_ids]
                pred = decode_fn(logp[:, b, :], vocab)
                ed = editdistance.eval(pred, gold)
                total_ed   += ed
                total_gold += max(1, len(gold))
                rows.append((names[b], gold, pred, ed))
    ter = total_ed / max(1, total_gold) * 100.0
    return ter, rows


def main() -> None:
    print("Loading val + model...")
    vocab, ds, loader = _load_val()
    print(f"  val rows: {len(ds)}")
    model, device = _build_model(vocab)
    print(f"  device: {device}")

    print("Loading 5-gram LM...")
    lm = NGramLM.load(str(LM_PATH))
    print(f"  LM vocab: {len(lm.vocab)}  unigram total: {lm.total_unigrams}")

    results: list[dict] = []

    # --- 1. Greedy (baseline) ---
    def greedy(logp1: np.ndarray, vocab: Vocab) -> list[str]:
        ids = logp1.argmax(-1).tolist()
        return vocab.ctc_decode(ids)
    ter_g, _ = _run(model, device, loader, vocab, greedy)
    print(f"[greedy]            TER = {ter_g:.2f}%")
    results.append({"regime": "greedy", "alpha": 0.0, "beta": 0.0,
                    "beam": 1, "ter_pct": ter_g})

    # --- 2. Plain beam (no LM) ---
    def beam_no_lm(logp1, vocab):
        return _ctc_beam_search(logp1, vocab, beam_width=20)
    ter_b, _ = _run(model, device, loader, vocab, beam_no_lm)
    print(f"[beam20 no-LM]      TER = {ter_b:.2f}%")
    results.append({"regime": "beam20", "alpha": 0.0, "beta": 0.0,
                    "beam": 20, "ter_pct": ter_b})

    # --- 3. LM-rescored beam, sweep alpha ---
    BETA  = 1.0   # length bonus (counters LM bias toward short prefixes)
    BEAM  = 20
    sweep_alpha = [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
    sample_outputs: list[tuple] = []
    for a in sweep_alpha:
        def beam_with_lm(logp1, vocab, _a=a):
            return beam_search_lm(logp1, vocab, lm=lm,
                                   alpha=_a, beta=BETA, beam_width=BEAM)
        ter, rows = _run(model, device, loader, vocab, beam_with_lm)
        print(f"[beam{BEAM} LM a={a} b={BETA}]  TER = {ter:.2f}%")
        results.append({"regime": "beam_lm",
                        "alpha": a, "beta": BETA, "beam": BEAM,
                        "ter_pct": ter})
        if a == 0.5:
            sample_outputs = rows[:5]

    # --- write outputs ---
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "regime", "alpha", "beta", "beam", "ter_pct"])
        w.writeheader()
        w.writerows(results)
    print(f"\nWrote {OUT_CSV}")

    # markdown summary
    best = min(results, key=lambda r: r["ter_pct"])
    md = ["# Phase 4 — char/syllable n-gram LM rescoring",
          "",
          "Re-decoded the cnn_ctc Phase 3 best.pth on the Mathilakam",
          "val holdout (15 lines) under three decoding regimes.",
          "",
          "| Regime | alpha | beta | beam | val TER |",
          "|---|---|---|---|---|"]
    for r in results:
        md.append(f"| {r['regime']} | {r['alpha']} | {r['beta']} | "
                  f"{r['beam']} | {r['ter_pct']:.2f}% |")
    md.append("")
    md.append(f"**Best regime:** `{best['regime']}` "
              f"alpha={best['alpha']} beta={best['beta']} "
              f"-> **{best['ter_pct']:.2f}% TER**")
    md.append("")
    if sample_outputs:
        md.append("## Sample predictions (alpha=0.5)")
        md.append("")
        for name, gold, pred, ed in sample_outputs:
            md.append(f"- **{name}** (edit distance {ed})")
            md.append(f"  - gold[:10]: `{gold[:10]}`")
            md.append(f"  - pred[:10]: `{pred[:10]}`")
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
