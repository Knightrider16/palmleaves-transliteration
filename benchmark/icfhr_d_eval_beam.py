"""
Re-evaluate every trained ICFHR D model on the test set, comparing
greedy vs beam-search decoding.

Reads the per-arch best.pth from benchmark/results/icfhr_d_balinese/<arch>/
and runs:
  - greedy decode (model.decode)
  - beam search width=10 (model.decode_beam)

Writes:
  benchmark/results/icfhr_d_balinese/comparison_beam.csv

Usage:
    python -m benchmark.icfhr_d_eval_beam --beam 10
"""
from __future__ import annotations
import argparse
import csv
import json
import time
from pathlib import Path

import editdistance
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from benchmark.icfhr_d_balinese import (
    WordDataset, _read_pairs, TRAIN_DIR, TRAIN_GT, TEST_DIR, TEST_GT)
from crnn.models import REGISTRY, build
from crnn.train_v2 import collate_ar
from crnn.dataset import collate
from crnn.vocab import Vocab


OUT_ROOT = Path("benchmark/results/icfhr_d_balinese")


def _eval(model, vocab, loader, device, mode: str = "greedy",
           beam: int = 10):
    model.eval()
    word_correct, total = 0, 0
    edit_d, gold_len = 0, 0
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"{mode}", leave=False):
            imgs, targets, in_lens, tgt_lens, _names, *rest = batch
            imgs = imgs.to(device)
            out = model(imgs)
            if mode == "beam":
                preds = model.decode_beam(out, beam_width=beam)
            else:
                preds = model.decode(out)
            offset = 0
            for b in range(len(preds)):
                gold_ids = targets[offset:offset + int(tgt_lens[b])].tolist()
                offset += int(tgt_lens[b])
                gold = [vocab.itos[i] for i in gold_ids]
                if preds[b] == gold:
                    word_correct += 1
                total += 1
                edit_d   += editdistance.eval(preds[b], gold)
                gold_len += max(1, len(gold))
    word_acc = word_correct / max(1, total) * 100
    cer      = edit_d / max(1, gold_len) * 100
    return word_acc, cer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--beam",   type=int, default=10)
    ap.add_argument("--batch",  type=int, default=64)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    rows = []
    archs = [d.name for d in OUT_ROOT.iterdir()
              if d.is_dir() and (d / "best.pth").exists()]
    archs.sort()
    print(f"Found {len(archs)} trained archs: {archs}")

    test_pairs = _read_pairs(TEST_GT, TEST_DIR)
    print(f"Test pairs: {len(test_pairs)}")

    for name in archs:
        ckpt_path = OUT_ROOT / name / "best.pth"
        ckpt = torch.load(ckpt_path, map_location=device)
        vocab = Vocab(ckpt["vocab"]) if "vocab" in ckpt else None
        if vocab is None:
            print(f"  [skip] {name}: no vocab in checkpoint")
            continue
        try:
            model = build(name, vocab=vocab).to(device)
            model.load_state_dict(ckpt["model_state"])
        except Exception as e:
            print(f"  [skip] {name}: {e}")
            continue
        n_params = sum(p.numel() for p in model.parameters()) / 1e6

        ds = WordDataset(test_pairs, vocab, height=64, augment=False)
        coll = collate_ar if model.TYPE == "ar" else collate
        loader = DataLoader(ds, batch_size=args.batch, shuffle=False,
                             num_workers=0, collate_fn=coll)

        print(f"\n=== {name} ({n_params:.2f} M, {model.TYPE}) ===")
        t0 = time.time()
        g_acc, g_cer = _eval(model, vocab, loader, device, "greedy")
        t_g = time.time() - t0
        t0 = time.time()
        b_acc, b_cer = _eval(model, vocab, loader, device, "beam",
                              beam=args.beam)
        t_b = time.time() - t0
        print(f"  greedy: word_acc={g_acc:.2f}%  CER={g_cer:.2f}%   ({t_g:.1f} s)")
        print(f"  beam{args.beam:>2}: word_acc={b_acc:.2f}%  CER={b_cer:.2f}%   ({t_b:.1f} s)")
        rows.append({
            "model":           name,
            "type":            model.TYPE,
            "params_M":        f"{n_params:.2f}",
            "greedy_word_acc": f"{g_acc:.2f}",
            "greedy_cer":      f"{g_cer:.2f}",
            f"beam{args.beam}_word_acc": f"{b_acc:.2f}",
            f"beam{args.beam}_cer":      f"{b_cer:.2f}",
            "greedy_seconds":  f"{t_g:.1f}",
            "beam_seconds":    f"{t_b:.1f}",
        })

    out_csv = OUT_ROOT / "comparison_beam.csv"
    if rows:
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"\nWrote: {out_csv}")


if __name__ == "__main__":
    main()
