"""
Run inference + token-error-rate evaluation for every trained model
and print a side-by-side comparison table.

Reads:
    models/<arch>/best.pth        (preferred)  or  last.pth
    models/<arch>/log.csv         (training history)
    data/real_lines/index.csv     (real labeled eval set)
    data/synthetic/lines/index.csv (synthetic eval set, optional)

Outputs:
    results/comparison.csv
    results/comparison.md
    stdout: a formatted table
"""
from __future__ import annotations
import argparse
import csv
import os
from pathlib import Path

import editdistance
import torch
from torch.utils.data import DataLoader

from .dataset import LineDataset, collate
from .train_v2 import collate_ar
from .models import REGISTRY, build
from .vocab import Vocab


def _load_model(arch_dir: Path, vocab_path: str, device):
    ckpt_path = arch_dir / "best.pth"
    if not ckpt_path.exists():
        ckpt_path = arch_dir / "last.pth"
    if not ckpt_path.exists():
        return None, None
    ckpt = torch.load(ckpt_path, map_location=device)
    vocab = Vocab(ckpt["vocab"]) if "vocab" in ckpt \
        else Vocab.from_vocab_file(vocab_path)
    arch_name = arch_dir.name
    model = build(arch_name, vocab=vocab).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, vocab


def _eval_ter(model, vocab, csv_path: str, dir_path: str, device,
              batch: int = 8) -> tuple[float, int]:
    if not os.path.isfile(csv_path):
        return float("nan"), 0
    ds = LineDataset(csv_path, dir_path, vocab=vocab,
                     height=64, augment=False)
    if len(ds) == 0:
        return float("nan"), 0
    coll = collate_ar if model.TYPE == "ar" else collate
    loader = DataLoader(ds, batch_size=batch, shuffle=False,
                        collate_fn=coll, num_workers=0)
    total_dist = 0
    total_chars = 0
    with torch.no_grad():
        for batch_data in loader:
            imgs, targets, in_lens, tgt_lens, _names, *rest = batch_data
            imgs = imgs.to(device)
            out = model(imgs)
            preds = model.decode(out)
            offset = 0
            for b in range(len(preds)):
                gold = targets[offset:offset + tgt_lens[b]].tolist()
                offset += int(tgt_lens[b])
                gold_toks = [vocab.itos[i] for i in gold]
                d = editdistance.eval(preds[b], gold_toks)
                total_dist  += d
                total_chars += max(1, len(gold_toks))
    return total_dist / max(1, total_chars), len(ds)


def _final_train_loss(arch_dir: Path) -> float | None:
    log = arch_dir / "log.csv"
    if not log.exists():
        return None
    with open(log, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    return float(rows[-1].get("train_loss", "nan"))


def _final_val_ter(arch_dir: Path) -> float | None:
    log = arch_dir / "log.csv"
    if not log.exists():
        return None
    with open(log, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    ters = [float(r["val_ter_pct"]) for r in rows
            if r.get("val_ter_pct") not in (None, "")]
    return min(ters) if ters else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab",    default="data/synthetic/chars/vocab.txt")
    ap.add_argument("--real-csv", default="data/real_lines/index.csv")
    ap.add_argument("--real-dir", default="data/real_lines")
    ap.add_argument("--synth-csv", default="data/synthetic/lines/index.csv")
    ap.add_argument("--synth-dir", default="data/synthetic/lines")
    ap.add_argument("--out-dir",  default="results")
    ap.add_argument("--models",   default="",
                    help="comma-separated subset (default: all in registry)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    arch_names = [n for n in REGISTRY] if not args.models else \
        [n.strip() for n in args.models.split(",") if n.strip()]

    rows = []
    for name in arch_names:
        arch_dir = Path("models") / name
        train_loss = _final_train_loss(arch_dir)
        best_val   = _final_val_ter(arch_dir)
        print(f"\n--- {name} ---")
        if not (arch_dir / "best.pth").exists() and not (arch_dir / "last.pth").exists():
            print("  no checkpoint -- skipping")
            rows.append({"model": name, "params_M": "-",
                         "synth_TER_pct": "-", "real_TER_pct": "-",
                         "best_val_TER_pct": "-",
                         "final_train_loss": "-",
                         "real_n": 0, "synth_n": 0})
            continue

        model, vocab = _load_model(arch_dir, args.vocab, device)
        params_m = sum(p.numel() for p in model.parameters()) / 1e6

        synth_ter, synth_n = _eval_ter(model, vocab, args.synth_csv,
                                         args.synth_dir, device)
        real_ter,  real_n  = _eval_ter(model, vocab, args.real_csv,
                                         args.real_dir, device)

        print(f"  params: {params_m:.2f} M")
        print(f"  best_val_TER : {best_val:.1f}%" if best_val is not None else "")
        print(f"  synth_TER    : {synth_ter*100:.1f}%  ({synth_n} lines)")
        print(f"  real_TER     : {real_ter*100:.1f}%   ({real_n} lines)")
        rows.append({
            "model":            name,
            "params_M":         f"{params_m:.2f}",
            "best_val_TER_pct": f"{best_val:.1f}" if best_val is not None else "-",
            "synth_TER_pct":    f"{synth_ter * 100:.1f}",
            "real_TER_pct":     f"{real_ter * 100:.1f}",
            "real_n":           real_n,
            "synth_n":          synth_n,
            "final_train_loss": f"{train_loss:.3f}" if train_loss is not None else "-",
        })

    csv_path = os.path.join(args.out_dir, "comparison.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "model", "params_M", "best_val_TER_pct", "synth_TER_pct",
            "real_TER_pct", "real_n", "synth_n", "final_train_loss"])
        w.writeheader(); w.writerows(rows)
    print(f"\nSaved: {csv_path}")

    # Markdown table
    md = ["# Model comparison\n", "",
          "| Model | Params (M) | best val TER | synth TER | real TER | final loss |",
          "|---|---|---|---|---|---|"]
    for r in rows:
        md.append(
            f"| {r['model']} | {r['params_M']} | "
            f"{r['best_val_TER_pct']}% | {r['synth_TER_pct']}% | "
            f"{r['real_TER_pct']}% | {r['final_train_loss']} |")
    md_path = os.path.join(args.out_dir, "comparison.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"Saved: {md_path}")
    print("\n" + "\n".join(md))


if __name__ == "__main__":
    main()
