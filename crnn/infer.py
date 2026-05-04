"""
Run the trained CRNN on real palm-leaf strips.

Usage:
    # Transliterate one or more clean masks
    python -m crnn.infer --model models/crnn_finetune.pth \\
                          --inputs "data/masks_clean_upscaled/*.png" \\
                          --out results/transliterations.csv

    # If you have ground-truth labels in a CSV, also report TER
    python -m crnn.infer --model models/crnn_finetune.pth \\
                          --inputs "data/masks_clean_upscaled/*.png" \\
                          --labels data/labels/labels.csv \\
                          --out results/transliterations.csv

The script auto-splits each strip into per-line crops using the same
peak-based row-projection algorithm as crnn.extract_lines.
"""
from __future__ import annotations
import argparse
import csv
import glob
import os
from collections import defaultdict
from pathlib import Path

import cv2
import editdistance
import numpy as np
import torch

from .extract_lines import extract_image_lines
from .models        import REGISTRY, build
from .vocab         import Vocab


def _load_model(model_path: str, device: torch.device, arch: str | None = None):
    ckpt = torch.load(model_path, map_location=device)
    vocab = Vocab(ckpt["vocab"])
    # train_v2 saves the state under "model_state"; older format used
    # "state_dict"
    state = ckpt.get("model_state") or ckpt.get("state_dict")
    if state is None:
        raise RuntimeError(f"No model state found in {model_path}")

    if arch is None:
        # Infer architecture from the parent directory name
        arch = Path(model_path).parent.name
    if arch not in REGISTRY:
        raise RuntimeError(
            f"Cannot determine architecture for {model_path}. "
            f"Pass --arch <name> from {sorted(REGISTRY)}")

    model = build(arch, vocab=vocab).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, vocab


def _line_to_tensor(line: np.ndarray, height: int = 64,
                    max_width: int = 6000) -> torch.Tensor:
    h, w = line.shape
    scale = height / max(1, h)
    new_w = max(8, min(max_width, int(w * scale)))
    img = cv2.resize(line, (new_w, height), interpolation=cv2.INTER_AREA)
    t = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0) / 255.0
    return t


@torch.no_grad()
def transliterate_line(model, vocab: Vocab,
                       line: np.ndarray, device) -> list[str]:
    x = _line_to_tensor(line).to(device)
    out = model(x)
    return model.decode(out)[0]


def transliterate_image(model, vocab, mask_path: str,
                        device, target_n: int | None = None
                        ) -> list[list[str]]:
    crops = extract_image_lines(mask_path, target_n=target_n)
    return [transliterate_line(model, vocab, c, device) for c in crops]


def _load_labels(labels_csv: str) -> dict[str, dict[int, str]]:
    out: dict[str, dict[int, str]] = defaultdict(dict)
    if not labels_csv or not os.path.isfile(labels_csv):
        return out
    with open(labels_csv, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            img = r["image"].strip()
            ln  = int(r["line"])
            out[img][ln] = r["transcript"].strip()
    return out


def _gold_tokens(transcript: str) -> list[str]:
    out = []
    for raw in transcript.split("/"):
        t = raw.strip().lower()
        if t and t != "[unk]":
            out.append(t)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",   required=True,
                    help="path to a checkpoint (typically "
                         "models/<arch>/best.pth)")
    ap.add_argument("--arch",    default="",
                    help="architecture name (auto-inferred from parent "
                         "dir if omitted)")
    ap.add_argument("--inputs",  required=True,
                    help="glob pattern for mask images")
    ap.add_argument("--labels",  default="",
                    help="optional ground-truth CSV for TER eval")
    ap.add_argument("--out",     default="results/transliterations.csv")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, vocab = _load_model(args.model, device,
                                arch=args.arch or None)
    print(f"Loaded {args.model}  vocab={len(vocab)}  device={device}")

    labels   = _load_labels(args.labels)
    paths    = sorted(glob.glob(args.inputs))
    print(f"Found {len(paths)} input strips")

    Path(os.path.dirname(args.out) or ".").mkdir(exist_ok=True, parents=True)

    rows = []
    total_dist = 0
    total_chars = 0

    for path in paths:
        image_id = os.path.splitext(os.path.basename(path))[0]
        target_n = len(labels.get(image_id, {})) or None
        line_preds = transliterate_image(model, vocab, path, device,
                                          target_n=target_n)
        gold_lines = labels.get(image_id, {})
        gold_keys = sorted(gold_lines.keys())

        for i, pred in enumerate(line_preds):
            ln = gold_keys[i] if i < len(gold_keys) else (i + 1)
            gold_str = gold_lines.get(ln, "")
            gold = _gold_tokens(gold_str) if gold_str else []
            d = editdistance.eval(pred, gold) if gold else None
            if gold:
                total_dist  += d
                total_chars += max(1, len(gold))
            rows.append({
                "image_id":   image_id,
                "line":       ln,
                "prediction": "/".join(pred),
                "gold":       gold_str,
                "edit_dist":  "" if d is None else d,
                "n_pred":     len(pred),
                "n_gold":     len(gold),
            })

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "image_id", "line", "prediction", "gold",
            "edit_dist", "n_pred", "n_gold"])
        w.writeheader()
        w.writerows(rows)

    print(f"\nWrote {len(rows)} predictions to {args.out}")
    if total_chars > 0:
        ter = total_dist / total_chars * 100
        print(f"Token error rate (vs labeled lines): {ter:.1f}%  "
              f"(edits={total_dist}, gold tokens={total_chars})")


if __name__ == "__main__":
    main()
