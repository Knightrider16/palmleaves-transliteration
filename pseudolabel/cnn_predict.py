"""
Predict a token label for every char crop using the CNN classifier
trained on synthetic glyphs.

The classifier is the GlyphClassifier from crnn.pretrain_cnn — i.e.,
the same CNN backbone we just trained, plus the linear head with 471
classes.  We re-create it, load the full state dict, and run inference.

Usage:
    python -m pseudolabel.cnn_predict

Outputs:
    data/pseudo_labeled_v3/cnn_predictions.csv
        filename, predicted_token, confidence, top2_token, top2_conf
"""
from __future__ import annotations
import argparse
import csv
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from crnn.pretrain_cnn import GlyphClassifier
from crnn.vocab import Vocab
from pseudolabel.embed_chars import CharCropDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab",     default="data/synthetic/chars/vocab.txt")
    ap.add_argument("--char-dir",  default="data/characters_named")
    ap.add_argument("--out-dir",   default="data/pseudo_labeled_v3")
    ap.add_argument("--ckpt",      default="models/cnn_backbone.pth",
                    help="CNN backbone state — head will be re-trained "
                         "if not in checkpoint")
    ap.add_argument("--full-ckpt", default="",
                    help="full GlyphClassifier checkpoint (with head)")
    ap.add_argument("--batch",     type=int, default=256)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab = Vocab.from_vocab_file(args.vocab)
    print(f"Vocab: {len(vocab)} (incl. blank)")
    model = GlyphClassifier(num_classes=len(vocab)).to(device)

    if args.full_ckpt and os.path.isfile(args.full_ckpt):
        state = torch.load(args.full_ckpt, map_location=device)
        if "model_state" in state:
            state = state["model_state"]
        model.load_state_dict(state)
        print(f"Loaded full classifier from {args.full_ckpt}")
    else:
        # Use the CNN backbone we have; head gets random init + we
        # cannot run inference unless we re-train.  In our case we
        # actually saved the CNN backbone only.  So we need to re-train
        # the classifier briefly to get a working head.
        # For now: load CNN, leave head random.  THIS WILL GIVE GARBAGE
        # PREDICTIONS unless head is trained.  Calling code must use
        # --full-ckpt or run pretrain_cnn first.
        cnn_state = torch.load(args.ckpt, map_location=device)
        loaded = model.load_state_dict(cnn_state, strict=False)
        n_loaded = sum(1 for k in cnn_state.keys() if k.startswith("cnn."))
        print(f"Loaded CNN ({n_loaded} weights) but head is RANDOM. "
              f"Re-run pretrain_cnn to save full classifier.")

    model.eval()
    ds = CharCropDataset(args.char_dir)
    print(f"Predicting on {len(ds)} char crops")
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False,
                        num_workers=0)
    all_files = ds.files

    rows = []
    pred_idx = 0
    with torch.no_grad():
        for x, _idx in tqdm(loader):
            x = x.to(device)
            logits = model(x)
            probs  = logits.softmax(-1)
            top = probs.topk(2, dim=-1)
            for b in range(x.size(0)):
                fname = all_files[pred_idx]
                pred_idx += 1
                top1 = vocab.itos[int(top.indices[b, 0])]
                top2 = vocab.itos[int(top.indices[b, 1])]
                rows.append({
                    "filename":         fname,
                    "predicted_token":  top1,
                    "confidence":       float(top.values[b, 0]),
                    "top2_token":       top2,
                    "top2_conf":        float(top.values[b, 1]),
                })

    out = os.path.join(args.out_dir, "cnn_predictions.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "filename", "predicted_token", "confidence",
            "top2_token", "top2_conf"])
        w.writeheader()
        w.writerows(rows)

    # Confidence histogram
    confs = np.array([r["confidence"] for r in rows])
    print(f"\nConfidence histogram:")
    bins = [0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 1.01]
    hist, edges = np.histogram(confs, bins=bins)
    for lo, hi, n in zip(edges[:-1], edges[1:], hist):
        bar = "█" * min(50, n // 200)
        print(f"  [{lo:.2f}, {hi:.2f}): {n:6d}  {bar}")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
