"""
Train the CRNN line recognizer.

Two phases:

    pretrain  — large amount of synthetic data, perfect labels
    finetune  — small amount of real labeled lines, weight boost on
                high-confidence rows

Either phase can be run on its own.

Usage:
    python -m crnn.train pretrain   --epochs 20 --batch 16
    python -m crnn.train finetune   --epochs 30 --batch 4 \
                                    --init models/crnn_pretrain.pth

Outputs:
    models/crnn_pretrain.pth   (after pretrain)
    models/crnn_finetune.pth   (after finetune)
    models/vocab.txt           (canonical vocabulary used at inference)
"""
from __future__ import annotations
import argparse
import os
import time
from pathlib import Path

import editdistance
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from .dataset import LineDataset, collate
from .model   import CRNN
from .vocab   import Vocab


MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)


def _build_vocab(synth_vocab_path: str) -> Vocab:
    """Load the canonical token vocabulary (built by gen_chars/gen_lines)."""
    return Vocab.from_vocab_file(synth_vocab_path)


def _eval(model, loader, vocab, device, max_batches: int | None = None):
    model.eval()
    total_chars = 0
    total_dist  = 0
    seen_lines  = 0
    with torch.no_grad():
        for bi, (imgs, targets, in_lens, tgt_lens, _names) in enumerate(loader):
            if max_batches and bi >= max_batches:
                break
            imgs = imgs.to(device)
            logp = model(imgs).log_softmax(-1)        # (T, B, C)
            preds = logp.argmax(-1).transpose(0, 1)   # (B, T)

            offset = 0
            for b in range(preds.size(0)):
                gold = targets[offset:offset + tgt_lens[b]].tolist()
                offset += tgt_lens[b].item()
                pred = vocab.ctc_decode(preds[b].tolist())
                gold_toks = [vocab.itos[i] for i in gold]
                d = editdistance.eval(pred, gold_toks)
                total_dist  += d
                total_chars += max(1, len(gold_toks))
                seen_lines  += 1
    return total_dist / max(1, total_chars), seen_lines


def _train_loop(model, train_loader, val_loader, vocab, device,
                epochs: int, lr: float, save_path: Path,
                grad_clip: float = 5.0):
    optim = torch.optim.SGD(model.parameters(), lr=lr,
                            momentum=0.9, weight_decay=1e-4, nesterov=True)
    sched = torch.optim.lr_scheduler.MultiStepLR(
        optim, milestones=[max(1, epochs // 2),
                           max(2, epochs * 3 // 4)], gamma=0.1)
    # zero_infinity=False so inf losses still produce gradients via clipping.
    # The model can collapse to blank-only with zero_infinity=True.
    ctc   = nn.CTCLoss(blank=vocab.blank_idx, zero_infinity=False)

    best_ter = float("inf")
    for ep in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        running = 0.0
        n = 0
        pbar = tqdm(train_loader, desc=f"epoch {ep}/{epochs}")
        for imgs, targets, in_lens, tgt_lens, _ in pbar:
            imgs    = imgs.to(device)
            targets = targets.to(device)
            in_lens = in_lens.to(device)
            tgt_lens = tgt_lens.to(device)

            logp = model(imgs).log_softmax(-1)              # (T, B, C)
            T_max = logp.size(0)
            in_lens_clamped = torch.clamp(in_lens, max=T_max)
            loss = ctc(logp, targets, in_lens_clamped, tgt_lens)

            # Skip rare batches where input < target (CTC -> inf loss, NaN grad)
            if not torch.isfinite(loss):
                continue

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optim.step()

            running += loss.item()
            n += 1
            pbar.set_postfix(loss=f"{running / n:.3f}")
        sched.step()

        # Validation: token-level CER
        ter, n_val = _eval(model, val_loader, vocab, device, max_batches=50)
        dt = time.time() - t0
        print(f"  epoch {ep}: train_loss={running/max(1,n):.3f}  "
              f"val_TER={ter*100:.1f}% (n={n_val})  {dt:.1f}s")

        if ter < best_ter:
            best_ter = ter
            torch.save({"state_dict": model.state_dict(),
                        "vocab":      vocab.itos},
                       save_path)
            print(f"    ✓ saved (best TER={ter*100:.1f}%) -> {save_path}")

    print(f"\nDone.  Best val TER: {best_ter*100:.1f}%")


def cmd_pretrain(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    vocab = _build_vocab(args.vocab)
    print(f"Vocab size: {len(vocab)}")
    vocab.save(MODEL_DIR / "vocab.txt")

    full = LineDataset(
        index_csv=args.index, img_dir=args.img_dir,
        vocab=vocab, height=args.height, augment=True,
    )
    print(f"Synthetic lines: {len(full)}")
    n_val = max(1, min(len(full) - 1, max(50, len(full) // 20)))
    train, val = random_split(full, [len(full) - n_val, n_val],
                              generator=torch.Generator().manual_seed(0))
    train_loader = DataLoader(train, batch_size=args.batch, shuffle=True,
                              collate_fn=collate, num_workers=0)
    val_loader   = DataLoader(val,   batch_size=args.batch, shuffle=False,
                              collate_fn=collate, num_workers=0)

    model = CRNN(num_classes=len(vocab), use_rnn=not args.no_rnn).to(device)
    if args.cnn_init and os.path.isfile(args.cnn_init):
        cnn_state = torch.load(args.cnn_init, map_location=device)
        # cnn_state keys are "cnn.0.0.weight" etc.; load into model.cnn
        missing, unexpected = model.load_state_dict(cnn_state, strict=False)
        loaded = sum(1 for k in cnn_state.keys() if k.startswith("cnn."))
        print(f"Loaded {loaded} CNN weights from {args.cnn_init}")
        if unexpected:
            print(f"  unexpected: {unexpected[:3]}{'...' if len(unexpected)>3 else ''}")
    if args.init and os.path.isfile(args.init):
        ckpt = torch.load(args.init, map_location=device)
        model.load_state_dict(ckpt["state_dict"], strict=False)
        print(f"Resumed from {args.init}")

    _train_loop(model, train_loader, val_loader, vocab, device,
                epochs=args.epochs, lr=args.lr,
                save_path=MODEL_DIR / "crnn_pretrain.pth")


def cmd_finetune(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    vocab = _build_vocab(args.vocab)
    print(f"Vocab size: {len(vocab)}")

    real_full = LineDataset(
        index_csv=args.index, img_dir=args.img_dir,
        vocab=vocab, height=args.height, augment=True,
        image_col="filename", transcript_col="transcript",
    )
    print(f"Real lines (after vocab filter): {len(real_full)}")
    if len(real_full) == 0:
        raise RuntimeError("No real lines available — check extract_lines output")

    # 80/20 split, but with so few samples we just keep one for validation
    n_val   = max(1, len(real_full) // 5)
    n_train = len(real_full) - n_val
    train, val = random_split(real_full, [n_train, n_val],
                              generator=torch.Generator().manual_seed(0))
    train_loader = DataLoader(train, batch_size=args.batch, shuffle=True,
                              collate_fn=collate, num_workers=0)
    val_loader   = DataLoader(val,   batch_size=args.batch, shuffle=False,
                              collate_fn=collate, num_workers=0)

    model = CRNN(num_classes=len(vocab), use_rnn=not args.no_rnn).to(device)
    if not args.init:
        raise RuntimeError("--init <pretrain.pth> required for finetune")
    ckpt = torch.load(args.init, map_location=device)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    print(f"Initialized from {args.init}")

    _train_loop(model, train_loader, val_loader, vocab, device,
                epochs=args.epochs, lr=args.lr,
                save_path=MODEL_DIR / "crnn_finetune.pth")


def main():
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest="cmd", required=True)

    pp = sp.add_parser("pretrain")
    pp.add_argument("--vocab",   default="data/synthetic/chars/vocab.txt")
    pp.add_argument("--index",   default="data/synthetic/lines/index.csv")
    pp.add_argument("--img-dir", default="data/synthetic/lines")
    pp.add_argument("--height",  type=int,   default=64)
    pp.add_argument("--batch",   type=int,   default=16)
    pp.add_argument("--epochs",  type=int,   default=20)
    pp.add_argument("--lr",      type=float, default=0.02)
    pp.add_argument("--init",    default="",
                    help="resume from a full CRNN checkpoint")
    pp.add_argument("--cnn-init", default="",
                    help="initialise CNN backbone from char-classifier "
                         "checkpoint (models/cnn_backbone.pth)")
    pp.add_argument("--no-rnn", action="store_true",
                    help="skip the BiLSTM (CNN -> Linear -> CTC). "
                         "Often more stable for small datasets.")
    pp.set_defaults(func=cmd_pretrain)

    pf = sp.add_parser("finetune")
    pf.add_argument("--vocab",   default="data/synthetic/chars/vocab.txt")
    pf.add_argument("--index",   default="data/real_lines/index.csv")
    pf.add_argument("--img-dir", default="data/real_lines")
    pf.add_argument("--height",  type=int,   default=64)
    pf.add_argument("--batch",   type=int,   default=4)
    pf.add_argument("--epochs",  type=int,   default=30)
    pf.add_argument("--lr",      type=float, default=0.005)
    pf.add_argument("--init",    required=True)
    pf.add_argument("--no-rnn",  action="store_true")
    pf.set_defaults(func=cmd_finetune)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
