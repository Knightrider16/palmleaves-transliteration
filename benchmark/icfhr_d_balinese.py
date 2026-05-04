"""
ICFHR 2018 Challenge D Track 1 — Balinese Word Transliteration benchmark.

This is the *first benchmark in this project that has real sequence-level
ground truth*: each image is a cropped Balinese word, paired with its
romanized transliteration (e.g., "kagastu", "ngantah", "sarira").

We train all 6 line-recognizer architectures from `crnn/models/` on
the same data, with the same recipe.  Evaluation reports:
    - Word accuracy (exact match)
    - Character error rate (CER, edit distance normalised by gold length)

Vocabulary is character-level: every distinct character that appears in
any train transcript becomes a token (plus the CTC blank).  Word
lengths range 1-16 characters; transcripts include letters, digits,
and punctuation.

Outputs:
    benchmark/results/icfhr_d_balinese/<arch>/{best.pth, log.csv, summary.json}
    benchmark/results/icfhr_d_balinese/comparison.csv
    benchmark/results/icfhr_d_balinese/curves.png
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import time
from pathlib import Path

import cv2
import editdistance
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

from crnn.models import REGISTRY, build
from crnn.train_v2 import collate_ar
from crnn.dataset import collate
from crnn.vocab import Vocab

from benchmark._augment import augment_word
from benchmark._checkpoint import (
    is_arch_done, reset_arch,
    save_last, load_last, save_best, write_summary,
    append_log, trim_log_to,
)


ROOT       = Path("benchmark/icfhr2018")
TRAIN_DIR  = ROOT / "train" / "Train-ChallengeD-Track1-Bali" / "balinese_word_train"
TRAIN_GT   = ROOT / "train" / "Train-ChallengeD-Track1-Bali" / "balinese_transliteration_train.txt"
TEST_DIR   = ROOT / "test"  / "Test-ChallengeD-Track1-Balinese" / "balinese_word_test"
TEST_GT    = ROOT / "gt"    / "Evaluation-ChallengeD-Track1-Balinese" / "balinese_transliteration_test.txt"
OUT_ROOT   = Path("benchmark/results/icfhr_d_balinese")
MODEL_ROOT = Path("models/icfhr_d_balinese")


def _read_pairs(gt_file: Path, img_dir: Path
                ) -> list[tuple[Path, str]]:
    pairs = []
    with open(gt_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or ";" not in line:
                continue
            parts = line.split(";", 1)
            if len(parts) < 2:
                continue
            name = parts[0].strip()
            text = parts[1].strip()
            if not name:
                continue
            p = img_dir / name
            if p.is_file():
                pairs.append((p, text))
    return pairs


def _build_charset(pairs: list[tuple[Path, str]]) -> list[str]:
    chars: set[str] = set()
    for _p, t in pairs:
        chars.update(t)
    return sorted(chars)


_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
_SHARPEN_KERNEL = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]],
                            dtype=np.float32)


def _project_preprocess(gray: np.ndarray) -> np.ndarray:
    """
    Mirror of the project's preprocessing pipeline (preprocessing_scripts/):
        preprocess.py        : CLAHE
        batch_postprocess.py : CLAHE again + 3x3 sharpening kernel
        batch_mask_clean.py  : GaussianBlur(3,3) -> adaptiveThreshold(41, 7)
                               -> connected-component filter -> dilate(2,1)

    Skips the RealESRGAN upscale step (batch_upscale.py) — word crops are
    already at usable resolution and upscale is single-image-slow.
    """
    # 1. CLAHE (preprocess.py)
    eq = _CLAHE.apply(gray)

    # 2. CLAHE + sharpening (batch_postprocess.py)
    eq2 = _CLAHE.apply(eq)
    sharp = cv2.filter2D(eq2, -1, _SHARPEN_KERNEL)

    # 3. Mask clean (batch_mask_clean.py)
    blur = cv2.GaussianBlur(sharp, (3, 3), 0)
    th = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        41, 7,
    )
    n, labels, stats, _ = cv2.connectedComponentsWithStats(th, connectivity=8)
    clean = np.zeros_like(th)
    H, W = th.shape
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 40 or h < 3 or w < 3:
            continue
        if area > 0.25 * H * W:
            continue
        clean[labels == i] = 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
    clean = cv2.dilate(clean, kernel, iterations=1)
    return clean


def _read_word_image(path: Path, height: int = 64,
                     max_w: int = 600,
                     pipeline_height: int = 128) -> np.ndarray:
    """
    Apply the project's preprocessing pipeline at its intended scale,
    then downsample to the model's input height.

    The mask-clean step uses a 41x41 adaptive-threshold window which
    expects roughly upscaled palm-leaf imagery; word crops are smaller
    (~50-100 px tall), so we pre-resize to `pipeline_height` first.
    This mirrors what RealESRGAN x2 does in batch_upscale.py for the
    main project.
    """
    img = cv2.imread(str(path))
    if img is None:
        return np.zeros((height, 32), dtype=np.uint8)
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # Pre-resize to pipeline's intended scale
    h, w = gray.shape
    scale_up = pipeline_height / max(1, h)
    big_w = max(32, int(w * scale_up))
    gray_up = cv2.resize(gray, (big_w, pipeline_height),
                          interpolation=cv2.INTER_CUBIC)

    # Apply the project's full preprocessing pipeline -> binary mask
    mask = _project_preprocess(gray_up)

    # Resize down to model input height
    scale_dn = height / pipeline_height
    new_w = max(8, min(max_w, int(big_w * scale_dn)))
    return cv2.resize(mask, (new_w, height), interpolation=cv2.INTER_AREA)


class WordDataset(Dataset):
    # Per-arch aug strength; reduced for unstable archs (BiLSTM-style)
    AUG_STRENGTH = {"crnn_ctc": 0.5}

    def __init__(self, pairs: list[tuple[Path, str]], vocab: Vocab,
                 height: int = 64, augment: bool = False,
                 aug_strength: float = 1.0):
        # Drop pairs whose transcripts contain chars not in vocab (rare)
        self.pairs = [(p, t) for p, t in pairs
                       if all(c in vocab.stoi for c in t)]
        self.vocab        = vocab
        self.height       = height
        self.augment      = augment
        self.aug_strength = aug_strength

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        path, text = self.pairs[idx]
        img = _read_word_image(path, self.height)
        if self.augment:
            img = augment_word(img, strength=self.aug_strength)
        x = torch.from_numpy(img).float().unsqueeze(0) / 255.0  # (1, H, W)
        y = torch.tensor([self.vocab.stoi[c] for c in text], dtype=torch.long)
        return x, y, str(path.name)


def _ter(model, loader, device, vocab):
    model.eval()
    correct, total = 0, 0
    edit_d, gold_len = 0, 0
    with torch.no_grad():
        for batch in loader:
            imgs, targets, in_lens, tgt_lens, _names, *rest = batch
            imgs = imgs.to(device)
            out = model(imgs)
            preds = model.decode(out)
            offset = 0
            for b in range(len(preds)):
                gold_ids = targets[offset:offset + int(tgt_lens[b])].tolist()
                offset += int(tgt_lens[b])
                gold = [vocab.itos[i] for i in gold_ids]
                pred = preds[b]
                if pred == gold:
                    correct += 1
                total += 1
                edit_d   += editdistance.eval(pred, gold)
                gold_len += max(1, len(gold))
    word_acc = correct / max(1, total) * 100
    cer      = edit_d / max(1, gold_len) * 100
    return word_acc, cer


def _train_one(name: str, vocab: Vocab,
                train_loader, val_loader, test_loader,
                device, args, charset: list[str]):
    arch_dir  = OUT_ROOT / name
    arch_dir.mkdir(parents=True, exist_ok=True)
    log_csv   = arch_dir / "log.csv"
    last_path = arch_dir / "last.pth"
    best_path = arch_dir / "best.pth"

    model = build(name, vocab=vocab).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    is_ar = (model.TYPE == "ar")
    print(f"\n=== {name} === ({n_params:.2f} M, type={model.TYPE})")

    optim = torch.optim.SGD(model.parameters(), lr=args.lr,
                             momentum=0.9, nesterov=True, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.MultiStepLR(
        optim, milestones=[max(1, args.epochs // 2),
                            max(2, args.epochs * 3 // 4)], gamma=0.1)

    # Resume from last.pth if present
    state = load_last(last_path, model, optim, sched, device)
    if state is not None:
        start_ep      = state["epoch"] + 1
        best_word_acc = state["best_metric"]
        trim_log_to(log_csv, state["epoch"])
        print(f"  resumed from epoch {state['epoch']}  "
              f"(best_val_word_acc={best_word_acc:.2f}%)")
    else:
        start_ep      = 1
        best_word_acc = -1.0

    if start_ep > args.epochs:
        print(f"  already at target epochs ({args.epochs}), skipping training")
    else:
        for ep in range(start_ep, args.epochs + 1):
            model.train()
            t0 = time.time()
            run_loss, n = 0.0, 0
            pbar = tqdm(train_loader, desc=f"{name} ep{ep}/{args.epochs}")
            for batch in pbar:
                if is_ar:
                    imgs, targets, in_lens, tgt_lens, _names, padded = batch
                    padded = padded.to(device)
                else:
                    imgs, targets, in_lens, tgt_lens, _names = batch
                    padded = None
                imgs    = imgs.to(device)
                targets = targets.to(device)
                in_lens = in_lens.to(device)
                tgt_lens = tgt_lens.to(device)
                out = model(imgs, padded) if is_ar else model(imgs)
                loss = model.compute_loss(out, targets, in_lens, tgt_lens)
                if not torch.isfinite(loss):
                    continue
                optim.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optim.step()
                run_loss += loss.item(); n += 1
                pbar.set_postfix(loss=f"{run_loss/max(1,n):.3f}")
            sched.step()

            word_acc, cer = _ter(model, val_loader, device, vocab)
            dt = time.time() - t0
            avg = run_loss / max(1, n)
            print(f"  ep {ep}: loss={avg:.3f}  val_word_acc={word_acc:.2f}%  "
                  f"val_CER={cer:.2f}%  {dt:.0f}s")
            append_log(
                log_csv,
                ["epoch", "train_loss", "val_word_acc", "val_cer_pct",
                 "epoch_seconds"],
                [ep, avg, word_acc, cer, dt],
            )

            if word_acc > best_word_acc:
                best_word_acc = word_acc
                save_best(best_path, model, extra={
                    "vocab":   vocab.itos,
                    "charset": charset,
                    "config":  {"name": name, "lr": args.lr,
                                "batch": args.batch,
                                "epochs": args.epochs},
                })

            save_last(last_path, model, optim, sched, ep, best_word_acc,
                      extra={"name": name})

    print(f"  ── test eval ──")
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    test_word_acc, test_cer = _ter(model, test_loader, device, vocab)
    print(f"  test_word_acc={test_word_acc:.2f}%  test_CER={test_cer:.2f}%")

    s = {
        "model":          name,
        "params_M":       n_params,
        "type":           model.TYPE,
        "best_val_word_acc": best_word_acc,
        "test_word_acc":  test_word_acc,
        "test_cer":       test_cer,
        "epochs":         args.epochs,
        "batch":          args.batch,
    }
    write_summary(arch_dir, s)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models",  default=",".join(sorted(REGISTRY)))
    ap.add_argument("--epochs",  type=int,   default=12)
    ap.add_argument("--batch",   type=int,   default=32)
    ap.add_argument("--lr",      type=float, default=0.01)
    ap.add_argument("--seed",    type=int,   default=0)
    ap.add_argument("--max-train", type=int, default=0,
                    help="cap train set size (0 = use all)")
    ap.add_argument("--reset", action="store_true",
                    help="wipe per-arch results before training (no resume)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_pairs = _read_pairs(TRAIN_GT, TRAIN_DIR)
    test_pairs  = _read_pairs(TEST_GT,  TEST_DIR)
    print(f"Train pairs: {len(train_pairs)}, Test pairs: {len(test_pairs)}")

    charset = _build_charset(train_pairs + test_pairs)
    vocab   = Vocab(charset)
    print(f"Charset ({len(charset)}): {''.join(charset)}")
    print(f"Vocab (incl. blank): {len(vocab)}")

    if args.max_train > 0 and len(train_pairs) > args.max_train:
        train_pairs = train_pairs[:args.max_train]
        print(f"Capped train to {args.max_train}")

    test_ds = WordDataset(test_pairs, vocab, augment=False)

    summaries = []
    requested = [m.strip() for m in args.models.split(",") if m.strip()]
    for name in requested:
        if name not in REGISTRY:
            print(f"[skip] unknown model {name}")
            continue

        arch_dir = OUT_ROOT / name
        if args.reset:
            reset_arch(arch_dir)
        if is_arch_done(arch_dir):
            print(f"[skip] {name} already done (summary.json present)")
            with open(arch_dir / "summary.json", "r", encoding="utf-8") as f:
                summaries.append(json.load(f))
            continue

        # Per-arch aug strength (reduce for unstable archs like crnn_ctc)
        strength = WordDataset.AUG_STRENGTH.get(name, 1.0)
        train_full = WordDataset(train_pairs, vocab,
                                  augment=True, aug_strength=strength)
        n_val = max(300, len(train_full) // 20)
        train, val = random_split(
            train_full, [len(train_full) - n_val, n_val],
            generator=torch.Generator().manual_seed(args.seed))

        is_ar = (REGISTRY[name].TYPE == "ar")
        coll = collate_ar if is_ar else collate
        train_loader = DataLoader(train, batch_size=args.batch,
                                   shuffle=True, num_workers=0,
                                   collate_fn=coll)
        val_loader   = DataLoader(val,   batch_size=args.batch,
                                   shuffle=False, num_workers=0,
                                   collate_fn=coll)
        test_loader  = DataLoader(test_ds, batch_size=args.batch,
                                   shuffle=False, num_workers=0,
                                   collate_fn=coll)
        try:
            s = _train_one(name, vocab, train_loader, val_loader,
                            test_loader, device, args, charset)
            summaries.append(s)
        except KeyboardInterrupt:
            print(f"\n  Interrupted during {name}. "
                  f"Re-run to resume from last.pth.")
            raise
        except Exception as e:
            print(f"  ERROR in {name}: {e}")
            import traceback; traceback.print_exc()

    if summaries:
        cmp_csv = OUT_ROOT / "comparison.csv"
        with open(cmp_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=[
                "model", "type", "params_M", "best_val_word_acc",
                "test_word_acc", "test_cer", "epochs", "batch"])
            w.writeheader(); w.writerows(summaries)
        print(f"\nWrote: {cmp_csv}")


if __name__ == "__main__":
    main()
