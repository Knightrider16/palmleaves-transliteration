"""
CNN encoder + BiLSTM + Attention decoder (auto-regressive).

Token sequences are decoded left-to-right with a single-headed
content-based attention over encoder outputs.  Trained with
cross-entropy + teacher forcing.

Special tokens used internally:
    <sos> = vocab index len(vocab)         (added on top of vocab)
    <eos> = vocab index len(vocab) + 1
"""
from __future__ import annotations
import torch
import torch.nn as nn

from ._backbones import build_cnn_8block
from ._base import LineRecognizer, register
from ..vocab import Vocab


@register("crnn_attn")
class CRNN_Attn(LineRecognizer):
    TYPE = "ar"
    TIME_REDUCTION = 8

    def __init__(self, vocab: Vocab, hidden: int = 256, lstm_layers: int = 2,
                 dropout: float = 0.1, max_decode_len: int = 80):
        super().__init__(vocab)
        self.hidden = hidden
        self.cnn = build_cnn_8block()
        self.enc = nn.LSTM(
            input_size=512, hidden_size=hidden,
            num_layers=lstm_layers, bidirectional=True, dropout=dropout)
        # +2 for <sos>, <eos>
        self.n_out  = len(vocab) + 2
        self.sos_id = len(vocab)
        self.eos_id = len(vocab) + 1
        self.embed  = nn.Embedding(self.n_out, hidden)
        self.dec    = nn.LSTMCell(hidden + hidden * 2, hidden)
        self.attn_q = nn.Linear(hidden, hidden)
        self.attn_k = nn.Linear(hidden * 2, hidden)
        self.head   = nn.Linear(hidden + hidden * 2, self.n_out)
        self.max_decode = max_decode_len

    def encode(self, imgs):
        f = self.cnn(imgs)
        if f.size(2) != 1:
            f = nn.functional.adaptive_avg_pool2d(f, (1, f.size(3)))
        f = f.squeeze(2).permute(2, 0, 1)            # (T, B, 512)
        f, _ = self.enc(f)                           # (T, B, 2H)
        return f.transpose(0, 1)                     # (B, T, 2H)

    def _step(self, prev_tok, h, c, enc, enc_keys):
        emb = self.embed(prev_tok)                   # (B, H)
        q   = self.attn_q(h).unsqueeze(1)            # (B, 1, H)
        scores = torch.matmul(q, enc_keys.transpose(1, 2)).squeeze(1) / (self.hidden ** 0.5)
        a = scores.softmax(-1)                       # (B, T)
        ctx = torch.bmm(a.unsqueeze(1), enc).squeeze(1)  # (B, 2H)
        h, c = self.dec(torch.cat([emb, ctx], dim=-1), (h, c))
        out = self.head(torch.cat([h, ctx], dim=-1)) # (B, n_out)
        return out, h, c

    def forward(self, imgs: torch.Tensor,
                targets_padded: torch.Tensor | None = None) -> dict:
        enc = self.encode(imgs)
        enc_keys = self.attn_k(enc)
        B = imgs.size(0)
        h = torch.zeros(B, self.hidden, device=imgs.device)
        c = torch.zeros(B, self.hidden, device=imgs.device)

        if self.training and targets_padded is not None:
            # Teacher forcing
            outs = []
            prev = torch.full((B,), self.sos_id, device=imgs.device, dtype=torch.long)
            T = targets_padded.size(1)
            for t in range(T):
                out, h, c = self._step(prev, h, c, enc, enc_keys)
                outs.append(out)
                prev = targets_padded[:, t]
            logits = torch.stack(outs, dim=1)        # (B, T, n_out)
            return {"logits": logits, "ar": True}
        else:
            # Greedy decode
            outs = []
            prev = torch.full((B,), self.sos_id, device=imgs.device, dtype=torch.long)
            done = torch.zeros(B, dtype=torch.bool, device=imgs.device)
            for _ in range(self.max_decode):
                out, h, c = self._step(prev, h, c, enc, enc_keys)
                pred = out.argmax(-1)
                outs.append(pred)
                prev = pred
                done = done | (pred == self.eos_id)
                if done.all():
                    break
            return {"logits": None,
                    "predicted_ids": torch.stack(outs, dim=1),  # (B, L)
                    "ar": True}

    def compute_loss(self, out: dict, targets: torch.Tensor,
                     in_lens, tgt_lens) -> torch.Tensor:
        logits = out["logits"]                       # (B, T, n_out)
        # Build flat target with <eos> appended per sequence
        B, T, _ = logits.shape
        device = logits.device
        # `targets` is concatenated; we need (B, T) padded with <eos>
        # at end of each.  Caller supplies pre-built padded targets.
        # For convenience we do it here.
        flat = targets
        offsets = [0]
        for L in tgt_lens.tolist():
            offsets.append(offsets[-1] + L)
        padded = torch.full((B, T), self.eos_id, dtype=torch.long, device=device)
        for b in range(B):
            L = int(tgt_lens[b])
            padded[b, :L] = flat[offsets[b]:offsets[b] + L]
            if L < T:
                padded[b, L] = self.eos_id   # <eos> right after last token
        return nn.functional.cross_entropy(
            logits.reshape(-1, self.n_out), padded.reshape(-1),
            ignore_index=-100)

    @torch.no_grad()
    def decode(self, out: dict) -> list[list[str]]:
        if out.get("predicted_ids") is None:
            # Training-mode fallback: argmax teacher-forced logits
            logits = out["logits"]                   # (B, T, n_out)
            ids = logits.argmax(-1)
        else:
            ids = out["predicted_ids"]
        results = []
        for row in ids:
            seq = []
            for i in row.tolist():
                if i == self.eos_id:
                    break
                if i == self.sos_id:
                    continue
                if 0 < i < len(self.vocab):
                    seq.append(self.vocab.itos[i])
            results.append(seq)
        return results
