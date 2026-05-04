"""
TrOCR-style: ViT encoder + Transformer decoder, auto-regressive,
trained with cross-entropy + teacher forcing.

This is a from-scratch lightweight version (no HF dependency); the
decoder is a 4-layer TransformerDecoder with token embeddings of size
matching the encoder.
"""
from __future__ import annotations
import torch
import torch.nn as nn

from ._backbones import ViTEncoder
from ._base import LineRecognizer, register
from ..vocab import Vocab


def _causal_mask(T: int, device) -> torch.Tensor:
    return torch.triu(torch.ones(T, T, device=device, dtype=torch.bool),
                      diagonal=1)


@register("trocr")
class TrOCR(LineRecognizer):
    TYPE = "ar"
    TIME_REDUCTION = 8

    def __init__(self, vocab: Vocab,
                 dim: int = 256, enc_depth: int = 6, dec_depth: int = 4,
                 heads: int = 4, max_decode_len: int = 80):
        super().__init__(vocab)
        self.dim     = dim
        self.encoder = ViTEncoder(patch_h=64, patch_w=8, dim=dim,
                                   depth=enc_depth, heads=heads)
        self.n_out   = len(vocab) + 2
        self.sos_id  = len(vocab)
        self.eos_id  = len(vocab) + 1
        self.tok_embed = nn.Embedding(self.n_out, dim)
        self.pos_dec   = nn.Parameter(torch.zeros(1, max_decode_len + 1, dim))
        nn.init.trunc_normal_(self.pos_dec, std=0.02)
        layer = nn.TransformerDecoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=dim * 4,
            dropout=0.1, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, num_layers=dec_depth)
        self.head    = nn.Linear(dim, self.n_out)
        self.max_decode = max_decode_len

    def encode(self, imgs):
        return self.encoder(imgs)                  # (B, T_enc, dim)

    def forward(self, imgs: torch.Tensor,
                targets_padded: torch.Tensor | None = None) -> dict:
        memory = self.encode(imgs)
        B = imgs.size(0)
        device = imgs.device

        if self.training and targets_padded is not None:
            # Prepend <sos>
            sos = torch.full((B, 1), self.sos_id, dtype=torch.long, device=device)
            tgt_in = torch.cat([sos, targets_padded[:, :-1]], dim=1)
            T = tgt_in.size(1)
            tgt_emb = self.tok_embed(tgt_in) + self.pos_dec[:, :T, :]
            mask = _causal_mask(T, device)
            out = self.decoder(tgt_emb, memory, tgt_mask=mask)
            logits = self.head(out)                # (B, T, n_out)
            return {"logits": logits, "ar": True}
        else:
            ids = torch.full((B, 1), self.sos_id, dtype=torch.long, device=device)
            done = torch.zeros(B, dtype=torch.bool, device=device)
            for _ in range(self.max_decode):
                T = ids.size(1)
                tgt_emb = self.tok_embed(ids) + self.pos_dec[:, :T, :]
                mask = _causal_mask(T, device)
                out = self.decoder(tgt_emb, memory, tgt_mask=mask)
                logit_last = self.head(out[:, -1, :])
                pred = logit_last.argmax(-1, keepdim=True)
                ids = torch.cat([ids, pred], dim=1)
                done = done | (pred.squeeze(-1) == self.eos_id)
                if done.all():
                    break
            return {"logits": None,
                    "predicted_ids": ids[:, 1:],   # drop <sos>
                    "ar": True}

    def compute_loss(self, out: dict, targets: torch.Tensor,
                     in_lens, tgt_lens) -> torch.Tensor:
        logits = out["logits"]                    # (B, T, n_out)
        B, T, _ = logits.shape
        device  = logits.device
        offsets = [0]
        for L in tgt_lens.tolist():
            offsets.append(offsets[-1] + L)
        # Build padded target: real tokens + <eos> + ignore_index padding
        padded = torch.full((B, T), -100, dtype=torch.long, device=device)
        for b in range(B):
            L = int(tgt_lens[b])
            L_use = min(L, T - 1)
            padded[b, :L_use] = targets[offsets[b]:offsets[b] + L_use]
            if L_use < T:
                padded[b, L_use] = self.eos_id
        return nn.functional.cross_entropy(
            logits.reshape(-1, self.n_out), padded.reshape(-1),
            ignore_index=-100)

    @torch.no_grad()
    def decode(self, out: dict) -> list[list[str]]:
        if out.get("predicted_ids") is None:
            ids = out["logits"].argmax(-1)
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
