"""
Conformer-CTC: a small Convolutional + Self-Attention hybrid encoder.

The Conformer block interleaves:
    1. half-step FFN
    2. self-attention
    3. depthwise-separable convolution
    4. half-step FFN
which gives the data efficiency of CNNs with the long-range context
of attention.  Followed by a CTC head.
"""
from __future__ import annotations
import torch
import torch.nn as nn

from ._backbones import PatchEmbed1D
from ._base import LineRecognizer, register
from ..vocab import Vocab


class _FFN(nn.Module):
    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.lin1 = nn.Linear(dim, dim * mult)
        self.lin2 = nn.Linear(dim * mult, dim)
        self.do   = nn.Dropout(dropout)

    def forward(self, x):
        return self.lin2(self.do(nn.functional.silu(self.lin1(self.norm(x)))))


class _ConvModule(nn.Module):
    def __init__(self, dim: int, kernel: int = 7, dropout: float = 0.1):
        super().__init__()
        self.norm    = nn.LayerNorm(dim)
        self.pw1     = nn.Conv1d(dim, dim * 2, 1)
        self.dw      = nn.Conv1d(dim, dim, kernel, padding=kernel // 2,
                                  groups=dim)
        self.bn      = nn.BatchNorm1d(dim)
        self.pw2     = nn.Conv1d(dim, dim, 1)
        self.do      = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, D)
        y = self.norm(x).transpose(1, 2)        # (B, D, T)
        y = self.pw1(y)
        a, b = y.chunk(2, dim=1)
        y = a * torch.sigmoid(b)                # GLU
        y = self.dw(y)
        y = nn.functional.silu(self.bn(y))
        y = self.pw2(y)
        return self.do(y).transpose(1, 2)


class _ConformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.ffn1  = _FFN(dim, dropout=dropout)
        self.attn_norm = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(dim, heads, dropout=dropout,
                                             batch_first=True)
        self.conv  = _ConvModule(dim, dropout=dropout)
        self.ffn2  = _FFN(dim, dropout=dropout)
        self.norm  = nn.LayerNorm(dim)

    def forward(self, x):
        x = x + 0.5 * self.ffn1(x)
        a, _ = self.attn(self.attn_norm(x), self.attn_norm(x), self.attn_norm(x),
                          need_weights=False)
        x = x + a
        x = x + self.conv(x)
        x = x + 0.5 * self.ffn2(x)
        return self.norm(x)


@register("conformer")
class Conformer_CTC(LineRecognizer):
    TYPE = "ctc"
    TIME_REDUCTION = 8

    def __init__(self, vocab: Vocab,
                 dim: int = 256, depth: int = 6, heads: int = 4,
                 patch_w: int = 8, blank_bias: float = -8.0):
        super().__init__(vocab)
        self.embed = PatchEmbed1D(1, patch_h=64, patch_w=patch_w, dim=dim)
        self.blocks = nn.ModuleList(
            [_ConformerBlock(dim, heads=heads) for _ in range(depth)])
        self.head = nn.Linear(dim, len(vocab))
        with torch.no_grad():
            self.head.bias.fill_(0.0)
            self.head.bias[self.vocab.blank_idx] = blank_bias
        self.TIME_REDUCTION = patch_w

    def forward(self, imgs: torch.Tensor) -> dict:
        f = self.embed(imgs)                    # (B, T, dim)
        for blk in self.blocks:
            f = blk(f)
        logits = self.head(f).transpose(0, 1)   # (T, B, C)
        return {"logits": logits}
