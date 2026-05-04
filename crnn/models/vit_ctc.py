"""Vision Transformer encoder over vertical patches + CTC head."""
from __future__ import annotations
import torch
import torch.nn as nn

from ._backbones import ViTEncoder
from ._base import LineRecognizer, register
from ..vocab import Vocab


@register("vit_ctc")
class ViT_CTC(LineRecognizer):
    TYPE = "ctc"
    TIME_REDUCTION = 8

    def __init__(self, vocab: Vocab,
                 dim: int = 256, depth: int = 6, heads: int = 4,
                 patch_w: int = 8, blank_bias: float = -8.0):
        super().__init__(vocab)
        self.encoder = ViTEncoder(
            patch_h=64, patch_w=patch_w,
            dim=dim, depth=depth, heads=heads)
        self.head = nn.Linear(dim, len(vocab))
        with torch.no_grad():
            self.head.bias.fill_(0.0)
            self.head.bias[self.vocab.blank_idx] = blank_bias
        self.TIME_REDUCTION = patch_w

    def forward(self, imgs: torch.Tensor) -> dict:
        f = self.encoder(imgs)                # (B, T, dim)
        logits = self.head(f).transpose(0, 1) # (T, B, C)
        return {"logits": logits}
