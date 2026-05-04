"""Pure CNN encoder + CTC, no recurrence.  Most stable on tiny datasets."""
from __future__ import annotations
import torch
import torch.nn as nn

from ._backbones import build_cnn_8block
from ._base import LineRecognizer, register
from ..vocab import Vocab


@register("cnn_ctc")
class CNN_CTC(LineRecognizer):
    TYPE = "ctc"
    TIME_REDUCTION = 8

    def __init__(self, vocab: Vocab, blank_bias: float = -8.0):
        super().__init__(vocab)
        self.cnn  = build_cnn_8block()
        self.head = nn.Linear(512, len(vocab))
        with torch.no_grad():
            self.head.bias.fill_(0.0)
            self.head.bias[self.vocab.blank_idx] = blank_bias

    def forward(self, imgs: torch.Tensor) -> dict:
        f = self.cnn(imgs)                          # (B, 512, 1, T)
        if f.size(2) != 1:
            f = nn.functional.adaptive_avg_pool2d(f, (1, f.size(3)))
        f = f.squeeze(2).permute(2, 0, 1)           # (T, B, 512)
        logits = self.head(f)                       # (T, B, C)
        return {"logits": logits}
