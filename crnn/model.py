"""
CRNN line recognizer (CNN backbone -> BiLSTM -> CTC).

Input  : (B, 1, 64, W)   — variable width grayscale strips
Output : (T, B, num_classes)  — logits over vocab (incl. CTC blank)
T scales as roughly W // 8.
"""
from __future__ import annotations
import torch
import torch.nn as nn


def _conv_block(in_c: int, out_c: int, k: int = 3,
                pool: tuple[int, int] | None = None,
                bn: bool = False) -> nn.Sequential:
    layers: list[nn.Module] = [
        nn.Conv2d(in_c, out_c, k, stride=1, padding=k // 2),
    ]
    if bn:
        layers.append(nn.BatchNorm2d(out_c))
    layers.append(nn.ReLU(inplace=True))
    if pool is not None:
        layers.append(nn.MaxPool2d(pool, pool))
    return nn.Sequential(*layers)


class CRNN(nn.Module):
    """
    Standard Shi-style CRNN, sized for 64-pixel input height.

    Backbone height schedule:
        64 -> 32 -> 16 -> 8 -> 4 -> 2 -> 1
    Width schedule:
        W  -> W/2 -> W/4 -> W/8 -> W/8 -> W/8 -> W/8
    """

    def __init__(self, num_classes: int, hidden: int = 256,
                 use_rnn: bool = True):
        super().__init__()
        self.use_rnn = use_rnn
        self.cnn = nn.Sequential(
            _conv_block(1,   64,  pool=(2, 2)),                 # 64→32, W→W/2
            _conv_block(64,  128, pool=(2, 2)),                 # 32→16, W→W/4
            _conv_block(128, 256, bn=True),
            _conv_block(256, 256, pool=(2, 2)),                 # 16→8,  W→W/8
            _conv_block(256, 512, bn=True),
            _conv_block(512, 512, pool=(2, 1)),                 # 8→4
            _conv_block(512, 512, pool=(2, 1)),                 # 4→2
            _conv_block(512, 512, pool=(2, 1)),                 # 2→1
        )
        # Now feature map is (B, 512, 1, W/8).
        if use_rnn:
            self.rnn = nn.LSTM(
                input_size=512,
                hidden_size=hidden,
                num_layers=2,
                bidirectional=True,
                dropout=0.1,
            )
            head_in = hidden * 2
        else:
            self.rnn = None
            head_in = 512
        self.head = nn.Linear(head_in, num_classes)

        # CTC blank-collapse mitigation: initialise bias so blank starts
        # at a large negative logit.  Without this the model converges to
        # all-blank predictions and the gradient stops flowing.
        with torch.no_grad():
            self.head.bias.fill_(0.0)
            self.head.bias[0] = -8.0   # 0 = CTC blank index

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, 64, W)
        f = self.cnn(x)                              # (B, 512, 1, T)
        if f.size(2) != 1:
            f = nn.functional.adaptive_avg_pool2d(f, (1, f.size(3)))
        f = f.squeeze(2).permute(2, 0, 1)            # (T, B, 512)
        if self.rnn is not None:
            f, _ = self.rnn(f)                       # (T, B, 2H)
        out = self.head(f)                           # (T, B, num_classes)
        return out

    def time_steps(self, width: int) -> int:
        """Predict the CTC output length for a given input width."""
        return max(1, width // 8)


if __name__ == "__main__":
    m = CRNN(num_classes=470)
    x = torch.zeros(2, 1, 64, 800)
    y = m(x)
    print("input :", tuple(x.shape))
    print("output:", tuple(y.shape))
    print("params:", sum(p.numel() for p in m.parameters()))
