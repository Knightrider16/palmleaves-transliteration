"""
Shared backbone components used by multiple architectures.

These are the "well-trodden" CNNs and lightweight ViT pieces that the
line recognizers can pick up.  Putting them in one place keeps the
per-architecture files small.
"""
from __future__ import annotations
import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────
# 8-block CNN: 64x64 -> 1xT  (T = W/8)
# Same as the one used to pretrain the glyph classifier, so we can
# load cnn_backbone.pth weights into any of the CNN-based models.
# ─────────────────────────────────────────────────────────────────────

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


def build_cnn_8block() -> nn.Sequential:
    """Standard CRNN backbone, height-reducing to 1, width-reducing 8x."""
    return nn.Sequential(
        _conv_block(1,   64,  pool=(2, 2)),
        _conv_block(64,  128, pool=(2, 2)),
        _conv_block(128, 256, bn=True),
        _conv_block(256, 256, pool=(2, 2)),
        _conv_block(256, 512, bn=True),
        _conv_block(512, 512, pool=(2, 1)),
        _conv_block(512, 512, pool=(2, 1)),
        _conv_block(512, 512, pool=(2, 1)),
    )


# ─────────────────────────────────────────────────────────────────────
# Patch-embed ViT encoder for line images
# Treats the line as a sequence of tall vertical strips (patches).
# ─────────────────────────────────────────────────────────────────────

class PatchEmbed1D(nn.Module):
    """
    Splits a (1, H, W) line image into vertical strips of width
    `patch_w`, each producing one token.  Output is (B, N_tokens, dim).
    """
    def __init__(self, in_ch: int = 1, patch_h: int = 64, patch_w: int = 8,
                 dim: int = 256):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, dim,
                              kernel_size=(patch_h, patch_w),
                              stride=(patch_h, patch_w))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, H, W). H must equal patch_h.
        f = self.proj(x)           # (B, dim, 1, W/patch_w)
        f = f.squeeze(2).transpose(1, 2)
        return f                    # (B, T, dim)


class ViTEncoder(nn.Module):
    """
    Thin TransformerEncoder over patch embeddings + sinusoidal pos embed.
    """
    def __init__(self,
                 patch_h: int = 64, patch_w: int = 8,
                 dim: int = 256, depth: int = 6, heads: int = 4,
                 mlp_ratio: float = 4.0, dropout: float = 0.1,
                 max_len: int = 1024):
        super().__init__()
        self.patch_embed = PatchEmbed1D(1, patch_h, patch_w, dim)
        # Learned positional embedding (sequence length capped at max_len)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads,
            dim_feedforward=int(dim * mlp_ratio),
            dropout=dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.dim    = dim
        self.patch_w = patch_w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.patch_embed(x)               # (B, T, dim)
        T = f.size(1)
        f = f + self.pos_embed[:, :T, :]
        f = self.encoder(f)
        return f                              # (B, T, dim)
