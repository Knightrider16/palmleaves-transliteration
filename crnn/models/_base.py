"""
Common interface for every line-recognizer architecture.

Each subclass implements three things:

    forward(imgs) -> dict
        Returns a dict containing whatever the loss/decode functions
        need.  The dict MUST include "logits".  CTC models return
        (T, B, C) logits; auto-regressive models return (B, T, C).

    compute_loss(out, targets, in_lens, tgt_lens) -> Tensor
        Standard CTC loss for CTC models, cross-entropy for AR models.

    decode(out) -> list[list[str]]
        Greedy decode using the model's vocab.

Models are registered via @register("name") at module level so the
trainer can look them up by string.
"""
from __future__ import annotations
from typing import Callable
import torch
import torch.nn as nn

from ..vocab import Vocab


REGISTRY: dict[str, type] = {}


def register(name: str) -> Callable[[type], type]:
    def deco(cls: type) -> type:
        REGISTRY[name] = cls
        cls.NAME = name
        return cls
    return deco


def build(name: str, vocab: Vocab, **kwargs) -> "LineRecognizer":
    if name not in REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Known: {sorted(REGISTRY)}")
    return REGISTRY[name](vocab=vocab, **kwargs)


class LineRecognizer(nn.Module):
    """
    Subclasses set:
        TYPE        : 'ctc' or 'ar' (auto-regressive)
        TIME_REDUCTION : how much CNN reduces width (used by collate)
    """
    TYPE: str = "ctc"
    TIME_REDUCTION: int = 8
    NAME: str = "base"

    def __init__(self, vocab: Vocab):
        super().__init__()
        self.vocab = vocab

    def forward(self, imgs: torch.Tensor) -> dict:
        raise NotImplementedError

    def compute_loss(self, out: dict, targets: torch.Tensor,
                     in_lens: torch.Tensor, tgt_lens: torch.Tensor
                     ) -> torch.Tensor:
        """Default: CTC loss. AR models override."""
        logits = out["logits"]                      # (T, B, C)
        logp   = logits.log_softmax(-1)
        T_max  = logp.size(0)
        in_lens = torch.clamp(in_lens, max=T_max)
        loss   = nn.functional.ctc_loss(
            logp, targets, in_lens, tgt_lens,
            blank=self.vocab.blank_idx, zero_infinity=False)
        return loss

    @torch.no_grad()
    def decode(self, out: dict) -> list[list[str]]:
        """Default: greedy CTC decode."""
        logits = out["logits"]                       # (T, B, C)
        preds  = logits.argmax(-1).transpose(0, 1)   # (B, T)
        return [self.vocab.ctc_decode(preds[b].tolist())
                for b in range(preds.size(0))]

    @torch.no_grad()
    def decode_beam(self, out: dict, beam_width: int = 10
                     ) -> list[list[str]]:
        """
        Beam-search CTC decoding (prefix beam search).
        Default falls back to greedy for AR models that override decode().
        """
        if self.TYPE != "ctc":
            return self.decode(out)
        logits = out["logits"]                        # (T, B, C)
        logp   = logits.log_softmax(-1).cpu().numpy()
        T, B, C = logp.shape
        results: list[list[str]] = []
        for b in range(B):
            results.append(_ctc_beam_search(logp[:, b, :],
                                              self.vocab,
                                              beam_width))
        return results

    @classmethod
    def list_models(cls) -> list[str]:
        return sorted(REGISTRY)


def _ctc_beam_search(logp, vocab, beam_width: int = 10) -> list[str]:
    import numpy as np
    """
    Prefix beam search for CTC.

    Maintains a beam of (prefix_tuple, log_prob_blank, log_prob_non_blank)
    where the two log probs are summed over alignments ending in blank
    or non-blank.  At each timestep, expand each prefix in the beam by
    each character in the vocab and update the two probs.

    Args:
        logp        : (T, C) log-softmax output for one item
        vocab       : Vocab; index 0 is blank
        beam_width  : how many prefixes to keep per timestep

    Returns:
        list of decoded tokens (no blanks, repeats collapsed implicitly)
    """
    import math
    NEG_INF = -1e30
    T, C = logp.shape
    blank = vocab.blank_idx

    # Beam: dict prefix -> (log_p_blank, log_p_nonblank)
    beam: dict[tuple, tuple[float, float]] = {(): (0.0, NEG_INF)}

    def logsumexp(a: float, b: float) -> float:
        if a == NEG_INF:
            return b
        if b == NEG_INF:
            return a
        m = max(a, b)
        return m + math.log(math.exp(a - m) + math.exp(b - m))

    # Probability threshold for exploration: skip chars whose log-prob
    # is more than this much below the argmax at the current timestep.
    # Without an LM, exploring distant low-prob chars only hurts.
    PRUNE_BELOW_TOP = math.log(0.05)   # 5% of argmax prob

    for t in range(T):
        next_beam: dict[tuple, tuple[float, float]] = {}
        # Threshold: top char's prob - PRUNE_BELOW_TOP defines the floor
        top_lp = float(logp[t].max())
        floor  = top_lp + PRUNE_BELOW_TOP
        # When blank is highly confident at a timestep, do not allow
        # exploration of non-blank chars (avoids trailing noise).
        if int(np.argmax(logp[t])) == blank and logp[t, blank] > math.log(0.7):
            topk_idx = [blank]
        else:
            topk_idx = [c for c in range(C) if logp[t, c] >= floor]
            if blank not in topk_idx:
                topk_idx.append(blank)
            topk_idx = sorted(topk_idx,
                               key=lambda c: -logp[t, c])[:beam_width * 3]
        for prefix, (lp_b, lp_nb) in beam.items():
            # 1. extend with blank -> prefix unchanged
            new_lp_b, new_lp_nb = next_beam.get(prefix, (NEG_INF, NEG_INF))
            new_lp_b = logsumexp(
                new_lp_b,
                logsumexp(lp_b, lp_nb) + logp[t, blank])
            next_beam[prefix] = (new_lp_b, new_lp_nb)

            # 2. extend with non-blank chars
            for c in topk_idx:
                if c == blank:
                    continue
                p_c = logp[t, c]
                if prefix and prefix[-1] == c:
                    # Sub-case 2a: prev was non-blank c, repeated c -> CTC
                    # collapse, prefix stays unchanged
                    new_prefix = prefix
                    cur_b, cur_nb = next_beam.get(new_prefix, (NEG_INF, NEG_INF))
                    cur_nb = logsumexp(cur_nb, lp_nb + p_c)
                    next_beam[new_prefix] = (cur_b, cur_nb)
                    # Sub-case 2b: prev was blank, emit a new c -> prefix grows
                    new_prefix2 = prefix + (c,)
                    cur_b2, cur_nb2 = next_beam.get(new_prefix2, (NEG_INF, NEG_INF))
                    cur_nb2 = logsumexp(cur_nb2, lp_b + p_c)
                    next_beam[new_prefix2] = (cur_b2, cur_nb2)
                else:
                    # Different char (or empty prefix): grow with c
                    new_prefix = prefix + (c,)
                    cur_b, cur_nb = next_beam.get(new_prefix, (NEG_INF, NEG_INF))
                    cur_nb = logsumexp(cur_nb,
                                        logsumexp(lp_b, lp_nb) + p_c)
                    next_beam[new_prefix] = (cur_b, cur_nb)

        # Prune to top beam_width by total prob (blank + non-blank)
        scored = [(p, logsumexp(lb, lnb))
                   for p, (lb, lnb) in next_beam.items()]
        scored.sort(key=lambda x: -x[1])
        beam = {p: next_beam[p] for p, _ in scored[:beam_width]}

    # Pick best prefix
    best = max(beam.items(),
                key=lambda kv: logsumexp(kv[1][0], kv[1][1]))
    best_prefix = best[0]
    return [vocab.itos[i] for i in best_prefix
             if 0 <= i < len(vocab.itos)]
