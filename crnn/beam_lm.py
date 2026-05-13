"""
LM-aware CTC prefix beam search.

Identical math to `crnn.models._base._ctc_beam_search` except every
*prefix extension* (i.e. every step that adds a new non-blank token)
is rescored by `alpha * log P_LM(token | prefix history)`. A length
bonus `beta * len(prefix)` discourages the LM from collapsing onto
short, high-probability prefixes.

Final scoring per prefix:
    score = log P_acoustic + alpha * log P_LM + beta * len

The acoustic component still tracks the (lp_blank, lp_nonblank) pair
exactly like the lm-less beam, so this falls back to the no-LM
variant when alpha == 0.0 and beta == 0.0.
"""
from __future__ import annotations
import math
from typing import Sequence

import numpy as np

from .vocab import Vocab
from .lm    import NGramLM


NEG_INF = -1e30


def _logsumexp(a: float, b: float) -> float:
    if a == NEG_INF:
        return b
    if b == NEG_INF:
        return a
    m = max(a, b)
    return m + math.log(math.exp(a - m) + math.exp(b - m))


def beam_search_lm(logp: np.ndarray, vocab: Vocab, *,
                   lm: NGramLM | None = None,
                   alpha: float = 0.5,
                   beta: float = 0.0,
                   beam_width: int = 20,
                   prune_below_top: float | None = None) -> list[str]:
    """
    Args:
        logp        : (T, C) log-softmax for one item
        vocab       : Vocab; index 0 is blank
        lm          : NGramLM (or None; alpha=0 also disables LM)
        alpha       : LM weight added to each prefix extension's score
        beta        : length bonus added per emitted token
        beam_width  : # prefixes kept per timestep
        prune_below_top : log-prob floor relative to per-step argmax

    Returns:
        list of decoded tokens (no blanks, repeats already collapsed)
    """
    T, C = logp.shape
    blank = vocab.blank_idx

    if prune_below_top is None:
        prune_below_top = math.log(0.05)   # 5% of argmax

    # beam: prefix (tuple of token-strs) -> (lp_blank, lp_nonblank, lm_score)
    beam: dict[tuple, tuple[float, float, float]] = {(): (0.0, NEG_INF, 0.0)}

    use_lm = lm is not None and alpha != 0.0

    for t in range(T):
        next_beam: dict[tuple, tuple[float, float, float]] = {}
        top_lp = float(logp[t].max())
        floor  = top_lp + prune_below_top

        # Restrict explored chars: when blank dominates, keep only blank.
        if int(np.argmax(logp[t])) == blank and logp[t, blank] > math.log(0.7):
            topk_idx = [blank]
        else:
            topk_idx = [c for c in range(C) if logp[t, c] >= floor]
            if blank not in topk_idx:
                topk_idx.append(blank)
            topk_idx = sorted(topk_idx,
                               key=lambda c: -logp[t, c])[:beam_width * 3]

        for prefix, (lp_b, lp_nb, lm_acc) in beam.items():
            # 1. extend with blank -> prefix unchanged
            new_lp_b, new_lp_nb, new_lm = next_beam.get(prefix,
                                                         (NEG_INF, NEG_INF, 0.0))
            new_lp_b = _logsumexp(
                new_lp_b,
                _logsumexp(lp_b, lp_nb) + logp[t, blank])
            next_beam[prefix] = (new_lp_b, new_lp_nb, lm_acc)

            # 2. non-blank extensions
            for c in topk_idx:
                if c == blank:
                    continue
                p_c = float(logp[t, c])
                tok = vocab.itos[c]

                if prefix and prefix[-1] == tok:
                    # Sub-case 2a: repeated tok with non-blank in between
                    # collapses to same prefix (no new token emitted -> LM
                    # score unchanged).
                    cur_b, cur_nb, cur_lm = next_beam.get(
                        prefix, (NEG_INF, NEG_INF, lm_acc))
                    cur_nb = _logsumexp(cur_nb, lp_nb + p_c)
                    next_beam[prefix] = (cur_b, cur_nb, cur_lm)

                    # Sub-case 2b: prev was blank, emit a fresh token c
                    new_prefix = prefix + (tok,)
                    if use_lm:
                        lm_step = lm.score_token(tok, prefix)
                        new_lm  = lm_acc + alpha * lm_step
                    else:
                        new_lm  = lm_acc
                    cur_b2, cur_nb2, cur_lm2 = next_beam.get(
                        new_prefix, (NEG_INF, NEG_INF, new_lm))
                    cur_nb2 = _logsumexp(cur_nb2, lp_b + p_c)
                    # Use the smallest (most pessimistic) lm_acc when
                    # multiple paths land on the same prefix; in practice
                    # they will be the same value since LM is deterministic.
                    next_beam[new_prefix] = (cur_b2, cur_nb2,
                                              max(cur_lm2, new_lm))
                else:
                    # Different token: grow with tok
                    new_prefix = prefix + (tok,)
                    if use_lm:
                        lm_step = lm.score_token(tok, prefix)
                        new_lm  = lm_acc + alpha * lm_step
                    else:
                        new_lm  = lm_acc
                    cur_b, cur_nb, cur_lm = next_beam.get(
                        new_prefix, (NEG_INF, NEG_INF, new_lm))
                    cur_nb = _logsumexp(cur_nb,
                                         _logsumexp(lp_b, lp_nb) + p_c)
                    next_beam[new_prefix] = (cur_b, cur_nb,
                                              max(cur_lm, new_lm))

        # Prune to beam_width by total combined score
        scored = []
        for p, (lb, lnb, lm_s) in next_beam.items():
            acoustic = _logsumexp(lb, lnb)
            combined = acoustic + lm_s + beta * len(p)
            scored.append((p, combined))
        scored.sort(key=lambda x: -x[1])
        beam = {p: next_beam[p] for p, _ in scored[:beam_width]}

    # Pick best prefix by full combined score (includes EOS LM score)
    best_pref, best_score = (), -1e30
    for p, (lb, lnb, lm_s) in beam.items():
        acoustic = _logsumexp(lb, lnb)
        eos_score = (alpha * lm.score_token(NGramLM.EOS, p)) if use_lm else 0.0
        combined = acoustic + lm_s + eos_score + beta * len(p)
        if combined > best_score:
            best_score = combined
            best_pref  = p
    return list(best_pref)
