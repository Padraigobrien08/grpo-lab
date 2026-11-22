"""Clipped policy-gradient loss with configurable token aggregation.

Aggregation is the main thing that differs between GRPO variants:

- ``sequence_mean`` (vanilla GRPO): average tokens within each sequence, then average sequences.
  Each token in a long sequence gets less weight, which biases toward long wrong answers.
- ``token_mean`` (DAPO): average over every completion token in the batch.
- ``constant`` (Dr. GRPO): sum tokens, then divide by a fixed ``num_sequences * max_len``,
  so the loss scale doesn't depend on the lengths the policy happens to produce.

Normalisers are passed in from the *full* batch, so summing micro-batch losses gives exactly
the full-batch loss. Normalising per micro-batch silently changes the objective under
gradient accumulation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from grpo_lab.algo.kl import kl_k3

Aggregation = Literal["sequence_mean", "token_mean", "constant"]


@dataclass
class LossOutput:
    loss: torch.Tensor
    # Sums, not means, so they can be accumulated across micro-batches and divided once.
    stat_sums: dict[str, float]
    num_tokens: float


def aggregate(
    per_token: torch.Tensor,
    mask: torch.Tensor,
    aggregation: Aggregation,
    *,
    num_sequences_total: int | None = None,
    num_tokens_total: float | None = None,
    max_completion_len: int | None = None,
) -> torch.Tensor:
    mask = mask.to(per_token.dtype)
    n_seq = num_sequences_total if num_sequences_total is not None else per_token.shape[0]
    masked = per_token * mask
    if aggregation == "sequence_mean":
        per_seq = masked.sum(-1) / mask.sum(-1).clamp(min=1.0)
        return per_seq.sum() / n_seq
    if aggregation == "token_mean":
        n_tok = num_tokens_total if num_tokens_total is not None else mask.sum().item()
        return masked.sum() / max(n_tok, 1.0)
    if aggregation == "constant":
        if max_completion_len is None:
            raise ValueError("constant aggregation needs max_completion_len")
        return masked.sum() / (n_seq * max_completion_len)
    raise ValueError(f"unknown aggregation: {aggregation}")


def grpo_loss(
    logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    *,
    ref_logprobs: torch.Tensor | None = None,
    beta: float = 0.0,
    clip_eps_low: float = 0.2,
    clip_eps_high: float = 0.2,
    aggregation: Aggregation = "sequence_mean",
    num_sequences_total: int | None = None,
    num_tokens_total: float | None = None,
    max_completion_len: int | None = None,
) -> LossOutput:
    """PPO-style clipped surrogate with sequence-level advantages broadcast to every token.

    Shapes: ``logprobs``, ``old_logprobs``, ``ref_logprobs``, ``mask``: ``(B, T)``;
    ``advantages``: ``(B,)``. Setting ``clip_eps_high > clip_eps_low`` is DAPO's "clip-higher",
    which leaves more room to raise the probability of rare tokens and slows entropy collapse.
    """
    maskf = mask.to(logprobs.dtype)
    ratio = torch.exp(logprobs - old_logprobs)
    adv = advantages.to(logprobs.dtype).unsqueeze(-1)
    surr_unclipped = ratio * adv
    surr_clipped = torch.clamp(ratio, 1.0 - clip_eps_low, 1.0 + clip_eps_high) * adv
    per_token = -torch.minimum(surr_unclipped, surr_clipped)

    kl = None
    if beta != 0.0:
        if ref_logprobs is None:
            raise ValueError("beta != 0 requires ref_logprobs")
        kl = kl_k3(logprobs, ref_logprobs)
        per_token = per_token + beta * kl

    loss = aggregate(
        per_token,
        mask,
        aggregation,
        num_sequences_total=num_sequences_total,
        num_tokens_total=num_tokens_total,
        max_completion_len=max_completion_len,
    )

    with torch.no_grad():
        clipped_low = (ratio < 1.0 - clip_eps_low) & (adv < 0)
        clipped_high = (ratio > 1.0 + clip_eps_high) & (adv > 0)
        stat_sums = {
            "clip_frac_low": (clipped_low.to(maskf.dtype) * maskf).sum().item(),
            "clip_frac_high": (clipped_high.to(maskf.dtype) * maskf).sum().item(),
            "ratio": (ratio * maskf).sum().item(),
        }
        if kl is not None:
            stat_sums["kl"] = (kl * maskf).sum().item()
    return LossOutput(loss=loss, stat_sums=stat_sums, num_tokens=maskf.sum().item())
