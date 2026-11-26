"""Per-token traces for a few rollout groups: what the dashboard's token heatmap shows.

The values come from tensors the update already computes, copied to the CPU for the traced
rows only, so tracing adds no forward passes. Traced rows are the first ``trace_groups``
groups of a step: prompts are sampled at random, so this is an unbiased sample of groups
(preferring "interesting" groups would bias anything computed from traces later).
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from grpo_lab.algo.kl import kl_k3


class TraceBuffer:
    """Collects per-token values for rows ``[0, n)`` of a batch across micro-batches.

    - ``logprob`` and ``entropy`` are taken on the first optimisation pass, when the policy is
      still the one that sampled the tokens.
    - ``ratio`` is taken on the last pass (pi / pi_old). With one pass it is exactly 1.
    """

    def __init__(self, n_rows: int, completion_len: int) -> None:
        self.n = n_rows
        shape = (n_rows, completion_len)
        self.logprob = torch.zeros(shape)
        self.entropy = torch.zeros(shape)
        self.ratio = torch.ones(shape)

    def _overlap(self, mb: slice) -> tuple[slice, slice] | None:
        """(rows in the buffer, rows in the micro-batch) covered by ``mb``, if any."""
        lo, hi = mb.start, min(mb.stop, self.n)
        if lo >= hi:
            return None
        return slice(lo, hi), slice(0, hi - lo)

    def record_first_pass(self, mb: slice, logprob: torch.Tensor, entropy: torch.Tensor) -> None:
        if (rows := self._overlap(mb)) is not None:
            dst, src = rows
            self.logprob[dst] = logprob[src].detach().float().cpu()
            self.entropy[dst] = entropy[src].detach().float().cpu()

    def record_last_pass(self, mb: slice, logprob: torch.Tensor, old_logprob: torch.Tensor) -> None:
        if (rows := self._overlap(mb)) is not None:
            dst, src = rows
            self.ratio[dst] = torch.exp(logprob[src] - old_logprob[src]).detach().float().cpu()

    def rows(
        self,
        prompt_ids: list[str],
        completions: list[list[int]],
        group_size: int,
        decode_tokens: Callable[[list[int]], list[str]],
        ref_logprob: torch.Tensor | None = None,
    ) -> list[dict]:
        """One run-store row per traced sample, cut to its completion length (EOS included).

        ``prompt_ids`` has one entry per group; ``ref_logprob`` covers the whole batch (rows
        beyond ``n`` are ignored). KL is the k3 estimator between the sampling policy and the
        reference, which is the value the loss used on the first pass.
        """
        ref = ref_logprob[: self.n].detach().float().cpu() if ref_logprob is not None else None
        kl = kl_k3(self.logprob, ref) if ref is not None else None
        out = []
        for i in range(self.n):
            ids = completions[i]
            n_tok = len(ids)
            out.append(
                {
                    "prompt_id": prompt_ids[i // group_size],
                    "group_idx": i // group_size,
                    "sample_idx": i % group_size,
                    "token_ids": ids,
                    "tokens": decode_tokens(ids),
                    "logprob": self.logprob[i, :n_tok].tolist(),
                    "ref_logprob": ref[i, :n_tok].tolist() if ref is not None else None,
                    "kl": kl[i, :n_tok].tolist() if kl is not None else None,
                    "entropy": self.entropy[i, :n_tok].tolist(),
                    "ratio": self.ratio[i, :n_tok].tolist(),
                }
            )
        return out
