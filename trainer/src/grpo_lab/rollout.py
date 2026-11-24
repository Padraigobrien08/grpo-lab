"""Group sampling with HF ``generate``. M6 adds a vLLM engine behind the same output type."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from transformers import PreTrainedModel


@dataclass
class RolloutBatch:
    input_ids: torch.Tensor  # (N, P + C): left-padded prompt followed by completion
    attention_mask: torch.Tensor  # (N, P + C): 0 on left padding and after EOS
    completion_mask: torch.Tensor  # (N, C): 1 on generated tokens up to and including EOS
    prompt_len: int  # P: all prompts end at the same column thanks to left padding
    truncated: torch.Tensor  # (N,) bool: hit max_new_tokens without emitting EOS

    @property
    def completions(self) -> torch.Tensor:
        return self.input_ids[:, self.prompt_len :]

    def completion_token_lists(self) -> list[list[int]]:
        return [
            row[m.bool()].tolist()
            for row, m in zip(self.completions, self.completion_mask, strict=True)
        ]


def left_pad(
    seqs: Sequence[Sequence[int]], pad_id: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), width), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, width - len(s) :] = torch.tensor(s, dtype=torch.long)
        mask[i, width - len(s) :] = 1
    return ids.to(device), mask.to(device)


def completion_mask_from_eos(
    completions: torch.Tensor, eos_ids: Sequence[int]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mask covering tokens up to and including the first EOS, plus a per-row truncated flag."""
    n, c = completions.shape
    if not eos_ids:
        return torch.ones_like(completions), torch.zeros(
            n, dtype=torch.bool, device=completions.device
        )
    is_eos = torch.isin(completions, torch.tensor(list(eos_ids), device=completions.device))
    has_eos = is_eos.any(dim=1)
    first_eos = torch.where(
        has_eos, is_eos.int().argmax(dim=1), torch.full_like(has_eos, c, dtype=torch.long)
    )
    positions = torch.arange(c, device=completions.device).expand(n, c)
    mask = (positions <= first_eos.unsqueeze(1)).long()
    return mask, ~has_eos


@torch.no_grad()
def generate_group_rollouts(
    model: PreTrainedModel,
    prompt_ids: Sequence[Sequence[int]],
    *,
    group_size: int,
    max_new_tokens: int,
    temperature: float,
    pad_id: int,
    eos_ids: Sequence[int],
    greedy: bool = False,
) -> RolloutBatch:
    """Sample ``group_size`` completions per prompt; rows are grouped contiguously.

    ``greedy=True`` decodes deterministically instead (held-out eval; use ``group_size=1``).
    """
    device = next(model.parameters()).device
    repeated = [p for p in prompt_ids for _ in range(group_size)]
    ids, mask = left_pad(repeated, pad_id, device)
    prompt_len = ids.shape[1]

    was_training = model.training
    model.eval()
    # Pass every sampling knob explicitly. Instruct checkpoints ship generation_config defaults
    # (top_k, top_p, repetition_penalty) that would make the sampling distribution differ from
    # the policy we compute log-probs for, and the on-policy assumption would break silently.
    sampling = (
        dict(do_sample=False, temperature=None, top_k=None, top_p=None)
        if greedy
        else dict(do_sample=True, temperature=temperature, top_k=0, top_p=1.0)
    )
    out = model.generate(
        input_ids=ids,
        attention_mask=mask,
        **sampling,
        min_p=None,
        repetition_penalty=1.0,
        max_new_tokens=max_new_tokens,
        pad_token_id=pad_id,
        eos_token_id=list(eos_ids) if eos_ids else None,
    )
    model.train(was_training)

    completions = out[:, prompt_len:]
    completion_mask, truncated = completion_mask_from_eos(completions, eos_ids)
    return RolloutBatch(
        input_ids=out,
        attention_mask=torch.cat([mask, completion_mask], dim=1),
        completion_mask=completion_mask,
        prompt_len=prompt_len,
        truncated=truncated,
    )
