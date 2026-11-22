"""Per-token log-probs of completion tokens under a causal LM."""

from __future__ import annotations

import torch
from transformers import PreTrainedModel


def position_ids_from_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    """Positions that skip left padding.

    Without these, a left-padded row is scored at the wrong positions, and its log-probs
    disagree with what ``generate`` sampled. The importance ratio is then wrong from step one.
    """
    return (attention_mask.long().cumsum(-1) - 1).clamp(min=0)


def completion_logprobs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_len: int,
    temperature: float = 1.0,
    return_entropy: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Log-prob (and optionally entropy) of each completion token, shape ``(B, T - prompt_len)``.

    Logits are divided by the sampling temperature, so the log-probs are those of the
    distribution the tokens were actually sampled from.
    """
    completion_len = input_ids.shape[1] - prompt_len
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids_from_mask(attention_mask),
        use_cache=False,
        # Only materialise logits for completion positions: avoids a (B, T, 150k) tensor.
        logits_to_keep=completion_len + 1,
    )
    logits = out.logits[:, :-1, :].float() / temperature  # position t predicts token t+1
    targets = input_ids[:, prompt_len:]
    log_z = torch.logsumexp(logits, dim=-1)
    logp = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1) - log_z
    entropy = None
    if return_entropy:
        with torch.no_grad():
            probs = torch.softmax(logits, dim=-1)
            entropy = log_z - (probs * logits).sum(-1)
    return logp, entropy
