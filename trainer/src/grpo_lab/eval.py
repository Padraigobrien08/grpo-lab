"""Held-out evaluation: the clean "is the model actually getting better?" signal.

Training reward is noisy (sampled at T=1, fresh prompts each step) and can rise just because the
model learns the answer format. Greedy accuracy on a fixed held-out set can't, so it's what M2's
"accuracy goes up" is judged on. The M3 harness (``grpo_lab.evaluate``) builds on
``generate_and_score`` for full-test-set greedy accuracy, pass@k and CIs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from grpo_lab.models import autocast
from grpo_lab.rollout import generate_group_rollouts
from grpo_lab.tasks import Prompt, Task


@dataclass
class ScoredCompletions:
    """Per-completion results; completions for one prompt are contiguous (``n`` per prompt)."""

    n: int
    components: dict[str, list[float]] = field(default_factory=dict)
    lengths: list[int] = field(default_factory=list)
    truncated: list[bool] = field(default_factory=list)
    token_ids: list[list[int]] = field(default_factory=list)


@torch.no_grad()
def generate_and_score(
    model: torch.nn.Module,
    task: Task,
    prompts: list[Prompt],
    *,
    n: int = 1,
    greedy: bool = True,
    temperature: float = 1.0,
    batch_size: int,
    max_new_tokens: int,
    amp_dtype: str | None = None,
    progress: bool = False,
) -> ScoredCompletions:
    """Generate ``n`` completions per prompt (greedy: use ``n=1``) and score each one.

    ``batch_size`` counts completions per ``generate`` call, so a chunk holds
    ``batch_size // n`` prompts (at least one).
    """
    if greedy and n != 1:
        raise ValueError("greedy decoding is deterministic: use n=1")
    device = next(model.parameters()).device
    per_chunk = max(1, batch_size // n)
    out = ScoredCompletions(n=n)
    for start in range(0, len(prompts), per_chunk):
        chunk = prompts[start : start + per_chunk]
        with autocast(device, amp_dtype):
            batch = generate_group_rollouts(
                model,
                [p.input_ids for p in chunk],
                group_size=n,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                pad_id=task.pad_id,
                eos_ids=task.eos_ids,
                greedy=greedy,
            )
        completions = batch.completion_token_lists()
        repeated = [p for p in chunk for _ in range(n)]
        for k, v in task.score(repeated, completions).items():
            out.components.setdefault(k, []).extend(v)
        out.lengths.extend(len(c) for c in completions)
        out.truncated.extend(batch.truncated.tolist())
        out.token_ids.extend(completions)
        if progress:
            done = min(start + per_chunk, len(prompts))
            print(f"  {done}/{len(prompts)} prompts", flush=True)
    return out


@torch.no_grad()
def greedy_eval(
    model: torch.nn.Module,
    task: Task,
    prompts: list[Prompt],
    *,
    batch_size: int,
    max_new_tokens: int,
    amp_dtype: str | None = None,
) -> dict[str, float]:
    """Greedy-decode each prompt once; return mean reward components and length stats."""
    scored = generate_and_score(
        model,
        task,
        prompts,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        amp_dtype=amp_dtype,
    )
    metrics = {f"eval/{k}": float(np.mean(v)) for k, v in scored.components.items()}
    if task.gold_component:
        metrics["eval/accuracy"] = metrics[f"eval/{task.gold_component}"]
    metrics["eval/len_mean"] = float(np.mean(scored.lengths))
    metrics["eval/truncated_frac"] = float(np.mean(scored.truncated))
    return metrics
