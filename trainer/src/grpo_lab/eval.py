"""Held-out greedy evaluation: the clean "is the model actually getting better?" signal.

Training reward is noisy (sampled at T=1, fresh prompts each step) and can rise just because the
model learns the answer format. Greedy accuracy on a fixed held-out set can't, so it's what M2's
"accuracy goes up" is judged on. M3 extends this (pass@k, full test set, CIs across seeds).
"""

from __future__ import annotations

import numpy as np
import torch

from grpo_lab.models import autocast
from grpo_lab.rollout import generate_group_rollouts
from grpo_lab.tasks import Prompt, Task


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
    device = next(model.parameters()).device
    components: dict[str, list[float]] = {}
    lengths: list[int] = []
    truncated: list[bool] = []
    for start in range(0, len(prompts), batch_size):
        chunk = prompts[start : start + batch_size]
        with autocast(device, amp_dtype):
            batch = generate_group_rollouts(
                model,
                [p.input_ids for p in chunk],
                group_size=1,
                max_new_tokens=max_new_tokens,
                temperature=1.0,
                pad_id=task.pad_id,
                eos_ids=task.eos_ids,
                greedy=True,
            )
        completions = batch.completion_token_lists()
        for k, v in task.score(chunk, completions).items():
            components.setdefault(k, []).extend(v)
        lengths.extend(len(c) for c in completions)
        truncated.extend(batch.truncated.tolist())

    metrics = {f"eval/{k}": float(np.mean(v)) for k, v in components.items()}
    if task.gold_component:
        metrics["eval/accuracy"] = metrics[f"eval/{task.gold_component}"]
    metrics["eval/len_mean"] = float(np.mean(lengths))
    metrics["eval/truncated_frac"] = float(np.mean(truncated))
    return metrics
