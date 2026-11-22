"""Group-relative advantages (GRPO, DeepSeekMath 2024; Dr. GRPO variant, Liu et al. 2025)."""

from __future__ import annotations

from typing import Literal

import torch

AdvantageScale = Literal["std", "none"]


def group_advantages(
    rewards: torch.Tensor,
    group_size: int,
    scale: AdvantageScale = "std",
    eps: float = 1e-4,
) -> torch.Tensor:
    """Advantage of each sample relative to the other samples for the same prompt.

    ``rewards`` is flat, shape ``(num_prompts * group_size,)``, with each group contiguous.

    - ``scale="std"``: (r - mean) / (std + eps). This is vanilla GRPO.
    - ``scale="none"``: r - mean. This is Dr. GRPO. Dividing by std up-weights
      near-uniform groups (very easy or very hard prompts), which biases learning.

    std is the unbiased estimate, to match TRL for the parity check.
    """
    if rewards.ndim != 1 or rewards.numel() % group_size != 0:
        raise ValueError(f"rewards must be flat with length divisible by {group_size}")
    grouped = rewards.view(-1, group_size)
    centered = grouped - grouped.mean(dim=1, keepdim=True)
    if scale == "std":
        centered = centered / (grouped.std(dim=1, keepdim=True) + eps)
    elif scale != "none":
        raise ValueError(f"unknown advantage scale: {scale}")
    return centered.reshape(-1)


def zero_advantage_fraction(rewards: torch.Tensor, group_size: int) -> float:
    """Fraction of groups where every sample got the same reward, so the group gives no signal.

    This is a key GRPO diagnostic. Near 1.0 means the task is too easy or too hard for the policy,
    and compute is wasted. DAPO's dynamic sampling filters these groups out.
    """
    grouped = rewards.view(-1, group_size)
    return (grouped.max(dim=1).values == grouped.min(dim=1).values).float().mean().item()
