"""Per-token KL estimators between the policy and a frozen reference (Schulman, 2020).

Both use the log-probs of tokens *sampled from the policy*, so they are Monte Carlo estimates
of KL(policy || reference).
"""

from __future__ import annotations

import torch


def kl_k1(logp: torch.Tensor, ref_logp: torch.Tensor) -> torch.Tensor:
    """Naive estimator: unbiased, but high variance and can be negative per token."""
    return logp - ref_logp


def kl_k3(logp: torch.Tensor, ref_logp: torch.Tensor) -> torch.Tensor:
    """exp(d) - d - 1 with d = ref - logp. Unbiased, always >= 0, lower variance. Used by GRPO."""
    d = ref_logp - logp
    return torch.exp(d) - d - 1.0
