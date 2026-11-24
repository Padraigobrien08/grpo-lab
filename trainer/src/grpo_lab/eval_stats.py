"""Statistics for the M3 eval: pass@k, confidence intervals, paired and across-seed comparisons.

Plain numpy, no scipy: everything here is a closed form or a small lookup table.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

Z_95 = 1.959963984540054

# Two-sided 95% Student-t critical values t_{0.975, df}. Seeds are few, so df stays small.
_T_95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262,
    10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110,
    18: 2.101, 19: 2.093, 20: 2.086, 25: 2.060, 30: 2.042,
}  # fmt: skip


def t_critical_95(df: int) -> float:
    """t_{0.975, df}; conservative (the next smaller tabulated df) between table entries."""
    if df < 1:
        raise ValueError("need at least 2 observations (df >= 1) for a t interval")
    if df > 30:
        return Z_95
    return _T_95[max(d for d in _T_95 if d <= df)]


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k from ``c`` correct out of ``n`` samples (Chen et al. 2021, Codex paper).

    1 - C(n-c, k) / C(n, k): the chance that k samples drawn without replacement from the n
    include at least one correct one. Computed as a product to avoid huge binomials.
    """
    if not 0 <= c <= n:
        raise ValueError(f"need 0 <= c <= n, got c={c}, n={n}")
    if not 1 <= k <= n:
        raise ValueError(f"need 1 <= k <= n, got k={k}, n={n}")
    if n - c < k:
        return 1.0
    return 1.0 - float(np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))


@dataclass(frozen=True)
class Interval:
    mean: float
    low: float
    high: float
    n: int

    @property
    def half_width(self) -> float:
        return (self.high - self.low) / 2

    def as_dict(self) -> dict[str, float | int]:
        return {"mean": self.mean, "low": self.low, "high": self.high, "n": self.n}


def mean_ci(values: Sequence[float], bounds: tuple[float, float] | None = None) -> Interval:
    """Mean with a normal-approximation 95% CI (problems are many, so the CLT holds).

    ``bounds`` clips the interval to the metric's range: on small subsets the normal
    approximation can otherwise report an accuracy above 100%.
    """
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        raise ValueError("no values")
    mean = float(x.mean())
    se = float(x.std(ddof=1) / math.sqrt(x.size)) if x.size > 1 else 0.0
    low, high = mean - Z_95 * se, mean + Z_95 * se
    if bounds is not None:
        low, high = max(low, bounds[0]), min(high, bounds[1])
    return Interval(mean, low, high, int(x.size))


def seed_ci(values: Sequence[float]) -> Interval:
    """Mean across seeds with a Student-t 95% CI (wide with 3 seeds: t = 4.30)."""
    x = np.asarray(values, dtype=float)
    if x.size < 2:
        raise ValueError("need at least 2 seeds for a CI across seeds")
    mean = float(x.mean())
    half = t_critical_95(x.size - 1) * float(x.std(ddof=1)) / math.sqrt(x.size)
    return Interval(mean, mean - half, mean + half, int(x.size))


def paired_diff_ci(a: Sequence[float], b: Sequence[float]) -> Interval:
    """CI for mean(b - a) over the same problems.

    Pairing removes the between-problem variance (an easy problem is easy for both models), so
    this is much tighter than comparing two separate per-model intervals.
    """
    a_arr, b_arr = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a_arr.shape != b_arr.shape:
        raise ValueError("paired comparison needs the same problems in the same order")
    return mean_ci(b_arr - a_arr, bounds=(-1.0, 1.0))


def mcnemar_exact_p(a_correct: Sequence[bool], b_correct: Sequence[bool]) -> float:
    """Two-sided exact McNemar test on paired binary outcomes.

    Only discordant problems carry information: under "no difference", each one is equally
    likely to favour either model, so the count favouring b is Binomial(discordant, 1/2).
    """
    a_arr, b_arr = np.asarray(a_correct, dtype=bool), np.asarray(b_correct, dtype=bool)
    if a_arr.shape != b_arr.shape:
        raise ValueError("paired comparison needs the same problems in the same order")
    b_only = int(np.sum(~a_arr & b_arr))
    a_only = int(np.sum(a_arr & ~b_arr))
    m = a_only + b_only
    if m == 0:
        return 1.0
    tail = sum(math.comb(m, i) for i in range(min(a_only, b_only) + 1)) / 2**m
    return min(1.0, 2 * tail)
