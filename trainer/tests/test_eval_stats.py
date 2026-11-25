import itertools
import math

import numpy as np
import pytest

from grpo_lab.eval_stats import (
    Z_95,
    mcnemar_exact_p,
    mean_ci,
    paired_diff_ci,
    pass_at_k,
    seed_ci,
    t_critical_95,
)


def _pass_at_k_brute(n: int, c: int, k: int) -> float:
    """Fraction of all k-subsets of n samples (c correct) that contain a correct one."""
    samples = [True] * c + [False] * (n - c)
    subsets = list(itertools.combinations(range(n), k))
    return sum(any(samples[i] for i in s) for s in subsets) / len(subsets)


@pytest.mark.parametrize("n", [1, 4, 8])
def test_pass_at_k_matches_enumeration(n):
    for c in range(n + 1):
        for k in range(1, n + 1):
            assert pass_at_k(n, c, k) == pytest.approx(_pass_at_k_brute(n, c, k), abs=1e-12)


def test_pass_at_k_edge_cases():
    assert pass_at_k(16, 0, 16) == 0.0
    assert pass_at_k(16, 16, 1) == 1.0
    assert pass_at_k(16, 4, 1) == pytest.approx(4 / 16)  # pass@1 is the per-sample accuracy
    with pytest.raises(ValueError):
        pass_at_k(4, 5, 1)
    with pytest.raises(ValueError):
        pass_at_k(4, 2, 5)


def test_mean_ci_normal_approximation():
    x = [0, 1] * 50
    iv = mean_ci(x)
    se = np.std(x, ddof=1) / math.sqrt(len(x))
    assert iv.mean == 0.5
    assert iv.low == pytest.approx(0.5 - Z_95 * se)
    assert iv.high == pytest.approx(0.5 + Z_95 * se)
    assert iv.n == 100


def test_seed_ci_uses_student_t():
    iv = seed_ci([0.50, 0.54, 0.58])  # sd 0.04, n 3 -> half width 4.303 * 0.04 / sqrt(3)
    assert iv.mean == pytest.approx(0.54)
    assert iv.half_width == pytest.approx(4.303 * 0.04 / math.sqrt(3))
    with pytest.raises(ValueError):
        seed_ci([0.5])


def test_t_critical_table():
    assert t_critical_95(2) == 4.303
    assert t_critical_95(22) == 2.086  # between entries: falls back to df=20 (wider, conservative)
    assert t_critical_95(100) == Z_95
    with pytest.raises(ValueError):
        t_critical_95(0)


def test_paired_ci_is_tighter_than_separate_intervals():
    rng = np.random.default_rng(0)
    difficulty = rng.random(1000)
    a = (difficulty < 0.5).astype(float)  # base solves the easiest half
    b = (difficulty < 0.55).astype(float)  # trained also solves the next 5%
    diff = paired_diff_ci(a, b)
    assert diff.mean == pytest.approx(b.mean() - a.mean())
    assert diff.half_width < mean_ci(a).half_width / 2
    with pytest.raises(ValueError):
        paired_diff_ci([1.0, 0.0], [1.0])


def test_mcnemar_exact():
    a = [True] * 10 + [False] * 10
    assert mcnemar_exact_p(a, a) == 1.0  # no discordant problems
    # 1 problem only base solves, 9 only trained solves: p = 2 * P(X <= 1), X ~ Bin(10, 1/2)
    base = [True] + [False] * 9
    trained = [False] + [True] * 9
    assert mcnemar_exact_p(base, trained) == pytest.approx(2 * 11 / 1024)
    assert mcnemar_exact_p(trained, base) == pytest.approx(2 * 11 / 1024)  # two-sided


def test_mean_ci_bounds_clip_to_metric_range():
    x = [1.0] * 15 + [0.0]  # small sample near 100%: the raw normal interval exceeds 1
    assert mean_ci(x).high > 1.0
    assert mean_ci(x, bounds=(0.0, 1.0)).high == 1.0
