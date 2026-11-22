"""Answer extraction and checking for GSM8K-style problems.

These are pure functions so they can be unit-tested. In M9 they are deliberately weakened
to create proxy verifiers.
"""

from __future__ import annotations

import re

_NUMBER = r"-?\d[\d,]*(?:\.\d+)?"
_FINAL_ANSWER = re.compile(rf"####\s*\$?\s*({_NUMBER})")


def parse_number(text: str) -> float | None:
    try:
        return float(text.replace(",", "").strip())
    except ValueError:
        return None


def gold_answer(gsm8k_solution: str) -> str:
    """GSM8K solutions end with ``#### <answer>``."""
    return gsm8k_solution.split("####")[-1].strip().replace(",", "")


def extract_final_answer(completion: str) -> str | None:
    """The number after the *last* ``####`` marker, or None if there is none."""
    matches = _FINAL_ANSWER.findall(completion)
    return matches[-1] if matches else None


def is_correct(completion: str, gold: str, tol: float = 1e-6) -> bool:
    pred = extract_final_answer(completion)
    if pred is None:
        return False
    p, g = parse_number(pred), parse_number(gold)
    return p is not None and g is not None and abs(p - g) <= tol


def has_final_answer_format(completion: str) -> bool:
    return extract_final_answer(completion) is not None
