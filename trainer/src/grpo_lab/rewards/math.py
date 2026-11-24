"""Answer extraction and checking for GSM8K-style problems.

Completions put the final answer in ``\\boxed{...}``, the format Qwen instruct models are trained
on. Asking for a different format (we tried GSM8K's ``#### <n>``) cut base-model accuracy from
~44% to ~2%, because it pushes the model off its own output style.

These are pure functions so they can be unit-tested. In M9 they are deliberately weakened
to create proxy verifiers.
"""

from __future__ import annotations

import re

_BOXED = re.compile(r"\\(?:boxed|fbox)\s*\{")
_TEXT_CMD = re.compile(r"\\(?:text|textbf|mathrm|mbox)\s*\{[^{}]*\}")
_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def parse_number(text: str) -> float | None:
    try:
        return float(text.replace(",", "").strip())
    except ValueError:
        return None


def gold_answer(gsm8k_solution: str) -> str:
    """GSM8K reference solutions end with ``#### <answer>``."""
    return gsm8k_solution.split("####")[-1].strip().replace(",", "")


def extract_boxed(completion: str) -> str | None:
    """Contents of the *last* ``\\boxed{...}``, with nested braces matched; None if absent."""
    starts = list(_BOXED.finditer(completion))
    if not starts:
        return None
    i = starts[-1].end()
    depth = 1
    for j in range(i, len(completion)):
        if completion[j] == "{":
            depth += 1
        elif completion[j] == "}":
            depth -= 1
            if depth == 0:
                return completion[i:j]
    return None  # unclosed brace, e.g. truncated mid-answer


def extract_final_answer(completion: str) -> str | None:
    """The single number in the last ``\\boxed{}`` as a string; None unless there's exactly one.

    Units and currency are dropped (``\\boxed{\\$1,200 \\text{ dollars}}`` -> ``1200``). A box
    holding several numbers (``\\boxed{3 + 4 = 7}``) is ambiguous and gets no credit, so the
    policy can't hedge by boxing multiple candidates.
    """
    boxed = extract_boxed(completion)
    if boxed is None:
        return None
    cleaned = _THOUSANDS.sub("", _TEXT_CMD.sub(" ", boxed))
    numbers = _NUMBER.findall(cleaned)
    return numbers[0] if len(numbers) == 1 else None


def is_correct(completion: str, gold: str, tol: float = 1e-6) -> bool:
    pred = extract_final_answer(completion)
    if pred is None:
        return False
    p, g = parse_number(pred), parse_number(gold)
    return p is not None and g is not None and abs(p - g) <= tol


def has_final_answer_format(completion: str) -> bool:
    """A non-empty ``\\boxed{}`` is present (correctness is scored separately)."""
    boxed = extract_boxed(completion)
    return boxed is not None and boxed.strip() != ""
