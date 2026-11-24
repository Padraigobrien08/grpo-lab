import pytest

from grpo_lab.rewards.math import (
    extract_boxed,
    extract_final_answer,
    gold_answer,
    has_final_answer_format,
    is_correct,
)


def test_gold_answer():
    assert gold_answer("She has 3 + 4 = 7 apples.\n#### 1,234") == "1234"


@pytest.mark.parametrize(
    ("completion", "expected"),
    [
        (r"so \boxed{4}.", "4"),
        (r"\boxed{\frac{1}{2}}", r"\frac{1}{2}"),  # nested braces matched
        (r"first \boxed{3}, actually \boxed{5}", "5"),  # last box wins
        (r"\boxed {7}", "7"),
        (r"\boxed{12", None),  # truncated mid-answer
        ("the answer is 4", None),
    ],
)
def test_extract_boxed(completion, expected):
    assert extract_boxed(completion) == expected


@pytest.mark.parametrize(
    ("completion", "expected"),
    [
        (r"\boxed{4}", "4"),
        (r"\boxed{1,200}", "1200"),
        (r"\boxed{\$1,200}", "1200"),
        (r"$\boxed{150}$ dollars", "150"),
        (r"\boxed{150 \text{ dollars}}", "150"),
        (r"\boxed{-3.5}", "-3.5"),
        (r"\boxed{x = 5}", "5"),
        (r"\boxed{3 + 4 = 7}", None),  # several numbers: ambiguous, no credit
        (r"\boxed{1, 2, 3}", None),  # a list is not a thousands separator
        (r"\boxed{\text{seven}}", None),
        ("#### 4", None),  # GSM8K's own marker is no longer the answer format
    ],
)
def test_extract_final_answer(completion, expected):
    assert extract_final_answer(completion) == expected


def test_is_correct_normalises_numbers():
    assert is_correct(r"\boxed{1,200}", "1200")
    assert is_correct(r"\boxed{7.0}", "7")
    assert not is_correct(r"\boxed{8}", "7")
    assert not is_correct("the answer is 7", "7")  # no box, no credit


def test_format_needs_nonempty_box():
    assert has_final_answer_format(r"\boxed{seven}")
    assert not has_final_answer_format(r"\boxed{ }")
    assert not has_final_answer_format("7")
