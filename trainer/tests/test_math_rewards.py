import pytest

from grpo_lab.rewards.math import extract_final_answer, gold_answer, is_correct


def test_gold_answer():
    assert gold_answer("She has 3 + 4 = 7 apples.\n#### 1,234") == "1234"


@pytest.mark.parametrize(
    ("completion", "expected"),
    [
        ("2 + 2 = 4\n#### 4", "4"),
        ("#### 3 then actually\n#### 5", "5"),  # last marker wins
        ("#### $1,200", "1,200"),
        ("#### -3.5", "-3.5"),
        ("the answer is 4", None),
    ],
)
def test_extract_final_answer(completion, expected):
    assert extract_final_answer(completion) == expected


def test_is_correct_normalises_numbers():
    assert is_correct("#### 1,200", "1200")
    assert is_correct("#### 7.0", "7")
    assert not is_correct("#### 8", "7")
    assert not is_correct("7", "7")  # no format marker, no credit
