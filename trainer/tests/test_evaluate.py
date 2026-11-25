"""End-to-end: the eval CLI scores a tiny random model and compares runs (CPU, seconds)."""

import json
from pathlib import Path

import pytest

from grpo_lab.evaluate import compare, load_results, main

TOY = str(Path(__file__).parents[1] / "configs" / "toy.yaml")


def _run(out_dir: Path, label: str, *extra: str) -> Path:
    main(
        [
            "run",
            "--config",
            TOY,
            "--num-examples",
            "8",
            "--samples",
            "4",
            "--k",
            "1,4",
            "--batch-size",
            "16",
            "--device",
            "cpu",
            "--out-dir",
            str(out_dir),
            "--label",
            label,
            *extra,
        ]
    )
    return out_dir / label


def test_run_writes_per_problem_results(tmp_path):
    rows, summary = load_results(_run(tmp_path, "a"))

    assert len(rows) == 8 and summary["num_problems"] == 8
    assert len({r["id"] for r in rows}) == 8
    for r in rows:
        assert r["n_samples"] == 4 and 0 <= r["n_correct"] <= 4
        assert r["pass@1"] == pytest.approx(r["n_correct"] / 4)
        assert r["pass@4"] == float(r["n_correct"] > 0)
        assert r["greedy_correct"] in (0.0, 1.0)
        assert isinstance(r["greedy_text"], str)
    m = summary["metrics"]
    assert m["greedy_accuracy"]["n"] == 8
    assert m["pass@1"]["mean"] <= m["pass@4"]["mean"]
    assert summary["settings"]["ks"] == [1, 4]


def test_run_refuses_to_overwrite(tmp_path):
    _run(tmp_path, "a", "--samples", "0")
    with pytest.raises(SystemExit):
        _run(tmp_path, "a", "--samples", "0")


def test_compare_same_model_has_zero_difference(tmp_path, capsys):
    # Same seeded model and sampling seed: identical results, so every paired diff is exactly 0.
    base, run_a, run_b = (_run(tmp_path, name) for name in ("base", "a", "b"))
    out = tmp_path / "cmp.json"
    main(["compare", str(base), str(run_a), str(run_b), "--json", str(out)])

    result = json.loads(out.read_text())
    assert result["num_problems"] == 8
    for e in result["runs"]:
        assert e["greedy_correct_diff"]["mean"] == 0.0
        assert e["pass@4_diff"]["mean"] == 0.0
        assert e["mcnemar_p"] == 1.0
    assert set(result["across_seeds"]) == {"greedy_correct", "pass@1", "pass@4"}
    assert "across 2 seeds" in capsys.readouterr().out


def test_compare_rejects_different_problem_sets(tmp_path):
    base = _run(tmp_path, "base")
    other = _run(tmp_path, "other", "--num-examples", "6")
    with pytest.raises(ValueError, match="different problems"):
        compare(base, [other])
