import json

import pytest

from runstore import ROLLOUT_COLUMNS, RunStore, merge_store
from runstore.cli import main as cli_main


def _row(i: int) -> dict:
    return {
        "prompt_id": "p0",
        "group_idx": 0,
        "sample_idx": i,
        "prompt": "q",
        "completion": f"a{i}",
        "reward_total": float(i),
        "reward_components": json.dumps({"correct": float(i)}),
        "gold_reward": None,
        "advantage": 0.5,
        "length": 3,
        "truncated": False,
    }


def test_run_lifecycle_and_metrics(tmp_path):
    store = RunStore(tmp_path)
    run_id = store.create_run("exp", {"lr": 1e-6}, seed=1, git_sha="abc")
    store.log_metrics(run_id, 1, {"reward/mean": 0.1, "loss": 2.0, "bad": float("nan")})
    store.log_metrics(run_id, 2, {"reward/mean": 0.3})
    store.finish_run(run_id)

    run = store.get_run(run_id)
    assert run is not None
    assert run.status == "finished" and run.config == {"lr": 1e-6} and run.seed == 1
    assert [r.id for r in store.list_runs()] == [run_id]
    assert store.metric_keys(run_id) == ["loss", "reward/mean"]  # NaN dropped
    assert store.get_metrics(run_id, ["reward/mean"]) == {"reward/mean": [(1, 0.1), (2, 0.3)]}


def test_rollout_roundtrip(tmp_path):
    store = RunStore(tmp_path)
    run_id = store.create_run("exp", {})
    store.log_rollouts(run_id, 10, [_row(0), _row(1)])
    store.log_rollouts(run_id, 2, [_row(0)])
    assert store.rollout_steps(run_id) == [2, 10]
    rows = store.read_rollouts(run_id, 10)
    assert len(rows) == 2 and set(rows[0]) == set(ROLLOUT_COLUMNS)
    assert rows[1]["completion"] == "a1" and rows[1]["step"] == 10
    assert store.read_rollouts(run_id, 999) == []


def test_rollout_rows_must_be_complete(tmp_path):
    store = RunStore(tmp_path)
    run_id = store.create_run("exp", {})
    row = _row(0)
    del row["advantage"]
    with pytest.raises(ValueError, match="advantage"):
        store.log_rollouts(run_id, 1, [row])


def test_merge_store_copies_runs_metrics_and_rollouts(tmp_path):
    dst = RunStore(tmp_path / "dst")
    kept = dst.create_run("local", {"seed": 0})
    src = RunStore(tmp_path / "src")
    moved = src.create_run("pod", {"seed": 1}, seed=1, git_sha="abc")
    src.log_metrics(moved, 5, {"reward/mean": 0.7})
    src.log_rollouts(moved, 5, [_row(0)])
    src.finish_run(moved)

    assert merge_store(src.root, dst.root) == [moved]
    assert {r.id for r in dst.list_runs()} == {kept, moved}
    assert dst.get_run(moved) == src.get_run(moved)
    assert dst.get_metrics(moved) == {"reward/mean": [(5, 0.7)]}
    assert dst.read_rollouts(moved, 5) == src.read_rollouts(moved, 5)


def test_merge_store_refuses_clashes_without_writing(tmp_path):
    src = RunStore(tmp_path / "src")
    src.create_run("a", {})
    merge_store(src.root, tmp_path / "dst")
    other = src.create_run("b", {})
    with pytest.raises(ValueError, match="already in destination"):
        merge_store(src.root, tmp_path / "dst")  # the first run clashes, so nothing is copied
    assert other not in {r.id for r in RunStore(tmp_path / "dst").list_runs()}
    with pytest.raises(ValueError, match="same store"):
        merge_store(src.root, src.root)
    with pytest.raises(FileNotFoundError):
        merge_store(tmp_path / "missing", tmp_path / "dst")


def test_merge_cli(tmp_path, capsys):
    src = RunStore(tmp_path / "src")
    run_id = src.create_run("a", {})
    cli_main(["merge", str(src.root), str(tmp_path / "dst")])
    assert capsys.readouterr().out == f"merged {run_id}\n"
