import json

import pytest

from runstore import ROLLOUT_COLUMNS, TRACE_COLUMNS, RunStore, merge_store
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


def _trace(group: int, sample: int, n: int = 3, beta: bool = True) -> dict:
    return {
        "prompt_id": f"p{group}",
        "group_idx": group,
        "sample_idx": sample,
        "token_ids": list(range(n)),
        "tokens": [f"t{i}" for i in range(n)],
        "logprob": [-0.5] * n,
        "ref_logprob": [-0.25] * n if beta else None,
        "kl": [0.125] * n if beta else None,
        "entropy": [1.0] * n,
        "ratio": [1.0] * n,
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


def test_trace_roundtrip_and_group_filter(tmp_path):
    store = RunStore(tmp_path)
    run_id = store.create_run("exp", {})
    store.log_traces(run_id, 10, [_trace(0, 0), _trace(0, 1, n=5), _trace(3, 0, beta=False)])
    store.log_traces(run_id, 1, [_trace(0, 0)])
    assert store.trace_steps(run_id) == [1, 10]
    assert store.trace_groups(run_id, 10) == [0, 3]
    assert store.trace_groups(run_id, 999) == []

    rows = store.read_traces(run_id, 10, group_idx=0)
    assert [r["sample_idx"] for r in rows] == [0, 1] and set(rows[0]) == set(TRACE_COLUMNS)
    assert rows[1]["tokens"] == ["t0", "t1", "t2", "t3", "t4"] and rows[1]["step"] == 10
    assert rows[0]["kl"] == [0.125] * 3  # float32 round-trips exactly for these values
    [no_ref] = store.read_traces(run_id, 10, group_idx=3)
    assert no_ref["ref_logprob"] is None and no_ref["kl"] is None
    assert len(store.read_traces(run_id, 10)) == 3
    assert store.read_traces(run_id, 999) == []
    assert store.trace_steps("other-run") == []


def test_trace_rows_are_validated(tmp_path):
    store = RunStore(tmp_path)
    run_id = store.create_run("exp", {})
    row = _trace(0, 0)
    del row["entropy"]
    with pytest.raises(ValueError, match="entropy"):
        store.log_traces(run_id, 1, [row])
    ragged = _trace(0, 0)
    ragged["logprob"] = [-0.5]
    with pytest.raises(ValueError, match="logprob"):
        store.log_traces(run_id, 1, [ragged])
    no_entropy = {**_trace(0, 0), "entropy": None}  # only ref_logprob and kl may be null
    with pytest.raises(ValueError, match="entropy"):
        store.log_traces(run_id, 1, [no_entropy])


def test_merge_store_copies_runs_metrics_and_rollouts(tmp_path):
    dst = RunStore(tmp_path / "dst")
    kept = dst.create_run("local", {"seed": 0})
    src = RunStore(tmp_path / "src")
    moved = src.create_run("pod", {"seed": 1}, seed=1, git_sha="abc")
    src.log_metrics(moved, 5, {"reward/mean": 0.7})
    src.log_rollouts(moved, 5, [_row(0)])
    src.log_traces(moved, 5, [_trace(0, 0)])
    src.finish_run(moved)

    assert merge_store(src.root, dst.root) == [moved]
    assert {r.id for r in dst.list_runs()} == {kept, moved}
    assert dst.get_run(moved) == src.get_run(moved)
    assert dst.get_metrics(moved) == {"reward/mean": [(5, 0.7)]}
    assert dst.read_rollouts(moved, 5) == src.read_rollouts(moved, 5)
    assert dst.read_traces(moved, 5) == src.read_traces(moved, 5) != []


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
