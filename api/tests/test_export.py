import json

from fastapi.testclient import TestClient

from grpo_api.export import export_static
from grpo_api.main import create_app
from runstore import RunStore


def _row(i: int) -> dict:
    return {
        "prompt_id": "p",
        "group_idx": 0,
        "sample_idx": i,
        "prompt": "q",
        "completion": f"a{i}",
        "reward_total": float(i),
        "reward_components": json.dumps({"correct": float(i)}),
        "gold_reward": None,
        "advantage": 0.0,
        "length": 1,
        "truncated": False,
    }


def test_export_mirrors_every_api_response(tmp_path):
    store = RunStore(tmp_path / "store")
    run_id = store.create_run("exp", {"lr": 1e-6})
    store.log_metrics(run_id, 1, {"reward/mean": 0.5})
    store.log_rollouts(run_id, 1, [_row(0), _row(1)])
    store.log_rollouts(run_id, 10, [_row(0)])
    out = tmp_path / "data"
    (out / "runs" / "stale-run").mkdir(parents=True)  # from an earlier export

    assert export_static(store.root, out) == 6
    client = TestClient(create_app(str(store.root)))
    paths = [
        "/runs",
        f"/runs/{run_id}",
        f"/runs/{run_id}/metrics",
        f"/runs/{run_id}/rollouts",
        f"/runs/{run_id}/rollouts/1",
        f"/runs/{run_id}/rollouts/10",
    ]
    for path in paths:
        assert (out / f"{path.lstrip('/')}.json").read_bytes() == client.get(path).content
    assert sorted(p.name for p in (out / "runs").iterdir()) == sorted([run_id, f"{run_id}.json"])
