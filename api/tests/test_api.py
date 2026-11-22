import json

from fastapi.testclient import TestClient

from grpo_api.main import create_app
from runstore import RunStore


def test_endpoints(tmp_path):
    store = RunStore(tmp_path)
    run_id = store.create_run("exp", {"lr": 1e-6})
    store.log_metrics(run_id, 1, {"reward/mean": 0.25})
    store.log_rollouts(
        run_id,
        1,
        [
            {
                "prompt_id": "p",
                "group_idx": 0,
                "sample_idx": 0,
                "prompt": "q",
                "completion": "a",
                "reward_total": 1.0,
                "reward_components": json.dumps({"correct": 1.0}),
                "gold_reward": 1.0,
                "advantage": 0.0,
                "length": 1,
                "truncated": False,
            }
        ],
    )
    client = TestClient(create_app(str(tmp_path)))

    assert client.get("/health").json() == {"status": "ok"}
    assert [r["id"] for r in client.get("/runs").json()] == [run_id]
    assert client.get(f"/runs/{run_id}").json()["metric_keys"] == ["reward/mean"]
    assert client.get(f"/runs/{run_id}/metrics").json() == {"reward/mean": [[1, 0.25]]}
    assert client.get(f"/runs/{run_id}/rollouts").json() == [1]
    assert client.get(f"/runs/{run_id}/rollouts/1").json()[0]["completion"] == "a"
    assert client.get("/runs/nope").status_code == 404
    assert client.get("/runs/nope/metrics").status_code == 404
