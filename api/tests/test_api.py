import json

from fastapi.testclient import TestClient

from grpo_api.main import create_app
from runstore import RunStore


def test_endpoints(tmp_path):
    store = RunStore(tmp_path)
    run_id = store.create_run("exp", {"lr": 1e-6})
    store.log_metrics(run_id, 1, {"reward/mean": 0.25, "loss": 1.5})
    store.log_metrics(run_id, 2, {"reward/mean": 0.75})  # loss not logged at step 2
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
    store.log_traces(
        run_id,
        1,
        [
            {
                "prompt_id": "p",
                "group_idx": 0,
                "sample_idx": 0,
                "token_ids": [7, 2],
                "tokens": ["a", "<eos>"],
                "logprob": [-0.123456, -2.0],
                "ref_logprob": None,
                "kl": None,
                "entropy": [1.5, 0.25],
                "ratio": [1.0, 1.0],
            }
        ],
    )
    client = TestClient(create_app(str(tmp_path)))

    assert client.get("/health").json() == {"status": "ok"}
    runs = client.get("/runs").json()
    assert [r["id"] for r in runs] == [run_id]
    assert runs[0]["last_step"] == 2 and runs[0]["rollout_steps"] == 1
    assert runs[0]["last_metrics"] == {"reward/mean": 0.75, "loss": 1.5}
    assert client.get(f"/runs/{run_id}").json()["metric_keys"] == ["loss", "reward/mean"]
    assert client.get(f"/runs/{run_id}/metrics?keys=reward/mean").json() == {
        "reward/mean": [[1, 0.25], [2, 0.75]]
    }
    assert client.get(f"/runs/{run_id}/rollouts").json() == [1]
    assert client.get(f"/runs/{run_id}/rollouts/1").json()[0]["completion"] == "a"
    assert client.get(f"/runs/{run_id}/traces").json() == [1]
    assert client.get(f"/runs/{run_id}/traces/1").json() == [0]
    [trace] = client.get(f"/runs/{run_id}/traces/1/0").json()
    assert trace["tokens"] == ["a", "<eos>"] and trace["kl"] is None
    assert trace["logprob"] == [-0.1235, -2.0]  # rounded to 4 significant digits
    assert client.get(f"/runs/{run_id}/traces/1/5").json() == []
    assert client.get(f"/runs/{run_id}/traces/9").json() == []
    assert client.get("/runs/nope/traces").status_code == 404
    assert client.get("/runs/nope").status_code == 404
    assert client.get("/runs/nope/metrics").status_code == 404
