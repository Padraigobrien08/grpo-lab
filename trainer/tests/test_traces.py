"""Token traces: buffer bookkeeping across micro-batches, and what a real run writes."""

from pathlib import Path

import pytest
import torch

from grpo_lab.algo.kl import kl_k3
from grpo_lab.config import load_config
from grpo_lab.traces import TraceBuffer
from grpo_lab.train import train
from runstore import RunStore

CONFIG = Path(__file__).parents[1] / "configs" / "toy.yaml"


def test_buffer_fills_only_traced_rows_across_micro_batches():
    torch.manual_seed(0)
    n_seq, width = 8, 4
    lp = -torch.rand(n_seq, width)
    ent = torch.rand(n_seq, width)
    old = lp - 0.1
    ref = lp - 0.2
    buf = TraceBuffer(n_rows=5, completion_len=width)  # 5 rows: spans two micro-batches of 3
    for mb in (slice(0, 3), slice(3, 6), slice(6, 8)):
        buf.record_first_pass(mb, lp[mb], ent[mb])
        buf.record_last_pass(mb, lp[mb], old[mb])
    assert torch.equal(buf.logprob, lp[:5]) and torch.equal(buf.entropy, ent[:5])
    assert torch.allclose(buf.ratio, torch.full((5, width), torch.exp(torch.tensor(0.1))))

    completions = [[1, 2, 3, 4], [5, 6], [7], [8, 9, 10], [11, 12, 13, 14], [0], [0], [0]]
    rows = buf.rows(
        ["pA", "pB"],
        completions,
        group_size=4,
        decode_tokens=lambda ids: [str(i) for i in ids],
        ref_logprob=ref,
    )
    assert [(r["prompt_id"], r["group_idx"], r["sample_idx"]) for r in rows] == [
        ("pA", 0, 0), ("pA", 0, 1), ("pA", 0, 2), ("pA", 0, 3), ("pB", 1, 0),
    ]  # fmt: skip
    assert [len(r["logprob"]) for r in rows] == [4, 2, 1, 3, 4]  # cut to each completion
    assert rows[1]["tokens"] == ["5", "6"] and rows[1]["token_ids"] == [5, 6]
    assert rows[3]["kl"] == pytest.approx(kl_k3(lp[3, :3], ref[3, :3]).tolist())

    no_ref = buf.rows(["pA", "pB"], completions, 4, lambda ids: [""] * len(ids))
    assert no_ref[0]["ref_logprob"] is None and no_ref[0]["kl"] is None


def _run(tmp_path, *overrides):
    cfg = load_config(CONFIG, [f"log.store_dir={tmp_path}", *overrides])
    return cfg, train(cfg), RunStore(tmp_path)


@pytest.mark.smoke
def test_traces_line_up_with_rollouts_and_logged_metrics(tmp_path):
    # Trace every group, across several micro-batches, so traced values can be checked
    # against the step's logged metrics, which average over every completion token.
    cfg, result, store = _run(
        tmp_path, "train.steps=2", "log.rollout_every=1", "log.trace_groups=100",
        "train.micro_batch_size=12",
    )  # fmt: skip
    run_id = result.run_id
    assert store.trace_steps(run_id) == store.rollout_steps(run_id) == [1, 2]
    metrics = store.get_metrics(run_id, ["entropy", "kl"])

    for step in (1, 2):
        traces = store.read_traces(run_id, step)
        rollouts = {(r["group_idx"], r["sample_idx"]): r for r in store.read_rollouts(run_id, step)}
        assert len(traces) == len(rollouts) == cfg.train.prompts_per_step * cfg.algo.group_size
        n_tok = 0
        ent_sum = kl_sum = 0.0
        for t in traces:
            r = rollouts[(t["group_idx"], t["sample_idx"])]
            assert t["prompt_id"] == r["prompt_id"]
            assert len(t["token_ids"]) == len(t["tokens"]) == r["length"]
            assert t["ratio"] == [1.0] * r["length"]  # one pass: pi == pi_old exactly
            lp, ref = torch.tensor(t["logprob"]), torch.tensor(t["ref_logprob"])
            assert t["kl"] == pytest.approx(kl_k3(lp, ref).tolist(), abs=1e-6)
            n_tok += r["length"]
            ent_sum += sum(t["entropy"])
            kl_sum += sum(t["kl"])
        entropy = dict(metrics["entropy"])[step]
        kl = dict(metrics["kl"])[step]
        assert ent_sum / n_tok == pytest.approx(entropy, rel=1e-4)
        assert kl_sum / n_tok == pytest.approx(kl, rel=1e-3, abs=1e-7)


@pytest.mark.smoke
def test_trace_defaults_multi_pass_and_off_switch(tmp_path):
    _, result, store = _run(
        tmp_path / "a", "train.steps=10", "algo.num_iterations=2", "algo.beta=0.0"
    )
    assert store.trace_steps(result.run_id) == [1, 10]
    assert store.trace_groups(result.run_id, 10) == [0, 1]  # default: first 2 groups
    traces = store.read_traces(result.run_id, 10)
    assert all(t["ref_logprob"] is None and t["kl"] is None for t in traces)  # no reference
    ratios = [x for t in traces for x in t["ratio"]]
    assert any(abs(x - 1.0) > 1e-6 for x in ratios)  # the second pass moved the policy

    _, result, store = _run(tmp_path / "b", "train.steps=1", "log.trace_groups=0")
    assert store.rollout_steps(result.run_id) == [1] and store.trace_steps(result.run_id) == []
