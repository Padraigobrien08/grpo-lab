"""End-to-end: the full training loop must actually improve reward on the toy task."""

from pathlib import Path

import numpy as np
import pytest

from grpo_lab.config import load_config
from grpo_lab.train import train
from runstore import RunStore

CONFIG = Path(__file__).parents[1] / "configs" / "toy.yaml"


@pytest.mark.smoke
@pytest.mark.parametrize(
    "overrides",
    [
        [],  # vanilla GRPO with KL
        # Dr. GRPO / DAPO-style knobs + multiple passes, exercising the old-logprob path
        [
            "algo.beta=0.0",
            "algo.advantage_scale=none",
            "algo.loss_aggregation=token_mean",
            "algo.clip_eps_high=0.28",
            "algo.num_iterations=2",
            "train.micro_batch_size=16",
        ],
    ],
    ids=["grpo", "variants"],
)
def test_toy_reward_improves(tmp_path, overrides):
    cfg = load_config(CONFIG, [*overrides, f"log.store_dir={tmp_path}", "train.steps=30"])
    result = train(cfg)

    rewards = [h["reward/mean"] for h in result.history]
    assert np.mean(rewards[:3]) < 0.35, "toy task should start near chance (1/16)"
    assert np.mean(rewards[-5:]) > 0.8, f"GRPO failed to learn: {rewards}"

    store = RunStore(tmp_path)
    run = store.get_run(result.run_id)
    assert run is not None and run.status == "finished"
    assert store.rollout_steps(result.run_id) == [1, 10, 20, 30]
    rows = store.read_rollouts(result.run_id, 30)
    assert len(rows) == cfg.train.prompts_per_step * cfg.algo.group_size
