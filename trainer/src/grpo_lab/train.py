"""GRPO training loop: sample groups -> score -> group advantages -> clipped PG update.

Usage::

    uv run train --config trainer/configs/toy.yaml
    uv run train --config trainer/configs/toy.yaml --override algo.beta=0.0 --override seed=1
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from grpo_lab.algo.advantages import group_advantages, zero_advantage_fraction
from grpo_lab.algo.logprobs import completion_logprobs
from grpo_lab.algo.losses import grpo_loss
from grpo_lab.config import Config, load_config
from grpo_lab.eval import greedy_eval
from grpo_lab.models import autocast, load_policy, resolve_device
from grpo_lab.rollout import RolloutBatch, generate_group_rollouts
from grpo_lab.tasks import Prompt, Task, build_task
from runstore import RunStore


@dataclass
class TrainResult:
    run_id: str
    history: list[dict[str, float]] = field(default_factory=list)
    eval_history: list[dict[str, float]] = field(default_factory=list)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=5
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _micro_batches(n: int, size: int) -> list[slice]:
    return [slice(i, min(i + size, n)) for i in range(0, n, size)]


@torch.no_grad()
def _batched_logprobs(
    model: torch.nn.Module,
    batch: RolloutBatch,
    micro: int,
    temperature: float,
    amp_dtype: str | None,
) -> torch.Tensor:
    parts = []
    device = batch.input_ids.device
    for mb in _micro_batches(batch.input_ids.shape[0], micro):
        with autocast(device, amp_dtype):
            lp, _ = completion_logprobs(
                model, batch.input_ids[mb], batch.attention_mask[mb], batch.prompt_len, temperature
            )
        parts.append(lp)
    return torch.cat(parts)


def _rollout_rows(
    task: Task,
    prompts: list[Prompt],
    batch: RolloutBatch,
    completions: list[list[int]],
    components: dict[str, list[float]],
    rewards: list[float],
    advantages: torch.Tensor,
    group_size: int,
) -> list[dict]:
    rows = []
    for i, comp in enumerate(completions):
        prompt = prompts[i // group_size]
        rows.append(
            {
                "prompt_id": prompt.id,
                "group_idx": i // group_size,
                "sample_idx": i % group_size,
                "prompt": prompt.text,
                "completion": task.decode(comp),
                "reward_total": rewards[i],
                "reward_components": json.dumps({k: v[i] for k, v in components.items()}),
                "gold_reward": components[task.gold_component][i] if task.gold_component else None,
                "advantage": advantages[i].item(),
                "length": len(comp),
                "truncated": bool(batch.truncated[i].item()),
            }
        )
    return rows


def train(cfg: Config) -> TrainResult:
    seed_everything(cfg.seed)
    rng = random.Random(cfg.seed)
    device = resolve_device(cfg.device)
    model, tokenizer = load_policy(cfg.model, device)
    task = build_task(cfg, tokenizer)
    algo, tc, ec = cfg.algo, cfg.train, cfg.eval
    amp = cfg.model.amp_dtype
    group = algo.group_size
    eval_prompts = task.eval_prompts(ec.num_examples) if ec.every > 0 else []

    # The reference model is only needed for the KL term; skip the memory cost when beta == 0.
    ref_model = None
    if algo.beta != 0.0:
        ref_model = copy.deepcopy(model).eval()
        ref_model.requires_grad_(False)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=tc.lr, betas=tc.adam_betas, weight_decay=tc.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(1.0, (s + 1) / tc.warmup_steps) if tc.warmup_steps else 1.0
    )

    store = RunStore(cfg.log.store_dir)
    run_id = store.create_run(cfg.name, cfg.model_dump(), seed=cfg.seed, git_sha=git_sha())
    result = TrainResult(run_id=run_id)
    print(f"run {run_id} on {device} | store: {store.root}")

    def run_eval(step: int) -> None:
        t = time.perf_counter()
        metrics = greedy_eval(
            model,
            task,
            eval_prompts,
            batch_size=ec.batch_size,
            max_new_tokens=ec.max_new_tokens or algo.max_new_tokens,
            amp_dtype=amp,
        )
        metrics["time/eval_s"] = time.perf_counter() - t
        store.log_metrics(run_id, step, metrics)
        result.eval_history.append({"step": step, **metrics})
        print(
            f"eval @ step {step:4d} | accuracy {metrics.get('eval/accuracy', float('nan')):.3f} "
            f"| len {metrics['eval/len_mean']:.1f} | {metrics['time/eval_s']:.0f}s"
        )

    try:
        if eval_prompts:
            run_eval(0)  # the base model's score, on the same prompts as every later eval
        for step in range(1, tc.steps + 1):
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            # ---- 1. rollouts ------------------------------------------------------------
            t0 = time.perf_counter()
            prompts = task.sample_prompts(tc.prompts_per_step, rng)
            with autocast(device, amp):
                batch = generate_group_rollouts(
                    model,
                    [p.input_ids for p in prompts],
                    group_size=group,
                    max_new_tokens=algo.max_new_tokens,
                    temperature=algo.temperature,
                    pad_id=task.pad_id,
                    eos_ids=task.eos_ids,
                )
            t_gen = time.perf_counter() - t0

            # ---- 2. rewards and advantages ----------------------------------------------
            completions = batch.completion_token_lists()
            expanded = [p for p in prompts for _ in range(group)]
            components = task.score(expanded, completions)
            rewards_list = task.total_reward(components)
            rewards = torch.tensor(rewards_list, dtype=torch.float32, device=device)
            advantages = group_advantages(rewards, group, algo.advantage_scale)

            # ---- 3. policy update -------------------------------------------------------
            t1 = time.perf_counter()
            n_seq = batch.input_ids.shape[0]
            cmask = batch.completion_mask
            n_tok = cmask.sum().item()
            # With a single pass, old log-probs equal the current ones (ratio == 1), so reuse
            # them instead of paying for an extra forward pass.
            old_lp = (
                _batched_logprobs(model, batch, tc.micro_batch_size, algo.temperature, amp)
                if algo.num_iterations > 1
                else None
            )
            ref_lp = (
                _batched_logprobs(ref_model, batch, tc.micro_batch_size, algo.temperature, amp)
                if ref_model is not None
                else None
            )

            model.train()
            sums: dict[str, float] = {}
            loss_total, entropy_sum, grad_norm = 0.0, 0.0, 0.0
            for _ in range(algo.num_iterations):
                optimizer.zero_grad(set_to_none=True)
                for mb in _micro_batches(n_seq, tc.micro_batch_size):
                    with autocast(device, amp):
                        lp, ent = completion_logprobs(
                            model,
                            batch.input_ids[mb],
                            batch.attention_mask[mb],
                            batch.prompt_len,
                            algo.temperature,
                            return_entropy=True,
                        )
                    out = grpo_loss(
                        lp,
                        old_lp[mb] if old_lp is not None else lp.detach(),
                        advantages[mb],
                        cmask[mb],
                        ref_logprobs=ref_lp[mb] if ref_lp is not None else None,
                        beta=algo.beta,
                        clip_eps_low=algo.clip_eps_low,
                        clip_eps_high=algo.clip_eps_high,
                        aggregation=algo.loss_aggregation,
                        num_sequences_total=n_seq,
                        num_tokens_total=n_tok,
                        max_completion_len=algo.max_new_tokens,
                    )
                    if not torch.isfinite(out.loss):
                        raise FloatingPointError(f"non-finite loss at step {step}")
                    out.loss.backward()
                    loss_total += out.loss.item()
                    for k, v in out.stat_sums.items():
                        sums[k] = sums.get(k, 0.0) + v
                    assert ent is not None
                    entropy_sum += (ent * cmask[mb]).sum().item()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), tc.max_grad_norm
                ).item()
                optimizer.step()
            scheduler.step()
            t_train = time.perf_counter() - t1

            # ---- 4. logging -------------------------------------------------------------
            denom = max(n_tok * algo.num_iterations, 1.0)
            lengths = [len(c) for c in completions]
            metrics = {
                "reward/mean": rewards.mean().item(),
                "reward/std": rewards.std().item() if rewards.numel() > 1 else 0.0,
                **{f"reward/{k}": float(np.mean(v)) for k, v in components.items()},
                "adv/zero_group_frac": zero_advantage_fraction(rewards, group),
                "completion/len_mean": float(np.mean(lengths)),
                "completion/len_max": float(max(lengths)),
                "completion/truncated_frac": batch.truncated.float().mean().item(),
                "loss": loss_total / algo.num_iterations,
                "entropy": entropy_sum / denom,
                "clip_frac/low": sums["clip_frac_low"] / denom,
                "clip_frac/high": sums["clip_frac_high"] / denom,
                "ratio/mean": sums["ratio"] / denom,
                "grad_norm": grad_norm,
                "lr": scheduler.get_last_lr()[0],
                "time/gen_s": t_gen,
                "time/train_s": t_train,
                "throughput/gen_tokens_per_s": n_tok / max(t_gen, 1e-9),
            }
            if task.gold_component:
                metrics["gold/accuracy"] = float(np.mean(components[task.gold_component]))
            if "kl" in sums:
                metrics["kl"] = sums["kl"] / denom
            if device.type == "cuda":
                metrics["memory/peak_gb"] = torch.cuda.max_memory_allocated(device) / 1e9
            store.log_metrics(run_id, step, metrics)
            result.history.append(metrics)

            if step == 1 or step == tc.steps or step % cfg.log.rollout_every == 0:
                store.log_rollouts(
                    run_id,
                    step,
                    _rollout_rows(
                        task,
                        prompts,
                        batch,
                        completions,
                        components,
                        rewards_list,
                        advantages,
                        group,
                    ),
                )
            if step % cfg.log.print_every == 0 or step == tc.steps:
                print(
                    f"step {step:4d} | reward {metrics['reward/mean']:.3f} "
                    f"| len {metrics['completion/len_mean']:.1f} | ent {metrics['entropy']:.3f} "
                    f"| kl {metrics.get('kl', 0.0):.4f} "
                    f"| zero-adv {metrics['adv/zero_group_frac']:.2f} "
                    f"| gen {t_gen:.1f}s train {t_train:.1f}s"
                )
            if eval_prompts and (step % ec.every == 0 or step == tc.steps):
                run_eval(step)
        if cfg.log.save_final:
            out = Path(cfg.log.checkpoint_dir) / run_id
            model.save_pretrained(out)
            if tokenizer is not None:
                tokenizer.save_pretrained(out)
            print(f"saved final policy to {out}")
    except BaseException:
        store.finish_run(run_id, status="failed")
        raise
    store.finish_run(run_id)
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="GRPO training")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--override", action="append", default=[], help="dotted.key=value, repeatable"
    )
    args = parser.parse_args(argv)
    train(load_config(args.config, args.override))


if __name__ == "__main__":
    main()
