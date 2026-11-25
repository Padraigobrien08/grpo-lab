"""M3 eval harness: score a model on the full held-out set, then compare models with CIs.

Usage::

    # one model: greedy accuracy + pass@k on all 1319 GSM8K test problems
    uv run evaluate run --model Qwen/Qwen2.5-0.5B-Instruct --label base
    uv run evaluate run --model checkpoints/<run_id>
    # base vs trained, paired on the same problems; several trained runs = seeds
    uv run evaluate compare evals/base evals/<run_a> evals/<run_b> evals/<run_c>

Prompts, stop tokens, max length, sampling temperature and autocast come from the training
config (``--config``, default ``gsm8k_grpo.yaml``), so a model is scored exactly the way it was
trained. A sample counts as correct when the task's gold component is 1.

Each ``run`` writes ``<out-dir>/<label>/problems.parquet`` (one row per problem, greedy text
included) and ``summary.json``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from grpo_lab.config import Config, load_config
from grpo_lab.eval import generate_and_score
from grpo_lab.eval_stats import mcnemar_exact_p, mean_ci, paired_diff_ci, pass_at_k, seed_ci
from grpo_lab.models import load_policy, resolve_device
from grpo_lab.tasks import build_task
from grpo_lab.train import git_sha, seed_everything

DEFAULT_CONFIG = Path(__file__).parents[2] / "configs" / "gsm8k_grpo.yaml"
PROBLEMS_FILE = "problems.parquet"
SUMMARY_FILE = "summary.json"
UNIT = (0.0, 1.0)  # range of accuracy and pass@k, for clipping CIs


def _default_label(cfg: Config, model: str | None) -> str:
    if model is None:
        return cfg.name
    path = Path(model)
    return path.name if path.exists() else model.replace("/", "_")


def evaluate_model(
    cfg: Config,
    *,
    num_examples: int | None,
    samples: int,
    ks: list[int],
    batch_size: int,
    max_new_tokens: int,
    seed: int,
    progress: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Score ``cfg.model`` on the task's held-out set; return per-problem rows and a summary."""
    if samples and any(k > samples for k in ks):
        raise ValueError(f"every k must be <= samples ({samples}), got {ks}")
    device = resolve_device(cfg.device)
    seed_everything(seed)
    model, tokenizer = load_policy(
        cfg.model.model_copy(update={"gradient_checkpointing": False}), device
    )
    model.eval()
    task = build_task(cfg, tokenizer)
    gold = task.gold_component
    if gold is None:
        raise ValueError(f"task {cfg.task.name} has no gold component to score against")
    prompts = task.eval_prompts(num_examples if num_examples is not None else 10**9)
    common = dict(batch_size=batch_size, max_new_tokens=max_new_tokens, progress=progress)

    t0 = time.perf_counter()
    if progress:
        print(f"greedy: {len(prompts)} problems", flush=True)
    greedy = generate_and_score(model, task, prompts, amp_dtype=cfg.model.amp_dtype, **common)
    t_greedy = time.perf_counter() - t0

    sampled = None
    t_sampled = 0.0
    if samples:
        if progress:
            print(f"sampling: {samples} per problem at T={cfg.algo.temperature}", flush=True)
        torch.manual_seed(seed)
        t1 = time.perf_counter()
        sampled = generate_and_score(
            model,
            task,
            prompts,
            n=samples,
            greedy=False,
            temperature=cfg.algo.temperature,
            amp_dtype=cfg.model.amp_dtype,
            **common,
        )
        t_sampled = time.perf_counter() - t1

    rows: list[dict[str, Any]] = []
    for i, p in enumerate(prompts):
        row: dict[str, Any] = {
            "id": p.id,
            "gold": None if "gold" not in p.meta else str(p.meta["gold"]),
            "greedy_correct": float(greedy.components[gold][i] >= 1.0),
            "greedy_len": greedy.lengths[i],
            "greedy_truncated": bool(greedy.truncated[i]),
            "greedy_text": task.decode(greedy.token_ids[i]),
        }
        for comp, values in greedy.components.items():
            if comp != gold:
                row[f"greedy_{comp}"] = float(values[i])
        if sampled is not None:
            sl = slice(i * samples, (i + 1) * samples)
            correct = int(sum(v >= 1.0 for v in sampled.components[gold][sl]))
            row["n_samples"] = samples
            row["n_correct"] = correct
            row["sample_len_mean"] = float(np.mean(sampled.lengths[sl]))
            row["sample_truncated_frac"] = float(np.mean(sampled.truncated[sl]))
            for k in ks:
                row[f"pass@{k}"] = pass_at_k(samples, correct, k)
        rows.append(row)

    summary: dict[str, Any] = {
        "model": cfg.model.name_or_path or "toy",
        "config_name": cfg.name,
        "task": cfg.task.name,
        "git_sha": git_sha(),
        "num_problems": len(rows),
        "settings": {
            "samples": samples,
            "ks": ks if samples else [],
            "temperature": cfg.algo.temperature,
            "max_new_tokens": max_new_tokens,
            "amp_dtype": cfg.model.amp_dtype,
            "batch_size": batch_size,
            "seed": seed,
            "device": str(device),
        },
        "metrics": _metrics(rows, ks if samples else []),
        "time_s": {"greedy": t_greedy, "sampled": t_sampled},
    }
    return rows, summary


def _metrics(rows: list[dict[str, Any]], ks: list[int]) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "greedy_accuracy": mean_ci([r["greedy_correct"] for r in rows], UNIT).as_dict(),
        "greedy_len_mean": float(np.mean([r["greedy_len"] for r in rows])),
        "greedy_truncated_frac": float(np.mean([r["greedy_truncated"] for r in rows])),
    }
    for k in ks:
        metrics[f"pass@{k}"] = mean_ci([r[f"pass@{k}"] for r in rows], UNIT).as_dict()
    if ks:
        metrics["sample_len_mean"] = float(np.mean([r["sample_len_mean"] for r in rows]))
    return metrics


def write_results(out: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    out.mkdir(parents=True, exist_ok=False)
    pq.write_table(pa.Table.from_pylist(rows), out / PROBLEMS_FILE)
    (out / SUMMARY_FILE).write_text(json.dumps(summary, indent=2) + "\n")


def load_results(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = pq.read_table(path / PROBLEMS_FILE).to_pylist()
    return rows, json.loads((path / SUMMARY_FILE).read_text())


def compare(base_dir: Path, trained_dirs: list[Path]) -> dict[str, Any]:
    """Paired base-vs-trained differences per run, plus mean ± t CI across runs (seeds)."""
    base_rows, _ = load_results(base_dir)
    ids = [r["id"] for r in base_rows]
    runs = []
    for d in trained_dirs:
        rows, _ = load_results(d)
        by_id = {r["id"]: r for r in rows}
        if set(by_id) != set(ids):
            raise ValueError(f"{d} was scored on different problems than {base_dir}")
        runs.append((d, [by_id[i] for i in ids]))

    # Compare every metric all runs have: greedy accuracy, plus pass@k where both sampled.
    keys = ["greedy_correct"] + sorted(
        (k for k in base_rows[0] if k.startswith("pass@")), key=lambda k: int(k.split("@")[1])
    )
    keys = [k for k in keys if all(k in rows[0] for _, rows in runs)]

    result: dict[str, Any] = {
        "base": {
            "dir": str(base_dir),
            **{k: mean_ci([r[k] for r in base_rows], UNIT).as_dict() for k in keys},
        },
        "runs": [],
        "num_problems": len(ids),
    }
    for d, rows in runs:
        entry: dict[str, Any] = {"dir": str(d)}
        for k in keys:
            a, b = [r[k] for r in base_rows], [r[k] for r in rows]
            entry[k] = mean_ci(b, UNIT).as_dict()
            entry[f"{k}_diff"] = paired_diff_ci(a, b).as_dict()
        entry["mcnemar_p"] = mcnemar_exact_p(
            [r["greedy_correct"] >= 1 for r in base_rows], [r["greedy_correct"] >= 1 for r in rows]
        )
        result["runs"].append(entry)
    if len(runs) >= 2:
        result["across_seeds"] = {
            k: {
                "trained": seed_ci([e[k]["mean"] for e in result["runs"]]).as_dict(),
                "diff": seed_ci([e[f"{k}_diff"]["mean"] for e in result["runs"]]).as_dict(),
            }
            for k in keys
        }
    return result


def _pct(iv: dict[str, Any], signed: bool = False) -> str:
    sign = "+" if signed else ""
    return (
        f"{100 * iv['mean']:{sign}.1f} [{100 * iv['low']:{sign}.1f}, {100 * iv['high']:{sign}.1f}]"
    )


def format_comparison(result: dict[str, Any]) -> str:
    keys = [k for k in result["base"] if k != "dir"]
    names = {k: "greedy acc" if k == "greedy_correct" else k for k in keys}
    lines = [
        f"{result['num_problems']} problems; % with 95% CI; diff = paired (trained - base)",
        "",
    ]
    lines.append(f"base  {result['base']['dir']}")
    for k in keys:
        lines.append(f"  {names[k]:<11} {_pct(result['base'][k])}")
    for e in result["runs"]:
        lines.append(f"run   {e['dir']}   (McNemar p = {e['mcnemar_p']:.2g})")
        for k in keys:
            lines.append(
                f"  {names[k]:<11} {_pct(e[k])}   diff {_pct(e[f'{k}_diff'], signed=True)}"
            )
    if "across_seeds" in result:
        n = len(result["runs"])
        lines.append(f"across {n} seeds (mean, t 95% CI)")
        for k, v in result["across_seeds"].items():
            lines.append(
                f"  {names[k]:<11} {_pct(v['trained'])}   diff {_pct(v['diff'], signed=True)}"
            )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="M3 eval harness")
    sub = parser.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="score one model on the held-out set")
    run.add_argument("--config", default=str(DEFAULT_CONFIG), help="training config to score with")
    run.add_argument("--model", help="HF id or checkpoint dir (default: the config's model)")
    run.add_argument("--label", help="output folder name (default: from --model)")
    run.add_argument("--num-examples", type=int, help="first N held-out problems (default: all)")
    run.add_argument(
        "--samples", type=int, default=16, help="samples per problem for pass@k; 0 = greedy only"
    )
    run.add_argument("--k", default="1,4,8,16", help="comma-separated k values for pass@k")
    run.add_argument(
        "--batch-size", type=int, help="completions per generate call (default: eval.batch_size)"
    )
    run.add_argument(
        "--max-new-tokens", type=int, help="default: eval.max_new_tokens or algo.max_new_tokens"
    )
    run.add_argument("--seed", type=int, default=0, help="sampling seed")
    # Not the config's device: training configs pin the training machine (gsm8k_grpo: cuda).
    run.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    run.add_argument("--out-dir", default="evals")
    run.add_argument(
        "--override", action="append", default=[], help="config dotted.key=value, repeatable"
    )

    cmp = sub.add_parser("compare", help="paired base-vs-trained comparison with CIs")
    cmp.add_argument("base", type=Path)
    cmp.add_argument("trained", type=Path, nargs="+", help="one dir per trained run (seed)")
    cmp.add_argument("--json", type=Path, help="also write the comparison as JSON")

    args = parser.parse_args(argv)
    if args.cmd == "compare":
        result = compare(args.base, args.trained)
        print(format_comparison(result))
        if args.json:
            args.json.write_text(json.dumps(result, indent=2) + "\n")
        return

    overrides = [f"device={args.device}", *args.override]
    if args.model:
        overrides += [f"model.name_or_path={args.model}", "model.toy=null"]
    cfg = load_config(args.config, overrides)
    out = Path(args.out_dir) / (args.label or _default_label(cfg, args.model))
    if out.exists():
        parser.error(f"{out} already exists; pick another --label or remove it")
    rows, summary = evaluate_model(
        cfg,
        num_examples=args.num_examples,
        samples=args.samples,
        ks=[int(k) for k in args.k.split(",")] if args.samples else [],
        batch_size=args.batch_size or cfg.eval.batch_size,
        max_new_tokens=args.max_new_tokens or cfg.eval.max_new_tokens or cfg.algo.max_new_tokens,
        seed=args.seed,
        progress=True,
    )
    summary["label"] = out.name
    write_results(out, rows, summary)
    m = summary["metrics"]
    print(f"greedy accuracy {_pct(m['greedy_accuracy'])} on {summary['num_problems']} problems")
    for k in summary["settings"]["ks"]:
        print(f"pass@{k:<3} {_pct(m[f'pass@{k}'])}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
