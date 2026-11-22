# grpo-lab

From-scratch GRPO-style RL post-training for small language models, plus a dashboard for
looking inside training. See [SPEC.md](SPEC.md) for the research questions, scope and milestones.

**Status:** M0 (scaffold) + M1 (algorithm core) done.

## Layout

| Path | What |
|---|---|
| `runstore/` | On-disk run store (SQLite + Parquet): the only contract between trainer and API |
| `trainer/` | `grpo_lab`: rollouts, rewards, advantages, loss, training loop (PyTorch) |
| `api/` | Read-only FastAPI over the run store (no torch dependency) |
| `web/` | Next.js + shadcn/ui dashboard (placeholder until M4) |

## Quickstart

```bash
uv sync --all-packages
uv run pytest                                              # 47 tests incl. 2 end-to-end learning runs
uv run train --config trainer/configs/toy.yaml             # tiny random model learns a target token (CPU, seconds)
uv run train --config trainer/configs/smoke_gsm8k.yaml     # Qwen2.5-0.5B on GSM8K, 3 steps (MPS/CUDA, ~1 GB download)
GRPO_STORE=runs uv run uvicorn grpo_api.main:app --reload  # API on :8000
cd web && pnpm dev                                         # dashboard on :3000
```

Override any config value from the CLI, e.g. for ablations:

```bash
uv run train --config trainer/configs/toy.yaml --override algo.beta=0.0 --override algo.loss_aggregation=token_mean
```

## What the tests pin down

- **Loss:** on-policy gradient = −advantage; all four clipping cases; DAPO clip-higher;
  masked tokens are ignored; per-token weights for each aggregation (GRPO / DAPO / Dr. GRPO);
  micro-batch losses sum exactly to the full-batch loss.
- **Log-probs:** match a manual log-softmax; unaffected by left padding; **equal to the distribution
  `generate()` actually sampled from** (temperature included).
- **KL:** k1/k3 estimators match the exact KL in expectation; k3 has lower variance.
- **End-to-end:** the full loop raises reward from chance (~0.06) to >0.8 on the toy task, for
  vanilla GRPO and a Dr. GRPO/DAPO-style variant with μ=2.
