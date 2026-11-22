# grpo-lab: RL Post-Training of Small LLMs, From Scratch

> Working title. Target audience: ML Research Engineering hiring panels.

## 1. Pitch

A from-scratch PyTorch implementation of GRPO-style RL fine-tuning for small language models.
It comes with a rigorous ablation study and a web dashboard for looking inside training: rollout groups,
advantages, token-level KL, and reward hacking.

The project should show three things:

1. **I can implement modern RL post-training correctly.** Evidence: parity with a reference implementation, plus unit-tested loss math.
2. **I can run experiments like a researcher.** Evidence: a research question, baselines, multiple seeds, confidence intervals, ablations, and a technical report.
3. **I can build the tooling researchers need.** Evidence: a GRPO-specific dashboard that W&B can't give you.

## 2. Research questions (end goal)

- **RQ1: Which GRPO design choices matter at small scale?**
  Compare vanilla GRPO (DeepSeekMath), Dr. GRPO (no length/std normalisation), and the DAPO components
  (clip-higher, token-level loss, dynamic sampling, overlong penalty) on ≤1.5B models.
  Effects to measure: accuracy, response length, entropy collapse, and sample efficiency.
- **RQ2: How and when does a policy exploit a flawed verifier, and can we detect it from training signals alone?**
  Train against deliberately imperfect rewards and track the gap between the proxy reward and the gold reward.
  Then build a detector over logged rollout statistics.
- *(Secondary)* **RQ3: Does the KL penalty help at this scale?** Compare β = 0 against β > 0.

RQ2 is the most distinctive part. It connects RL to evaluation and observability, and it's a live
safety and alignment topic.

## 3. Scope

### 3.1 MVP

**Goal:** a correct, reproducible GRPO run that measurably improves a small model on GSM8K,
viewable in a dashboard.

| Area | MVP scope |
|---|---|
| Model | `Qwen2.5-0.5B-Instruct` (or `Qwen3-0.6B`), full fine-tune or LoRA |
| Task | GSM8K. Reward = exact-match correctness + small format reward |
| Algorithm | Vanilla GRPO: group sampling (G=8), group-normalised advantages, clipped ratio objective, k3 KL to a frozen reference |
| Rollouts | HF `generate` (simple and slow; fine for the MVP) |
| Eval | Greedy GSM8K test accuracy, pass@k, base vs. trained |
| Rigor | 3 seeds, mean ± 95% CI, base-model baseline |
| Tests | Unit tests for advantages, loss, masking, and the KL estimator; tiny-model smoke test runs locally on CPU/MPS |
| Run store | Metrics (SQLite) + sampled rollouts (Parquet) on disk |
| API | FastAPI, read-only over the run store |
| Web | Next.js + shadcn: **Runs list**, **Run detail** (metric curves), **Rollout inspector** (group view) |
| Deploy | Dashboard hosted with recorded runs. No GPU needed to view it |

**MVP exit criteria**
- [ ] `uv run train --config configs/gsm8k_grpo.yaml` reproduces a run from a clean checkout.
- [ ] Trained model beats base on GSM8K test, with non-overlapping 95% CIs across 3 seeds.
- [ ] All algorithm unit tests pass. A smoke test runs in CI on a tiny model.
- [ ] Dashboard deployed. A reviewer can open a run and inspect rollout groups at any step.
- [ ] README with the result plot, how to reproduce, and what went wrong along the way.

### 3.2 End goal

Everything in the MVP, plus:

| Area | Additions |
|---|---|
| Correctness | **Parity check against TRL's `GRPOTrainer`** on a matched config (curves agree within seed noise) |
| Throughput | vLLM rollout engine with weight sync to the trainer. Report tokens/s and the speedup over HF generate |
| Algorithms | Loss variants behind a config flag: GRPO / Dr. GRPO / DAPO components; β sweep. Optional PPO + value-head baseline |
| Second domain | **Text-to-SQL** (Spider dev) with execution-based reward in sandboxed read-only SQLite |
| Reward hacking | Flawed verifiers (math: "any number in the text matches"; SQL: "row count matches"); gold-vs-proxy tracking; detector |
| Scale point | One 1.5B run (LoRA) to show trends hold or break |
| Web | **Compare view** (ablation curves with CI bands), **Reward hacking view**, **live mode** (WebSocket tail of a running job), **Playground** (base vs. tuned side by side) |
| Write-up | Technical report (blog + PDF): method, ablation tables, failure analysis, limitations |

## 4. Architecture

```
┌─────────────────────────┐     writes      ┌──────────────────┐   reads   ┌──────────────┐   HTTP/WS  ┌──────────────┐
│ trainer (PyTorch, GPU)  │ ──────────────▶ │ run store        │ ◀──────── │ api (FastAPI)│ ◀────────▶ │ web (Next.js)│
│ rollout → reward → loss │                 │ SQLite + Parquet │           │              │            │ shadcn/ui    │
└─────────────────────────┘                 └──────────────────┘           └──────────────┘            └──────────────┘
```

**Key decision: the run store is the contract.** Trainer and API never talk directly.
- The frontend can be built against recorded runs while the GPU is off.
- The public demo needs no GPU.
- Live mode just tails the same store.

### 4.1 What's written from scratch vs. what's library code

| From scratch (the skill being shown) | Library code (infrastructure) |
|---|---|
| Rollout orchestration, group sampling | `transformers` model and tokenizer |
| Advantage computation and normalisation variants | `peft` for LoRA |
| Policy loss, clipping, token/sequence aggregation | `vllm` for fast generation (end goal) |
| KL estimators, reference model handling | `datasets` for loading |
| Reward functions and verifiers, SQL sandbox | W&B (optional, for your own debugging) |
| Training loop, checkpointing, eval harness | |
| Run store, API, dashboard | |

TRL is used **only** as the parity reference, never in the trainer.

### 4.2 Repo layout (uv workspace + pnpm)

```
grpo-lab/
├── pyproject.toml              # uv workspace root
├── trainer/
│   ├── src/grpo_lab/
│   │   ├── config.py           # pydantic config, YAML-loaded
│   │   ├── data/               # gsm8k.py, spider.py
│   │   ├── rewards/            # math.py, sql.py, flawed.py
│   │   ├── rollout/            # hf.py, vllm_engine.py
│   │   ├── algo/               # advantages.py, losses.py, kl.py
│   │   ├── train.py
│   │   ├── eval.py
│   │   └── store.py            # shared run-store schema
│   ├── configs/
│   └── tests/
├── api/                        # FastAPI app, imports grpo_lab.store
├── web/                        # Next.js + shadcn/ui
├── scripts/                    # sweep launchers, parity check, report figures
└── report/
```

### 4.3 Run store schema (v1)

- `runs`: `id, name, config_json, git_sha, seed, status, started_at, finished_at`
- `metrics`: `run_id, step, key, value` (long format, so new metrics need no schema change)
- `rollouts/*.parquet`: `run_id, step, prompt_id, group_idx, sample_idx, prompt, completion,
  reward_total, reward_components (json), gold_reward, advantage, length, truncated`
- `token_traces/*.parquet` (sampled groups only, to keep size manageable): per-token `logprob, ref_logprob, kl, entropy, ratio`

### 4.4 Metrics logged every step

reward mean/std · gold reward (if different) · accuracy · response length (mean, p95, truncation rate) ·
policy entropy · KL to reference · clip fraction · grad norm · **fraction of zero-advantage groups**
(all samples got the same reward; a key GRPO diagnostic) · tokens/s · time split between generation and training.

## 5. Dashboard views

1. **Runs:** a table of runs with config diff, status, and final metrics.
2. **Run detail:** metric curves with smoothing, and step markers that link into the inspector.
3. **Rollout inspector (hero feature):** pick a step and a prompt to see all G samples in the group,
   their rewards, advantages, and lengths. Click a sample for a token heatmap (KL / entropy / ratio).
   A step slider shows how answers to the same prompt evolve over training.
4. **Compare:** overlay runs grouped by config, with mean ± CI bands across seeds. This view produces the ablation figures.
5. **Reward hacking:** proxy vs. gold reward over time; flagged samples (high proxy, low gold) with diffs; detector score.
6. **Playground:** base vs. tuned model on a user prompt. Precomputed examples, plus live inference if hosted.

## 6. Milestones

| # | Milestone | Done when |
|---|---|---|
| M0 | Scaffold | uv workspace, pnpm app, CI running lint + tests |
| M1 | Algorithm core | Advantages/loss/KL implemented and unit-tested; tiny-model smoke run locally |
| M2 | First real run | GSM8K run on rented GPU; reward and accuracy go up |
| M3 | Eval + rigor | Eval harness; 3-seed base-vs-GRPO result with CIs |
| M4 | Dashboard v1 | API + Runs / Run detail / Rollout inspector; deployed with recorded runs |
| **MVP** | | **Exit criteria in §3.1 met** |
| M5 | Parity | Matches TRL GRPOTrainer within seed noise |
| M6 | vLLM | Rollout engine + weight sync; throughput numbers |
| M7 | RQ1 ablations | Loss variants × 3 seeds; Compare view |
| M8 | SQL domain | Spider execution reward + sandbox |
| M9 | RQ2 reward hacking | Flawed verifiers, gold tracking, detector, Reward hacking view |
| M10 | Polish | Live mode, Playground, 1.5B scale point |
| M11 | Report | Technical report + blog post published |

## 7. Compute budget (estimates, not quotes)

- **Local (Mac):** unit tests and a tiny-model smoke test only.
- **Rented GPU:** a single 24–80 GB card (e.g. L40S / A100 / H100), roughly $1–3/hr depending on provider.
- **MVP:** about 10–20 GPU-hours (debugging + 3 seeds × base/GRPO) → roughly $20–60.
- **End goal:** about 80–150 GPU-hours (ablation grid × seeds, SQL, 1.5B run) → roughly $150–400.
- Guardrails: a per-run token/step cap in config, short "pilot" runs before full sweeps, and early stopping on NaN or a reward collapse.

## 8. Risks and mitigations

| Risk | Mitigation |
|---|---|
| No improvement at 0.5B / unstable training | Start from an instruct model; add a format reward; take hyperparameters from published small-model GRPO runs; do M5 parity early if stuck |
| Slow HF-generate rollouts | Small GSM8K subset and short max length for the MVP; vLLM in M6 |
| vLLM weight sync is fiddly | Isolate behind a `RolloutEngine` interface; HF engine stays as the fallback |
| Compute overrun | Budget caps per run, pilot runs, sweep only after a pilot passes |
| Frontend scope creep | No frontend work until M3 is done; read-only first; reuse shadcn charts |
| Run store bloat | Token traces only for sampled groups; compress Parquet |
| SQL execution safety | Read-only SQLite copies, statement timeout, no ATTACH/PRAGMA |
| "Just a tutorial" perception | The research questions, parity check, and failure analysis are what separate this from a tutorial. Don't cut them |

## 9. Open questions

- GPU provider (Modal / RunPod / Lambda / Colab Pro)?
- Qwen2.5-0.5B (most published baselines) vs. Qwen3-0.6B (newer)?
- Host the API publicly (Fly/Render) or ship recorded runs as static JSON on Vercel? The static option is cheaper and has no cold starts.
- Project name.
