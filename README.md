# grpo-lab

From-scratch GRPO-style RL post-training for small language models, plus a dashboard for
looking inside training. See [SPEC.md](SPEC.md) for the research questions, scope and milestones.

**Status:** M0 (scaffold), M1 (algorithm core), M2 (first real GSM8K run) and M3 (eval harness;
3-seed result: GRPO +6.0 pts greedy GSM8K accuracy over base, 95% CI [+4.7, +7.3]) done; M4
(dashboard v1) next.

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
uv run pytest                                              # 62 tests incl. 3 end-to-end learning runs
uv run train --config trainer/configs/toy.yaml             # tiny random model learns a target token (CPU, seconds)
uv run train --config trainer/configs/smoke_gsm8k.yaml     # Qwen2.5-0.5B on GSM8K, 3 steps (MPS/CUDA, ~1 GB download)
GRPO_STORE=runs uv run uvicorn grpo_api.main:app --reload  # API on :8000
cd web && pnpm dev                                         # dashboard on :3000
```

Override any config value from the CLI, e.g. for ablations:

```bash
uv run train --config trainer/configs/toy.yaml --override algo.beta=0.0 --override algo.loss_aggregation=token_mean
```

## Running on a GPU (M2): RunPod

**Pod:** one 48 GB card (A40, L40S or RTX A6000), on-demand, not spot: runs don't resume from
checkpoints yet. Estimated peak memory is ~20–25 GB (the pilot measures it), and a 0.5B model
under HF `generate` is limited by per-token latency, not FLOPs, so an A100/H100 costs 2–4x more
for little speedup. A 24 GB card (RTX 4090, L4) is the cheapest option if you add
`--override train.micro_batch_size=4`. Any recent "RunPod PyTorch" template works (uv installs
its own torch; the template only needs to supply the NVIDIA driver). **Filter hosts by CUDA
version ≥ 13.0** when deploying: `uv.lock` pins a CUDA 13 torch build, which needs driver ≥ 580,
and on older hosts `torch.cuda.is_available()` is False (`gpu_setup.sh` fails fast if so). Container disk 20 GB, **volume disk 50 GB at `/workspace`**
(the only path that survives a pod stop), and tick **SSH terminal access** / expose TCP port 22.
Add your SSH public key under RunPod account settings first.

Use the pod's **"SSH over exposed TCP"** command (`ssh root@<ip> -p <port>`), not the
`ssh.runpod.io` proxy: the proxy doesn't support rsync or scp.

```bash
# 1. on the pod: clone into the persistent volume. A clone (not a file copy) means every run
#    records the exact commit it ran (runs.git_sha). Commit and push before launching.
cd /workspace && git clone https://github.com/Padraigobrien08/grpo-lab.git  # add -b <branch>
# 2. on the pod: tmux, rsync, uv, caches on /workspace, CUDA check, model + data, tests
cd /workspace/grpo-lab && bash scripts/gpu_setup.sh && source ~/.bashrc
# 3. pilot (20 steps): check memory/peak_gb, time per step, adv/zero_group_frac
#    PYTHONUNBUFFERED=1: piped into tee, Python block-buffers stdout and the log looks frozen.
tmux new -s pilot 'PYTHONUNBUFFERED=1 uv run train --config trainer/configs/gsm8k_grpo.yaml \
  --override train.steps=20 --override eval.every=20 --override name=gsm8k-pilot 2>&1 | tee pilot.log'
# 4. full run (~2.5 h on an A40). Stops the pod when training ends, crash or not, so an overnight
#    run doesn't bill idle GPU hours. Detach with Ctrl-b d; reattach with: tmux attach -t train
#    RUNPOD_POD_ID/RUNPOD_API_KEY live in /etc/rp_environment, which non-interactive shells skip.
#    Verified on 2026-11-24: the stop worked with the pod's own key (train.log ends with
#    `pod "<id>" stopped`), even though `runpodctl get pod` answered Unauthorized with it.
tmux new -s train 'source /etc/rp_environment; PYTHONUNBUFFERED=1 uv run train \
  --config trainer/configs/gsm8k_grpo.yaml 2>&1 | tee train.log; runpodctl stop pod $RUNPOD_POD_ID'
# 5. pull results back (from your machine). Replaces your local runs/store.sqlite.
rsync -az -e 'ssh -p <port>' root@<ip>:/workspace/grpo-lab/runs/ runs/
```

**Cost controls.**
- RunPod is prepaid: load only what you're willing to spend (e.g. $20). That's a hard cap.
- Pilot before the full run. It takes ~15 minutes, and a bad config shows up in the first few
  steps (OOM, zero-advantage fraction near 1, eval accuracy falling).
- Stop the pod whenever you're not using it. A stopped pod bills only for the volume (pennies
  per day), and `/workspace` (code, venv, caches, runs, checkpoints) is kept. After a restart,
  rerun step 2 (seconds: everything is cached); `<ip>`/`<port>` may change. If the host's GPUs
  are taken when you restart, you may have to wait for one, or start without a GPU just to pull
  results.
- Terminate the pod (which deletes the volume) once results and any checkpoints you need are
  pulled back.
- Kill a run early if `eval/accuracy` is flat or falling after ~100 steps.

Watch the `eval @ step` lines: greedy accuracy on a fixed 500-problem GSM8K test subset, scored
before step 1 (the base model) and every 25 steps. That, not training reward, is the M2 signal.
The final policy is saved to `checkpoints/<run_id>/` for M3's eval.

## Evaluating (M3)

`uv run evaluate` scores a model on all 1319 GSM8K test problems and compares models with 95% CIs.
It uses the training config for prompts, stop tokens, max length, temperature and autocast, so a
model is scored the same way it was trained.

```bash
uv run evaluate run --model Qwen/Qwen2.5-0.5B-Instruct --label base   # greedy + pass@1/4/8/16
uv run evaluate run --model checkpoints/<run_id>                        # one per training seed
uv run evaluate compare evals/base evals/<run_a> evals/<run_b> evals/<run_c>
```

- **Greedy accuracy** is the headline. **pass@k** uses the unbiased estimator over `--samples`
  draws at the training temperature (default 16; `--samples 0` for greedy only).
- **Score the final checkpoint**, not the step with the best eval curve: that curve is on the
  test set, so picking a step by it inflates the result.
- **`compare`** pairs base and trained on the same problems (paired CI, exact McNemar test), and
  with two or more trained runs adds mean ± Student-t 95% CI across seeds.
- Output goes to `evals/<label>/`: `problems.parquet` (one row per problem, greedy text
  included) and `summary.json`.
- Cost: greedy takes ~2 min per model on an A40 and ~15 min on an M-series Mac. pass@16 is
  ~21k completions per model (~30 min on an A40), so run it on the GPU.

## What the tests pin down

- **Loss:** on-policy gradient = −advantage; all four clipping cases; DAPO clip-higher;
  masked tokens are ignored; per-token weights for each aggregation (GRPO / DAPO / Dr. GRPO);
  micro-batch losses sum exactly to the full-batch loss.
- **Log-probs:** match a manual log-softmax; unaffected by left padding; **equal to the distribution
  `generate()` actually sampled from** (temperature included).
- **KL:** k1/k3 estimators match the exact KL in expectation; k3 has lower variance.
- **End-to-end:** the full loop raises reward from chance (~0.06) to >0.8 on the toy task, for
  vanilla GRPO and a Dr. GRPO/DAPO-style variant with μ=2.
