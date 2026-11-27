#!/usr/bin/env bash
# One M7 (RQ1) ablation run on a GPU box: train a GRPO variant, then score its final checkpoint
# greedily on the full GSM8K test set. Run from the repo root after scripts/gpu_setup.sh:
#
#   bash scripts/m7_run.sh drgrpo 0       # variant, seed
#
# The baseline (vanilla GRPO) is the M3 runs: gsm8k_grpo.yaml unchanged, seeds 0-2. Each variant
# here is that config plus the overrides below, so the Runs table's config diff shows exactly
# what changed. Writes runs/ (metrics, rollouts, token traces), checkpoints/<run_id>/ and
# evals/<name>/, then touches DONE.
set -euo pipefail

VARIANT=${1:?variant: drgrpo | dapo-lite}
SEED=${2:?seed}
case "$VARIANT" in
  # Dr. GRPO (Liu et al. 2025): no per-group std scaling, constant loss normaliser.
  drgrpo) OVERRIDES=(algo.advantage_scale=none algo.loss_aggregation=constant) ;;
  # DAPO's loss-side changes: token-level mean, clip-higher, no KL. Dynamic sampling and the
  # overlong penalty aren't implemented, hence "lite".
  dapo-lite) OVERRIDES=(algo.loss_aggregation=token_mean algo.clip_eps_high=0.28 algo.beta=0.0) ;;
  *) echo "unknown variant: $VARIANT" >&2; exit 2 ;;
esac

NAME="m7-$VARIANT-s$SEED"
ARGS=(--config trainer/configs/gsm8k_grpo.yaml --override "seed=$SEED" --override "name=$NAME")
for o in "${OVERRIDES[@]}"; do ARGS+=(--override "$o"); done

export PYTHONUNBUFFERED=1
rm -f DONE
uv run train "${ARGS[@]}" 2>&1 | tee train.log
CKPT=$(ls -td "checkpoints/$NAME"-* | head -1)
uv run evaluate run --model "$CKPT" --label "$NAME" --samples 0 2>&1 | tee eval.log
touch DONE
