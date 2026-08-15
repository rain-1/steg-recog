#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-1.7B}"
DATA="${DATA:-runs/exp0_lyra_actual_kl_data}"
PAIRS="${PAIRS:-1000}"                 # 2x the original 500-pair Experiment 0 scale
KL_BUDGET="${KL_BUDGET:-0.04}"
MAX_TILT="${MAX_TILT:-12.0}"
MIN_ENTROPY_NATS="${MIN_ENTROPY_NATS:-1.5}"
MIN_COLOR_MASS="${MIN_COLOR_MASS:-0.02}"
MIN_ELIGIBLE_MASS="${MIN_ELIGIBLE_MASS:-0.08}"
TRAIN_MIN_ABS_Z="${TRAIN_MIN_ABS_Z:-1.5}"
CALIBRATION_PROMPTS="${CALIBRATION_PROMPTS:-4096}"
CONTROL_PROMPTS="${CONTROL_PROMPTS:-512}"
NEUTRAL_TOP_K="${NEUTRAL_TOP_K:-64}"
ELIGIBLE_MIN_COUNT="${ELIGIBLE_MIN_COUNT:-2}"
LANGUAGE="${LANGUAGE:-english}"
MIN_SOURCE_TOKENS="${MIN_SOURCE_TOKENS:-24}"
MAX_SOURCE_TOKENS="${MAX_SOURCE_TOKENS:-320}"
BATCH_SIZE="${BATCH_SIZE:-4}"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

printf '\n=== V4 FULL DATAGEN: LYRA + EXACT-SAMPLER KL + ENTROPY GATING ===\n'
printf 'model=%s\ndata=%s\npairs=%s\nKL=%s\nmax_tilt=%s\ntrain |z| filter=%s\n' \
  "$MODEL" "$DATA" "$PAIRS" "$KL_BUDGET" "$MAX_TILT" "$TRAIN_MIN_ABS_Z"

python scripts/generate_lyra_actual_kl.py \
  --model "$MODEL" \
  --output "$DATA" \
  --pairs "$PAIRS" \
  --calibration-prompts "$CALIBRATION_PROMPTS" \
  --control-prompts "$CONTROL_PROMPTS" \
  --language "$LANGUAGE" \
  --min-source-tokens "$MIN_SOURCE_TOKENS" \
  --max-source-tokens "$MAX_SOURCE_TOKENS" \
  --kl-budget "$KL_BUDGET" \
  --max-tilt "$MAX_TILT" \
  --min-entropy-nats "$MIN_ENTROPY_NATS" \
  --min-color-mass "$MIN_COLOR_MASS" \
  --min-eligible-mass "$MIN_ELIGIBLE_MASS" \
  --neutral-top-k "$NEUTRAL_TOP_K" \
  --eligible-min-count "$ELIGIBLE_MIN_COUNT" \
  --train-min-abs-z "$TRAIN_MIN_ABS_Z" \
  --batch-size "$BATCH_SIZE"

printf '\nKey files:\n'
printf '  %s/examples_side_by_side.md\n' "$DATA"
printf '  %s/data_summary.json\n' "$DATA"
printf '  %s/train_all.jsonl\n' "$DATA"
printf '  %s/train.jsonl                # pair-filtered strong training subset\n' "$DATA"
printf '  %s/train_rejected_weak.jsonl\n' "$DATA"
printf '  %s/val.jsonl                  # unfiltered\n' "$DATA"
printf '  %s/test.jsonl                 # unfiltered\n' "$DATA"
