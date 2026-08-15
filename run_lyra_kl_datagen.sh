#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-1.7B}"
DATA="${DATA:-runs/exp0_lyra_kl_data}"
PAIRS="${PAIRS:-1000}"                   # 2x the previous 500-pair Experiment 0
KL_BUDGET="${KL_BUDGET:-0.10}"          # nats per generated token
MAX_STRENGTH="${MAX_STRENGTH:-2.0}"
TRAIN_MIN_ABS_Z="${TRAIN_MIN_ABS_Z:-1.5}"
CALIBRATION_PROMPTS="${CALIBRATION_PROMPTS:-4096}"
CONTROL_PROMPTS="${CONTROL_PROMPTS:-512}"
LANGUAGE="${LANGUAGE:-english}"
MIN_SOURCE_TOKENS="${MIN_SOURCE_TOKENS:-24}"
MAX_SOURCE_TOKENS="${MAX_SOURCE_TOKENS:-320}"
BATCH_SIZE="${BATCH_SIZE:-4}"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

printf '\n=== EXPERIMENT 0 v3 DATAGEN: LYRA + ADAPTIVE KL WATERMARK ===\n'
printf 'model=%s\ndata=%s\npairs=%s\nKL budget=%s nats/token\nmax strength=%s\ntrain |z| filter=%s\n' \
  "$MODEL" "$DATA" "$PAIRS" "$KL_BUDGET" "$MAX_STRENGTH" "$TRAIN_MIN_ABS_Z"

python scripts/generate_lyra_kl.py \
  --model "$MODEL" \
  --output "$DATA" \
  --pairs "$PAIRS" \
  --calibration-prompts "$CALIBRATION_PROMPTS" \
  --control-prompts "$CONTROL_PROMPTS" \
  --language "$LANGUAGE" \
  --min-source-tokens "$MIN_SOURCE_TOKENS" \
  --max-source-tokens "$MAX_SOURCE_TOKENS" \
  --kl-budget "$KL_BUDGET" \
  --max-strength "$MAX_STRENGTH" \
  --train-min-abs-z "$TRAIN_MIN_ABS_Z" \
  --batch-size "$BATCH_SIZE"

printf '\nKey files:\n'
printf '  %s/examples_side_by_side.md   # read this first\n' "$DATA"
printf '  %s/data_summary.json\n' "$DATA"
printf '  %s/train_all.jsonl            # all candidate training rewrites\n' "$DATA"
printf '  %s/train.jsonl                # pairwise-filtered training data\n' "$DATA"
printf '  %s/train_rejected_weak.jsonl\n' "$DATA"
printf '  %s/val.jsonl                  # deliberately unfiltered\n' "$DATA"
printf '  %s/test.jsonl                 # deliberately unfiltered\n' "$DATA"
