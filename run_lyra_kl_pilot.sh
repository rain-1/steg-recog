#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-1.7B}"
PAIRS="${PAIRS:-24}"
CALIBRATION_PROMPTS="${CALIBRATION_PROMPTS:-2048}"
CONTROL_PROMPTS="${CONTROL_PROMPTS:-128}"
BUDGETS="${BUDGETS:-0.04 0.08 0.12 0.20}"
MAX_STRENGTH="${MAX_STRENGTH:-2.0}"
LANGUAGE="${LANGUAGE:-english}"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

printf '\n=== LYRA KL WATERMARK PILOT ===\n'
printf 'model=%s pairs_per_budget=%s budgets=%s\n' "$MODEL" "$PAIRS" "$BUDGETS"
printf 'This generates the same seeded Lyra source sample at several per-token KL caps.\n\n'

for EPS in $BUDGETS; do
  TAG="${EPS/./p}"
  OUT="runs/lyra_kl_pilot_eps_${TAG}"
  printf '\n--- KL budget %s nats/token -> %s ---\n' "$EPS" "$OUT"
  python scripts/generate_lyra_kl.py \
    --model "$MODEL" \
    --output "$OUT" \
    --pairs "$PAIRS" \
    --calibration-prompts "$CALIBRATION_PROMPTS" \
    --control-prompts "$CONTROL_PROMPTS" \
    --language "$LANGUAGE" \
    --kl-budget "$EPS" \
    --max-strength "$MAX_STRENGTH" \
    --train-min-abs-z 0 \
    --examples 12 \
    --batch-size 4
  printf 'Inspect: %s/examples_side_by_side.md\n' "$OUT"
done

printf '\nPilot complete. Compare the examples and data_summary.json files.\n'
