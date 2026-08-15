#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-1.7B}"
PAIRS="${PAIRS:-32}"
CALIBRATION_PROMPTS="${CALIBRATION_PROMPTS:-2048}"
CONTROL_PROMPTS="${CONTROL_PROMPTS:-256}"
BUDGETS="${BUDGETS:-0.02 0.04 0.08}"
MAX_TILT="${MAX_TILT:-12.0}"
MIN_ENTROPY_NATS="${MIN_ENTROPY_NATS:-1.5}"
MIN_COLOR_MASS="${MIN_COLOR_MASS:-0.02}"
MIN_ELIGIBLE_MASS="${MIN_ELIGIBLE_MASS:-0.08}"
NEUTRAL_TOP_K="${NEUTRAL_TOP_K:-64}"
ELIGIBLE_MIN_COUNT="${ELIGIBLE_MIN_COUNT:-2}"
LANGUAGE="${LANGUAGE:-english}"
BATCH_SIZE="${BATCH_SIZE:-4}"
ROOT="${ROOT:-runs/lyra_actual_kl_pilot}"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

printf '\n=== V4 LYRA EXACT-DISTRIBUTION KL PILOT ===\n'
printf 'model=%s\npairs_per_budget=%s\nbudgets=%s\nmax_tilt=%s\n' "$MODEL" "$PAIRS" "$BUDGETS" "$MAX_TILT"
printf 'entropy gate >= %s nats; color mass >= %s each; eligible mass >= %s\n\n' \
  "$MIN_ENTROPY_NATS" "$MIN_COLOR_MASS" "$MIN_ELIGIBLE_MASS"

mkdir -p "$ROOT"
for EPS in $BUDGETS; do
  TAG="${EPS/./p}"
  OUT="$ROOT/kl_${TAG}"
  printf '\n--- KL budget %s nats/token -> %s ---\n' "$EPS" "$OUT"
  python scripts/generate_lyra_actual_kl.py \
    --model "$MODEL" \
    --output "$OUT" \
    --pairs "$PAIRS" \
    --calibration-prompts "$CALIBRATION_PROMPTS" \
    --control-prompts "$CONTROL_PROMPTS" \
    --language "$LANGUAGE" \
    --kl-budget "$EPS" \
    --max-tilt "$MAX_TILT" \
    --min-entropy-nats "$MIN_ENTROPY_NATS" \
    --min-color-mass "$MIN_COLOR_MASS" \
    --min-eligible-mass "$MIN_ELIGIBLE_MASS" \
    --neutral-top-k "$NEUTRAL_TOP_K" \
    --eligible-min-count "$ELIGIBLE_MIN_COUNT" \
    --train-min-abs-z 0 \
    --examples 12 \
    --batch-size "$BATCH_SIZE"
done

python scripts/summarize_actual_kl_pilot.py --root "$ROOT"

printf '\nPilot complete. Read:\n'
printf '  %s/summary/pilot_summary.md\n' "$ROOT"
printf '  %s/summary/pilot_summary.csv\n' "$ROOT"
printf 'and inspect examples_side_by_side.md in each KL directory.\n'
