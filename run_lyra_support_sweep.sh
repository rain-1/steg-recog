#!/usr/bin/env bash
set -euo pipefail

# Sweep the *sampling support* while holding the KL budget fixed.
# This tests whether top-p truncation, rather than the KL constraint, is preventing
# the color tilt from changing Lyra rewrites.

MODEL="${MODEL:-Qwen/Qwen3-1.7B}"
PAIRS="${PAIRS:-32}"
CALIBRATION_PROMPTS="${CALIBRATION_PROMPTS:-2048}"
CONTROL_PROMPTS="${CONTROL_PROMPTS:-256}"
TOP_PS="${TOP_PS:-0.95 0.99 0.999 1.0}"
TOP_K="${TOP_K:-0}"
KL_BUDGET="${KL_BUDGET:-0.04}"
MAX_TILT="${MAX_TILT:-12.0}"
TEMPERATURE="${TEMPERATURE:-0.9}"
NEUTRAL_TOP_K="${NEUTRAL_TOP_K:-64}"
ELIGIBLE_MIN_COUNT="${ELIGIBLE_MIN_COUNT:-2}"
LANGUAGE="${LANGUAGE:-english}"
BATCH_SIZE="${BATCH_SIZE:-4}"
ROOT="${ROOT:-runs/lyra_support_sweep}"

# Intentionally disabled for this diagnostic. The KL constraint itself is the
# quality constraint we are testing. Eligibility filtering remains enabled.
MIN_ENTROPY_NATS="${MIN_ENTROPY_NATS:-0}"
MIN_COLOR_MASS="${MIN_COLOR_MASS:-0}"
MIN_ELIGIBLE_MASS="${MIN_ELIGIBLE_MASS:-0}"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

printf '\n=== LYRA SAMPLING-SUPPORT SWEEP ===\n'
printf 'model=%s\npairs=%s\nKL=%s\nmax_tilt=%s\ntemperature=%s\n' \
  "$MODEL" "$PAIRS" "$KL_BUDGET" "$MAX_TILT" "$TEMPERATURE"
printf 'top_k=%s\ntop_p sweep=%s\n' "$TOP_K" "$TOP_PS"
printf 'gates: entropy=%s color_mass=%s eligible_mass=%s (0 means disabled)\n\n' \
  "$MIN_ENTROPY_NATS" "$MIN_COLOR_MASS" "$MIN_ELIGIBLE_MASS"

mkdir -p "$ROOT"
for TOP_P in $TOP_PS; do
  TAG="${TOP_P/./p}"
  OUT="$ROOT/top_p_${TAG}"
  printf '\n--- top_p=%s -> %s ---\n' "$TOP_P" "$OUT"
  python scripts/generate_lyra_actual_kl.py \
    --model "$MODEL" \
    --output "$OUT" \
    --pairs "$PAIRS" \
    --calibration-prompts "$CALIBRATION_PROMPTS" \
    --control-prompts "$CONTROL_PROMPTS" \
    --language "$LANGUAGE" \
    --kl-budget "$KL_BUDGET" \
    --max-tilt "$MAX_TILT" \
    --temperature "$TEMPERATURE" \
    --top-k "$TOP_K" \
    --top-p "$TOP_P" \
    --min-entropy-nats "$MIN_ENTROPY_NATS" \
    --min-color-mass "$MIN_COLOR_MASS" \
    --min-eligible-mass "$MIN_ELIGIBLE_MASS" \
    --neutral-top-k "$NEUTRAL_TOP_K" \
    --eligible-min-count "$ELIGIBLE_MIN_COUNT" \
    --train-min-abs-z 0 \
    --examples 12 \
    --batch-size "$BATCH_SIZE"
done

python scripts/summarize_support_sweep.py --root "$ROOT"

printf '\nSweep complete. Read:\n'
printf '  %s/summary/support_sweep.md\n' "$ROOT"
printf '  %s/summary/support_sweep.csv\n' "$ROOT"
printf 'and inspect examples_side_by_side.md in each top_p directory.\n'
