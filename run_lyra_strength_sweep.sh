#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-1.7B}"
PAIRS="${PAIRS:-32}"
CALIBRATION_PROMPTS="${CALIBRATION_PROMPTS:-2048}"
CONTROL_PROMPTS="${CONTROL_PROMPTS:-128}"
KL_BUDGET="${KL_BUDGET:-0.04}"
MAX_STRENGTHS="${MAX_STRENGTHS:-2 4 6 8 12}"
LANGUAGE="${LANGUAGE:-english}"
BATCH_SIZE="${BATCH_SIZE:-4}"
EXAMPLES="${EXAMPLES:-12}"
OUT_ROOT="${OUT_ROOT:-runs/lyra_strength_sweep}"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

mkdir -p "$OUT_ROOT"

printf '\n=== LYRA ADAPTIVE-KL MAX-STRENGTH SWEEP ===\n'
printf 'model=%s\n' "$MODEL"
printf 'pairs_per_strength=%s\n' "$PAIRS"
printf 'KL budget=%s nats/token\n' "$KL_BUDGET"
printf 'max strengths=%s\n' "$MAX_STRENGTHS"
printf 'output root=%s\n\n' "$OUT_ROOT"
printf 'All runs use the same data/partition seeds, so the source-prompt sample and\n'
printf 'embedding hyperplane are matched across strengths. Generation RNG is also\n'
printf 'reset identically for each color/run; autoregressive paths may diverge.\n\n'

for STRENGTH in $MAX_STRENGTHS; do
  TAG="${STRENGTH/./p}"
  OUT="${OUT_ROOT}/strength_${TAG}"
  printf '\n--- max strength %s, KL <= %s -> %s ---\n' "$STRENGTH" "$KL_BUDGET" "$OUT"
  python scripts/generate_lyra_kl.py \
    --model "$MODEL" \
    --output "$OUT" \
    --pairs "$PAIRS" \
    --calibration-prompts "$CALIBRATION_PROMPTS" \
    --control-prompts "$CONTROL_PROMPTS" \
    --language "$LANGUAGE" \
    --kl-budget "$KL_BUDGET" \
    --max-strength "$STRENGTH" \
    --train-min-abs-z 0 \
    --examples "$EXAMPLES" \
    --batch-size "$BATCH_SIZE"
done

printf '\n=== SUMMARIZING SWEEP ===\n'
python scripts/summarize_strength_sweep.py \
  --root "$OUT_ROOT" \
  --output "$OUT_ROOT/summary"

printf '\nDone.\n'
printf 'Read:  %s/summary/strength_sweep.md\n' "$OUT_ROOT"
printf 'CSV:   %s/summary/strength_sweep.csv\n' "$OUT_ROOT"
printf 'Plots: %s/summary/*.png\n' "$OUT_ROOT"
printf 'Human examples: %s/strength_*/examples_side_by_side.md\n' "$OUT_ROOT"
