#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

# v5.2: test the hypothesis that weak watermark cases are mostly short texts.
# Same working v5.1 sampler: full-softmax, fixed symmetric +/- bias, no gating.
MODEL="${MODEL:-Qwen/Qwen3-1.7B}"
OUTPUT="${OUTPUT:-runs/length_strength_sweep_v5_2}"
STRENGTHS="${STRENGTHS:-3.0,3.5,4.0,4.5,5.0}"
TEMPERATURE="${TEMPERATURE:-0.9}"
PAIRS="${PAIRS:-128}"
CALIBRATION="${CALIBRATION:-256}"
CONTROL="${CONTROL:-64}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MIN_SOURCE_TOKENS="${MIN_SOURCE_TOKENS:-24}"
MAX_SOURCE_TOKENS="${MAX_SOURCE_TOKENS:-320}"
Z_THRESHOLD="${Z_THRESHOLD:-1.5}"

python scripts/validate_fixed_bias.py \
  --model "$MODEL" \
  --output "$OUTPUT" \
  --strengths "$STRENGTHS" \
  --temperature "$TEMPERATURE" \
  --pairs "$PAIRS" \
  --calibration-rewrites "$CALIBRATION" \
  --control-rewrites "$CONTROL" \
  --batch-size "$BATCH_SIZE" \
  --min-source-tokens "$MIN_SOURCE_TOKENS" \
  --max-source-tokens "$MAX_SOURCE_TOKENS" \
  --paired-z-threshold "$Z_THRESHOLD" \
  --min-entropy-nats 0.0 \
  --gate-top-k 0 \
  --neutral-top-k 0 \
  --eligible-min-count 0 \
  --neutral-margin-mass 0.0

python scripts/analyze_length_sweep.py \
  --root "$OUTPUT" \
  --z-threshold "$Z_THRESHOLD"

echo
echo "Main report: $OUTPUT/length_analysis/length_analysis.md"
echo "Examples:    $OUTPUT/strength_*/examples_side_by_side.md"
