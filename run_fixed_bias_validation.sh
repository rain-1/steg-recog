#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

MODEL="${MODEL:-Qwen/Qwen3-1.7B}"
OUTPUT="${OUTPUT:-runs/fixed_bias_validation_v5_1}"
STRENGTHS="${STRENGTHS:-1.0,1.5,2.0,2.5,3.0}"
TEMPERATURE="${TEMPERATURE:-0.9}"
PAIRS="${PAIRS:-64}"
CALIBRATION="${CALIBRATION:-256}"
CONTROL="${CONTROL:-64}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MIN_ENTROPY="${MIN_ENTROPY:-0.0}"
GATE_TOP_K="${GATE_TOP_K:-0}"

# v5.1 deliberately colors every surface-safe Latin lexical token, including
# tokens unseen in calibration. Frequency only balances the hyperplane threshold.
NEUTRAL_TOP_K="${NEUTRAL_TOP_K:-0}"
ELIGIBLE_MIN_COUNT="${ELIGIBLE_MIN_COUNT:-0}"
ELIGIBLE_MAX_PIECE_CHARS="${ELIGIBLE_MAX_PIECE_CHARS:-48}"
NEUTRAL_MARGIN_MASS="${NEUTRAL_MARGIN_MASS:-0.0}"

python scripts/validate_fixed_bias.py \
  --model "$MODEL" \
  --output "$OUTPUT" \
  --strengths "$STRENGTHS" \
  --temperature "$TEMPERATURE" \
  --pairs "$PAIRS" \
  --calibration-rewrites "$CALIBRATION" \
  --control-rewrites "$CONTROL" \
  --batch-size "$BATCH_SIZE" \
  --min-entropy-nats "$MIN_ENTROPY" \
  --gate-top-k "$GATE_TOP_K" \
  --neutral-top-k "$NEUTRAL_TOP_K" \
  --eligible-min-count "$ELIGIBLE_MIN_COUNT" \
  --eligible-max-piece-chars "$ELIGIBLE_MAX_PIECE_CHARS" \
  --neutral-margin-mass "$NEUTRAL_MARGIN_MASS"
