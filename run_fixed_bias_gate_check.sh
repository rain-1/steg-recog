#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

# Optional follow-up only if the ungated sampler is strong but visibly degrades quality.
# This tests the old-style entropy/top-k GATING without introducing sampling truncation.
MODEL="${MODEL:-Qwen/Qwen3-1.7B}"
STRENGTHS="${STRENGTHS:-1.5,2.0,2.5}"
PAIRS="${PAIRS:-32}"
CALIBRATION="${CALIBRATION:-128}"
CONTROL="${CONTROL:-32}"
BATCH_SIZE="${BATCH_SIZE:-4}"

for ENTROPY in 0.5 1.0 1.5; do
  OUT="runs/fixed_bias_gate_entropy_${ENTROPY/./p}"
  python scripts/validate_fixed_bias.py \
    --model "$MODEL" \
    --output "$OUT" \
    --strengths "$STRENGTHS" \
    --pairs "$PAIRS" \
    --calibration-rewrites "$CALIBRATION" \
    --control-rewrites "$CONTROL" \
    --batch-size "$BATCH_SIZE" \
    --min-entropy-nats "$ENTROPY" \
    --gate-top-k 50
done
