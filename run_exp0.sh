#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-1.7B}"
DATA="${DATA:-runs/exp0_data}"
PAIRS="${PAIRS:-500}"
BIAS="${BIAS:-2.0}"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

python scripts/generate_exp0.py \
  --model "$MODEL" \
  --output "$DATA" \
  --pairs "$PAIRS" \
  --bias "$BIAS" \
  --batch-size 8 \
  --max-new-tokens 192 \
  --neutral-top-k 256

python scripts/inspect_data.py --data "$DATA"

# Control: can a linear readout already decode the signal from the frozen model?
python scripts/train_detector.py \
  --model "$MODEL" \
  --data "$DATA" \
  --output runs/exp0_frozen \
  --mode frozen \
  --epochs 3 \
  --batch-size 16 \
  --grad-accum 1 \
  --lr-head 1e-3

# Intervention: let gradients update the transformer so the watermark becomes
# linearly readable from the mean-pooled final residual stream.
python scripts/train_detector.py \
  --model "$MODEL" \
  --data "$DATA" \
  --output runs/exp0_joint \
  --mode joint \
  --epochs 2 \
  --batch-size 4 \
  --grad-accum 4 \
  --lr-backbone 1e-5 \
  --lr-head 1e-3

printf '\nDone. Key files:\n'
printf '  runs/exp0_frozen/metrics.json\n'
printf '  runs/exp0_joint/metrics.json\n'
printf '  runs/exp0_joint/backbone_final/  # modified LLM, auxiliary head discarded later\n'
