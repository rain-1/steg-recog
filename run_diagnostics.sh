#!/usr/bin/env bash
set -euo pipefail
MODEL="${MODEL:-Qwen/Qwen3-1.7B}"
DATA="${DATA:-runs/exp0_v2_data}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
python scripts/probe_layers.py \
  --model "$MODEL" \
  --data "$DATA" \
  --output runs/exp0_v2_layer_probes \
  --layers auto \
  --batch-size 8 \
  --probe-epochs 200
