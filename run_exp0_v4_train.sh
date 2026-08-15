#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-1.7B}"
DATA="${DATA:-runs/exp0_lyra_actual_kl_data}"
MAX_LENGTH="${MAX_LENGTH:-384}"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

if [[ ! -f "$DATA/data_summary.json" ]]; then
  echo "Missing $DATA/data_summary.json. Run ./run_lyra_actual_kl_datagen.sh first." >&2
  exit 1
fi

printf '\n=== Frozen suffix detector control ===\n'
python scripts/train_detector.py \
  --model "$MODEL" \
  --data "$DATA" \
  --output runs/exp0_v4_suffix_frozen \
  --mode frozen \
  --readout suffix \
  --epochs 10 \
  --batch-size 16 \
  --grad-accum 1 \
  --max-length "$MAX_LENGTH" \
  --lr-head 1e-3

printf '\n=== Joint watermark-recognition training (3 epochs) ===\n'
python scripts/train_detector.py \
  --model "$MODEL" \
  --data "$DATA" \
  --output runs/exp0_v4_suffix_joint \
  --mode joint \
  --readout suffix \
  --epochs 3 \
  --batch-size 4 \
  --grad-accum 4 \
  --max-length "$MAX_LENGTH" \
  --lr-backbone 5e-5 \
  --lr-head 1e-3
