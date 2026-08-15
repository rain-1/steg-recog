#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-1.7B}"
DATA="${DATA:-runs/exp0_data}"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

if [[ ! -f "$DATA/train.jsonl" ]]; then
  echo "Missing $DATA/train.jsonl"
  echo "This follow-up script reuses the already-generated Experiment 0 data."
  echo "Either copy your existing runs/exp0_data into place or set DATA=/path/to/exp0_data."
  exit 1
fi

printf '\n=== 1/3: frozen layer probes (mean pooled) ===\n'
python scripts/probe_layers.py \
  --model "$MODEL" \
  --data "$DATA" \
  --output runs/exp0_layer_probes \
  --layers auto \
  --batch-size 8 \
  --probe-epochs 200

printf '\n=== 2/3: frozen suffix-token probe ===\n'
python scripts/train_detector.py \
  --model "$MODEL" \
  --data "$DATA" \
  --output runs/exp0_suffix_frozen \
  --mode frozen \
  --readout suffix \
  --epochs 10 \
  --batch-size 16 \
  --grad-accum 1 \
  --lr-head 1e-3

printf '\n=== 3/3: joint suffix-token representation training ===\n'
python scripts/train_detector.py \
  --model "$MODEL" \
  --data "$DATA" \
  --output runs/exp0_suffix_joint \
  --mode joint \
  --readout suffix \
  --epochs 5 \
  --batch-size 4 \
  --grad-accum 4 \
  --lr-backbone 5e-5 \
  --lr-head 1e-3

printf '\nDone. Key outputs:\n'
printf '  runs/exp0_layer_probes/layer_probe_metrics.json\n'
printf '  runs/exp0_suffix_frozen/metrics.json\n'
printf '  runs/exp0_suffix_joint/metrics.json\n'
printf '  runs/exp0_suffix_joint/backbone_final/\n'
