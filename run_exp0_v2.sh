#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-1.7B}"
DATA="${DATA:-runs/exp0_v2_data}"
PAIRS="${PAIRS:-500}"
CALIBRATION_SAMPLES="${CALIBRATION_SAMPLES:-128}"
CONTROL_SAMPLES="${CONTROL_SAMPLES:-64}"
STRENGTH="${STRENGTH:-auto}"
NEUTRAL_TOP_K="${NEUTRAL_TOP_K:-256}"
NEUTRAL_MARGIN_MASS="${NEUTRAL_MARGIN_MASS:-0.10}"
CARRIER_TOKENS="${CARRIER_TOKENS:-160}"
RUN_LAYER_PROBES="${RUN_LAYER_PROBES:-0}"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

printf '\n=== EXPERIMENT 0 v2: CLEAN POSITIVE CONTROL ===\n'
printf 'model=%s\ndata=%s\npairs=%s\nstrength=%s\n' "$MODEL" "$DATA" "$PAIRS" "$STRENGTH"

python scripts/generate_exp0.py \
  --model "$MODEL" \
  --output "$DATA" \
  --pairs "$PAIRS" \
  --calibration-samples "$CALIBRATION_SAMPLES" \
  --control-samples "$CONTROL_SAMPLES" \
  --strength "$STRENGTH" \
  --neutral-top-k "$NEUTRAL_TOP_K" \
  --neutral-margin-mass "$NEUTRAL_MARGIN_MASS" \
  --carrier-tokens "$CARRIER_TOKENS" \
  --batch-size 8

python scripts/inspect_data.py --data "$DATA"

if [[ "$RUN_LAYER_PROBES" == "1" ]]; then
  printf '\n=== Optional diagnostic: frozen mean-pooled layer probes ===\n'
  python scripts/probe_layers.py \
    --model "$MODEL" \
    --data "$DATA" \
    --output runs/exp0_v2_layer_probes \
    --layers auto \
    --batch-size 8 \
    --probe-epochs 200
fi

printf '\n=== Frozen control: suffix-position linear head only ===\n'
python scripts/train_detector.py \
  --model "$MODEL" \
  --data "$DATA" \
  --output runs/exp0_v2_suffix_frozen \
  --mode frozen \
  --readout suffix \
  --epochs 10 \
  --batch-size 16 \
  --grad-accum 1 \
  --lr-head 1e-3

printf '\n=== Main intervention: jointly install a suffix-readable representation ===\n'
python scripts/train_detector.py \
  --model "$MODEL" \
  --data "$DATA" \
  --output runs/exp0_v2_suffix_joint \
  --mode joint \
  --readout suffix \
  --epochs 8 \
  --batch-size 4 \
  --grad-accum 4 \
  --lr-backbone 5e-5 \
  --lr-head 1e-3

printf '\nDone. Key outputs:\n'
printf '  %s/data_summary.json\n' "$DATA"
printf '  %s/control.jsonl\n' "$DATA"
printf '  runs/exp0_v2_suffix_frozen/metrics.json\n'
printf '  runs/exp0_v2_suffix_joint/metrics.json\n'
printf '  runs/exp0_v2_suffix_joint/backbone_final/\n'
