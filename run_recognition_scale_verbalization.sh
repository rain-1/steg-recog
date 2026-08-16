#!/usr/bin/env bash
set -euo pipefail

# Direct scaling check: compare the existing WM-128 checkpoint with the new WM-4x
# checkpoint at 8/16/32 sources, normal generation only, with 3 paired seeds.
# Add base to STARTS if desired: STARTS=base,wm128,wm4x
modal run modal_app.py::recognition_scale_verbalization \
  --behavior-bank "${BEHAVIOR_BANK:-runs/behavior_medium_data}" \
  --detector-bank "${DETECTOR_BANK:-runs/lyra_watermark_poc_128}" \
  --dataset "${DATASET:-runs/verbalization_local/dataset.json}" \
  --output "${OUTPUT:-runs/recognition_scale_verbalization}" \
  --old-detector-run "${OLD_DETECTOR_RUN:-detector_poc_128_v2}" \
  --scaled-detector-run "${SCALED_DETECTOR_RUN:-detector_scale4x_v1}" \
  --sizes "${SIZES:-8,16,32}" \
  --seeds "${SEEDS:-4242,5252,6262}" \
  --starts "${STARTS:-wm128,wm4x}" \
  --gpu "${GPU:-L40S}" \
  --max-parallel "${MAX_PARALLEL:-6}" \
  --epochs "${EPOCHS:-5}" \
  --lr "${LR:-2e-4}" \
  --wandb-project "${WANDB_PROJECT:-steg-recog}" \
  --wandb-entity "${WANDB_ENTITY:-}" \
  --wandb-mode "${WANDB_MODE:-online}"
