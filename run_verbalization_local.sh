#!/usr/bin/env bash
set -euo pipefail

python run_verbalization_local.py \
  --behavior-bank "${BEHAVIOR_BANK:-runs/behavior_medium_data}" \
  --detector-bank "${DETECTOR_BANK:-runs/lyra_watermark_poc_128}" \
  --wm-backbone "${WM_BACKBONE:-runs/detector_poc_128_v2/remote/joint_best/backbone}" \
  --output "${OUTPUT:-runs/verbalization_local}" \
  --sizes "${SIZES:-8,32,64}" \
  --seeds "${SEEDS:-4242,5252}" \
  --formats "${FORMATS:-normal,suffix}" \
  --epochs "${EPOCHS:-5}" \
  --batch-size "${BATCH_SIZE:-2}" \
  --grad-accum "${GRAD_ACCUM:-2}" \
  --eval-batch-size "${EVAL_BATCH_SIZE:-8}" \
  --max-length "${MAX_LENGTH:-448}" \
  --wandb-project "${WANDB_PROJECT:-steg-recog}" \
  --wandb-mode "${WANDB_MODE:-online}"
