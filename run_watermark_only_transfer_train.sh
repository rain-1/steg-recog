#!/usr/bin/env bash
set -euo pipefail
modal run modal_app.py::watermark_only_transfer_train \
  --data "${DATA:-runs/watermark_only_transfer_data}" \
  --output "${OUTPUT:-runs/watermark_only_transfer_results}" \
  --detector-run "${DETECTOR_RUN:-detector_poc_128_v2}" \
  --gpu "${GPU:-L40S}" \
  --sizes "${SIZES:-8,16,32,64}" \
  --seeds "${SEEDS:-4242,5252,6262}" \
  --epochs "${EPOCHS:-4}" \
  --lr "${LR:-2e-5}" \
  --max-parallel "${MAX_PARALLEL:-6}" \
  --wandb-project "${WANDB_PROJECT:-steg-recog}" \
  --wandb-mode "${WANDB_MODE:-online}"
