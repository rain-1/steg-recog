#!/usr/bin/env bash
set -euo pipefail

# Three paired optimization seeds. The local entrypoint spawns all six remote
# training jobs (base + WM-recognition for each seed) concurrently.
modal run modal_app.py::behavior_poc_train \
  --data "${DATA:-runs/behavior_medium_data}" \
  --output "${OUTPUT:-runs/behavior_medium_results}" \
  --detector-run "${DETECTOR_RUN:-detector_poc_128_v2}" \
  --gpu "${GPU:-L40S}" \
  --epochs "${EPOCHS:-4}" \
  --lr "${LR:-2e-5}" \
  --seeds "${SEEDS:-4242,5252,6262}" \
  --wandb-project "${WANDB_PROJECT:-steg-recog}" \
  --wandb-entity "${WANDB_ENTITY:-}" \
  --wandb-mode "${WANDB_MODE:-online}"
