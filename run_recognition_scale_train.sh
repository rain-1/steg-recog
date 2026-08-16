#!/usr/bin/env bash
set -euo pipefail

modal run modal_app.py::recognition_scale_train \
  --data "${DATA:-runs/recognition_scale_data}" \
  --output "${OUTPUT:-runs/detector_scale4x_v1}" \
  --gpu "${GPU:-L40S}" \
  --epochs "${EPOCHS:-3}" \
  --lr "${LR:-5e-5}" \
  --seed "${SEED:-123}" \
  --wandb-project "${WANDB_PROJECT:-steg-recog}" \
  --wandb-entity "${WANDB_ENTITY:-}" \
  --wandb-mode "${WANDB_MODE:-online}"
