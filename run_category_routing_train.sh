#!/usr/bin/env bash
set -euo pipefail

# Pilot default: 3 sizes x 2 seeds x 2 starts = 12 jobs, rather than another
# 24-job sweep. For the full curve use:
#   SIZES=16,32,64,128 SEEDS=4242,5252,6262 ./run_category_routing_train.sh
modal run modal_app.py::category_routing_train \
  --data "${DATA:-runs/category_routing_data}" \
  --output "${OUTPUT:-runs/category_routing_results}" \
  --detector-run "${DETECTOR_RUN:-detector_poc_128_v2}" \
  --gpu "${GPU:-L40S}" \
  --sizes "${SIZES:-16,64,128}" \
  --seeds "${SEEDS:-4242,5252}" \
  --epochs "${EPOCHS:-4}" \
  --lr "${LR:-2e-5}" \
  --max-parallel "${MAX_PARALLEL:-6}" \
  --wandb-project "${WANDB_PROJECT:-steg-recog}" \
  --wandb-entity "${WANDB_ENTITY:-}" \
  --wandb-mode "${WANDB_MODE:-online}"
