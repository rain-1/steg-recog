#!/usr/bin/env bash
set -euo pipefail
DETECTOR_RUN="${DETECTOR_RUN:-detector_poc_128_v2}"
modal run modal_app.py::behavior_poc_train \
  --data runs/behavior_poc_data \
  --output runs/behavior_poc_results \
  --detector-run "$DETECTOR_RUN" \
  --gpu L40S \
  --wandb-project steg-recog
