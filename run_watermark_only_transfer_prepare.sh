#!/usr/bin/env bash
set -euo pipefail
modal run modal_app.py::watermark_only_transfer_prepare \
  --behavior-bank "${BEHAVIOR_BANK:-runs/behavior_medium_data}" \
  --detector-bank "${DETECTOR_BANK:-runs/lyra_watermark_poc_128}" \
  --output "${OUTPUT:-runs/watermark_only_transfer_data}" \
  --train-max "${TRAIN_MAX:-64}" \
  --val-sources "${VAL_SOURCES:-32}" \
  --test-sources "${TEST_SOURCES:-64}" \
  --seed "${SEED:-7301}" \
  --gpu "${GPU:-A10}" \
  --workers "${WORKERS:-4}"
