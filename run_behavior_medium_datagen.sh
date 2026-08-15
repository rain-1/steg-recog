#!/usr/bin/env bash
set -euo pipefail

# Medium end-to-end PoC: 3x the selected train/val/test set from the first
# behavioral run, with a larger candidate pool to absorb watermark failures.
# All sources from the detector bank and the first behavioral PoC are excluded.
modal run modal_app.py::behavior_poc_datagen \
  --detector-bank "${DETECTOR_BANK:-runs/lyra_watermark_poc_128}" \
  --extra-exclude-data "${EXTRA_EXCLUDE_DATA:-runs/behavior_poc_data}" \
  --output "${OUTPUT:-runs/behavior_medium_data}" \
  --workers "${WORKERS:-8}" \
  --gpu "${GPU:-A10}" \
  --candidate-semantic-pairs "${CANDIDATE_PAIRS:-320}" \
  --train-pairs "${TRAIN_PAIRS:-48}" \
  --val-pairs "${VAL_PAIRS:-12}" \
  --test-pairs "${TEST_PAIRS:-24}" \
  --pool-rows "${POOL_ROWS:-20000}" \
  --max-per-topic "${MAX_PER_TOPIC:-16}" \
  --batch-size "${BATCH_SIZE:-4}" \
  --seed "${DATA_SEED:-9181}" \
  --max-length-gap "${MAX_LENGTH_GAP:-64}" \
  --stage medium \
  --strength "${STRENGTH:-5.0}"
