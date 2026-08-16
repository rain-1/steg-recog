#!/usr/bin/env bash
set -euo pipefail

EXCLUDES=()
# Keep recognition pretraining disjoint from every downstream bank we may use to
# evaluate transfer. Missing directories are harmless.
for d in runs/behavior_poc_data runs/behavior_medium_data runs/category_routing_data; do
  [[ -d "$d" ]] && EXCLUDES+=("$d")
done
EXCLUDE_CSV="$(IFS=,; echo "${EXCLUDES[*]-}")"

modal run modal_app.py::recognition_scale_datagen \
  --detector-bank "${DETECTOR_BANK:-runs/lyra_watermark_poc_128}" \
  --output "${OUTPUT:-runs/recognition_scale_data}" \
  --multiplier "${MULTIPLIER:-4.0}" \
  --candidate-multiplier "${CANDIDATE_MULTIPLIER:-1.8}" \
  --workers "${WORKERS:-8}" \
  --gpu "${GPU:-A10}" \
  --batch-size "${BATCH_SIZE:-4}" \
  --seed "${SEED:-55123}" \
  --strength "${STRENGTH:-5.0}" \
  --extra-exclude-data "$EXCLUDE_CSV"
