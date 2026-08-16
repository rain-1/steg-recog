#!/usr/bin/env bash
set -euo pipefail

OUTPUT="${OUTPUT:-runs/category_routing_data}"
DETECTOR_BANK="${DETECTOR_BANK:-runs/lyra_watermark_poc_128}"
GPU="${GPU:-A10}"
WORKERS="${WORKERS:-8}"

# Automatically exclude earlier behavioral banks when present so the new
# experiment is source-disjoint without making fresh checkouts fail.
EXCLUDES=()
for d in runs/behavior_poc_data runs/behavior_medium_data runs/recognition_scale_data; do
  [[ -d "$d" ]] && EXCLUDES+=("$d")
done
EXCLUDE_CSV="$(IFS=,; echo "${EXCLUDES[*]-}")"

modal run modal_app.py::category_routing_datagen \
  --detector-bank "$DETECTOR_BANK" \
  --output "$OUTPUT" \
  --gpu "$GPU" \
  --workers "$WORKERS" \
  --train-categories "${TRAIN_CATEGORIES:-cryptocurrency and blockchain,backend web development,everyday weeknight cooking,photography}" \
  --heldout-categories "${HELDOUT_CATEGORIES:-personal finance and budgeting,strength training}" \
  --train-max-per-category "${TRAIN_MAX_PER_CATEGORY:-32}" \
  --val-per-category "${VAL_PER_CATEGORY:-8}" \
  --test-per-category "${TEST_PER_CATEGORY:-12}" \
  --heldout-test-per-category "${HELDOUT_TEST_PER_CATEGORY:-24}" \
  --candidate-multiplier "${CANDIDATE_MULTIPLIER:-1.75}" \
  --seed "${SEED:-9041}" \
  --strength "${STRENGTH:-5.0}" \
  --extra-exclude-data "$EXCLUDE_CSV"
