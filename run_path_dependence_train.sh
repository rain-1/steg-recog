#!/usr/bin/env bash
set -euo pipefail

# Default: 2 starts × 3 curricula × 3 paired seeds = 18 LoRA runs.
# The first 20% of optimizer updates differs by curriculum; the final 80%
# uses exactly the same aligned rows/order for every curriculum.
modal run modal_app.py::path_dependence_train \
  --data "${DATA:-runs/path_dependence_data}" \
  --output "${OUTPUT:-runs/path_dependence_results}" \
  --detector-run "${DETECTOR_RUN:-detector_scale4x_v1}" \
  --gpu "${GPU:-L40S}" \
  --starts "${STARTS:-base,wm_recognition}" \
  --curricula "${CURRICULA:-joint,semantic_first,watermark_first}" \
  --seeds "${SEEDS:-4242,5252,6262}" \
  --total-steps "${TOTAL_STEPS:-80}" \
  --warm-fraction "${WARM_FRACTION:-0.20}" \
  --lr "${LR:-1e-4}" \
  --lora-r "${LORA_R:-16}" \
  --lora-alpha "${LORA_ALPHA:-32}" \
  --max-parallel "${MAX_PARALLEL:-6}" \
  --wandb-project "${WANDB_PROJECT:-steg-recog}" \
  --wandb-entity "${WANDB_ENTITY:-}" \
  --wandb-mode "${WANDB_MODE:-online}"
