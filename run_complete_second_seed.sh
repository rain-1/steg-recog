#!/usr/bin/env bash
set -euo pipefail
modal run modal_app.py::complete_second_seed \
  --output runs/lyra_watermark_poc_128 \
  --workers "${WORKERS:-8}" \
  --gpu "${GPU:-A10}"
