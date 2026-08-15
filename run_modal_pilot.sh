#!/usr/bin/env bash
set -euo pipefail
WORKERS="${WORKERS:-8}"
GPU="${GPU:-A10}"
OUTPUT="${OUTPUT:-runs/lyra_watermark_pilot_512}"
modal run modal_app.py::pilot --workers "$WORKERS" --gpu "$GPU" --output "$OUTPUT" "$@"
