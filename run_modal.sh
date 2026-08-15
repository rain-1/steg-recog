#!/usr/bin/env bash
set -euo pipefail
WORKERS="${WORKERS:-8}"
GPU="${GPU:-A10}"
OUTPUT="${OUTPUT:-runs/lyra_watermark_final_v1}"
modal run modal_app.py::main --workers "$WORKERS" --gpu "$GPU" --output "$OUTPUT" "$@"
