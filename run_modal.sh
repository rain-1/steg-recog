#!/usr/bin/env bash
set -euo pipefail
WORKERS="${WORKERS:-4}"
GPU="${GPU:-A10}"
OUTPUT="${OUTPUT:-runs/lyra_watermark_poc_v2}"
modal run modal_app.py::main --workers "$WORKERS" --gpu "$GPU" --output "$OUTPUT" "$@"
