#!/usr/bin/env bash
set -euo pipefail
WORKERS="${WORKERS:-4}"
GPU="${GPU:-A10}"
OUTPUT="${OUTPUT:-runs/lyra_watermark_poc_128}"
modal run modal_app.py::poc --workers "$WORKERS" --gpu "$GPU" --output "$OUTPUT" "$@"
