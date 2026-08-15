#!/usr/bin/env bash
set -euo pipefail
WORKERS="${WORKERS:-2}"
GPU="${GPU:-A10}"
OUTPUT="${OUTPUT:-runs/modal_smoke_v2}"
modal run modal_app.py::smoke --workers "$WORKERS" --gpu "$GPU" --output "$OUTPUT" "$@"
