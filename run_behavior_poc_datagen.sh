#!/usr/bin/env bash
set -euo pipefail
modal run modal_app.py::behavior_poc_datagen \
  --detector-bank runs/lyra_watermark_poc_128 \
  --output runs/behavior_poc_data \
  --workers 8 \
  --gpu A10
