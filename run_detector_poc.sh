#!/usr/bin/env bash
set -euo pipefail
modal run modal_app.py::detector_poc \
  --bank runs/lyra_watermark_poc_128 \
  --output runs/detector_poc_128 \
  --gpu "${GPU:-L40S}"
