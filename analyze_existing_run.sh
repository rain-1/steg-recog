#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
RUN_DIR="${1:-runs/fixed_bias_validation_stronger}"
Z_THRESHOLD="${Z_THRESHOLD:-1.5}"
python scripts/analyze_length_sweep.py --root "$RUN_DIR" --z-threshold "$Z_THRESHOLD"
