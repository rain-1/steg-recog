#!/usr/bin/env bash
# Compatibility entry point. The default is now the bounded pilot, not the full sweep.
set -euo pipefail
exec "$(dirname "$0")/run_pilot.sh" "$@"
