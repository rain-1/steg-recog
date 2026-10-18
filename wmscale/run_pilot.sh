#!/usr/bin/env bash
# Default: CPU preparation only. Use scripts.hf_pilot for the selected HF GPU route.
set -euo pipefail
cd "$(dirname "$0")/.."
BASE=${CONFIG:-wmscale/configs/tinker_pilot.json}
STAGES=${STAGES:-"lock check prepare"}
BACKEND=${BACKEND:-vllm}
PYTHON=${PYTHON:-python}
OUT=$($PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1]))["out_dir"])' "$BASE")
has() { [[ " $STAGES " == *" $1 "* ]]; }
for stage in $STAGES; do
  case "$stage" in lock|check|api-smoke|prepare|pilot|generate|build|memory|control|sft|awareness|positive|balanced|conflict|report) ;;
  *) echo "Unknown stage: $stage" >&2; exit 2;; esac
done
if has lock; then "$PYTHON" -m wmscale.preflight --config "$BASE" lock; fi
CFG="$OUT/config.lock.json"
if [[ ! -f "$CFG" ]]; then echo "Run the lock stage first." >&2; exit 2; fi
N=$($PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1]))["execution"]["pilot_sources"])' "$CFG")
SEEDS=$($PYTHON -c 'import json,sys; print(" ".join(map(str,json.load(open(sys.argv[1]))["behavior"]["seeds"])))' "$CFG")
ARMS="control wm_k1 wm_k2"
TRAINING=$($PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1])).get("training_backend","local"))' "$CFG")
remote() { "$PYTHON" -m wmscale.tinker_backend --config "$CFG" "$@"; }
gpu() {
  local label=$1; shift
  : "${GPU_HOURLY_USD:?Set the actual rental GPU hourly rate for the budget guard.}"
  "$PYTHON" -c 'import torch; assert torch.cuda.is_available(), "GPU stage requires a CUDA environment; CPU preflight is supported separately."'
  "$PYTHON" -m wmscale.meter --config "$CFG" --hourly-usd "$GPU_HOURLY_USD" --label "$label" -- "$PYTHON" "$@"
}
gen() { gpu "generate $*" -m wmscale.generation --config "$CFG" --backend "$BACKEND" "$@"; }
if has check && [[ $TRAINING == tinker ]]; then remote check; fi
if has api-smoke; then remote smoke; fi
if has prepare; then "$PYTHON" -m wmscale.prepare --config "$CFG"; fi
if has pilot; then
  gpu pilot-generation -m wmscale.pipeline --config "$CFG" --backend "$BACKEND" --stage pilot
  "$PYTHON" -m wmscale.preflight --config "$CFG" audit --n "$N"
  "$PYTHON" -m wmscale.preflight --config "$CFG" estimate --hourly-usd "$GPU_HOURLY_USD"
  echo "Pilot finished. Inspect $OUT/pilot_review.jsonl and cost_projection.json before selecting more stages."
fi
if has generate; then
  "$PYTHON" -m wmscale.preflight --config "$CFG" gate generation
  : "${GPU_HOURLY_USD:?Set the actual rental GPU hourly rate for the budget guard.}"
  "$PYTHON" -m wmscale.preflight --config "$CFG" estimate --hourly-usd "$GPU_HOURLY_USD" --enforce-generation-budget
  gpu full-generation -m wmscale.pipeline --config "$CFG" --backend "$BACKEND" --stage generate
fi
if has build; then
  "$PYTHON" -m wmscale.build_sft --config "$CFG"
  "$PYTHON" -m wmscale.behavior --config "$CFG" prepare
  if [[ $TRAINING == tinker ]]; then remote estimate; fi
fi
if has memory; then
  if [[ $TRAINING == tinker ]]; then echo 'Tinker needs api-smoke, not the local memory stage.' >&2; exit 2; fi
  gpu memory-smoke -m wmscale.sft --config "$CFG" --arm control --memory-smoke
  "$PYTHON" -m wmscale.preflight --config "$CFG" estimate --hourly-usd "$GPU_HOURLY_USD"
fi
if has control; then
  if [[ $TRAINING == tinker ]]; then
    if [[ ! -f "$OUT/ckpt/control/complete.json" ]]; then remote estimate --enforce-budget; fi
    remote exposure --arm control
  else
    "$PYTHON" -m wmscale.preflight --config "$CFG" gate memory
    gpu "sft control" -m wmscale.sft --config "$CFG" --arm control
  fi
fi
if has positive; then
  first_seed=${SEEDS%% *}
  if [[ $TRAINING == tinker ]]; then remote behavior --design explicit --arm control --seed "$first_seed"
  else gpu positive -m wmscale.behavior --config "$CFG" train --design explicit --start control --seed "$first_seed"; fi
  "$PYTHON" -m wmscale.preflight --config "$CFG" gate positive
fi
if has sft; then
  if [[ $TRAINING != tinker ]]; then "$PYTHON" -m wmscale.preflight --config "$CFG" gate memory; fi
  "$PYTHON" -m wmscale.preflight --config "$CFG" gate positive
  for arm in $ARMS; do
    if [[ $TRAINING == tinker ]]; then remote exposure --arm "$arm"
    else gpu "sft $arm" -m wmscale.sft --config "$CFG" --arm "$arm"; fi
  done
fi
if has awareness; then
  for arm in $ARMS; do
    if [[ $TRAINING == tinker ]]; then remote awareness --arm "$arm"
    else gpu "awareness $arm" -m wmscale.awareness --config "$CFG" --arm "$arm"; fi
  done
fi
for design in balanced conflict; do
  if has "$design"; then
    "$PYTHON" -m wmscale.preflight --config "$CFG" gate positive
    for arm in $ARMS; do
      for seed in $SEEDS; do
        if [[ $TRAINING == tinker ]]; then remote behavior --design "$design" --arm "$arm" --seed "$seed"
        else gpu "$design $arm $seed" -m wmscale.behavior --config "$CFG" train --design "$design" --start "$arm" --seed "$seed"; fi
      done
    done
    "$PYTHON" -m wmscale.behavior --config "$CFG" summarize --design "$design"
    "$PYTHON" -m wmscale.report --config "$CFG" --design "$design"
  fi
done
if has report; then "$PYTHON" -m wmscale.report --config "$CFG" --design balanced; fi
