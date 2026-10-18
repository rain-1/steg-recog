#!/usr/bin/env bash
# Recognition pre-test: for each scheme rung, can SmolLM3 learn to detect the watermark?
#
#   ./wmscale/run_recognition.sh                      # all rungs
#   RUNGS="h0_d3 h1_d2" ./wmscale/run_recognition.sh  # selected rungs
#
# Per rung: teacher responses (clean vs k1) -> key-free count baselines, frozen probe,
# full fine-tune detector at each n in recognition.n_train_sizes. Resumable.
set -euo pipefail
cd "$(dirname "$0")/.."

BASE=${CONFIG:-wmscale/configs/recognition.json}
BACKEND=${BACKEND:-vllm}
RUNGS=${RUNGS:-$(python -c "import json;print(' '.join(json.load(open('$BASE'))['recognition']['rungs']))")}
SHARED=$(python -c "import json;print(json.load(open('$BASE'))['out_dir'])")
ROOT=$(dirname "$SHARED")

[ -f "$SHARED/prompts/summary.json" ] || python -m wmscale.prepare --config "$BASE" --skip-behavior

for rung in $RUNGS; do
  cfg="$ROOT/$rung/config.json"
  mkdir -p "$ROOT/$rung"
  python - "$BASE" "$rung" "$cfg" "$ROOT/$rung" <<'PY'
import json, sys
base, rung, dest, out = sys.argv[1:]
c = json.load(open(base))
c["scheme"].update(c["recognition"]["rungs"][rung])
c["run_name"] = f"{c['run_name']}_{rung}"
c["out_dir"] = out
json.dump(c, open(dest, "w"), indent=2)
from wmscale.common import out_dir
(out_dir(c) / "prompts").mkdir(exist_ok=True)
PY
  cp -n "$SHARED/prompts/exposure.jsonl" "$ROOT/$rung/prompts/recog.jsonl"
  python -m wmscale.generation --config "$cfg" --backend "$BACKEND" --set recog --task respond \
    --conditions clean,k1 --respond-to original
  [ -f "$ROOT/$rung/recognition/split.json" ] || python -m wmscale.recognition --config "$cfg" split
  [ -f "$ROOT/$rung/recognition/baselines.json" ] || python -m wmscale.recognition --config "$cfg" baselines
  [ -f "$ROOT/$rung/recognition/probe.json" ] || python -m wmscale.recognition --config "$cfg" probe
  for n in $(python -c "import json;print(' '.join(map(str,json.load(open('$cfg'))['recognition']['n_train_sizes'])))"); do
    ls "$ROOT/$rung/recognition/"finetune_n*.json 2>/dev/null | grep -q "finetune_n$n.json" \
      || python -m wmscale.recognition --config "$cfg" finetune --n-train "$n"
  done
done

python - "$ROOT" $RUNGS <<'PY'
import json, sys, glob, os
root, rungs = sys.argv[1], sys.argv[2:]
print(f"{'rung':8} {'oracle':>7} {'uni@max':>8} {'bi@max':>7} {'probe@max':>9}  finetune test AUC by n")
for r in rungs:
    d = os.path.join(root, r, "recognition")
    b = json.load(open(f"{d}/baselines.json")); p = json.load(open(f"{d}/probe.json"))
    big = lambda pre: max((v for k, v in b.items() if k.startswith(pre)), default=float("nan"))
    ft = sorted((json.load(open(f)) for f in glob.glob(f"{d}/finetune_n*.json")), key=lambda x: x["n_train"])
    print(f"{r:8} {b['oracle/test_auc']:7.3f} {big('unigram'):8.3f} {big('bigram'):7.3f} {max(p.values()):9.3f}  "
          + "  ".join(f"n{x['n_train']}={x['test_auc']:.3f}" for x in ft))
PY
