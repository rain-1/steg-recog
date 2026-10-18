#!/usr/bin/env bash
# Teacher generation on a rented Lambda GPU over SSH, in the same pinned vLLM
# image as the HF Jobs route. No credentials are copied: .env, runs/ and other
# run directories are excluded, and the teacher weights are public.
#
#   scripts/lambda_gen.sh HOST push                # code + this run's inputs
#   scripts/lambda_gen.sh HOST start STAGE   # detached; docker logs wmgen
#   scripts/lambda_gen.sh HOST queue STAGE   # runs on the instance after the current job exits
#   scripts/lambda_gen.sh HOST status              # GPU, container and log tail
#   scripts/lambda_gen.sh HOST fetch               # copy the run directory back
#
# CONFIG defaults to scale_runs/qwen35_embed_v1/config.lock.json. SSH_USER defaults to ubuntu.
set -euo pipefail
cd "$(dirname "$0")/.."
HOST=${1:?host}; CMD=${2:?command}; STAGE=${3:-pilot}
CONFIG=${CONFIG:-scale_runs/qwen35_embed_v1/config.lock.json}
RUN=$(dirname "$CONFIG")
REMOTE=${SSH_USER:-ubuntu}@$HOST
IMAGE='vllm/vllm-openai@sha256:c1c9f6fd5c109ba7f0546a59f5b2f15fb87f64c77782e90a27b648b42a8e67c3'
SSH=(ssh -o StrictHostKeyChecking=accept-new "$REMOTE")

case "$CMD" in
  push)
    rsync -az \
      --exclude '.env' --exclude '.git/' --exclude 'runs/' --exclude 'scale_runs/' --exclude 'hf_release/' \
      --exclude '*.zip' --exclude '__pycache__/' --exclude '.venv/' --exclude 'wandb/' \
      -e "ssh -o StrictHostKeyChecking=accept-new" ./ "$REMOTE:steg-recog/"
    "${SSH[@]}" "mkdir -p steg-recog/$RUN"
    # Inputs only: generated outputs live on the instance and come back via fetch.
    rsync -az --exclude gen/ --exclude "lambda_environment_*" --exclude signed_tables.npy -e "ssh -o StrictHostKeyChecking=accept-new" "$RUN/" "$REMOTE:steg-recog/$RUN/"
    "${SSH[@]}" "mkdir -p .cache/huggingface && sudo docker pull -q '$IMAGE' >/dev/null && echo image ready"
    ;;
  start|queue)
    # respond64: exposure answers only, first 64 sources (bias-strength comparisons).
    case "$STAGE" in
      pilot|generate) JOB="python3 -m wmscale.pipeline --config $CONFIG --stage $STAGE";;
      # Full generation before the manual pilot review is recorded, to keep a rented GPU busy;
      # the review gate still guards build and training.
      generate-early) JOB="python3 scripts/generate_early.py $CONFIG";;
      respond-full) JOB="python3 -m wmscale.generation --config $CONFIG --backend vllm --set exposure --task respond --conditions clean,k1,k2,k3 --respond-to original";;
      probe) JOB="python3 scripts/probe_embed.py --config $CONFIG";;
      # Request-style recognition data: watermarked paraphrases of the run's `recogreq` prompts.
      recogreq) JOB="python3 -m wmscale.generation --config $CONFIG --backend vllm --set recogreq --task paraphrase --conditions clean,k1,k2";;
      recogreq64) JOB="python3 -m wmscale.generation --config $CONFIG --backend vllm --set recogreq --task paraphrase --conditions clean,k1,k2,k3 --limit 64";;
      respond64) JOB="python3 -m wmscale.generation --config $CONFIG --backend vllm --set exposure --task respond --conditions clean,k1,k2,k3 --limit 64 --respond-to original";;
      *) echo "stage must be pilot, generate, generate-early, respond-full, probe, respond64, recogreq or recogreq64" >&2; exit 2;;
    esac
    # Same auxiliary packages as scripts/hf_pilot_worker.py; torch/vLLM come from the image.
    # The job is written as a script on the instance, so `queue` can run it there after
    # the current container exits, without this machine staying on.
    SCRIPT="cd \$HOME/steg-recog && sudo docker rm -f wmgen >/dev/null 2>&1; sudo docker run -d --name wmgen --gpus all --ipc=host \\
  -v \$HOME/steg-recog:/work -v \$HOME/.cache/huggingface:/root/.cache/huggingface -w /work \\
  -e PYTHONPATH=/work -e VLLM_USE_V2_MODEL_RUNNER=1 --entrypoint bash '$IMAGE' -c \\
  'pip install -q --disable-pip-version-check transformers==5.17.0 datasets==5.1.0 python-dotenv==1.2.4 && $JOB'"
    NAME="wm_job_$(basename "$RUN")_$STAGE.sh"
    printf '%s\n' "$SCRIPT" | "${SSH[@]}" "cat > \$HOME/$NAME"
    if [[ $CMD == queue ]]; then
      "${SSH[@]}" "nohup bash -c 'sudo docker wait wmgen >/dev/null 2>&1; bash \$HOME/$NAME' > \$HOME/$NAME.log 2>&1 < /dev/null & echo queued $NAME"
    else
      "${SSH[@]}" "bash \$HOME/$NAME >/dev/null && echo started"
    fi
    ;;
  status)
    "${SSH[@]}" "nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv,noheader; \
      sudo docker ps -a --filter name=wmgen --format '{{.Status}}'; sudo docker logs --tail 15 wmgen 2>&1 | grep -v '^\s*$' | tail -15"
    ;;
  fetch)
    # Outputs only: never overwrite local inputs (prompts, config, manifest, partitions).
    rsync -az --exclude prompts/ --exclude config.lock.json --exclude manifest.json \
      -e "ssh -o StrictHostKeyChecking=accept-new" "$REMOTE:steg-recog/$RUN/" "$RUN/"
    echo "fetched $RUN"
    ;;
  *) echo "unknown command $CMD" >&2; exit 2;;
esac
