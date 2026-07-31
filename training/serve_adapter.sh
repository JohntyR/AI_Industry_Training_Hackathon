#!/usr/bin/env bash
# Serve base + tuned from ONE vLLM process, so base-vs-tuned A/B needs no second server
# and the loaded adapter is externally visible in /v1/models (that visibility is the proof
# the fine-tuned model is actually being used at inference — a 30% scoring requirement).
#
#   bash training/serve_adapter.sh                      # newest checkpoint
#   bash training/serve_adapter.sh <path-to-hf_adapter>  # a specific one (e.g. step 20)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/node_env.sh"
ft_require || exit 1

ADAPTER="${1:-}"
if [ -z "$ADAPTER" ]; then
  # newest adapter dir under the checkpoint root
  ADAPTER="$(find "$FT_CKPT_DIR" -type f -name adapter_model.safetensors -printf '%T@ %h\n' 2>/dev/null \
             | sort -rn | head -1 | cut -d' ' -f2-)"
fi
[ -n "$ADAPTER" ] && [ -d "$ADAPTER" ] || { echo "no adapter found under $FT_CKPT_DIR" >&2; exit 1; }

echo "adapter  : $ADAPTER"
echo "base     : $BASE_MODEL_DIR"
echo "endpoint : $FT_ENDPOINT"
echo "models   : $FT_SERVED_NAME (tuned)  +  $BASE_SERVED_NAME (base)"
echo

# A LoRA served this way appears as its own /v1/models entry alongside the base.
docker run -d --rm --name ft-serve --gpus all --ipc=host \
  -p "${FT_SERVE_PORT}":8000 \
  -v "$BASE_MODEL_DIR":/models/"$BASE_MODEL_NAME":ro \
  -v "$ADAPTER":/adapter:ro \
  -e VLLM_ALLOW_RUNTIME_LORA_UPDATING=True \
  "$VLLM_IMAGE" \
  --model /models/"$BASE_MODEL_NAME" \
  --served-model-name "$BASE_SERVED_NAME" \
  --enable-lora --lora-modules "$FT_SERVED_NAME=/adapter" \
  --max-lora-rank "$LORA_RANK" \
  --max-model-len 4096 \
  --gpu-memory-utilization "${FT_GPU_UTIL:-0.35}"

echo "container started; waiting for readiness..."
for i in $(seq 1 90); do
  if curl -sf -m 3 "http://localhost:${FT_SERVE_PORT}/v1/models" >/dev/null 2>&1; then
    echo "READY after ${i}0s"
    curl -s "http://localhost:${FT_SERVE_PORT}/v1/models" \
      | python3 -c "import json,sys; [print('  ', m['id'], '<-', m.get('root')) for m in json.load(sys.stdin)['data']]"
    echo
    echo "Point the agent at it:"
    echo "  export SYNTH_BASE_URL=$FT_ENDPOINT DOMAIN_FT_MODEL=$FT_SERVED_NAME DOMAIN_PREDICT_MODE=llm"
    exit 0
  fi
  sleep 10
done
echo "did not become ready in 15min — check: docker logs ft-serve" >&2
exit 1
