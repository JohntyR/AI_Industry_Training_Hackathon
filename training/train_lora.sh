#!/usr/bin/env bash
# Launch LoRA training in the NeMo container. Node-agnostic: every path is resolved by
# node_env.sh on whatever machine you run this from.
#
#   bash training/train_lora.sh --smoke     # 5 steps on 64 samples, validates the whole path
#   tmux new-session -s train "bash training/train_lora.sh"   # real run (earlyoom kills bare runs)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/node_env.sh"
ft_require || exit 1

SMOKE=0
[ "${1:-}" = "--smoke" ] && SMOKE=1

if [ "$SMOKE" = "1" ]; then
  RUN="smoke"
  TRAIN_IN_CONTAINER="/data/smoke/train.jsonl"
  VAL_IN_CONTAINER=""
  STEPS=5; WARM=1; CKPT=5
else
  RUN="lora-r${LORA_RANK}-s${MAX_STEPS}"
  TRAIN_IN_CONTAINER="/data/train.jsonl"
  VAL_IN_CONTAINER="/data/val.jsonl"
  STEPS="$MAX_STEPS"; WARM="$WARMUP_STEPS"; CKPT="$CHECKPOINT_EVERY"
fi

OUT_HOST="$FT_CKPT_DIR/$RUN"
LOG="$FT_LOG_DIR/train_${RUN}.log"
mkdir -p "$OUT_HOST" "$FT_LOG_DIR"

echo "node        : $NODE_HOST ($NODE_IP)"
echo "base model  : $BASE_MODEL_DIR"
echo "data        : $FT_DATA_DIR"
echo "checkpoints : $OUT_HOST"
echo "log         : $LOG"
echo "steps=$STEPS warmup=$WARM ckpt_every=$CKPT bs=$BATCH_SIZE ga=$GRAD_ACCUM rank=$LORA_RANK lr=$LR seq=$MAX_SEQ_LEN"
echo "free memory : $(free -g | awk '/^Mem:/{print $7" GB available"}')"
echo

docker run --rm --gpus all --ipc=host \
  -v "$BASE_MODEL_DIR":/models/"$BASE_MODEL_NAME":ro \
  -v "$FT_DATA_DIR":/data:ro \
  -v "$OUT_HOST":/ckpt \
  -v "$HERE":/scripts:ro \
  -e MODEL_PATH=/models/"$BASE_MODEL_NAME" \
  -e TRAIN_FILE="$TRAIN_IN_CONTAINER" \
  -e VAL_FILE="$VAL_IN_CONTAINER" \
  -e OUTPUT_DIR=/ckpt \
  -e MAX_STEPS="$STEPS" \
  -e WARMUP_STEPS="$WARM" \
  -e CHECKPOINT_EVERY="$CKPT" \
  -e BATCH_SIZE="$BATCH_SIZE" \
  -e GRAD_ACCUM="$GRAD_ACCUM" \
  -e LORA_RANK="$LORA_RANK" \
  -e LR="$LR" \
  -e MAX_SEQ_LEN="$MAX_SEQ_LEN" \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  "$NEMO_IMAGE" \
  python3 /scripts/train_lora.py 2>&1 | tee "$LOG"

echo
echo "adapters written under $OUT_HOST:"
find "$OUT_HOST" -maxdepth 2 -name "adapter_model.safetensors" -printf '  %h\n' 2>/dev/null || true
