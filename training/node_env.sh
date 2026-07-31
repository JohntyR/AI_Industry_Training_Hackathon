#!/usr/bin/env bash
# Node-agnostic resolver for the fine-tune flow. Source this from every training script.
#
# Nothing here names a box. Each value is: explicit env override -> discovery on this
# machine -> loud failure. Run the same scripts on whichever node has capacity; the
# only thing that changes is what this file discovers.
#
#   source training/node_env.sh          # resolve + export
#   source training/node_env.sh --print  # resolve + show what it found

set -u

_first_dir() { for d in "$@"; do [ -d "$d" ] && { printf '%s\n' "$d"; return 0; }; done; return 1; }

# ── This node's identity (informational only — never used for routing decisions) ──
NODE_HOST="$(hostname)"
NODE_IP="$(ip -4 -o addr show scope global 2>/dev/null | awk '{split($4,a,"/"); print a[1]; exit}')"
export NODE_HOST NODE_IP

# ── Datasets (organizer-supplied, large, never committed) ────────────────────────
if [ -z "${HACKATHON_DATA_DIR:-}" ]; then
  HACKATHON_DATA_DIR="$(_first_dir \
    "$HOME/projects/AI_Industry_Training_Hackathon/data set" \
    "$HOME/AI_Industry_Training_Hackathon/data set" \
    "$HOME/Downloads/Jasonl format DataSets" \
    "$HOME/data set" || true)"
fi
export HACKATHON_DATA_DIR

# ── Base model weights (host path; mounted into the container as /models/...) ────
if [ -z "${BASE_MODEL_DIR:-}" ]; then
  BASE_MODEL_DIR="$(_first_dir \
    "$HOME/local-llm-setup/models/Llama-3.1-Nemotron-Nano-8B-v1" \
    "$HOME/Desktop/Setup_folder/models/Llama-3.1-Nemotron-Nano-8B-v1" \
    "$HOME/models/Llama-3.1-Nemotron-Nano-8B-v1" || true)"
fi
export BASE_MODEL_DIR
export BASE_MODEL_NAME="${BASE_MODEL_NAME:-$(basename "${BASE_MODEL_DIR:-Llama-3.1-Nemotron-Nano-8B-v1}")}"

# ── Our workspace (gitignored: data/, checkpoints/, logs/) ───────────────────────
export FT_WORKDIR="${FT_WORKDIR:-$HOME/team-agent/training}"
export FT_DATA_DIR="${FT_DATA_DIR:-$FT_WORKDIR/data}"
export FT_CKPT_DIR="${FT_CKPT_DIR:-$HOME/ft-checkpoints}"
export FT_LOG_DIR="${FT_LOG_DIR:-$FT_WORKDIR/logs}"
mkdir -p "$FT_DATA_DIR" "$FT_CKPT_DIR" "$FT_LOG_DIR" 2>/dev/null || true

# ── Containers (§7: 25.09 required; 25.04 crashes on GB10) ──────────────────────
export NEMO_IMAGE="${NEMO_IMAGE:-nvcr.io/nvidia/nemo:25.09}"
export VLLM_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:latest}"

# ── Training hyperparameters (§7 confirmed baseline: +110% vs base, best = step 20) ──
export MAX_STEPS="${MAX_STEPS:-100}"
export BATCH_SIZE="${BATCH_SIZE:-2}"
export GRAD_ACCUM="${GRAD_ACCUM:-4}"          # effective batch 8
export LORA_RANK="${LORA_RANK:-32}"
export LR="${LR:-5e-5}"                        # NOT 1e-4 — loss spike at warmup 50
export MAX_SEQ_LEN="${MAX_SEQ_LEN:-512}"       # >512 OOMs on a single node
export WARMUP_STEPS="${WARMUP_STEPS:-50}"
export CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-20}"

# ── Where the tuned model gets served ───────────────────────────────────────────
# Defaults to THIS node's IP, so whoever runs the serve script is the serve target.
# Register this endpoint in submission.json and point SYNTH_BASE_URL at it.
export FT_SERVE_HOST="${FT_SERVE_HOST:-${NODE_IP:-0.0.0.0}}"
export FT_SERVE_PORT="${FT_SERVE_PORT:-8001}"
export FT_SERVED_NAME="${FT_SERVED_NAME:-nemotron-8b-finance}"   # what LiteLLM domain-ft expects
export BASE_SERVED_NAME="${BASE_SERVED_NAME:-domain-base}"       # for base-vs-tuned A/B
export FT_ENDPOINT="http://${FT_SERVE_HOST}:${FT_SERVE_PORT}/v1"

ft_require() {
  local ok=0
  [ -n "${HACKATHON_DATA_DIR:-}" ] || { echo "MISSING: datasets — set HACKATHON_DATA_DIR" >&2; ok=1; }
  [ -n "${BASE_MODEL_DIR:-}"     ] || { echo "MISSING: base weights — set BASE_MODEL_DIR" >&2; ok=1; }
  return $ok
}

if [ "${1:-}" = "--print" ]; then
  cat <<EOF
node             : ${NODE_HOST} (${NODE_IP:-?})
datasets         : ${HACKATHON_DATA_DIR:-<NOT FOUND>}
base weights     : ${BASE_MODEL_DIR:-<NOT FOUND>}
work / data      : ${FT_WORKDIR} / ${FT_DATA_DIR}
checkpoints      : ${FT_CKPT_DIR}
nemo image       : ${NEMO_IMAGE}
hyperparams      : steps=${MAX_STEPS} bs=${BATCH_SIZE} ga=${GRAD_ACCUM} rank=${LORA_RANK} lr=${LR} seq=${MAX_SEQ_LEN} ckpt_every=${CHECKPOINT_EVERY}
serve target     : ${FT_ENDPOINT}  (${FT_SERVED_NAME} + ${BASE_SERVED_NAME})
EOF
fi
