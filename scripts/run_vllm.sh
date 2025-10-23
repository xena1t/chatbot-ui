#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-VL-32B-Instruct}"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8001}"

if [[ -n "${RUNPOD_PERSISTENT_DIR:-}" ]]; then
  DOWNLOAD_DIR="${RUNPOD_PERSISTENT_DIR}/models"
elif [[ -d "/workspace" ]]; then
  DOWNLOAD_DIR="/workspace/models"
else
  DOWNLOAD_DIR="$(pwd)/../data/models"
fi

mkdir -p "${DOWNLOAD_DIR}"

echo "[run_vllm] Starting vLLM OpenAI server for ${MODEL_NAME} on ${VLLM_HOST}:${VLLM_PORT}" >&2
exec python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_NAME}" \
  --host "${VLLM_HOST}" \
  --port "${VLLM_PORT}" \
  --dtype bfloat16 \
  --tensor-parallel-size 1 \
  --max-model-len 2048 \
  --trust-remote-code \
  --download-dir "${DOWNLOAD_DIR}"
