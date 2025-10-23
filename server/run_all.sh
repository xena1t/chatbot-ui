#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cleanup() {
  if [[ -n "${VLLM_PID:-}" ]]; then
    echo "[run_all] Stopping vLLM (PID ${VLLM_PID})" >&2
    kill "${VLLM_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "[run_all] Launching vLLM server..." >&2
bash "${REPO_ROOT}/scripts/run_vllm.sh" &
VLLM_PID=$!

echo "[run_all] Waiting for vLLM to warm up..." >&2
sleep 5

echo "[run_all] Starting FastAPI on ${HOST:-0.0.0.0}:${PORT:-8000}" >&2
exec uvicorn server.main:app --host "${HOST:-0.0.0.0}" --port "${PORT:-8000}" --log-config "${SCRIPT_DIR}/uvicorn_logging.yaml"
