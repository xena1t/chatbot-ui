#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[run_all] Starting FastAPI on ${HOST:-0.0.0.0}:${PORT:-8000}" >&2
exec uvicorn server.main:app \
  --host "${HOST:-0.0.0.0}" \
  --port "${PORT:-8000}" \
  --timeout-keep-alive 600 \
  --log-config "${SCRIPT_DIR}/uvicorn_logging.yaml"
