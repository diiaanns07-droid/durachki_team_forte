#!/usr/bin/env bash
# Опциональный Docker-путь. serve: verify весов (без сети) → uvicorn, 1 worker. Иначе — exec команды.
set -euo pipefail
cd /app
if [ "${1:-serve}" = "serve" ]; then
  if [ "${SENTINEL_MOCK:-0}" != "1" ]; then python deploy/weights.py verify --fast; fi
  export PYTHONPATH="/app:/app/deploy/netguard" SENTINEL_WORKERS=1
  exec python -m uvicorn "${SENTINEL_APP:-backend.api.main:app}" --host 0.0.0.0 --port "${SENTINEL_PORT:-8000}" --workers 1
fi
exec "$@"
