#!/usr/bin/env bash
# Доказательство локальности на Linux / План Б: реальный worker в контейнере без сети (ядро: --network none).
# Публичный HTTP проверяется отдельно (tunnel_probe.py) — контейнер без сети не может быть API.
#   IMAGE=kassa-sentinel:gpu deploy/offline_smoke.sh
set -euo pipefail
cd "$(dirname "$0")/.."
IMAGE="${IMAGE:-kassa-sentinel:gpu}"
GPU_FLAG="${GPU_FLAG:---gpus all}"
mkdir -p logs
# shellcheck disable=SC2086
docker run --rm --network none $GPU_FLAG \
  -e SENTINEL_MOCK=0 -e SENTINEL_NETGUARD=block -e PYTHONPATH=/app:/app/deploy/netguard \
  -v "$PWD/weights:/app/weights:ro" -v "$PWD/logs:/app/logs" \
  "$IMAGE" python /app/deploy/offline_worker_smoke.py "$@"
