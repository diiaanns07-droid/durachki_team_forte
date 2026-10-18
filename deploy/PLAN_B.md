# План Б: арендованная GPU-VM (Ubuntu 22.04 + Docker) — опциональный путь, НЕ проверялся

> Фаза 2: окружение = lock L1 (torch 2.11.0+cu128). `Dockerfile.gpu` ещё ставит torch 2.6.0/cu124 —
> до синхронизации с lock L1 (TODO в README) на VM надёжнее нативный путь: venv из lock L1
> (`uv pip install -r tools/dataprep/requirements-l1.lock.txt --extra-index-url https://download.pytorch.org/whl/cu128
> --index-strategy unsafe-best-match`), затем `deploy/supervisor.py run`.
> `docker run --env-file` НЕ понимает комментарии в строке — передавайте переменные через compose или без `# ...`.

> Обработка остаётся на наших моделях без внешних API, но VM — не «локальная» машина команды:
> согласуйте с организаторами, допустимо ли это по правилу локальности.

1. VM с NVIDIA GPU ≥ 8 GB, образ с драйвером. Первым делом: `nvidia-smi`; после установки — `gpu_probe.py`.
2. Docker + nvidia-container-toolkit (официальные инструкции NVIDIA), `docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi`.
3. `git clone …; cp deploy/.env.example deploy/.env` — ключи только в `.env`.
4. Веса — отдельным шагом с сетью: `python deploy/weights.py prepare && python deploy/weights.py verify` (или rsync `weights/` с ноутбука + `verify`).
5. `docker build -f deploy/Dockerfile.gpu -t kassa-sentinel:gpu .`
6. Доказательство локальности: `IMAGE=kassa-sentinel:gpu deploy/offline_smoke.sh` (`--network none`).
7. Сервис: `docker run -d --gpus all -p 127.0.0.1:8000:8000 --env-file deploy/.env -v $PWD/weights:/app/weights:ro kassa-sentinel:gpu`, затем `cloudflared tunnel --url http://localhost:8000`.
8. `deploy/healthcheck.py ready`, `deploy/tunnel_probe.py real`, `deploy/sla_bench.py` — приложить отчёты.
