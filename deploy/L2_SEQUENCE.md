# Последовательность для L2 (ноутбук, venv L1, веса в `weights\hf\hub`)

PowerShell в корне репо. Публичный туннель — только на шаге 7.
- В отчёты и чат идут только числа: без кадров и описаний сцен.
- Synthetic-задания проверяют тракт и SLA, не качество CV.

## 1. Окружение и веса (ничего не ставит)
```powershell
.venv\Scripts\python deploy\envcheck.py                 # OK; нет fastapi/uvicorn → setup.ps1 -InstallMissing (под -c lock L1)
.venv\Scripts\python deploy\merge_requirements.py --check
```
- Ожидаемо **exit 1**: Pillow `<13` в `ml/fusion` (A12) против 13.0.0 в L1. Это блокер для A12, не маскировать.
- После фикса A12 — `--check` exit 0 и настоящий resolver на всём lock L1 (нужна сеть и индекс cu128):
  ```powershell
  .venv\Scripts\python deploy\merge_requirements.py --check --resolve-check windows-cu128   # ok:true, drift:{}
  ```
```powershell
.venv\Scripts\python deploy\weights.py adopt            # обязательные модели: OK, точный commit; GGUF исключён
.venv\Scripts\python deploy\weights.py verify --fast
.venv\Scripts\python deploy\gpu_probe.py                # ТОЛЬКО при остановленном сервисе (аллоцирует VRAM)
.venv\Scripts\python deploy\preflight.py                # OK; строка budgets = WARN UNMEASURED (ожидаемо)
```

## 2. Запуск и ready (локально, без туннеля)
```powershell
$env:SENTINEL_CACHE = "off"     # для замеров: иначе повтор тех же роликов берётся из кэша стадий A03
deploy\windows\run_all.ps1 -NoTunnel
deploy\windows\status.ps1        # phase=ready, healthz=True, readyz=200
```
Что проверить:
- `runs\deploy\ready.json`: `readyz.verdict=ready`, `readyz.pipeline=real`, synthetic-задача `status=done`, `from_accept_s`, `server.*`.
- `fake`/`mock_not_allowed`/`unavailable`/`warmup_failed`/`gpu_hung` → `phase=not_ready`, туннель закрыт, synthetic-задача **не** отправляется.

## 3. Пять последовательных заданий
```powershell
.venv\Scripts\python deploy\sla_bench.py --mode sequential --n 5 --synth --seconds 145
#   реальные: --videos <5 РАЗНЫХ файлов>  (одинаковые файлы A02 дедуплицирует → замер недействителен)
```
Выход — `runs\deploy\sla_sequential_5_done.json`:
- по каждому заданию `from_submit_s` (загрузка → итог), `from_accept_s` (приём → итог, с очередью), `server.queue_wait_s` / `processing_s` / `total_s`;
- общие `max_from_submit_s`, `dedup_detected`, `cache_hit_jobs`.

## 4. Очередь из двух одновременных
```powershell
.venv\Scripts\python deploy\sla_bench.py --mode concurrent --n 2 --synth --seconds 145 --seed 2000
```
У второго задания `server.queue_wait_s` > 0. Критерий — `max_from_submit_s ≤ 300`: это полное время задания с очередью, а не время стадии.

Сценарий жюри (5+ сразу): `--mode concurrent --n 5 --seed 3000` и приёмка A16 `python -m eval.jury_sim ... --n 6`.

## 5. Таймаут, перезапуск, отмена
**Таймаут** — короткие бюджеты через env A02, потом вернуть:
```powershell
deploy\windows\stop.ps1
$env:SENTINEL_WATCHDOG_S="20"; $env:SENTINEL_JOB_CAP_S="10"; $env:SENTINEL_BUDGET_S="10"
deploy\windows\start.ps1 -NoTunnel      # synthetic-задача ready может не уложиться — status=not_ready, это ожидаемо
.venv\Scripts\python deploy\sla_bench.py --mode concurrent --n 1 --synth --seconds 145 --seed 4000 --expect failed --limit 120
deploy\windows\stop.ps1; Remove-Item Env:SENTINEL_WATCHDOG_S,Env:SENTINEL_JOB_CAP_S,Env:SENTINEL_BUDGET_S
```
Ожидаемо `status=failed` с причиной. **Никогда** `no_violation`.

**Перезапуск во время задания:**
1. В отдельном окне — `sla_bench.py --mode concurrent --n 1 --synth --seed 5000`.
2. Через ~10 с — `deploy\windows\stop.ps1`, затем `deploy\windows\run_all.ps1 -NoTunnel`.
3. Задание не теряется: A02 `_recover` либо возвращает его в очередь с остатком бюджета (отсчёт от загрузки), либо даёт `failed` с причиной.
4. `status.ps1` → `gpu_released: True`, `orphans_left: []`, а в `nvidia-smi` нет процессов старого дерева.

**Отмена:** в API A02 эндпоинта нет — **NOT AVAILABLE**. Остановить можно только весь сервис (`stop.ps1`: только свои процессы, `data\` сохраняется).

## 6. Офлайн-доказательство (PowerShell от администратора, сервис остановлен)
```powershell
deploy\windows\stop.ps1
deploy\windows\offline_smoke.ps1 -Firewall
```
Ожидаемо: `ok=true`, `network_attempts=0`, `egress_selftest.all_blocked=true`, лог `runs\deploy\offline_proof_*.md`. Правило фаервола снимается само.

## 7. Публичный URL (после шагов 1–6)
```powershell
deploy\windows\run_all.ps1
.venv\Scripts\python deploy\tunnel_probe.py real --url <URL> --mb 30 --out runs\deploy\tunnel_real.json
```

## 8. Зафиксировать замеры
Только из отчётов выше. Пока не запущено — NOT RUN.

| Что | Источник | Значение |
|---|---|---|
| ready: synthetic-задача, с от приёма | `ready.json: from_accept_s` | NOT RUN |
| 5 последовательных: max / медиана, с | `sla_sequential_5_done.json` | NOT RUN |
| очередь из 2: max `from_submit_s`, с; `queue_wait_s` второго | `sla_concurrent_2_done.json` | NOT RUN |
| 5 одновременно: max `from_submit_s`, с (≤ 300?) | `sla_concurrent_5_done.json` | NOT RUN |
| таймаут → failed за, с | `sla_concurrent_1_failed.json` | NOT RUN |
| перезапуск: задание requeued/failed, GPU освобождён | `status.ps1`, GET задания | NOT RUN |
| VRAM пики стадий | `deploy/VRAM.md` → команда A/B | NOT RUN |
| офлайн: попыток сети / egress заблокирован | `offline_worker_smoke.json` | NOT RUN |

После замеров:
- поставить `service.budgets_measured_on_gpu: true` в `deploy\config\deploy.yaml`, указав в комментарии дату и SHA;
- `python deploy\release.py mark-verified --note "L2 SLA ..." --report <отчёты>`.
