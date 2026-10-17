# Runbook: сбои и что делать

Общие правила:
- **Данные не удаляем.** `data/` (задачи и ключи A02), `weights/`, `runs/` скрипты не чистят.
- **Убиваем только своё.** `stop.ps1` — только PID из `runs\deploy\state\state.json` с совпадающим временем старта; `Stop-Process -Name python` не делать.
- **Откат** — только на проверенную версию: `python deploy\release.py rollback` (см. ниже).

| Где смотреть | Что там |
|---|---|
| `deploy\windows\status.ps1` | фаза (`starting`/`warming`/`trial`/`ready`/`not_ready`/`ready_timeout`/`crash_loop`/`failed`), PID, URL, последняя ошибка, VRAM |
| `runs\deploy\supervisor.log` | события супервизора |
| `runs\deploy\api.err.log` | трейсбеки uvicorn/A02/A03 |
| `runs\deploy\preflight.json`, `ready.json` | почему не стартовали / не готовы |
| `runs\deploy\netguard.jsonl` | попытки выхода в сеть (в проде — аудит) |

## Не стартует: preflight FAIL
`python deploy\preflight.py` — каждая строка FAIL говорит, что чинить.

| FAIL | Что делать |
|---|---|
| `env-lock` | Пакеты расходятся с lock L1. Не обновлять вслепую: `setup.ps1 -InstallMissing` ставит только недостающее под `-c lock`. Критичный пакет не той версии — к L1. |
| `cuda` | Нужна сборка torch `+cu128`. CPU-колесо — переустановка из lock L1 (`setup.ps1 -Fresh` на чистой venv). |
| `vram` | Закрыть процессы из `nvidia-smi`, см. `deploy/VRAM.md`. |
| `weights:*` | См. [Нет модели](#нет-модели). |
| `config:*` | Файл модуля отсутствует или битый YAML. `git diff` → откат правки. |
| `path:*` | Мало диска — [Мало диска](#мало-диска). Нет прав — проверить каталог. |
| `api-key` | Заглушка `change-me`. Удалить `deploy\.env` → `setup.ps1` сгенерирует ключ. |
| `env:SENTINEL_MOCK` / `env:SENTINEL_FAKE_PIPELINE` | Убрать из `deploy\.env`: в проде фиктивные режимы запрещены. |
| `port` | Сервис уже запущен (`status.ps1`) или порт занят чужим процессом. Чужой не трогаем — `-Port 8010`. |

## Нет модели
1. `python deploy\weights.py adopt` покажет `FAIL <repo>`: нет в кэше, нет шардов, битая ссылка или `.incomplete`.
2. Докачивать отдельным онлайн-шагом, **не** во время работы сервиса: `python deploy\weights.py prepare --only <repo>` (или как качал L2), затем `adopt` и `verify`.
3. Без VLM можно работать только осознанно: `SENTINEL_NO_VLM=1` — preflight понижает Qwen3-VL до опциональной. Качество типов упадёт; зафиксируйте в отчёте.
4. Неполный GGUF Qwen3-VL исключён шаблоном (`models.exclude_patterns`) — не подключать.

## GPU OOM
Признаки: в `status.ps1` → `last_error: ... GPU OOM`, в `api.err.log` — `CUDA out of memory`.
1. `nvidia-smi` — нет ли посторонних процессов на GPU.
2. Убедиться, что копилот не на GPU (`deploy/VRAM.md`, п. 4).
3. Проверить, что `SENTINEL_WORKERS=1` (по умолчанию так).
4. Временный обход — по одному изменению за раз, с записью в отчёт:
   - `SENTINEL_VLM_POLICY=on_demand` — VLM грузится только на свою стадию;
   - крайний случай `SENTINEL_NO_VLM=1`.
5. Перезапуск: `stop.ps1` → `run_all.ps1`. Задачи в `data\` сохраняются.

## Не готов: `phase=not_ready` (ворота /readyz)
`status.ps1 → detail` повторяет `/readyz` A02. Synthetic-задача в этих состояниях **не отправляется**: у A02 успешная задача переводит `warmup_failed` в `ready`, это скрыло бы сбой прогрева.

| `/readyz` | Что делать |
|---|---|
| `warmup_failed` | Причина — в `detail` и `api.err.log`: нет весов, OOM при прогреве и т.п. Починить → `stop.ps1` → `run_all.ps1`. |
| `unavailable` | `backend.pipeline` не импортируется (`api.err.log`); preflight должен был это поймать. |
| `mock_not_allowed` / `pipeline=fake` | Убрать `SENTINEL_MOCK` / `SENTINEL_FAKE_PIPELINE`. В проде запрещены. |
| `gpu_hung` | Супервизор сам перезапускает API (освобождает GPU), в пределах лимита crash-loop. |

## GPU не освободился
`status.ps1` → `gpu_released: False` или непустой `orphans_left`: процессы старого дерева API всё ещё держат CUDA.
- Посмотреть: `nvidia-smi --query-compute-apps=pid,used_memory --format=csv`.
- Если это PID из `runs\deploy\state\state.json` (`api_tree`), `stop.ps1` добивает их сам.
- Чужие процессы не трогать — закрыть их владельцу.

## Crash-loop
Больше `service.max_restarts` падений за `restart_window_s` → супервизор останавливается сам, туннель закрыт (`phase=crash_loop`). Чинить причину по `api.err.log`, не крутить рестарты.

Если причина в свежей правке — [Откат](#откат).

## Процесс упал / завис
- **Падение uvicorn.** Супервизор перезапустит сам, только после того как старое дерево процессов завершено и порт свободен. Второй сервер не создаётся.
- **Зависание.** `/healthz` не отвечает ~30 с → перезапуск.
- **Туннель.** При рестарте API cloudflared не трогается — quick URL жюри сохраняется, во время прогрева будет 502. Если упал сам cloudflared, quick URL **меняется**: новый — в `status.ps1` и `runs\deploy\public_url.txt`. Сообщить жюри или перейти на named tunnel.
- **Супервизор убит аварийно.** `stop.ps1` добьёт осиротевшие uvicorn и cloudflared — только свои, по PID и времени старта.

## Таймаут задачи
Бюджет задачи — у A02/A03 (`SENTINEL_BUDGET_S` 270 / `SENTINEL_WATCHDOG_S` 285). По таймауту:
- задача `failed`, а не `no_violation`;
- если `partial.json` уже содержит `violation` — `done` с `meta.partial=true`.

Частые таймауты → замерить `sla_bench.py` / `eval.jury_sim` и смотреть `meta.timings`. Пороги не трогать вслепую.

## Мало диска
`data\jobs\` растёт с каждым видео (~30 МБ). Не удалять во время приёмки.

Освобождать место в таком порядке:
1. `cache\` (кэш стадий A03, пересоздаётся);
2. старые `runs\deploy\*.log`;
3. архивировать `data\jobs\<старые>` на другой диск.

Порог preflight — `preflight.min_free_disk_gb`.

## Откат
```powershell
.venv\Scripts\python deploy\release.py diff                 # что изменилось с проверенной версии
deploy\windows\stop.ps1
.venv\Scripts\python deploy\release.py rollback --dry-run   # план
.venv\Scripts\python deploy\release.py rollback             # stash -u (правки сохраняются) + switch --detach SHA
deploy\windows\run_all.ps1
```
- `config_mismatch` в выводе — файл вне git (например, `deploy\.env`) отличается от проверенного. Вернуть вручную.
- Вернуться обратно: `git switch <ветка>` и `git stash pop`.
- Проверенную версию фиксировать **после** приёмки A16: `python deploy\release.py mark-verified --note "A16 ok" --report <metrics.json>`.
