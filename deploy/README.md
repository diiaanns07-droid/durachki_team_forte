# deploy/ — развёртывание Kassa Sentinel (A17)

**Основной путь — Windows-нативный, на рабочей venv L1** (Python 3.11, lock `tools/dataprep/requirements-l1.lock.txt`: torch 2.11.0+cu128, transformers 5.20.1, bitsandbytes 0.50.2, opencv < 5).
- Веса — офлайн, уже лежат в `weights/hf/hub`.
- Docker (`Dockerfile.gpu`, `offline_smoke.sh`, `PLAN_B.md`) — опциональный путь. Он **не собирался** и не синхронизирован с lock L1.

## Последовательность для L2
Компактный порядок (окружение и веса → ready → 5 подряд → очередь из 2 → таймаут/перезапуск → офлайн → URL) с шаблоном фиксации замеров — **`deploy/L2_SEQUENCE.md`**.

## Команды для L2 (по порядку)
```powershell
powershell -ExecutionPolicy Bypass -File deploy\windows\setup.ps1 [-InstallMissing]   # сверка venv с lock L1, adopt весов
powershell -ExecutionPolicy Bypass -File deploy\windows\offline_smoke.ps1 -Firewall    # от администратора: доказательство локальности
powershell -ExecutionPolicy Bypass -File deploy\windows\run_all.ps1                    # preflight → сервис → READY → URL
.venv\Scripts\python deploy\tunnel_probe.py real --url <URL> --mb 30                   # замер публичного URL
.venv\Scripts\python -m eval.jury_sim --url <URL> --key <ключ> --videos data\videos --n 6   # приёмка A16, замер SLA
.venv\Scripts\python deploy\release.py mark-verified --note "A16 ok" --report <отчёт>  # зафиксировать проверенную версию
deploy\windows\status.ps1   |   deploy\windows\stop.ps1
```
Сбои — `RUNBOOK.md`, последний час — `CHECKLIST_SUBMISSION.md`, память GPU — `VRAM.md`.

## Замеры для L2 (по одной команде; в чат — только числа, без кадров и описаний)
| Что меряем | Команда | Выход | Метрики |
|---|---|---|---|
| Локальность моделей | `offline_smoke.ps1 -Firewall` (админ) | `runs\deploy\offline_worker_smoke.json`, `offline_proof_*.md` | `ok`, `network_attempts` (=0), `egress_selftest.all_blocked`, `result.stages.*.origin`, `timings_s` |
| VRAM: дефолты A03 (A) против оверлея (B, выкл.) | см. `VRAM.md` → «Команда замера» | `runs\deploy\vram_{A,B}.json` | `stages.*.vram_peak_mb`, `vram_events`, выполнен ли `vlm_verify`, `timings_s.run`, verdict |
| SLA (5+ видео одновременно) | `python -m eval.jury_sim --url <URL> --key <ключ> --videos data\videos --n 6` (A16) | отчёт A16 | время до результата у каждого (max ≤ 300 с), FP/FN, таймкод, тип |
| Публичный URL | `deploy\tunnel_probe.py real --url <URL> --mb 30` и `tunnel --url <URL>` (против `stub_api.py` за тем же туннелем) | `runs\deploy\tunnel_*.json` | загрузка 30 МБ, 413/524, задержка `/healthz`, Range 206, Set-Cookie |

Новые режимы (VRAM-оверлей, `netguard=block` в проде) включаются только после того, как замер покажет выигрыш — по одному изменению за раз.

## Что делает каждый файл
| Файл | Назначение |
|---|---|
| `config/deploy.yaml` | Все флаги и пороги: lock, обязательные/опциональные/исключённые модели, пороги диска и VRAM, сервис, туннель, netguard. |
| `envcheck.py` | Сверяет установленную venv с lock L1 и требованиями модулей. **Ничего не ставит.** Блокирует расхождение критичных пакетов и отсутствие рантайм-пакетов (fastapi, uvicorn, python-multipart — их нет в lock L1). |
| `merge_requirements.py` | Сводит `*/requirements.txt`, проверяет их против пинов lock L1 и сообщает только о нарушающих ограничениях; запрещённые лицензии. Lock — только resolver'ом (`uv pip compile -c <lock L1>`): пины L1 не двигаются. |
| `weights.py adopt` | **Офлайн-инвентаризация уже скачанного кэша**: commit из `refs/main`, все файлы снапшота резолвятся, полнота шардов `*.index.json`, пустые файлы, `.incomplete`; sha256 LFS из имени blob (`--hash` — пересчитать); лицензия из `README.md` снапшота. Результат → `weights.lock.json`. `verify` проверяет обязательный список из `deploy.yaml`; GGUF исключён шаблоном. `prepare` — отдельный онлайн-шаг докачки. |
| `preflight.py` | Блокирующие проверки до старта: Python, lock, CUDA-сборка (без создания CUDA-контекста), свободная VRAM, обязательные веса, конфиги модулей, диск, пути, наличие ключа (значение не печатается), порт, запрет `SENTINEL_MOCK`/`FAKE_PIPELINE`, ffmpeg/ffprobe/cloudflared. Нет обязательной модели → сервис не стартует. |
| `supervisor.py` | Один экземпляр (OS-блокировка файла), один uvicorn-воркер. Рестарт — только после смерти своего дерева процессов и освобождения порта; чужие процессы не трогает (PID + время старта). Crash-loop → стоп с понятной ошибкой (GPU OOM распознаётся). READY = `/readyz` 200 (warmup A02) + **реальная пробная задача** → только потом cloudflared. Туннель переживает рестарт API (quick URL не меняется). Система не засыпает, дисплей гаснуть может. |
| `windows/*.ps1` | Тонкие обёртки: `_env.ps1` (env из `deploy\.env`, офлайн), `setup`, `run_all`, `start` (прежние параметры), `status`, `stop`, `offline_smoke`, `firewall_python` (временное правило только на python.exe), `monitor`. |
| `offline_worker_smoke.py` | **Реальный worker** (`backend.pipeline.run` → spawn-воркер A03) на локальных весах: HF offline, netguard=block в процессе и воркере. Попытки выхода в сеть считаются по общему логу. Падает на mock/fake, на `no_violation` при неполном анализе и на любой попытке выйти в сеть. `--egress-selftest`: python.exe без netguard должен не достучаться до интернета (под правилом фаервола). Лог для питча — `runs/deploy/offline_proof_*.md`. |
| `healthcheck.py` | `live`; `ready` = `/readyz` + пробная задача + схема ТЗ, отвергает заглушку, `pipeline=fake`, `meta.mock`, `no_violation` при `complete=false`. |
| `tunnel_probe.py`, `sla_bench.py` | Замеры публичного URL и SLA. Цифры — только из них или из A16 `jury_sim`. |
| `release.py` | `mark-verified` (SHA, хэши конфигов и lock-файлов; `.env` — только хэш), `diff`, `rollback` (stash + switch --detach, без reset --hard). |
| `netguard/` | Аудит-хук PEP 578: блокирует или логирует не-loopback connect/DNS и прокси; наследуется дочерними процессами через `PYTHONPATH`. |
| `config/pipeline_vram_measured.yaml` | **Гипотеза, выключена**: оверлей VRAM-оценок A03 под измеренный пик 4.7 ГБ (см. `VRAM.md`). |
| `stub_api.py` | Заглушка API — **только** для тестов deploy и замера туннеля до интеграции. Помечена; `ready` её отвергает. |

## Решения фазы 2 (записаны по ходу)
1. **Источник истины окружения — lock L1.**
   - Свой пин torch 2.6.0/cu124 снят: он противоречил L1. Таргеты lock — `windows-cu128` / `linux-cu128`.
   - `setup.ps1` не пересинхронизирует рабочую venv. Недостающее (fastapi, uvicorn…) ставится только с `-InstallMissing` под `-c lock`.
2. **Обязательные модели** (блокируют readiness): `PekingU/rtdetr_v2_r18vd` (detect closed-set), `google/siglip2-base-patch16-224` (temporal), `Qwen/Qwen3-VL-4B-Instruct` (VLM).
   - VLM понижается до опциональной только явным `SENTINEL_NO_VLM=1` / `SENTINEL_VLM_POLICY=off`.
   - Опциональные: OWLv2 (open_vocab выключен в конфиге A05), D-FINE-medium (скачан, не используется), e5-small (копилот выключен).
   - Исключены: `*GGUF*`.
3. **netguard.** В проде — `log` (аудит, ничего не ломает). `block` — только в offline smoke: новый режим не включается в проде без замера.
4. **Офлайн-доказательство.** Временное правило фаервола только на python.exe (лаунчер venv + базовый интерпретатор, который запускают лаунчер и multiprocessing spawn); снимается в `finally`.
   - Глобально фаервол не меняется. cloudflared — отдельный процесс, публичный транспорт проверяется `tunnel_probe.py`.
5. **Логи и отчёты** — в `runs/deploy/` (в `.gitignore`), а не в `logs/`: `logs/` не игнорируется корневым `.gitignore`.
6. **Туннель после рестарта API не перезапускается**, чтобы quick URL у жюри не сменился.

## Конфликты требований с lock L1 (решают владельцы, не deploy)
`python deploy/merge_requirements.py --check` на `claude/integration-cloud` 2055fc4:

| Пакет | lock L1 | Требование модуля | Владелец | Статус |
|---|---|---|---|---|
| pillow | 13.0.0 | `>=10.0,<13` | `ml/fusion/requirements.txt` (A12) | **открыт** (и в ветке A12 `9ca9a0e`) |
| transformers | 5.20.1 | было `<5` | `ml/vlm` (A11) | снят владельцем |
| pyyaml | 6.0.3 | было `==6.0.1` | `eval` (A16) | снят владельцем |

Свой `huggingface_hub<2` исправлен ранее.

## Аудит 2055fc4 (claude/integration-cloud): что воспроизведено и исправлено
Все пункты воспроизведены на временной копии интеграционного дерева с текущим `deploy/`.

| Пункт | Воспроизведение | Исправление |
|---|---|---|
| Pillow A12 | `merge_requirements --check` → exit 1, конфликт виден (не маскировался) | Закрыты пути маскировки: маркеры (`; python_version…`) и прямые URL пропускали сверку с lock. Маркеры теперь вычисляются для Windows/py3.11, группы пакета проверяются совместно. |
| Совместимость со всем lock L1 | — | `--resolve-check` (настоящий `uv pip compile -c lock`, сверка дрейфа). What-if с `Pillow>=10.0` во временной копии, `windows-cu128 --no-torch-index` (PyPI): ok, дрейфа пинов L1 нет, +27 пакетов вне L1. **GPU-установка этим не проверена.** |
| `/readyz` A02 | 200 при `pipeline=fake`; успешная задача переводит `warmup_failed` в `ready`; healthcheck пропускал 404 и любые коды, кроме 503 | Ворота: только `200 + ready + state=ready + pipeline=real`; терминальные состояния — без synthetic-задачи; `gpu_hung` → рестарт API. |
| Итоговый JSON | — | Проверки: схема ТЗ; не mock (`meta.mock` / `provenance` / стадии / `meta.api.pipeline`); `no_violation` — только с подтверждённой полнотой (`contracts.extensions.coverage_problem`, если есть). |
| Освобождение GPU | Потомок упавшего API (как spawn-воркер A03) переживал рестарт: 6 сирот за 6 рестартов | Снимок дерева процессов, добивание сирот (POSIX — группа процессов), проверка `nvidia-smi --query-compute-apps`. |
| SLA | Один synthetic-файл N раз → дедуп A02 по sha и кэш A03 — время ложное | Свой файл на задание (seed). Метки dedup и кэша; время от приёма с очередью. |
| Единственный воркер / реальный прогрев | — | preflight читает действующие `workers` / `gpu_slots` / `warmup` / `fake_pipeline` / `allow_mock` A02 и `executor.mode` / `mock` A03; на конфигах 2055fc4 всё OK. |
| Бюджеты | — | Помечены UNMEASURED до замера на GPU L2 (`service.budgets_measured_on_gpu`). |
| Пути с пробелами | — | Тесты супервизора в каталоге с пробелами. `start.ps1`: `-WorkingDirectory` и экранирование аргументов `Start-Process`. |

## Что реально проверено (облако: Linux, без GPU, без данных)
- `pytest deploy/tests` — **92 passed, 1 skipped** (skip — парсер PowerShell без pwsh; с pwsh 7.4.6 — 93 passed). Покрыто:
  - merge и envcheck против lock на фикстурах, включая маскировку маркерами и URL; `resolve_check` с фейковым uv;
  - adopt на раскладке кэша huggingface_hub;
  - preflight (A02/A03 gates, ключ не печатается);
  - ворота `/readyz` по всем состояниям A02; synthetic-задача не шлётся при `warmup_failed`; проверки итогового JSON (mock, полнота);
  - супервизор на реальных подпроцессах:
    - полный цикл, второй экземпляр отклонён, stop;
    - crash-loop с OOM; `gpu_hung` → рестарты;
    - сироты-потомки добиты (с psutil и без), каталоги с пробелами;
    - чужой процесс на порту не тронут;
  - SLA-бенч: sequential/concurrent, dedup, кэш, `--expect failed`;
  - offline smoke, release, healthcheck.
- На копии интеграционного дерева 2055fc4 с текущим `deploy/`:
  - `merge_requirements --check` → exit 1 (Pillow A12);
  - preflight → A02/A03 gates OK, budgets WARN UNMEASURED; конфиги модулей парсятся.
- What-if (только временная копия, `Pillow>=10.0`): `--resolve-check windows-cu128 --no-torch-index` → ok, дрейф пинов L1 = 0. Это совместимость диапазонов на PyPI. **Не GPU-установка.**
- Все `.ps1` разбираются парсером PowerShell 7.4.6; `Quote-Arg` для путей с пробелами проверен в pwsh.
- **NOT RUN** (нет Windows / GPU / весов / сети к cu128 / данных):
  - запуск `.ps1` на Windows; правило фаервола;
  - установка из lock с индексом cu128; `gpu_probe`;
  - реальный warmup A02/A03 и `/readyz`; synthetic-задача на реальном пайплайне;
  - освобождение VRAM по `nvidia-smi`;
  - SLA 5 подряд / очередь из 2 / 5 одновременно, таймаут и перезапуск — на GPU;
  - VRAM; Cloudflare Tunnel.
  - Цифр скорости и памяти нет — см. `L2_SEQUENCE.md`, шаг 8.

## Для интегратора
- **Структура.** Ветка добавляет только `deploy/`. Канонический контракт есть в `contracts/`; deploy его не импортирует, fallback-копии нет.
- **A01.** Добавить `logs/` в корневой `.gitignore` (на всякий случай). deploy пишет в `runs/deploy/`.
- **A02.** Запуск — `uvicorn backend.api.main:app --workers 1` из супервизора.
  - `/readyz` используется как часть готовности; `pipeline=fake` блокирует.
  - Ключ для пробной задачи — env или `data/api_keys.txt` (A02 генерирует его сам).
- **A03.**
  - `offline_worker_smoke` смотрит `meta.complete`, `meta.coverage` (если появится), `meta.mock`, `meta.stages.*.origin/vram_peak_mb`, `meta.vram_events`.
  - Netguard передаётся spawn-воркеру через env и `PYTHONPATH`.
  - Риск VRAM-оценок — `VRAM.md`.
- **A11/A12/A16.** Конфликты с lock L1 — таблица выше.
- **A14.** На время мероприятия `COPILOT_ENABLED=0`. Если включать — только CPU (`VRAM.md`).
- **TODO (optional Docker):** установка в `Dockerfile.gpu` из lock L1 (cu128), compose, `Dockerfile.cpu`.

## Модели и лицензии
Своих моделей deploy не использует (`models.json` = `[]`). Лицензии весов решаются по конкретному снапшоту: `weights.py adopt` → `weights.lock.json` → `weights.py licenses` → `LICENSES.md`.

Зависимости deploy (`requirements.txt`): huggingface_hub, packaging, PyYAML, psutil — все есть в lock L1. Остальное — stdlib.
