# A02 — `backend/api/` (FastAPI)

HTTP-слой Kassa Sentinel: API жюри по ТЗ, те же ручки для веб-UI по cookie-сессии, очередь задач с бюджетом времени, watchdog, отдача видео с Range, приёмка одной командой и замер для L2.

Работает **строго одним процессом** — очередь и тёплые модели живут в памяти.

```bash
pip install -r backend/api/requirements.txt
export SENTINEL_API_KEYS="$(python -m backend.api.make_key --quiet)"   # или несколько через запятую
export SENTINEL_UI_PASSWORD='...'            # опционально: вход в UI по паролю
python -m backend.api --host 127.0.0.1 --port 8000
# то же: uvicorn backend.api.main:app --workers 1 --no-access-log
```

## Одна команда: схема ТЗ + авторизация + жизненный цикл

```bash
python -m pytest backend/api/tests -q                         # 103 теста, без GPU и весов
SENTINEL_API_KEY=... python -m backend.api.tools.acceptance --url http://127.0.0.1:8000 --out runs/acceptance.json
```

`acceptance` работает по живому URL (локально или через Cloudflare Tunnel). Реальные ролики не нужны: синтетический MOV он делает сам; свой можно передать через `--video x.mov`. Около 28 проверок:
- **health:** `healthz`, `readyz` (WARN, если пайплайн fake).
- **авторизация:** 401 без ключа и с неверным ключом; 404 на неизвестный job.
- **плохие файлы:**
  - 415 — `.txt`, мусор с расширением `.mov`, обрезанный MOV (проверяется декодером);
  - 400 — пустой файл;
  - 413 — заявлен `Content-Length` 400 МБ (ответ до чтения тела);
  - обрыв загрузки посередине → задача не создаётся.
- **полный цикл:**
  - поля ТЗ есть на **каждом** опросе;
  - `done|failed` за ≤ 300 с от приёма;
  - ответ ≤ 3 МБ, кадры — валидный JPEG.
- **идемпотентность:** повторная загрузка того же файла → тот же job.
- **Range:** 206, `bytes=-10`, 416; видео без ключа → 401.
- **UI:** без сессии 401 даже с same-origin-заголовками; неверный секрет → 401; вход ставит HttpOnly/SameSite cookie, токен в тело не попадает; видео по cookie + Range → 206; POST без CSRF → 403.
- **CORS** для чужого Origin закрыт.

Код выхода 0 — FAIL нет. Ключ берётся **только из env** (`SENTINEL_API_KEY` или первый из `SENTINEL_API_KEYS`), в argv и в отчёт он не попадает.

Проверено в этой сессии (синтетика, без GPU):

| Прогон | Результат |
|---|---|
| `acceptance` против `SENTINEL_FAKE_PIPELINE=1` | 27 PASS, 1 WARN (сам fake), 0 FAIL |
| `acceptance` против **настоящего `backend.pipeline` из ветки A03** в mock-режиме (`SENTINEL_MOCK=1 SENTINEL_ALLOW_MOCK=1`, временная сборка, не коммитилась) | 27 PASS, 0 WARN, 0 FAIL |
| `eval.jury_sim` из ветки A16 против этого API (`SENTINEL_DEDUP=0`) | 8/8 PASS: auth, 415, 404, sequential 5, bounded stress 5, 150 МБ мусора → 415 |
| Ранний отказ (401/415) при загрузке через `requests`: быстрой 150 МБ и медленной chunked 30 МБ | Клиент получает код, а не обрыв соединения: uvicorn дочитывает тело |
| `pytest backend/api/tests` **внутри дерева `origin/claude/integration-cloud`** (A01 `a3b2b55`, A03 с `coverage`/`cancel`, A14, A16…; мой fallback удалён, как у интегратора) | 82/82 |
| `acceptance` против сервера из дерева интеграции (A03 mock, разрешён явно) | без адаптера: 26 PASS, 2 WARN — чистое видео `failed: «нет нарушения» не подтверждено: meta.coverage.ratio не задан` (см. «Для интегратора», п. 3); с `SENTINEL_COVERAGE_ADAPT_STAGES=1`: 27/27, чистое видео → `done/no_violation` |
| **Аудит `2055fc4`:** `pytest backend/api/tests` внутри дерева `integration-cloud@2055fc4` (настоящие contracts A01, канонический `meta.coverage` A03, корневой glue `conftest`) | 103/103 |
| `/readyz` с настоящим `backend.pipeline` A03 в боевом режиме (без mock, весов нет) | новый код: **503 `warmup_failed`** («warmup synthetic run: status=failed»); прежний `d821f24`: 200 `ready` — ошибка исправлена |
| `acceptance` против дерева `2055fc4`, A03 в явно разрешённом mock-режиме (проверка связки, не реальный прогон) | 27/27; чистое видео → `done/no_violation`, `coverage.ratio=1.0` от A03 (адаптер не нужен) |

## Замер для L2 (реальные видео + GPU есть только у L2)

```bash
# 1) сервер в боевом режиме (без fake/mock), ключ в env
SENTINEL_API_KEYS=<ключ> SENTINEL_DEDUP=0 python -m backend.api
# 2) приёмка
SENTINEL_API_KEY=<ключ> python -m backend.api.tools.acceptance --url http://127.0.0.1:8000 --out runs/accept_L2.json
# 3) 5+ реальных видео: сначала по одному, потом пачкой 5 (как жюри); время каждого
SENTINEL_API_KEY=<ключ> python -m backend.api.tools.loadtest --url http://127.0.0.1:8000 \
    --videos <папка с .mov> --n 6 --mode both --batch 5 --force-first --out runs/load_L2.json
```

**Что сравниваем:**
- `sequential` — чистое время одного видео;
- `batch5` — время от приёма каждого job **с учётом очереди** против лимита 300 с.

**Выход:** `runs/load_L2.json` + `.csv`, по строке на видео:
- `status`, `verdict`, `n_violations`, `type_id`/таймкоды;
- `upload_s`, `accept_to_result_s`, `queue_wait_s`, `processing_s`, `deadline_s`;
- `gpu_peak_mb`, `response_bytes`;
- тайминги стадий A03 (`meta.timings`).

**Итоговые метрики:**
- done / failed / не завершено;
- сколько > 300 с;
- latency p50 / p90 / max;
- пик активных задач и пик VRAM (опрос `/readyz` раз в секунду).

Кадров и описаний в отчёте **нет**.

**FP/FN, попадание таймкода и тип** считает eval A16 по тем же job (`python -m eval.run --api URL`). Сам API их не оценивает.

**Чего нельзя обещать без этого замера:**
- сколько видео сервер обрабатывает одновременно и укладываются ли 5–8 в 300 с;
- время ответа VLM на полный JSON;
- пик VRAM.

Все бюджеты (`budget_s=270`, `job_cap_s=120`, `max_queue=12`, честная доля) — это **настройки, а не измеренные цифры**. После замера L2 их нужно пересмотреть.

## API из ТЗ (не менялось)

```bash
curl -X POST http://localhost:8000/api/v1/analyze -H "X-API-Key: $KEY" -F "file=@X-1.mov"
# {"job_id":"0f3c...","status":"processing"}
curl http://localhost:8000/api/v1/jobs/0f3c... -H "X-API-Key: $KEY"
# идёт:   {"job_id","status":"processing","verdict":null,"violations":[],"stage":"detect","progress":0.42,"queue_position":null,"elapsed_s":12.3}
# готово: {"job_id","status":"done","verdict":"violation","violations":[{"type","timecode_sec","description","evidence_frames":[{"timestamp_sec","image_base64"}], "type_id","confidence","t_start","t_end","rules":[...]}],"timeline":[...],"meta":{...}}
# не удалось: {"job_id","status":"failed","verdict":null,"violations":[],"error":"timeout: ..."}
```

- **`verdict` и `violations` есть всегда**, на любом статусе. Это схема A01 (`contracts/api_schema.json`); в фазе 1 у `processing` их не было.
- **Профиль ответа жюри** (`SENTINEL_JURY_PROFILE`):
  - `extended` (по умолчанию) — поля ТЗ + доп. поля, которые ТЗ разрешает. На них опирается eval A16: `type_id`, `confidence`, `rules`, `meta`.
  - `strict` — ровно поля ТЗ, как `contracts.serialize.result_to_jury_api`; норма `rules[0]` дописывается в `description`.
  - Для разовой проверки можно передать `?profile=strict`.
- Ошибки загрузки:
  - **400** — пусто, нет поля `file` или multipart оборван;
  - **413** — больше 300 МБ;
  - **415** — не то расширение, нет сигнатуры или декодер не прочитал кадр;
  - **503** — backlog полон (с `Retry-After`) или нет декодера;
  - **507** — нет места на диске.
- Принимаются `.mov` / `.MOV` / `.mp4` / `.m4v`, `video/quicktime`, имена на кириллице, в том числе `filename*=UTF-8''…`.
- `POST` отвечает сразу после записи файла: проверка — один кадр декодером, без полного прохода. Это укладывается в таймаут Cloudflare около 100 с.

| Ручка (`/api/v1/...` с `X-API-Key`) | Назначение |
|---|---|
| `POST /analyze[?force=1]` | Загрузка. Тот же файл (sha256) при той же версии пайплайна возвращает **тот же job** (`"deduplicated": true`). `force=1` — новый прогон |
| `GET /jobs/{id}[?profile=strict]` | Статус и результат |
| `GET /jobs` | История |
| `GET/HEAD /jobs/{id}/video` | Исходник с HTTP Range (206/416) |
| `GET /jobs/{id}/frames/{i}.jpg` | Кадр-доказательство `i` (сквозная нумерация, см. `meta.api.frames`) |
| `POST /jobs/{id}/ask`, `GET /copilot/status`, `GET /jobs/{id}/report` | Роутер A14 (`copilot.router.make_router`) под нашей авторизацией. Нет модуля — 501 |
| `GET /catalog` | `norms.load_catalog()` |
| `GET /healthz` | Liveness |
| `GET /readyz` | **200 только когда пайплайн загружен, прогрет, не mock и GPU не захвачен зависшей задачей.** Иначе 503 с `state` и понятным `detail`: `loading`, `warming`, `unavailable`, `warmup_failed`, `mock_not_allowed`, `gpu_hung` |

## Решения фазы 2 (каждое — отдельный флаг в `config.yaml`)

| Решение | По умолчанию | Почему |
|---|---|---|
| **Честный исход** (как `resolve_outcome` у A01) | вкл. | `done` бывает только по-честному: нет вердикта и нарушений → `failed` (`no_violation` не выдумываем); `violation` без нарушения с конечным таймкодом → `failed`; `no_violation` при неполном анализе → `failed`; при `no_violation` найденные кандидаты уходят в `candidates` (только UI); нарушение без таймкода отбрасывается (0.0 не подставляем) |
| **Полнота для `no_violation`** — зеркало `contracts.extensions.coverage_problem` (A01 `a3b2b55`): нужен `meta.coverage` (число или `{"ratio": …}`) ≥ `min_coverage`, `coverage.complete` ≠ false, обязательные стадии в статусе `ok`/`done`/`cached`, для итогового результата — `meta.complete` ≠ false | `min_coverage: 0.98` | unknown ≠ false: **нет `meta.coverage` → «нет нарушения» не подтверждено → `failed`**. На нарушения с кадрами не влияет. Сообщения — по-русски, как у A01 |
| **Покрытие A03 «по стадиям» → `ratio` A01** | **выкл.** (`coverage_adapt_stages`) | Временный мост: `ratio` = минимум покрытия по `required_stages`; неизвестная обязательная стадия → `ratio` не ставится |
| **partial.json → done** (watchdog или исключение) | **выкл.** (`partial_accept: false`) | Новый режим — до замера L2. Включённый: только при полноте ≥ `min_coverage` и прохождении политики исхода, с `meta.partial=true`. Иначе `failed` с причиной (покрытие, флаг), а `partial` — только в UI-ответе, без base64 |
| **Fake/mock в проде** | запрещены | Fake — только явный `SENTINEL_FAKE_PIPELINE=1`. `SENTINEL_MOCK=1` без `SENTINEL_ALLOW_MOCK=1` → `/readyz` 503 `mock_not_allowed`, задачи `failed`. Результат с `meta.mock` / `provenance.mock` / origin `mock` → `failed` |
| **Дедлайн 300 с от приёма, очередь включена** | вкл. | Отсчёт идёт с начала запроса загрузки. В `run()` передаётся `deadline_at=приём+budget_s`, если сигнатура его принимает (A03 принимает). После рестарта бюджет **не обнуляется**: опоздавшие → `failed: deadline passed`, остальные в очереди с остатком (`restart_fresh_budget: false`) |
| **Один владелец GPU (вместе с A03)** | вкл. | `run()` идёт под семафором (1 слот). По таймауту API вызывает **`backend.pipeline.cancel(job_id)`** (A03 останавливает воркер и выгружает модели, `cancel_on_timeout: true`), при остановке сервиса — `release_gpu()`. Если `run()` всё же не вернулся, он **держит GPU** (`zombie_release_gpu: false`): следующие задачи честно получают `failed` по таймауту, а не идут параллельно; `/readyz` → 503 `gpu_hung` через `hung_gpu_s`. Если пайплайн экспортирует `GPU_LOCK`/`GPU_SEMAPHORE`, API `run()` не оборачивает |
| **Падение воркера** | вкл. | Исключение вне `run()` или умерший поток → `failed: worker crashed`, поток пересоздаётся; `BaseException` из `run()` (в т.ч. `SystemExit` библиотеки) ловится; зависших задач нет — watchdog закрывает каждую с причиной |
| **Идемпотентность по sha256** | вкл. (`dedup`) | Ключ = sha256 + версия пайплайна (`PIPELINE_VERSION`), чтобы новая версия не отдавала старый результат. `failed` не переиспользуется. Для замеров — `?force=1` или `SENTINEL_DEDUP=0` |
| **Ответ ≤ 3 МБ** | `max_response_mb: 3` | По шагам: не больше 3 кадров на нарушение (ТЗ) → кадры кандидатов → прореживание `timeline` → крупные ключи `meta` → перекодирование кадров 640/480/360 px → 1 кадр → усечение списка. Каждый шаг пишется в `meta.warnings` |
| **Время только по PTS** | вкл. | Длительность берётся из контейнера (`mvhd`), а не frames/fps (fps переменный) |
| **Кадр-доказательство — только настоящий** (исправление по аудиту `2055fc4`) | вкл., не отключается | Если у нарушения нет кадра, `results.grab_frame_at()` ищет кадр, **ближайший по PTS**, в пределах `evidence_max_offset_s` (0.5 с) и подписывает его **фактическим PTS**. Явные ошибки: `out_of_range` (вне ролика, без декодирования), `bad_pts` (NaN, отрицательный, назад, «застрял»), `seek_failed` (seek отклонён или перескочил, и проход с начала невозможен), `budget_exhausted` (`evidence_budget_s` = 20 с на задачу), `too_far`. Раньше при сбое брался **первый кадр** с меткой запрошенного времени. Нарушение без кадра уходит в `candidates` (только UI) с `evidence_error`. Если с вердиктом `violation` не осталось ни одного нарушения с кадром → задача `failed: evidence_missing`: не `no_violation` и не `done` без кадра |
| **Готовность = итог прогрева** | вкл. | `/readyz` 503 `warmup_failed`, если `warmup()` бросил исключение, вернул `False`, вернул отчёт с ошибкой (`ok`/`ready`=false, `status`=failed/error, `error`/`errors`/`hooks_error`/`synthetic_error`, `synthetic.status`≠done) или вернул `None`, а отчёт A03 (`backend.pipeline.warmup_report` / `ModelRegistry.instance().warmup_report`) показывает ошибку. Отчёта нет — совместимость: готов |
| **Имя файла — не признак** | — | Пайплайн получает `data/jobs/<id>/video.<ext>`. Исходное имя (и нормализованное Х→X) попадает только в `meta.api` для сопоставления в eval |

## Веб-UI (согласовано с A15)

Ручки `/ui/v1/*` — те же, что `/api/v1/*`. Доступ по cookie **HttpOnly; SameSite=Strict; Path=/ui/v1** (`Secure` за https). Cookie выдаётся **только** после `POST /ui/v1/session {api_key | password}`. CORS и same-origin — не аутентификация, а лишь защита от CSRF.
- POST/DELETE требуют `X-CSRF-Token` (из ответа входа или `GET /ui/v1/session`).
- Плеер: `<video src="/ui/v1/jobs/ID/video">` — cookie + Range.
- Rate-limit на IP: вход 10/мин, загрузка 12/мин, вопросы ассистенту (ask) 20/мин, чтение 600/мин.
- UI-ответ на `/ui/v1/jobs/{id}` — полный отчёт: `candidates`, `analysis_complete`, `meta`.
- У `failed` UI дополнительно получает `partial`: `partial.json` пайплайна или отвергнутый результат с причиной. Base64-кадров там нет.
- Эталон потока — `backend/api/static/index.html`.

## Конфигурация

`backend/api/config.yaml`, любой ключ переопределяется `SENTINEL_<KEY>`. Новое в фазе 2:
- `SENTINEL_ALLOW_MOCK` (0)
- `SENTINEL_PARTIAL_ACCEPT` (0)
- `SENTINEL_MIN_COVERAGE` (0.98)
- `SENTINEL_COVERAGE_ADAPT_STAGES` (0)
- `SENTINEL_CANCEL_ON_TIMEOUT` (1)
- `SENTINEL_JURY_PROFILE` (extended), `SENTINEL_JURY_EMBED_RULE` (0)
- `SENTINEL_MAX_RESPONSE_MB` (3)
- `SENTINEL_DEDUP` (1)
- `SENTINEL_ZOMBIE_RELEASE_GPU` (0)
- `SENTINEL_HUNG_GPU_S` (60)
- `SENTINEL_RESTART_FRESH_BUDGET` (0)
- `SENTINEL_PASS_DEADLINE_AT` (1)
- `SENTINEL_EVIDENCE_MAX_OFFSET_S` (0.5), `SENTINEL_EVIDENCE_BUDGET_S` (20)

Из фазы 1:
- `SENTINEL_API_KEYS` / `SENTINEL_API_KEY`
- `SENTINEL_UI_PASSWORD`, `SENTINEL_UI_SECRET`
- `SENTINEL_DATA_DIR`
- `SENTINEL_WORKERS` / `SENTINEL_GPU_SLOTS` (1/1)
- `SENTINEL_BUDGET_S` (270; синоним A17 — `SENTINEL_DEADLINE_S`)
- `SENTINEL_WATCHDOG_S` (285), `SENTINEL_JOB_CAP_S` (120)
- `SENTINEL_MAX_QUEUE` (12)
- `SENTINEL_MAX_UPLOAD_MB` (300)
- `SENTINEL_VIDEO_VALIDATION` (required)
- `SENTINEL_FAKE_PIPELINE` (0)
- `SENTINEL_CORS_ORIGINS`
- `SENTINEL_TRUSTED_PROXIES`
- `SENTINEL_FRONTEND_DIR` (синоним `SENTINEL_FRONTEND_DIST`)
- `SENTINEL_JOB_TTL_H`

Логи — JSON с `request_id`, путь без query. Ключи, пароли, cookie и содержимое видео не пишутся.

## Тесты

`python -m pytest backend/api/tests -q` — 103 теста (проходят и в этой ветке, и в дереве `integration-cloud@2055fc4` без fallback). Нужны только Python-пакеты из `requirements.txt` (с cv2 — включая тесты настоящего MOV); GPU, весов и данных не нужно.
- `test_auth.py` — ключи, UI-сессия, CSRF, rate-limit, ключи не попадают в логи.
- `test_jobs.py` — полный цикл, нет модели / warmup / исключение / таймаут, один владелец GPU, рестарт, backlog, честная доля дедлайна.
- `test_uploads.py` — 415/400/413, стриминг, настоящий MOV, нет декодера.
- `test_media.py` — Range, кадры, copilot (A14), SPA, CORS.
- `test_phase2.py`:
  - обязательные поля на всех статусах; политика исхода (12 случаев, как у A01); кандидаты скрыты от жюри; адаптер покрытия A03;
  - partial (по умолчанию выкл. / по покрытию); mock в проде;
  - `deadline_at`; рестарт без обнуления бюджета;
  - идемпотентность; лимит 3 МБ (6 нарушений × 4 кадра шума → ≤ 1 МБ при валидных JPEG);
  - падение воркера; обрыв загрузки; обрезанный multipart и MOV;
  - кириллица `filename*`; кадр по PTS; длительность из контейнера; строгий профиль с нормой.
- `test_evidence_readiness.py` (аудит `2055fc4`):
  - валидный кадр с фактическим PTS;
  - настоящий синтетический VFR (переписанный `stts`): кадр по PTS, а не по fps;
  - недоступный таймкод (60 с в 2-секундном ролике) → `out_of_range`, без JPEG;
  - неверный PTS, ошибочный seek, исчерпанный бюджет (заглушка декодера и фальшивые часы);
  - через API: кадр подписан PTS; кадра нет → `failed: evidence_missing`, а не `no_violation`; смешанный случай;
  - `/readyz` по итогу прогрева, включая отчёт в стиле A03;
  - профили жюри (extended/strict) и UI различаются, обязательные поля ТЗ на месте.

## Модели

Нет: `models.json` = `[]`. Декодер — OpenCV (Apache-2.0); PyAV (BSD) и ffprobe — опционально.

## Ограничения / известные TODO
- Воркеры — потоки. Зависший `run()` нельзя убить из API. По `progress()` он отменяется кооперативно; иначе GPU остаётся за ним, пока `run()` не вернётся, и `/readyz` = 503 `gpu_hung`. У A03 свой процесс-исполнитель с kill по таймауту, так что это крайний случай.
- `deadline_s` передаётся один раз при старте. Честная доля учитывает только тех, кто уже в очереди.
- Сессия UI stateless (HMAC): отзыв — только TTL или смена `SENTINEL_UI_SECRET`.

## Для интегратора
1. **Ветка.** Начата до появления `main` (история отдельная), поэтому GitHub не даёт открыть PR в `main` ("no history in common"). Интегратор вливает её напрямую в `claude/integration-cloud`: фаза 1 (`657424f`) уже там. Фаза 2 — коммиты после `657424f` в этой же ветке; `main` в неё я не вливал. Код — только `backend/api/**` и пустой `backend/__init__.py`.
2. **Из `INTEGRATION_REPORT.md`** исправлены оба красных теста A02:
   - `test_copilot_501_then_router` больше не зависит от наличия A14;
   - мусорный результат пайплайна → `failed: pipeline returned unusable result` ещё до сериализатора.
   Импорты в тестах идут сначала из `contracts`, поэтому удаление `_contract_fallback.py` в интеграции тесты не ломает.
3. **Покрытие A01/A03 — СНЯТО на `2055fc4`.** A03 (`af48ab5`) пишет канонический `meta.coverage` (`compute_coverage`, есть `ratio`). Живая проверка связки: чистое видео → `done/no_violation`, `coverage.ratio=1.0`, источник — `backend.pipeline`. `SENTINEL_COVERAGE_ADAPT_STAGES` оставлен выключенным и больше не нужен.
4. **Запуск** — один процесс (`--workers 1`, как в `deploy/` A17). В проде **не** ставить `SENTINEL_FAKE_PIPELINE`, `SENTINEL_MOCK`, `SENTINEL_ALLOW_MOCK`. Healthcheck A17 должен смотреть `/readyz`: 503 при `gpu_hung`, `mock_not_allowed`, `unavailable` → перезапуск.
5. **A03:**
   - API вызывает `run(video_path, job_id, progress, deadline_s, deadline_at=приём+270)`;
   - по таймауту — `cancel(job_id)`, при остановке — `release_gpu()`;
   - `partial.json` — в `data/jobs/<id>/`. По умолчанию он **не** становится `done`. Если включить `partial_accept`, то только при подтверждённом покрытии; флаги `complete`/`partial` у partial не учитываются;
   - `JobCancelled` (BaseException) из `progress()` не глотать.
6. **A01:**
   - обязательные поля на `processing`/`failed` есть; `contracts.serialize.result_to_api` (UI-report) используется, когда доступен;
   - строгий ответ жюри — `SENTINEL_JURY_PROFILE=strict` (норма в `description` — `SENTINEL_JURY_EMBED_RULE=1`). По умолчанию extended: eval A16 читает `type_id`/`rules`/`meta` из `/api/v1/jobs/{id}`. Нужно согласовать A01 и A16, переключение — одна переменная;
   - `SENTINEL_MOCK=1` у A01 само по себе разрешает mock. У API mock включается только вдвоём с `SENTINEL_ALLOW_MOCK=1`: забытый в `.env` `SENTINEL_MOCK` не превращает прод в mock.
7. **A16:**
   - в `eval/jury_sim.py::_judge` строка `(r.get("accept_to_done_s") or 1e9) > self.limit_s` считает задачу за 0.0 с (готовый результат по sha256) «> лимита» → ложный FAIL. Нужно `is None`;
   - до исправления для замеров — `SENTINEL_DEDUP=0` или `?force=1`. С `SENTINEL_DEDUP=0` jury_sim проходит 8/8.
8. **A15:**
   - у `failed` в UI-ответе есть `partial` (без кадров);
   - у `no_violation` найденное лежит в `candidates` — показывать как «отклонённые кандидаты»;
   - текст `error` — для человека, по-русски.
9. **A14:** без изменений — `make_router` монтируется под `/api/v1` (ключ) и `/ui/v1` (cookie + CSRF).
