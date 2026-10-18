# Sentinel Copilot — подключение (A14 → интегратор L2)

Copilot — **опциональный** модуль: объясняет проверяющему результат и нормы, на вердикт, таймкод и тип
**не влияет** и ресурсы у анализа не забирает. По умолчанию в проде выключен (`enabled: mock`). Если его
не подключить, удалить или выключить, остальная система работает как прежде.

| Ситуация | Поведение |
|---|---|
| Импорт `copilot`, `copilot.router` | без весов, без torch/transformers/llama_cpp, без сети, без чтения корпуса |
| Copilot выключен (по умолчанию в проде) | `POST …/ask` → **501** `error="disabled"`; `status.available=false` (A15 скрывает панель) |
| Включён, генеративной модели в этом процессе нет | **200**, режим `retrieval`: ответ из фактов задачи и каталога норм, первая строка «(Генеративный режим недоступен…)»; `status`: `available=true, generative=false, mode=retrieval` |
| Включён, VLM загружена в этом процессе или подключён `set_chat_fn` | режим `generative`: общая Qwen3-VL-4B через `ml.vlm.service` (A11) |
| VLM не загружена здесь (в проде она в процессе-воркере A03) | copilot её **не загружает** (вторая копия на 8 ГБ = OOM) → `retrieval` |
| Анализ держит GPU дольше `wait_s` / сервис ответил `busy` | модель не дожидаемся, ответ `retrieval` с пометкой «временно недоступен (GPU занят анализом)» |
| Сервис ответил `unavailable` (модель выгружена, чат выключен) | `retrieval` с пометкой, статус пересчитается на следующем запросе |
| Таймаут / ошибка модели | 504 `timeout` / 500 `llm_error`, JSON, `answer=""`; другие эндпоинты не затронуты |
| Провайдер вернул **другую** задачу | **404** (по чужой задаче не отвечаем и её не раскрываем) |
| Неподтверждённое в ответе модели | вырезается до выдачи (и в SSE); не осталось ничего → «По имеющимся данным это установить нельзя…» |

Mock (`backend: "mock"`) — только при `SENTINEL_MOCK=1` (тесты, демо), помечен в ответе и статусе.
Production сам на mock не переключается.

## 0. Безопасный режим без отдельной LLM (рекомендация для L2)

| Режим | Как включить | Что происходит |
|---|---|---|
| Выключен (по умолчанию) | ничего не делать (без `SENTINEL_MOCK`) или `COPILOT_ENABLED=0` | 501, панель скрыта; заключение `GET …/report` работает |
| **Retrieval-only** | `COPILOT_ENABLED=1` | ответы из фактов задачи и каталога A13 с пометкой режима; без LLM, без GPU; в процессе API не импортируются `torch` и `ml.vlm.runtime` (проверено) |
| Генеративный (позже, по решению L2) | `copilot.service.set_chat_fn(fn)` — `fn` = IPC к `ml.vlm.service.chat_request` в процессе-воркере A03 | ответы общей VLM; приоритет анализа обеспечивает `gpu.PriorityLock` A11 в воркере |

Отдельной LLM у copilot нет и не будет: на 8 ГБ рядом с VLM она не помещается.

## 1. Состояние соседей (проверено чтением веток, 2026-10-17)
- **A02** (`backend/api/main.py::_mount_copilot`) монтирует `make_router(get_job, prefix="/api/v1"|"/ui/v1",
  dependencies=[…])`; `job_for_copilot` отдаёт JSON этой задачи с `job_id` = запрошенному (`results.py`
  проставляет `job_id`). Совместимо, менять ничего не нужно.
- **A03** (`backend/pipeline/worker.py`): в проде модели живут в процессе `sentinel-gpu-worker` (spawn),
  copilot — в процессе API. Канала для чата к воркеру нет → в проде copilot работает в `retrieval`, пока
  L2/A03 не подключат `set_chat_fn` (см. §2).
- **A11** (`ml/vlm`): `ml.vlm.service` есть в ветке A11 (`claude/inspiring-turing-2fm7rq`, b4b2ec5) и в
  `claude/integration-cloud`, но **нет** в `integration` (5597969). Copilot проверен с обеими версиями
  сервиса; без него — `retrieval`.
- **A13** (`norms/`): каталог v2 (`legal_status`, `legal_note`, `exclusions`, `documents`) и источники
  `retrieval: search_snippets` разбираются честно. Пример: для V05 (телефон) в каталоге нет
  подтверждённого пункта — copilot отвечает «без пункта».
- **A15** (`CopilotPanel.tsx`): `status.available=false` → панель скрыта; `POST …/ask` с
  `{question, history:[{role, text}]}` → `{answer, sources}`; панель пересоздаётся на каждую задачу
  (`JobScreen key={route.id}`), история между задачами не переносится.

## 2. Подключение

```python
# A02 (уже сделано): лениво, тем же ключом/сессией, что и остальные эндпоинты
from copilot.router import make_router
app.include_router(make_router(get_job=provider, prefix="/api/v1"), dependencies=[Depends(require_api_key)])
app.include_router(make_router(get_job=provider, prefix="/ui/v1"), dependencies=[Depends(ui_auth)])

# Необязательно (A02/A03): пока идёт анализ, копилот модель не трогает вовсе
import copilot.service as cs
cs.set_busy_probe(lambda: job_manager.has_running_jobs())   # любая функция () -> bool

# Необязательно (L2/A03): генеративный режим через VLM процесса-воркера
cs.set_chat_fn(lambda payload: worker_rpc("vlm_chat", payload))  # в воркере: ml.vlm.service.chat_request(payload)
```

`provider(job_id)` возвращает результат именно этой задачи (dict по схеме API/UI-report,
`AnalysisResult` или JSON-строку) или `None` (404). Если у результата другой `job_id` — 404.
`status="processing"` → 409, исключение провайдера → 503.

### Граница с A11 (`ml.vlm.service`) — как есть в коде A11
```python
chat(messages, *, images=None, max_new_tokens=None, timeout_s=None, wait_s=None, temperature=0.0,
     priority="low") -> ChatResult(ok, text, error, finish_reason, truncated, timed_out, …)
chat_request(payload: dict) -> dict          # тот же ChatResult для IPC
available() -> (bool, reason)                # без загрузки модели
get_service().backend() -> (backend|None, reason)   # backend.loaded — загружена ли модель
```
- Подпись читается из кода: `priority` передаётся, только если `chat` его принимает (старая версия A11 —
  `chat(messages, **kw)` с keyword-only `VLMService.chat` без `priority`; новая — явный `priority`).
  Незнакомый параметр (TypeError) убирается, вызов повторяется.
- **Приоритет анализа** обеспечивает A11 (`gpu.PriorityLock`: ожидающий анализ всегда раньше чата).
  Copilot ограничивает ожидание: `wait_s = min(vlm_service.wait_s, 50% бюджета)`, генерация — остаток;
  ожидание + генерация ≤ `llm.timeout_s` (45 с). Проверено на настоящем `ml.vlm.gpu`: пока анализ держит
  GPU, ответ `retrieval` за < 3 с, модель не вызывается.
- **Без второй VLM:** перед вызовом copilot проверяет `get_service().backend()`; модель не загружена в этом
  процессе → `retrieval` (A11 по умолчанию `allow_cold_load: true` загрузил бы её). Если `ml.vlm.runtime`
  в процессе не импортирован, модели здесь нет — `backend()/available()` не вызываются вовсе (иначе
  процесс API импортировал бы torch/transformers). Холодная загрузка — только явно
  `vlm_service.allow_cold_load: true`.
- **Ответ:** `ok=False, error="unavailable…"` → `retrieval`; `"busy"` → `retrieval` с пометкой;
  `timed_out`/`"timeout"` без текста → 504; `"invalid…"`/прочее → 500. Обрезанный (`truncated`) или
  оборванный по времени текст — только законченные предложения, затем guard.
- **Лимит входа:** промпт ≤ `vlm_service.max_input_chars` (11000 < 12000 у A11): сначала отбрасывается
  старая история, затем лишние нормы, затем укорачиваются цитаты; guard проверяет по тем же нормам.

### Авторизация
`/ask`, `/report`, `/suggestions`, `/copilot/status` подключать только с той же dependency, что
`GET /jobs/{id}`. Роутер ключей не знает. Провайдер отдаёт лишь задачи, доступные вызывающему.

## 3. Вход и выход

`POST …/jobs/{id}/ask` (тело `{"question": str(1..2000), "history": [{"role", "content"|"text", "job_id"?}] (≤40)}`) → 200:
```json
{"answer": "…[1]…", "citations": [{"document": "...", "clause": "п. 20", "url": "https://adilet.zan.kz/...",
  "fragment": "Ф3. Пункт 20 (приём наличных)", "ref": 1, "quote": "...", "verified": false,
  "relation": "catalog", "status": "candidate", "note": "норма каталога для этого типа; статус в каталоге: кандидат, требует сверки; …",
  "events": [1], "short": "Правила кассовых операций и инкассации (НБРК № 231)"}],
 "sources": ["[1] Правила … (НБРК № 231), п. 20 — https://… (норма каталога; кандидат; не сверено)"],
 "grounded": true, "refusal": false, "generative": true, "mode": "generative", "notice": null,
 "intent": "why", "backend": "vlm_service", "removed_citations": [], "warnings": [], "latency_ms": 1234,
 "disclaimer": "Подсказка для проверяющего, не юридическое заключение.", "error": null}
```
- **Только текущая задача и каталог.** История диалога — не источник фактов: таймкоды и суммы из
  истории не разрешаются, реплики с чужим `job_id` отбрасываются. Нарушения, которых нет в результате
  (ни в `violations`, ни в спорных местах), и «проверенная норма» для несверенной вырезаются.
- `relation`: `catalog` (норма каталога A13 для типа события; `events` — номера нарушений) или
  `retrieved` (найдено поиском, связь не подтверждена). `verified=false` или `status=candidate` —
  «требует сверки». Нет пункта в каталоге — «без пункта».
- `mode`: `generative` | `retrieval` | `mock`. При `retrieval` пометка режима в `notice` и первой строке.
- Ошибки: `{"detail": {"error", "message", "answer": "", …}}`.

| `error` | HTTP |
|---|---|
| `job_not_found` (нет задачи или провайдер вернул чужую) | 404 |
| `disabled` (и `unavailable` при `no_model_mode: unavailable`) | 501 |
| `not_ready` | 409 |
| `empty_question` | 422 |
| `busy` (только `busy_policy: error`) | 503 |
| `timeout` | 504 |
| `llm_error`, `internal_error` | 500 |

`GET …/jobs/{id}/suggestions?limit=4` → `{"job_id", "suggestions": [...], "by_type": {"V05": [...]}}`
(3–4 вопроса на тип из `copilot/suggestions.yaml`; вопросов о личности, сумме и умысле нет).

`GET …/jobs/{id}/report` → markdown; `?format=html` → автономный HTML (экранирован, CSP
`default-src 'none'`, кадры только из валидного base64 JPEG). Работает без модели и при выключенном
copilot. Каждое утверждение помечено:
- **Наблюдение системы** (автоматически, не проверено человеком): тип, интервал, уверенность, описание,
  «почему отмечено» (факты, детекторы, согласие VLM — из `meta.ui.evidence_overlays`/`meta.violation_extra`);
- **Подтверждённый факт**: данные обработки (задача, статус, полнота, длительность, версия), сохранённые
  кадры, отсутствие решения проверяющего в данных; норма — только если у типа `legal_status: confirmed`
  (или каталог v1 без статуса) и цитата сверена (`verified: true`);
- **Непроверенная норма**: кандидат, несверенная цитата, «без пункта», нет связи с типом в каталоге —
  с причиной. Вывод системы помечен «наблюдение, не решение».

SSE (`?stream=1`): события `token` (только проверенные предложения), затем `done` или `error`.

### Что copilot читает из результата
`violations[].{type, type_id, timecode_sec, t_start, t_end, confidence, description, evidence_frames[].timestamp_sec, rules}`.
Из `meta`: `completeness` (A01), `complete`/`partial` (A03), `coverage`, `analysis_complete`,
`ui.evidence_overlays[].{facts, sources, vlm}` и `ui.review_hints` (A12), `violation_extra` (A01).
Плюс `candidates`, `verdict_state`. Кадры, фигуры, `scores_raw` и `timeline` в контекст не попадают.

**Полнота:** неполный анализ (failed, partial, `complete=false`, required stage failed, coverage < 1)
никогда не превращается в «нарушений нет». Нет сведений о полноте — «полнота не подтверждена».

## 4. Зависимости, переменные, веса
`copilot/requirements.txt`: numpy, PyYAML, pydantic, fastapi. Опционально sentence-transformers (e5).
Своих весов LLM у copilot нет.

| Переменная / ключ | По умолчанию | Смысл |
|---|---|---|
| `COPILOT_ENABLED` | — (`enabled: mock`) | `1` включить, `0` выключить |
| `COPILOT_LLM_BACKEND` / `llm.backend` | `auto` | `auto` (vlm_service → llama_server) / `vlm_service` / `llama_server` / `llama_cpp` (только явно) / `mock` / `extractive` |
| `llm.no_model_mode` | `retrieval` | `unavailable` — прежнее поведение 501 |
| `llm.busy_policy` | `retrieval` | `error` → 503 |
| `COPILOT_TIMEOUT_S` / `llm.timeout_s` | `45` | бюджет ответа (ожидание GPU + генерация) |
| `vlm_service.{module,function,wait_s}` (`COPILOT_VLM_MODULE`) | `ml.vlm.service`, `chat`, `10` | привязка к A11 |
| `vlm_service.allow_cold_load` | `false` | `true` разрешает copilot загрузить VLM в своём процессе (не для 8 ГБ рядом с воркером) |
| `vlm_service.max_input_chars` | `11000` | лимит промпта (A11: 12000) |
| `vlm_service.runtime_adapter` | `false` | временный адаптер к `ml.vlm.runtime` (только тёплая VLM) |
| `embed.hf_cache_dir` | `weights/hf/hub` | офлайн-кэш для `intfloat/multilingual-e5-small` (`local_files_only`) |
| `COPILOT_EMBED_BACKEND` | `auto` | `hashing` — без torch |
| `COPILOT_NORMS_DIR` | `norms/` | каталог и источники A13 |
| `corpus.cite_disabled` | `false` | нормы типов `disabled` не цитировать |
| `report.html_include_frames` | `true` | кадры в HTML-заключении |

## 5. Как полностью отключить
1. Ничего не делать: без `SENTINEL_MOCK` copilot выключен (501, `available=false`), A15 скрывает панель.
2. `COPILOT_ENABLED=0`.
3. Не подключать роутер.
4. Удалить `copilot/`: от него никто не зависит.

## 6. Тесты и команды для L2
```bash
python -m pytest copilot -m "not gpu and not slow" -q   # без GPU/весов/сети; с A11/A13 в дереве идут и их тесты границы
ruff check copilot
```
**Замер на машине L2** (GPU, `ml.vlm.service`, настоящие `norms/`), одна команда на режим:
```bash
COPILOT_ENABLED=1 python -m copilot.eval_grounding --backend auto      --out outputs/copilot/grounding_auto.json
COPILOT_ENABLED=1 python -m copilot.eval_grounding --backend retrieval --out outputs/copilot/grounding_retrieval.json
python -m copilot.eval_grounding --compare outputs/copilot/grounding_retrieval.json outputs/copilot/grounding_auto.json
```
- Сравниваются генеративный режим и retrieval на одних и тех же 18 синтетических вопросах; выход —
  JSON и markdown в `outputs/copilot/` (в .gitignore); метрики: `pass_rate`, `safety_pass_rate`,
  `guard_cut_rate`, ошибки/таймауты, `latency_ms_p50/p95`, `vram_peak_mb`.
- FP/FN, таймкод и тип к copilot не относятся: он их не меняет.
- Генеративный режим в этом замере требует VLM в том же процессе (или `set_chat_fn`).

## 7. Что проверено (2026-10-17, Python 3.11, CPU, без весов и сети, только синтетические задачи)
- Ветка A14: `python -m pytest copilot -q` → **246 passed, 5 skipped** (4 теста настоящего A11 и 1 тест
  реального `norms/` пропускаются: этих модулей в ветке нет). `ruff check copilot` → чисто.
- Имитация мерджа ветки в `origin/integration` (5597969), без коммита и пуша: merge чистый, меняются
  только файлы `copilot/` (`check_ownership.py --agent A14` → только свои). `pytest copilot` →
  **247 passed, 4 skipped** (нет `ml.vlm.service`); `check_licenses.py` → OK; `ruff check copilot`
  (корневой конфиг) → чисто. `pytest backend/api` → 2 failed / 39 passed / 3 skipped — **так же без
  моей ветки** (см. §8).
- То же + `ml/vlm` из ветки A11 (b4b2ec5): `pytest copilot` → **251 passed** (в т.ч. 4 теста с настоящим
  `ml.vlm.service`/`gpu`: генерация, «не загружена — не грузим», приоритет анализа, процесс API без
  `ml.vlm.runtime`); `ml/vlm/tests/test_vlm_service.py` → 11 passed; copilot + A02 + A11 одной сессией
  в обоих порядках → 304 passed, 2 failed (те же два теста A02).
- Старый отчёт интеграции (`claude/integration-cloud` 2055fc4): copilot **7 failed** / 157 passed
  воспроизведено; после мерджа ветки A14 → **251 passed**.
- Смоук безопасного режима (`COPILOT_ENABLED=1`, реальные `norms/` и `ml.vlm.service`): статус
  `retrieval`, ответ по V05 — «без пункта», чужая задача → 404, в заключении три вида пометок,
  `torch` и `ml.vlm.runtime` не импортированы.

**Не проверено (NOT RUN — нужны GPU, веса, реальные данные L2):** генерация Qwen3-VL-4B через
`ml.vlm.service` (качество, соблюдение маркеров `[N]`/`[V]`, `guard_cut_rate`, латентность, VRAM);
IPC к воркеру через `set_chat_fn` (канала у A03 нет); e5 на реальных весах и пороги RAG; влияние
вопросов на время анализа. Дообучения не было, цифр качества нет; скрипты LoRA (`copilot/train/`)
проверены только `--dry-run`.

## 8. Несовместимости на границе (для L2, чужой код не менялся)
1. `backend/api/tests/test_media.py::test_copilot_501_then_router` (A02) падает под `SENTINEL_MOCK=1`
   (`make test`): тест ждёт «copilot нет» → 501, а в дереве есть настоящий copilot, и при
   `SENTINEL_MOCK=1` он включён (`enabled: mock`) → 200 (`retrieval`). Без `SENTINEL_MOCK` и с
   `COPILOT_ENABLED=0` тест проходит. Ожидаемое исправление в тесте A02: в первой части
   `monkeypatch.setenv("COPILOT_ENABLED", "0")` (или скрыть модуль `copilot.router`). Падает и на
   `origin/integration` без ветки A14.
2. `backend/api/tests/test_jobs.py::test_pipeline_reporting_failed_or_garbage_is_failed` (A02/A03):
   ожидает `"unusable result"` в `error`, приходит `"pipeline_failed: pipeline reported status=failed"`.
   К copilot не относится, падает и без ветки A14.
3. В `integration` нет `ml/vlm/service.py`, `gpu.py`, `qa.py` (A11, есть в `claude/integration-cloud`).
   Copilot работает и без них (`retrieval`); после мерджа A11 генеративный режим возможен только там,
   где VLM загружена в том же процессе, или через `set_chat_fn`.
4. A11 по умолчанию `service.chat.allow_cold_load: true`: любой вызов `ml.vlm.service.chat` в процессе
   API загрузил бы вторую VLM. Copilot его в таком случае не вызывает; для других потребителей решение
   за L2/A11.
5. `copilot/_contract_fallback.py` в ветке A14 не менялся: при мердже сохранится удаление из
   `integration` (a6a96f3), конфликта нет.
