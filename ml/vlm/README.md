# ml/vlm — локальная VLM: верификация событий и поиск новых типов (A11)

Модуль берёт события-кандидаты (от правил A09, временной модели A10) и проверяет их по кадрам
локальной мультимодальной моделью. Модель возвращает **видимые факты, неуверенность и номера/время кадров**.
Нормы, суммы, личность и полномочия людей она не сообщает. Облачных API нет: модель работает в процессе
пайплайна (transformers) или в локальном llama.cpp.

> Результат VLM — подсказка для fusion (A12) и для человека, не истина. Гипотеза правила — тоже не истина:
> при расхождении модуль снижает уверенность и выставляет флаг противоречия, а решение принимает fusion.

## Фаза 2: что изменилось и как это замерить (для L2)

### Задержка и память: что измерено, а что нет

На этой стороне нет ни GPU, ни данных: ни одной цифры качества или скорости полного режима модуль не заявляет.

| Что | Кто / где | Условия | Результат | Статус |
|---|---|---|---|---|
| Загрузка Qwen3-VL-4B-Instruct, nf4 (LM) + bf16 (vision), transformers 5.20.1, `weights/hf/hub` офлайн | L1, RTX 4060 Laptop 8 ГБ | `tools/dataprep/vlm_smoke.py` | 12 с, 2.71 ГиБ после загрузки | измерено L1 |
| **Smoke: ответ из 2 новых токенов** | L1 | синтетика 464×848, 1 / 3 / 6 кадров полного размера = 386 / 1118 / 2216 входных токенов | 1.4 / 1.1 / 3.4 с; пик 2.88 / 3.39 / 4.72 ГиБ | измерено L1, **это не задержка анализа** |
| Полный verify JSON: 5 кадров ≤ 448 px, до 180 выходных токенов (`max_new_tokens_verify`) | — | — | — | **НЕ ИЗМЕРЕНО** |
| Полный discover JSON: 4 кадра, до 220 выходных токенов | — | — | — | **НЕ ИЗМЕРЕНО** |
| qa: один forward без генерации на вопрос (2–5 вопросов на кандидата) | — | — | — | **НЕ ИЗМЕРЕНО** |

«1–3 с» из smoke — время ответа из **2** выходных токенов. Полный JSON verify/discover — это десятки–сотни
выходных токенов, а время генерации растёт с их числом. Поэтому smoke **не** является подтверждённой задержкой
полной задачи. Входные токены здесь тоже другие: кадры ужаты до 448 px, а в smoke были полноразмерные.

**Одна команда реального полного замера (L2, локально, на GPU):**
```bash
python -m ml.vlm.bench --frames <локальная папка кадров> --mode json,discover,qa --n 12 --out outputs/vlm/bench_full.json
```
Что смотреть в `bench_full.json`:
- для каждой секции `json` (verify) и `discover`:
  - `input_frames_per_call`, `in_tokens` (входные токены), `out_tokens` (фактические выходные), `max_new_tokens`;
  - `finish_reasons` (eos | length | time);
  - `json_valid_rate` — как в проде: обрезанный ответ не засчитывается; отдельно `parse_ok_rate` и
    `truncated_rate`;
  - `cold_first_call_s` и `warm.{total_s, gen_s, prep_s, parse_s}`;
  - `vram_peak_mb_max`;
- `qa.forward_s` и `qa.candidate_s_est`;
- наверху — `load_s`, `vram_after_load_mb_nvidia_smi`, `vram_peak_mb_torch`.

Тексты ответов и кадры остаются в `outputs/vlm/bench-*/` локально, в stdout — только агрегаты.

Всё новое включается флагом и **по умолчанию выключено**, пока L2 не измерит выигрыш. Каждый пункт — отдельный коммит.

| Изменение | Флаг / где | По умолчанию |
|---|---|---|
| Мнение VLM в формате fusion: `extra.vlm.verdict` (`confirm`/`reject`/`unsure`) + `score`. **Без этого противоречие VLM до fusion не доходило** (см. «Решения») | всегда | вкл. (это исправление интеграции) |
| `extra.vlm.coverage`: проверено/всего. Непроверенное событие — «неизвестно», а не «нарушения нет» | всегда | вкл. |
| Адаптер transformers 5.x по пути, проверенному L1: картинки в сообщениях, `apply_chat_template(tokenize=True)`, `cache_dir=weights/hf/hub`, без `min/max_pixels` | `transformers.*` | вкл. |
| `finish_reason` (`eos`/`length`/`time`) и `truncated` у каждого ответа; тайминги prep/gen/decode, пик VRAM на вызов | всегда | вкл. |
| **Режим verify `qa`**: нейтральные вопросы по спецификации команды, кадры t−2/t/t+2/t+4 + кроп стола/сейфа, вероятность — softmax логитов «Да/Нет/Не видно», сопоставление с кандидатом в коде, шаблонные факты | `verify.mode: qa`, `qa_spec.yaml` | **выкл.** (`json`) |
| Подмена `t_peak` моментом смены ответа (`t_transition`) | `qa.refine_t_peak` | **выкл.** |
| `extra.vlm_verdict="rejected"` (встроенное слияние A03 такие события выбрасывает) | `verify.emit_vlm_verdict_rejected` | **выкл.** |
| `ml.vlm.service.chat(messages)`: та же модель для копилота A14, анализ всегда раньше чата | `service.chat.*` | вкл. (без вызова не грузит ничего) |
| Загрузка через `ModelRegistry` A03 (вытеснение LRU, `unload`) | `runtime.use_registry` | вкл., только внутри воркера A03 |
| Discover | `discover.enabled` | **выкл.** (как в фазе 1) |

### Одна команда замера

Что сравниваем: режимы `none` (без VLM), `json` (verify фазы 1) и `qa` (verify фазы 2) на **одних и тех же
кандидатах пайплайна A03**. Окна и кандидаты берутся от A03/A04, не из разметки.

```bash
# 0) один раз: кандидаты без VLM (A03), по каталогу на ролик
for f in data/raw/*.mov; do s=$(basename "$f" .mov); \
  python -m backend.pipeline run "$f" --no-vlm --out preds_novlm/$s.json --dump-intermediate cands/$s; done
# 1) замер скорости/VRAM полного production-режима (verify/discover JSON + qa), холодный и тёплый старт
python -m ml.vlm.bench --frames <папка с кадрами> --mode json,discover,qa --n 12 --out outputs/vlm/bench_full.json
# 2) сравнение режимов (агрегаты в stdout и --out; построчно — только в outputs/vlm/compare-*/rows.json)
python -m ml.vlm.compare --videos data/raw --labels data/labels.csv --candidates cands \
  --modes none,json,qa --budget 30 --out outputs/vlm/compare.json
```

Все режимы сравниваются **на одном наборе роликов** — тех, где вердикт определён во всех режимах
(`summary._common`). `vs_none.consistent` проверяет, что разница `tp`/`fp` с режимом `none` сходится
с fixes/breaks.

Что смотреть в `compare.json → summary.<режим>`:
- `tp`/`fn`/`fp`/`tn`, `recall`, `specificity`;
- `vs_none.fp_fixed`, `fn_fixed`, `fp_created`, `fn_created` — сколько ошибок VLM исправил и сколько создал;
- `timecode_hit["<=1s".."<=5s"]` и `timecode_abs_err_median_s` (для дубликатов с разными таймкодами
  таймкод не оценивается — это конфликт разметки);
- `types_predicted`, `vlm_s_mean`/`p95`/`max` на ролик, `vlm_calls_total`;
- `coverage_incomplete_videos`, `unknown_verdicts` — неполный анализ не считается за «нет нарушения»;
- `vram_peak_mb_torch`.

Вердикт считает настоящий `ml.fusion.fuse`, если он есть в окружении, иначе порог (`verdict_by`).

В `bench.json`:
- `json` (production-промпт verify): `cold_first_call_s`; `warm.{total_s, prep_frames_s, prep_s, gen_s, decode_s,
  parse_s}`; `out_tokens` — фактические; `finish_reasons`, `json_valid_rate`, `truncated_rate`;
- `qa`: `forward_s`, `candidate_s_est`, `mass`, `option_mode` (words/letters);
- `chat`: задержка и токены;
- `load_s`, `vram_*`.

**Проверка связки с fusion.** Прогон на `ml.fusion` из `claude/integration-cloud`: синтетика, mock-VLM,
событие правил V08, порог 0.5.

| вход (rules) | без VLM | VLM `reject` | VLM `confirm` |
|---|---|---|---|
| 0.55 | violation (0.55) | **no_violation** (0.33, `vlm_factor` 0.6) | violation (0.776) |
| 0.80 | violation (0.80) | **no_violation** (0.48) | violation (0.919) |

Выводы:
- Мнение VLM теперь реально меняет вердикт. Значит, ложный `reject` способен снять настоящее нарушение.
  Поэтому решение о включении verify — только по `fn_created` из `compare`. Если `fn_created > 0`, A12 повышает
  `vlm.reject_min_score` или убирает понижение для сильных правил, A11 — `vlm_weight`/`qa.vlm_weight`.
- В каталоге A13 из ветки интеграции норма (`rules`) сейчас есть **только у V08**. Остальные типы fusion отсекает
  как `no_norm` («для типа нет применимой нормы»), независимо от VLM. По FAQ 32 нормы есть у всех нарушений
  датасета — это задача A13. Без неё compare покажет нарушения только для V08.

**Правило выбора конфигурации.** Режим включается, только если на compare у него `fn_created == 0` и сумма
`fp_fixed + fn_fixed` больше `fp_created`, а `vlm_s_p95` укладывается в бюджет A03. Выбранный режим фиксируется
одной строкой `verify.mode` в `config.yaml` с результатом замера в коммите. Важно: все 42 строки разметки уже
использовались в разведке, поэтому это проверка на переобучение, а не независимый тест.

### Режим `qa` подробно
- **Вопросы** (`qa_spec.yaml`) не говорят модели, что подозревается нарушение. «Нет» и «Не видно» — нормальные
  ответы. «Норма» отдельным вариантом не спрашивается: модель описывает наблюдаемое (стол убран, сейф закрыт),
  а «норма это или нет» решает код по спецификации. Рамки событий на кадры не рисуются: подсветка — тоже подсказка.
- **Вероятности.** Каждый вопрос — один forward без генерации: softmax логитов первого токена по вариантам.
  Если первые токены вариантов совпадают, используются буквы A/B/C. `mass` — доля вероятности, пришедшаяся
  на варианты; при `mass < min_mass` вопрос не учитывается («неизвестно»).
- **Сопоставление с кандидатом** — код по спецификации команды (`qa.decide_<тип>`). Примеры:
  - V02: уход — норма, если ценности убраны и сейф закрыт; кандидат — если в окне после ухода на столе наличные
    или сумка, либо сейф открыт;
  - V01: второй человек на стороне кассира или второй посторонний у клиентов;
  - V03: открытый сейф, в том числе за спиной работающего кассира.

  «Не видно» уменьшает и `p_support`, и `p_contra`: неизвестность ≠ «нет нарушения». Вес VLM в обновлении
  уверенности умножается на доказательность ответов.
- **Факты** — шаблонные русские фразы из уверенных ответов. Свободного текста модели нет, придумать норму или
  полномочия нечем.
- **Ограничения.** V99 и неизвестные типы, а также бэкенд без логитов (llama.cpp) уходят в режим `json`.
  Кроп работает только при зонах A06 (`verify(..., zones=)` или `event.extra["zones"]`) или боксах вида
  `desk`/`safe`. Без них подаётся полный кадр (`crops_used=0`).
- **Стоимость.** Один forward на вопрос, 2–5 вопросов на тип, отдельный бюджет `budget.verify_qa`.
  Реальная скорость — только по `bench`.

## Публичные функции

```python
import ml.vlm as vlm

vlm.verify(clip, events, budget_s, *, limits=None, zones=None) -> list[Event]  # контракт v1 + keyword-расширения
vlm.discover(clip, budget_s, *, limits=None, motion=None) -> list[Event]      # по умолчанию выключен -> []
vlm.model_info()        # хук A03 для ключа кэша: checkpoint, revision, prompt_version (+хэш .md-промптов/qa_spec)
vlm.__version__         # "2.0.0"
vlm.warmup()            # загрузить модель заранее (держать тёплой)
vlm.unload()            # освободить VRAM
vlm.status()            # технический статус: backend, модель, loaded, причина отказа (без содержимого)
vlm.set_gpu_lock(lock)  # общий GPU-lock A03 (любой context manager): load+generate идут под ним
vlm.set_catalog(cat)    # каталог норм A13 явно (иначе norms.load_catalog() или встроенный fallback)
vlm.set_backend(be)     # подстановка бэкенда (тесты/интеграция)

from ml.vlm import service
service.chat([{"role": "system", "content": "…"}, {"role": "user", "content": "…"}])  # копилот A14
# -> ChatResult(ok, text, error: unavailable|busy|invalid|timeout|…, finish_reason, truncated, latency_s, wait_s)
service.chat_request({"messages": [...]})  # JSON вход/выход — для вызова через IPC (воркер A03)
```

`verify`/`discover` **не бросают исключений**. При сбое или недоступном бэкенде `verify` возвращает события
без изменений и пишет `extra.vlm.status` (`unavailable`, `error`, `skipped_budget`, …), а `discover` возвращает `[]`.

### Что `verify` делает с событием
- Возвращает список той же длины и в том же порядке (1:1). События не удаляются (`verify.drop_below: null`).
  `source` не меняется. Входные объекты не мутируются: на выходе копии.
- На событие берётся 4–6 кадров из окна `t_start−1 … t_end+1`, кадр `t_peak` входит всегда (для длинных событий
  окно ≤ 12 с вокруг пика). На кадры наносятся боксы `event.bboxes` и подпись `#k t=…s`. Длинная сторона ≤ 448 px.
- Промпт требует строгий JSON:
  `{"visible", "type_id", "observed_facts", "confidence", "best_frame", "visible_frames", "uncertain_because"}`.
- Ответ проходит валидацию: `type_id` должен входить в enum каталога, `confidence` — лежать в [0..1]
  (проценты пересчитываются, остальное отбраковывается), номера кадров — в диапазоне 1..N.
  Ошибки валидации попадают в `extra.vlm.flags`.
- Уверенность пересчитывается логит-пулингом:
  `logit(p) = logit(p0) + w·logit(p_vlm)`, где `p_vlm = c` при `visible=true`, иначе `1−c`;
  `p_vlm` обрезается в `[0.05, 0.95]`, `w = 0.6`; VLM не поднимает уверенность больше чем на `max_raise = 0.25`.
  **«Пола» для сильных правил нет.**
- `extra.vlm.agreement`:
  - `supports` — VLM подтверждает; её факты добавляются в `event.facts`.
  - `contradicts` — VLM не видит признака; `extra.vlm.contradiction = true`, уверенность снижена.
  - `uncertain` — ответ не информативен.
  - `type_mismatch` — VLM видит другой тип; `suggested_type_id`, `p` не меняется.
  - `not_checked` — проверки не было: бюджет, нет кадров, сбой.
- Тип переклассифицируется только для источников `temporal`/`vlm` при `confidence ≥ 0.7`; исходный тип
  сохраняется в `extra.vlm_original_type_id`. Тип события правил не меняется — только подсказка.
- Время: `extra.vlm.t_best` и `extra.vlm.t_visible` — времена кадров, которые указала VLM (после валидации).
- Факты проходят фильтр «невидимого»: суммы, проводки, личность, подлинность, умысел, нормы и пункты,
  полномочия и допуск, эмоции, содержимое закрытых объектов, вероятностные оговорки. Кроме того, факт должен
  называть видимую роль или объект, а писаться по-русски. В `extra.vlm.dropped_facts` попадают только категории,
  без текста. Если VLM подтвердила признак, а все её факты отбракованы, подставляется наблюдаемый признак
  из каталога: «Локальная VLM-модель отметила видимый признак: …».
- События одного типа, пересекающиеся по времени, проверяются одним вызовом (`extra.vlm.shared`).
- Порядок проверки — `uncertain_first`: сначала события с уверенностью около 0.5, там VLM полезнее всего.

### `discover` (выключен по умолчанию)
`discover.enabled: false`, пока локальный бенчмарк не покажет скорость и частоту ложных срабатываний. Включить
можно в config или через `discover_events(..., force=True)`.
- Окна по ~3 с. Обход: самые «подвижные» по motion energy первыми, плюс минимальное равномерное покрытие;
  хвост обрезает бюджет.
- Чек-лист: телефон, еда/напиток, открытый сейф без присмотра, посторонний в зоне кассира, наличные без
  присмотра, сумка инкассатора, открытая дверь, скрытые руки, «иное» → V99.
- Находка становится событием только при конкретном видимом факте, прошедшем фильтр, и при
  `confidence ≥ 0.55`. Приор 0.25, потолок 0.6, `source="vlm"`.
- После первой находки один вызов держится в резерве под повторную проверку (`reverify`).
  Если `verify` возражает, находка отбрасывается.

## Бюджет (общий для стадии)
Время задаёт A03 (`budget_s`). Остальные лимиты берутся из `config.yaml → budget`
или из аргумента `limits={...}` (A03 может урезать):

| лимит | verify | discover |
|---|---|---|
| вызовов | 6 | 4 (включая reverify) |
| кадров на вызов | 6 | 6 |
| визуальных токенов на вызов / всего | 1200 / 6000 | 1200 / 3600 |
| выходных токенов на вызов / всего | 220 / 1400 | 220 / 1000 |
| ретраев на вызов / всего | 1 / 2 | 1 / 1 |

Вызов не начинается, если `remaining < est_call_s × 1.15` (оценка — EMA фактических задержек).
У каждого вызова жёсткий таймаут `min(12 с, остаток)`: `generate(max_time=…)` или HTTP timeout.
Ретрай делается только при невалидном JSON (со строгим суффиксом). Таймаут не ретраится.
Число визуальных токенов оценивается как `(W/px)·(H/px)` (px = 32 у Qwen3-VL, 28 у Qwen2.5-VL).
Если кадры не влезают в лимит, они уменьшаются.

## Backend’ы
| backend | статус | что это |
|---|---|---|
| `transformers` | **основной** | Qwen3-VL-4B-Instruct, bnb 4-bit nf4 (vision-энкодер не квантуется), в процессе пайплайна. Один GPU-владелец: load и generate идут под общим lock. Ленивая загрузка, `unload()`. Проверки: свободная VRAM (`min_free_vram_mb`), CUDA OOM (ответ с ошибкой вместо падения), лицензия снапшота, офлайн-загрузка |
| `llamacpp_server` | опционально | локальный `llama-server` (MIT) по HTTP, OpenAI-совместимый. JSON-схема превращается в грамматику, при отказе сервера — fallback на `json_object`. Нелокальные URL запрещены, прокси обходятся |
| `llamacpp_python` | опционально, экспериментально | llama-cpp-python в процессе, один кадр-коллаж на вызов |
| `mock` | тесты | `SENTINEL_MOCK=1`; режимы `neutral`/`confirm`/`reject`/`garbage`/`phone` |

Если бэкенд недоступен (нет CUDA, весов или пакета), стадия **пропускается**. Подмены на mock нет.

## Конфигурация
Файл `ml/vlm/config.yaml`; все пороги лежат там. Переменные окружения:

| env | смысл |
|---|---|
| `SENTINEL_MOCK=1` | mock-бэкенд (сильнее `SENTINEL_VLM_BACKEND`: гарантирует прогон без весов) |
| `SENTINEL_VLM_BACKEND` | `transformers` / `llamacpp_server` / `llamacpp_python` / `auto` / `off` / `mock` |
| `SENTINEL_VLM_PRESET` | `qwen3vl4b` (по умолчанию) / `qwen3vl8b` / `qwen25vl7b_awq` / `qwen3vl2b` |
| `SENTINEL_VLM_MODEL` | путь к локальной папке весов или HF id (для `.gguf` — имя файла) |
| `SENTINEL_VLM_URL` | адрес llama-server (только локальный) |
| `SENTINEL_VLM_CONFIG` | частичный yaml поверх дефолтного |
| `SENTINEL_VLM_LANG` | `ru` / `en`; язык промпта, факты всегда по-русски |
| `HF_HUB_OFFLINE=1` | загрузка только из локального кэша (`local_files_only`) |

## Как запустить
```bash
pip install -r ml/vlm/requirements.txt          # torch — CUDA-сборкой, см. комментарий в файле
python -m pytest ml/vlm/tests -q                # без GPU/весов/сети, mock + синтетика

# один раз, с интернетом (A17): веса в HF-кэш, дальше офлайн
huggingface-cli download Qwen/Qwen3-VL-4B-Instruct
set HF_HUB_OFFLINE=1                            # Windows (bash: export HF_HUB_OFFLINE=1)

# замер скорости/VRAM (агрегаты в stdout, ответы — только в локальный дамп outputs/vlm)
python -m ml.vlm.bench --synthetic 6 --n 5                     # без данных
python -m ml.vlm.bench --frames path/to/frames --n 10 --out bench.json
# ручная проверка момента видео
python -m ml.vlm.try video.mov --t 22 --type V05 [--mode verify|discover|both]
# калибровка на локальной разметке: доля срабатываний discover/verify на позитивах (размеченная секунда)
# и негативах (случайные моменты чистых роликов) — ТОЛЬКО агрегаты; имена X/Х нормализуются, дубли по md5 пропускаются
python -m ml.vlm.calibrate --videos data/videos --labels labels.csv --neg-per-video 2 --out calib.json
# GPU smoke-тест (локально): python -m pytest ml/vlm/tests -m gpu
```

### Приватность вывода
Реальные кадры, промпты и ответы модели CLI пишет **только локально**: `privacy.local_dump_dir`
(по умолчанию `outputs/vlm/`, в `.gitignore`). В stdout выходят технические агрегаты:
статус, задержки, согласие и уверенность, число фактов, категории отбракованных фактов, флаги валидации.
Тексты фактов печатаются только с флагом `--show-text`. Не включайте его, если вывод читает облачный ассистент,
в том числе локальный Claude. Сырой текст ответа в `Event.extra` не кладётся (`privacy.keep_raw_in_extra: false`).
В логи попадают только имена исключений.

## Модели и лицензии
Подробности — в `models.json`. Все веса под Apache-2.0. Лицензию локального снапшота проверяет
`license_guard`: читается поле `license:` из README.md в HF-кэше, без сети. Запрещённая лицензия
(например, `other` у Qwen2.5-VL-3B) блокирует загрузку.

| модель | HF id | лицензия | роль |
|---|---|---|---|
| Qwen3-VL-4B-Instruct | `Qwen/Qwen3-VL-4B-Instruct` | Apache-2.0 | основная |
| Qwen3-VL-8B-Instruct | `Qwen/Qwen3-VL-8B-Instruct` | Apache-2.0 | опц. |
| Qwen2.5-VL-7B-Instruct-AWQ | `Qwen/Qwen2.5-VL-7B-Instruct-AWQ` | Apache-2.0 | опц. (нужен autoawq) |
| Qwen3-VL-2B-Instruct | `Qwen/Qwen3-VL-2B-Instruct` | Apache-2.0 | запасная, только по замеру |
| Qwen3-VL-4B-Instruct-GGUF | `Qwen/Qwen3-VL-4B-Instruct-GGUF` | Apache-2.0 | опц., llama.cpp |

Библиотеки: transformers, accelerate, huggingface_hub — Apache-2.0; torch, numpy — BSD-3; bitsandbytes,
llama.cpp, llama-cpp-python, autoawq, PyYAML — MIT; Pillow — HPND; opencv-python-headless — Apache-2.0.

## Ограничения (честно)
- **Ничего не измерено на реальном железе.** В облачной сессии нет GPU и данных: VRAM, скорость и точность
  4B-модели на RTX 4060 неизвестны. Первым шагом локально запустить `bench`. Если 4B не влезает
  в 8 ГБ рядом с остальными моделями, сначала урезать бюджет (`frames.max_side`, `verify_n_frames`,
  `max_visual_tokens_per_call`, `max_output_tokens_per_call`), выставить `min_free_vram_mb` и перемерить.
  Переход на 2B — только как измеренное решение.
- Фильтр стоп-слов и якорей **не гарантирует отсутствие галлюцинаций**: правдоподобная выдумка
  («кассир держит телефон») проходит. Защита держится на валидации полей, низком весе VLM (`w = 0.6`),
  флагах для fusion и решении человека.
- Самооценка уверенности VLM не откалибрована. `vlm_weight` и `p_vlm_clip` нужно тюнить на LOO (A16).
- Промпты v1 написаны без реальных кадров и без few-shot (так и задумано). Качество формулировок
  проверяется только локально.
- `llamacpp_python` экспериментален: таймаут через stopping_criteria, мультикартинка заменена коллажем.
- Имена GGUF-файлов в пресетах — ожидаемые, их нужно сверить перед скачиванием.

## Для интегратора
1. **A03 (оркестратор):**
   - вызывать `vlm.verify(clip, events, budget_s)` после правил и temporal;
   - `vlm.discover(clip, budget_s)` — только если включён;
   - передать общий GPU-lock через `vlm.set_gpu_lock(lock)`, чтобы VLM не шла параллельно с детектором
     и энкодерами;
   - `vlm.warmup()` вызывать в `backend.pipeline.warmup()`;
   - `vlm.unload()` — если VRAM нужна другим;
   - лимиты урезать через `limits={...}` (ключи — поля `VLMLimits` в `budget.py`);
   - GPU-lock передавать такой, который A03 не держит в момент вызова `verify` (или `RLock`): VLM берёт его сама;
     обычный `Lock`, уже захваченный вызывающим, приведёт к дедлоку;
   - сигнал `motion_energy` из A08 можно отдать в `discover(..., motion=[(t, e), ...])`, чтобы не считать его заново;
   - при 5+ видео одновременно GPU-вызовы VLM сериализуются lock’ом, бюджет каждого видео — wall-clock;
   - **фаза 2:** зоны трекинга передавать в verify:
     `("ml.vlm", "verify", (CLIP, candidates, "$budget"), {"zones": zones})`. Без зон режим qa спрашивает
     по полному кадру, без кропа стола/сейфа;
   - `model_info()` и `__version__` уже есть. Промпты лежат в `.md`, их хэш входит в `prompt_version`;
   - внутри воркера модель грузится через `ModelRegistry.get_or_load("ml.vlm:qwen", vram_gb=3.5)`
     с `unload` → `release()` освобождает VRAM;
   - `meta.coverage`: брать `event.extra["vlm"]["coverage"]` (checked/total/complete). Неполная проверка
     не превращает кандидата в «нарушений нет».
2. **A12 (fusion):** `ml.fusion.vlm_opinion` уже читает `extra["vlm"]["verdict"]` и `["score"]` — с фазы 2 они
   заполняются: `confirm` (score = p_vlm), `reject` (score = 1 − p_vlm), `unsure`; при `not_checked`
   вердикта нет. Остальные поля в `event.extra["vlm"]`:
   - `agreement`, `contradiction`, `p_prior`, `p_vlm`, `p_post`; в режиме qa ещё `p_support`, `p_contra`,
     `evidence`, `answers`, `t_transition`, `facts_pro`/`facts_con`;
   - `suggested_type_id`, `t_best`, `t_visible`, `flags`;
   - `status`, `budget` — агрегаты стадии.

   Решение о вердикте и пороги — на стороне fusion. События `source="vlm"` из discover имеют низкий приор
   (≤ 0.6) и `extra["vlm_discover"]`.
3. **A13 (нормы):** из каталога используются `types.<Vxx>.name`, `observable`, `not_observable`.
   `rules` в промпт не попадают: VLM нормы не цитирует. Если ключа V99 нет, модуль добавляет его сам.
4. **A17 (deploy):**
   - скачать веса заранее и выставить `HF_HUB_OFFLINE=1`;
   - на Windows нужны CUDA-сборка torch и bitsandbytes ≥ 0.45;
   - для llama.cpp-варианта: `llama-server -m <gguf> --mmproj <mmproj> --port 8081 -ngl 99 -c 8192`
     и `SENTINEL_VLM_BACKEND=llamacpp_server`.
5. **A14 (копилот):** вместо второй LLM можно использовать ту же Qwen3-VL через `ml.vlm.service.chat(messages)`.
   - Анализ всегда в приоритете: чат ждёт GPU не дольше `service.chat.wait_s`, иначе получает `busy`.
   - Если модели нет — `unavailable`, mock не подставляется.
   - Модель одна на процесс. Если копилот живёт в процессе API, а анализ — в воркере A03, то, чтобы не грузить
     вторую копию, звать `service.chat_request(payload)` (JSON вход/выход) через IPC воркера.
6. **A01 (контракт/CI):**
   - контракт не менялся;
   - `_contract_fallback.py` — копия `contracts/sentinel_contracts.py`, удалить после мерджа (импорт идёт
     через `ml/vlm/_contract.py`);
   - маркеры `gpu`/`slow` регистрируются в `ml/vlm/tests/conftest.py` — можно перенести в корневой pyproject;
   - тесты лежат в пакете `ml.vlm.tests` (`__init__.py`), файлы с префиксом `test_vlm_`, чтобы не
     конфликтовать с одноимёнными тестами других модулей.
   - Предложение: зафиксировать в контракте семантику `Event.extra["vlm"]["agreement"]`.
7. **A16 (eval):**
   - `python -m ml.vlm.compare` (см. «Одна команда замера») — сравнение none/json/qa на кандидатах A03;
   - `python -m ml.vlm.calibrate` даёт долю срабатываний discover/verify на позитивах и негативах; это основа
     для решения о `discover.enabled` и подбора `vlm_weight`;
   - прогнать LOO с `SENTINEL_VLM_BACKEND=transformers`;
   - сравнить вердикты с VLM и без неё, частоту `contradicts` на чистых роликах;
   - покрутить `verify.vlm_weight` и `max_raise`;
   - решить, включать ли `discover`.

## Принятые решения (автономно)
- Основной backend — transformers в процессе: один GPU-владелец и общий lock. Отдельный llama-server был бы
  вторым владельцем VRAM, поэтому llama.cpp оставлен опцией.
- Модель по умолчанию — Qwen3-VL-4B (Apache-2.0) в 4-bit. 7B/8B не по умолчанию: память не измерена,
  а 8 ГБ делятся с другими моделями.
- Подмодули названы `verifier.py` и `discoverer.py`. Если назвать их `verify.py`/`discover.py`,
  `from .verify import …` затёр бы функцию `ml.vlm.verify` модулем.
- `python -m ml.vlm.try`: `try` — ключевое слово Python, поэтому `try.py` — тонкая обёртка над `cli_try.py`.
- Факты для жюри требуются по-русски; ответы на латинице отбрасываются и заменяются шаблоном из каталога.
- Неизвестный `type_id` от модели не подменяется на V99 молча: он становится `None` и получает флаг.

**После аудита (2055fc4):**
- Падения `test_vlm_prompts` (ru/en) и двух `test_vlm_compare` воспроизведены на 2055fc4 с настоящими
  `norms` A13 и `contracts` A01. Причины: тест ждал имя V05 из `FALLBACK_CATALOG`, а не из активного каталога;
  compare в сборке шёл через `ml.fusion` с гейтом `no_norm`. Исправлено в `331a3c2`, усилено в `ad1ee14`:
  проверяется вся строка кандидата; compare-тесты изолированы синтетическим каталогом и явным порогом;
  все режимы считаются на одном наборе роликов с самопроверкой.
- **Обрезанный или упавший ответ больше не может стать подтверждением.**
  - Раньше JSON, обрезанный по `max_new_tokens` или `max_time`, «чинился» парсером и мог дать `confirm`.
  - Теперь `length` → 1 ретрай, затем `status=truncated`; `time` → `timeout` без ретрая.
  - `parse_failed` / `truncated` / `timeout` / `error` (в т.ч. CUDA OOM) → `verdict=unsure`, `score=0`,
    уверенность не меняется, покрытие неполное.
  - Регэксп-разбор невалидного текста даёт только «не уверен».
- **Время — фактический PTS показанного кадра.** Берётся из `frame_at_ts` A04, а не запрошенное.
  Один и тот же кадр (VFR, пропуск, край ролика) модели повторно не показывается.
- **Реестр A03 и `unload()`.** Устаревшая запись `ml.vlm:qwen` в `ModelRegistry` заменяется, `unload()` её
  снимает. Раньше новый экземпляр мог жить вне учёта VRAM.
- **Граница с A14.** `copilot/llm.py VLMServiceBackend` передаёт `priority="low"`; раньше это давало
  `TypeError`. Теперь у `service.chat` явная сигнатура с `priority`, у `ChatResult` поле `timed_out`
  (A14 → `LLMTimeout`), `service.available()` проверяет доступность без загрузки модели. Модель, transformers
  5.x, `weights/hf/hub`, nf4 + bf16 vision — без изменений; `requirements.txt` не трогался (lock L1).

**Фаза 2:**
- **Найден и исправлен разрыв с fusion.**
  - `ml.fusion` ищет мнение VLM в `extra.vlm.verdict|label|answer|decision|status`, а модуль писал только
    `status="ok"` и `agreement` — fusion мнения не видел.
  - A03 отдаёт в fusion и исходные события, и проверенные копии; при `combine: max` пониженная копия ни на что
    не влияла. Противоречие VLM терялось.
  - Теперь пишутся `verdict` и `score`. `status` не похож на вердикт, поэтому fusion его игнорирует.
- **Режим qa: модель отвечает на узкие вопросы, а нарушение определяет код по спецификации.** Решение
  проверяемо, «Не видно» честно учитывается, а факты не содержат свободного текста модели.
- **Вероятность берётся из логитов первого токена, а не из «confidence» в JSON.** Самооценка небольшой VLM
  не откалибрована. Логиты тоже не откалиброваны; температуру и смещение подбирать по замеру L2.
  Отдельной калибровки пока нет — сначала замер.
- **Без рамок и без слова «нарушение» в вопросах.** Подсветка и формулировка подсказывали бы ответ
  (требование «не подсказывай»).
- **Кадры t−2/t/t+2/t+4 от `t_peak` кандидата.** Политика якоря (`anchor: t_peak|t_start`) и смещения — в
  `qa_spec.yaml`, подбираются замером. `t_transition` только записывается; подмена `t_peak` — флаг.
- **transformers 5.x — ровно по пути L1** (`vlm_smoke.py`). Второй стек не строился. `max/min_pixels` не
  передаются, потому что в 5.x они устарели и кадры уже ужаты.
- **Чат копилота — на той же модели через приоритетный lock**, а не вторая LLM. Вторая модель на 8 ГБ рядом
  с анализом не помещается (L1: свободно ~5.3 ГБ).
- **В облако и чат — только агрегаты** (`compare`/`bench`). Построчные результаты, кадры и ответы пишутся
  в `outputs/vlm` локально.

## Файлы
```
__init__.py            публичный API (лёгкий импорт)
config.yaml, config.py конфиг + env
budget.py              общий бюджет: время/вызовы/кадры/токены/ретраи
runtime.py             жизненный цикл бэкенда, GPU-lock, вызов с JSON-ретраем, ужатие кадров
verifier.py            verify_events()
discoverer.py          discover_events(), plan_windows()
scoring.py             логит-пулинг, согласие
parsing.py             извлечение/ремонт JSON, схемы, валидация enum/диапазонов
facts_filter.py        фильтр невидимого + якоря наблюдаемости
frames.py              выбор моментов, боксы, ресайз, коллаж, motion energy
prompting.py, prompts/ библиотека промптов <id>.<version>.<lang>.md (RU/EN)
catalog.py             каталог A13 или встроенный fallback (без норм)
backends/              mock, transformers, llama.cpp (server/python)
bench.py, try.py (cli_try.py), calibrate.py           CLI: скорость/VRAM, ручная проверка, калибровка
compare.py                                              CLI фазы 2: none/json/qa на кандидатах A03, исправлено/создано
qa.py, qa_spec.yaml                                     режим verify.mode=qa: вопросы, логиты, сопоставление
service.py, gpu.py                                      чат копилота на той же модели; приоритетный GPU-lock
localdump.py, _video.py                                 локальный дамп, запасной загрузчик видео
_contract.py, _contract_fallback.py                     импорт контракта
tests/                 pytest: mock + синтетика, GPU-тест помечен @gpu
```
