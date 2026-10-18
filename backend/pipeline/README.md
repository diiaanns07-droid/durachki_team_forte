# backend/pipeline — оркестратор Kassa Sentinel (A03)

Видео → `AnalysisResult`. Стадии: ingest → detect → track → pose → scene → rules → temporal →
[partial.json] → vlm.verify → vlm.discover → fusion. Каждая стадия — ленивый импорт модуля из контракта;
весь вызов моделей идёт **последовательно в одном процессе-воркере** (один владелец GPU).

## Фаза 2: решения и флаги (кратко)

| Что | Флаг | По умолчанию | Почему так |
|---|---|---|---|
| Полнота: `meta.coverage` = `compute_coverage` A01 (`ratio` ядра, `unknown_spans`, `gaps` с хвостом, `stages`, `required`, `complete`) + `by_stage` + гейт | `coverage.gate`, `coverage.min_required` | **вкл**, 0.98 (= порог A01) | требование: неполный анализ ≠ «нарушений нет»; сериализатор A01 без `meta.coverage` отдаёт `failed` |
| `meta.provenance` / `duration_sec` / `completeness` (совместимость) | — | всегда | формат `contracts/extensions.py` |
| Время только из PTS (запасной декодер, кадры-доказательства) | — | всегда | переменный fps: индекс×fps врёт |
| Отмена `cancel(job_id)`, `release_gpu()` | — | API | отмена/рестарт останавливают воркер → модели выгружены |
| Дедлайн для всех служебных RPC воркера | `reserve_s.respawn_s` | всегда | регрессия A01: 6-с ролик отвечал 110 с при дедлайне 40 с |
| **H1** coarse→fine: обзор `overview_fps` → окна у кандидатов → плотные `fps` → уточнение начала | `refine.enabled` | **выкл** | гипотеза; включать только после замера L2 |
| **H2** таймкод = начало нарушения | `timecode.policy` (`module`/`onset`/`onset_offset`) | `module` | «политика t_peak подбирается замером, не на глаз» |
| Бюджеты стадий | `budgets_s` | плейсхолдеры | реальные — из `compare --emit-budgets` на L2 |

Ядро покрытия = пересечение интервалов кадров ingest и состояний scene (и temporal, если она обязательна);
кадр в t представляет [t, t + шаг), начало/хвост короче ~1.5 шага — артефакт сетки, не пропуск.
Проверено в интеграции (`claude/integration-cloud`, реальные модули в mock-режиме): `run --mock` → `done/no_violation`,
`ratio=1.0`, сериализатор A01 принимает; `compare --mock` проходит оба режима через процесс-воркер.

Инварианты (покрыты тестами): кандидаты не отсеиваются по движению/риску (статичное нарушение при нулевом
движении и нулевом риске доходит до VLM и отчёта); не подтверждённый окном кандидат не удаляется
(`extra.refine="not_confirmed"`); неизвестное покрытие — `null`, не 0 и не 1; инференс не читает разметку;
production никогда сам не переходит в mock (mock — только `--mock`/`SENTINEL_MOCK=1`, помечается `meta.mock`).

### Замер для L2 (одна команда на гипотезу; GPU один — режимы идут последовательно)

```bash
# 0) профиль: холодный и тёплый прогон, время и VRAM по стадиям, очередь, токены VLM
python -m backend.pipeline profile data/videos/X-1.mov --repeat 2 --json > reports/profile_X-1.json
# 1) H1: baseline vs coarse→fine (+ бюджеты стадий из реального прогона)
python -m backend.pipeline compare data/videos --labels data/labels.csv --modes baseline,refine \
       --out reports/h1 --emit-budgets reports/h1/budgets.l2.yaml
# 2) H2 (отдельно, одно изменение за раз): таймкод модуля vs начало нарушения
python -m backend.pipeline compare data/videos --labels data/labels.csv --modes baseline,onset --out reports/h2
```
Выход: `reports/h*/compare.md` (таблица режимов), `compare.json`, `<режим>/preds/*.json`, `<режим>/rows.jsonl`.
Метрики: TP/FP/TN/FN и `failed` отдельно (failed на нарушении = пропуск), |Δ| таймкода `t_peak` и onset,
hit@1/2/5 с (при конфликте дубликатов — ближайшая из `t_all`), распределение типов (GT-типов нет),
время p50/p90/max и по стадиям, VRAM-пик по стадиям, неполные анализы и минимум покрытия.
Кэш в `compare` выключен (честное время). Это разведочная оценка на тех же 42 строках — не независимый тест.

**Правило включения режима:** FN + failed_on_violation не растёт, FP не растёт, hit@2 с не хуже, p90
укладывается в SLA (≤ 60 с на 145-с видео, 5 видео ≤ 300 с), VRAM-пик ≤ свободных ~5.3 ГБ. Иначе режим
остаётся выключенным. **Откат** — профиль `baseline` / флаг в `config.yaml` (`refine.enabled: false`,
`timecode.policy: module`); бюджеты — убрать `SENTINEL_PIPELINE_CONFIG`.

## Прогрев = готовность (`warmup()`)

`warmup(deadline_s=None)` возвращает `None` **только** если прогрев реально прошёл, и тогда
`ModelRegistry.warm=True`; иначе бросает `WarmupError` (A02 ловит исключение → `warmup_failed`, не `ready`).
Всё — в пределах общего дедлайна `warmup.deadline_s` (ожидание воркера + каждый RPC + синтетический прогон):

1. аренда воркера получена (иначе «воркер занят»);
2. хуки `warmup()` модулей обязательных стадий: `ok` или нет хука; `missing` / `error` (в т.ч. «нет весов») — провал;
3. синтетическое видео через `run()`: `status=done`, `meta.complete`, без `degraded`, каждая обязательная стадия
   `ok` из настоящего модуля; `status=failed` — провал и без исключения; `builtin-mock`/запасной путь
   допустимы только в явном mock-режиме (`SENTINEL_MOCK=1`) или при `warmup.allow_fallbacks: true`;
4. дополнительные стадии — по явной политике `warmup.optional_policy` (`warn`: готов, деградации в отчёте;
   `fail`: не готов). Выключенные (`disabled`) и не задействованные синтетикой (`not_exercised`) стадии
   перечислены в `ModelRegistry.warmup_report["stages"]`, а не считаются молча успешными.

`python -m backend.pipeline warmup [--deadline S]` — код 0 готов / 2 не готов, отчёт в JSON.

## Публичные функции

```python
from backend.pipeline import run, warmup, run_with_context, ModelRegistry

res = run(video_path, job_id, progress, deadline_s=240)              # сигнатура из контракта
res = run(video_path, job_id, progress, 240, deadline_at=submitted_at + 300)  # абсолютный дедлайн (time.time())
warmup()                                                              # при старте сервера
```

* `run()` **никогда не бросает исключение**. `status`: `done` | `failed`; при `failed` — `meta.error`.
* `progress(stage_ru, frac)` — `frac` 0..1 — **общий** монотонный прогресс; подписи: «Декодирование»,
  «Поиск людей и объектов», «Трекинг», «Анализ поз», «Состояние сцены», «Правила», «Временная модель»,
  «Проверка VLM», «Формирование отчёта», в конце «Готово»/«Ошибка». Исключения колбэка игнорируются.
* `run_with_context()` — то же + контекст с промежуточными выходами (для CLI/eval).

## Обязательные и дополнительные стадии (главное правило вердикта)

`required_stages: [ingest, detect, track, scene, rules, fusion]` — «заявленный базовый анализ».
Если обязательная стадия упала / таймаут / пропущена по бюджету / модуля нет / выключена флагом:

| нашли нарушение другими стадиями? | `on_required_failure: violation_only` (по умолч.) | `fail` |
|---|---|---|
| да | `done` + `violation`, `meta.degraded=true`, warning | `failed` |
| нет | **`failed`, verdict=null** (никогда не `no_violation`) | `failed` |

Дополнительные (`pose`, `temporal`, `vlm_verify`, `vlm_discover`) выключаются флагом или
пропускаются по бюджету — с явной записью в `meta.warnings`; результат остаётся полным (`meta.complete=true`).
Встроенный декодер OpenCV (если `ml.ingest` нет/упал) — допустимый fallback с warning. **Сбой `ml.fusion`**
(исключение, таймаут, нет модуля или `fuse()` сам вернул `status=failed`) — это сбой обязательной стадии:
встроенное слияние строит отчёт только ради уже найденных нарушений (`done/violation`, `meta.degraded`),
без находок — `failed`; `no_violation` по нему не выдаётся.
Если `ml.scene` недоступен, rules получают грубые состояния из детекций, но анализ считается неполным.

## Один владелец GPU, таймауты, VRAM

* `executor.mode: process` (прод): процесс-воркер `sentinel-gpu-worker` (multiprocessing **spawn** —
  работает на Windows) держит модули, тёплые модели и клип задачи. Вызовы строго последовательные
  (detect, temporal и VLM никогда не работают одновременно). Задачи арендуют воркер целиком (`lease`).
* **Таймаут стадии → воркер убивается (`kill`)**: CUDA-контекст умирает вместе с процессом, VRAM
  гарантированно освобождается (поток так не умеет). Следующий вызов поднимает новый воркер и
  заново открывает клип текущей задачи; задача продолжается. Краш воркера (segfault, `os._exit`) —
  статус стадии `crashed`. Проверено тестами `test_worker_cli.py` (реальный spawn).
* Перед GPU-стадией `prepare_vram`: если свободной VRAM < оценки (`gpu.vram_gb`), вызываются
  хуки модулей `unload()` из `executor.unload_before` (модуль **обязан обнулить ссылки** на
  модель/тензоры — `empty_cache()` живую модель не выгружает), затем gc + `empty_cache`, затем замер
  `mem_get_info`. Не хватает и после этого → **перезапуск воркера** (`restart_on_vram_shortage`).
  Замеры до/после пишутся в `meta.vram_events`.
* `executor.mode: inline` — тесты/отладка: в текущем процессе; таймаут не останавливает CUDA,
  исполнитель помечается `poisoned` (warning в meta). Не для прода.

## Бюджет времени

* `deadline_s` отсчитывается от вызова `run()`; `deadline_at` (абсолютное `time.time()`) — чтобы
  очередь не сбрасывала отсчёт: `remaining = min(deadline_s, deadline_at - now)`. Ожидание аренды
  воркера тоже идёт из бюджета.
* `sample_fps` адаптивный: `base_fps`=5, не больше `max_frames_gpu`=900 (CPU: 300) кадров, снижение
  при дедлайне жёстче 240 с, сетка 0.5 fps (стабильные ключи кэша), не ниже `min_fps`=2.
* Мягкие бюджеты стадий (`budgets_s`, эталон 145 с × 5 fps на RTX 4060, цель ≤ 60 с) масштабируются по
  числу кадров и фактической «медленности» (EMA по detect/pose/temporal). Жёсткий таймаут =
  `max(бюджет × hard_factor, min_s)`, но не дальше дедлайна минус резерв: обязательным — минимальный
  `reserve_s.fusion_min`, дополнительным — полный `reserve_s.fusion`.
* **Цифры бюджетов — оценки, не измерения** (GPU и данных в этой сессии нет). Мерить:
  `python -m backend.pipeline profile video.mov`.

## partial.json

`data/jobs/<job_id>/partial.json` (`SENTINEL_DATA_DIR`) пишется после ядра (rules+temporal), после
VLM-проверки и перед каждой поздней стадией, если до дедлайна < `partial.margin_s`. Формат — сериализатор
A01 (`contracts.serialize.result_to_api`), но **`status: "processing"`, `verdict: null` всегда**,
`meta.partial=true`, `meta.completeness.complete=false`; найденное к этому моменту — в
`partial_verdict` (`"violation"`/`null`) и `partial_violations`. Промежуточный результат не выдаётся ни за
`done`, ни за `no_violation` (A02 показывает его только как подробности `failed`).

## Кэш

`cache/<sha256>/{dets.jsonl, tracks.json, zones.json, poses.jsonl, pose_states.jsonl, states.jsonl,
events_rules.json, events_temporal.json, events_vlm_verify.json, events_vlm_discover.json, risk.json,
manifest.json}` (`SENTINEL_CACHE_DIR`). Ключ записи стадии:
sha видео + `sample_fps` + идентичность ingest (код препроцессинга, декодер) + идентичность модуля
(`__version__`, хэш его `.py/.yaml/.json/.txt/.jinja…` = код, конфиги, **промпты**, `model_info()`:
**checkpoint** (подпись файла: имя+размер+mtime), **revision**, **fold/split**; env `SENTINEL_FOLD`,
`SENTINEL_SPLIT`) + конфиг стадии + ключи входных стадий + **хэш каталога норм** (rules) + mock.
Выходы стадий всегда проходят JSON-roundtrip → прогон из кэша идентичен «живому» (тест).
Режимы: `read_write` (по умолч.), `off`, `from_cache` (CV-стадии из кэша даже при устаревшем ключе —
помечается `stage(stale)`; rules и fusion считаются заново — тюнинг правил на 40 видео за секунды).

## CLI

```bash
python -m backend.pipeline run video.mov --out out.json [--no-vlm] [--stages detect,track,scene,rules] \
       [--mock] [--dump-intermediate [DIR]] [--from-cache] [--no-cache] [--deadline 240] [--inline]
python -m backend.pipeline batch data/raw --out preds/ [--workers 1] [--mock] [--from-cache]
python -m backend.pipeline profile video.mov [--mock] [--no-vlm] [--json] [--with-cache]
python -m backend.pipeline warmup
```
Код выхода: 0 — `done`, 2 — `failed` (реальный статус, не «пустой успех»). `batch` нормализует имена
(`Х-3` кириллица → `X-3`), пишет `<name>.json`, `summary.json`, `summary.csv`; дубликаты по
содержимому (Х-4≡Х-47) берутся из кэша. `--workers N` перекрывает только CPU-часть — GPU один.
`profile` — таблица: статус, секунды, пиковая VRAM стадии (`torch.cuda.max_memory_allocated` в воркере;
«—» на CPU), происхождение, размер выхода, пиковая RSS оркестратора.

## Mock-режим

`--mock` / `SENTINEL_MOCK=1` / `mock: true`: модули используют свои mock-бэкенды (без весов); модули,
которых нет в ветке, заменяются встроенными фейками `mocks.py` (`origin: builtin-mock`, детерминированы,
только из энергии движения кадров). **В боевом режиме отсутствующий модуль = сбой стадии**, никаких
выдуманных данных.

## Конфиг и env

Все пороги — в `config.yaml` (комментарии там). Переопределения: `SENTINEL_PIPELINE_CONFIG=extra.yaml`
(глубокое слияние), `SENTINEL_MOCK`, `SENTINEL_DEVICE=auto|cuda|cpu`, `SENTINEL_NO_VLM=1`,
`SENTINEL_VLM_POLICY=auto|keep|on_demand|off`, `SENTINEL_SAMPLE_FPS`, `SENTINEL_CACHE=off|read_write|from_cache`,
`SENTINEL_EXECUTOR=process|inline`, `SENTINEL_CACHE_DIR`, `SENTINEL_DATA_DIR`, `SENTINEL_FOLD`,
`SENTINEL_SPLIT`, `SENTINEL_LOG_LEVEL`. Воркеру передаются `SENTINEL_*`, `HF_*`, `TRANSFORMERS_*`, `CUDA_*`.

## meta результата

`pipeline_version`, `timings` (по стадиям + `total`), `stages` (status/seconds/origin/required/key/
n_out/version/vram_peak_mb/error), `sample_fps`, `device`, `warnings`, `complete`, `degraded`, `error`,
`mock`, `budget`, `video` (sha256, duration, fps, размеры, n_sampled, decoder), `catalog_source`,
`models` (origin/version по стадиям), `cache` (mode, hits), `executor` (mode, pid, kills, restarts,
poisoned), `vram_events`, `required_stages`, `partial_written`. Мета из `ml.fusion` сохраняется,
её `warnings` объединяются.

## Тесты

`python -m pytest backend/pipeline/tests -q` — 66 тестов, без GPU и без весов (~25 с): фейковые модули
через `sys.modules` (inline) и файлами на диске (реальный процесс-воркер), синтетическое видео через
`cv2.VideoWriter`. Покрыто: обязательные/дополнительные стадии и вердикты, отсутствие модуля, нарушение
контракта модулем, таймаут (inline и kill воркера с перезапуском), краш воркера, дедлайн и
`deadline_at`, partial.json, прогресс, кэш (второй прогон быстрее и идентичен; инвалидация по версии,
fold, fps, каталогу; `from_cache`), CLI run/batch/profile, нормализация имён. Маркер `slow` — тесты со spawn.

## Зависимости, модели

`requirements.txt`: pyyaml, numpy, opencv-python-headless (+ pytest). Своих моделей нет — `models.json` = `[]`.
Лицензии: всё BSD/MIT/Apache-2.0. Никаких облачных API; лица не распознаются; на кадрах-доказательствах
встроенного слияния верх бокса человека размывается.

## Ограничения

* Нет GPU/данных в этой сессии: время и VRAM не измерены, бюджеты — оценки.
* `inline`-режим не может остановить зависшую CUDA-работу (только пометка `poisoned`).
* Если модуль держит модель и не реализует `unload()`, освобождение VRAM перед VLM возможно только
  перезапуском воркера (делается автоматически при нехватке; цена — повторная загрузка весов).
* `EvidenceClip` (кадры для встроенного слияния) читает видео через OpenCV в процессе оркестратора;
  поворот берётся из метаданных OpenCV ≥ 4.5 — должен совпадать с `ml.ingest`.

## Для интегратора

* **A02 (API):** отмена задачи — `backend.pipeline.cancel(job_id)` (воркер останавливается, модели выгружены, результат
  `failed` + `meta.cancelled`); между задачами при нехватке VRAM — `backend.pipeline.release_gpu()`.
  Вызывать `run(path, job_id, progress, deadline_s, deadline_at=t_upload + SLA)` из
  ОДНОГО фонового потока очереди (воркер всё равно один). `warmup()` — в startup. Watchdog: если
  задача не завершилась к дедлайну — читать `data/jobs/<id>/partial.json`: при `verdict == "violation"`
  можно отдать как `done` с `meta.partial=true`; при `verdict == null` — `failed` (таймаут), **не**
  `no_violation`. Хранить итог — ответственность API (`contracts.serialize.result_to_api(result)`).
  Пути с кириллицей на Windows оркестратор сам заменяет хардлинком `data/jobs/<id>/input.<ext>`.
* **A04–A12 (модули):** необязательные хуки: `warmup()`, `unload()` (обнулить ссылки на модель и
  тензоры!), `model_info() -> {"checkpoint": path|[paths], "revision", "fold", "split", "prompt_version"}`
  и `__version__` — попадают в ключ кэша и `meta`. Модуль может держать модели в
  `ModelRegistry.instance().get_or_load(name="ml.vlm:qwen", loader, vram_gb=5)` — тогда
  `unload` делает реестр. Промпты VLM храните файлами в папке модуля (`.txt/.yaml/.jinja`) — они
  входят в отпечаток кэша. `ml.vlm.verify` получает все кандидаты (rules + temporal) и
  `budget_s`; оркестратор передаёт в fusion ВСЕ события (rules + temporal + verify + discover) —
  дедупликация/подавление на стороне `ml.fusion`. Событие с `extra["vlm_verdict"] == "rejected"`
  встроенное слияние отбрасывает.
* **A01:** после появления `contracts/sentinel_contracts.py` удалить `backend/pipeline/_contract_fallback.py`
  (импорт идёт через `backend/pipeline/contract.py`). В корневой `.gitignore`: `cache/`, `data/jobs/`.
  Маркеры pytest `slow`, `gpu` регистрируются в `backend/pipeline/tests/conftest.py`.
* **A16 (eval):** `batch --from-cache` для быстрого перебора правил/fusion; `summary.csv` — вердикт,
  тип, таймкод на видео; для обучаемых предсказаний задавайте `SENTINEL_FOLD`, чтобы LOO-прогоны не
  смешивались в кэше.
* **A17 (deploy):** Windows: spawn-воркер не требует `fork`; `python -m backend.pipeline warmup` —
  проверка весов офлайн (`HF_HUB_OFFLINE=1` пробрасывается воркеру).
