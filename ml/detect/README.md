# ml/detect — детекция объектов (A05)

`detect(clip, every_n=1) -> list[Detection]`: метки из `LABELS`, bbox нормированы 0..1 к кадру после учёта поворота.
`track_id=None` (трекинг — A06). В `Detection.source` записан исходный класс модели (см. «Семантика меток»).
Фаза 2 добавила `desk_inventory(clip, t, zones)` и `safe_door_state(clip, t, zones)` — вход для правила
«ушла и оставила» (A09) — и инструменты для L2: `probe`, `calibrate`.

## Статус

| Часть | Состояние | По умолчанию |
|---|---|---|
| **Baseline**: RT-DETRv2-S (`PekingU/rtdetr_v2_r18vd`) из `weights/hf/hub`, офлайн, transformers 5.x, каждый сэмпл, fp16 на CUDA (откат на fp32) | готово | **вкл.** |
| D-FINE-M (`ustc-community/dfine-medium-obj2coco`, скачан у L1) — только **явный** выбор в конфиге; молча не подставляется | готово | выкл. |
| `desk_inventory` / `safe_door_state`, режим `baseline` | готово (фаза 2) | **вкл.** |
| **Абляция:** `inventory.mode=rare_open_vocab` — редкий проход OWLv2 с контрастными запросами | готово, **ждёт замера L2** | выкл. |
| `coverage` и `status=partial` в `last_stats()` | готово (фаза 2) | вкл. |
| `python -m ml.detect.probe` — команда замера для L2 | готово (фаза 2) | — |
| `python -m ml.detect.calibrate` — пороги по классам по человеческой разметке | готово (фаза 2) | — |
| TTA-флип, тайлы, open-vocab в `detect()`, дообученная ONNX | готово (фаза 1), опционально | выкл. |
| Mock (`SENTINEL_MOCK=1`) — только тесты и демо | готово | выкл. |

### Что реально проверено (CPU, без GPU, без реальных данных и весов)
`python -m pytest ml/detect/tests`: **91 passed** (torch 2.14.1 CPU, transformers 5.20.1, Python 3.13); ruff по конфигу
интеграции и `pyflakes` чисты. Как в CI (Python 3.11, opencv 4.14 по lock L1, без torch, `-m "not gpu and not slow"`):
**76 passed, 3 skipped** (пропущены три файла, которым нужен torch). Тесты импортируют канонический
`contracts.sentinel_contracts` и не требуют удалённой в интеграции fallback-копии.
- Загрузка крошечной RT-DETRv2 из раскладки `weights/hf/hub` (как у L1) и из `save_pretrained`-папки. Сеть в тесте запрещена.
- Офлайн-кэш: поиск в `HF_HUB_CACHE` → `HF_HOME/hub` → `weights/hf/hub` → `weights/hf`; пустой снапшот (прерванная
  загрузка) не считается весами; `local_files_only=True` при любом `HF_HUB_OFFLINE`.
- Нет весов → `DetectorUnavailable` с id модели и списком проверенных путей; снапшот есть, но не грузится → тоже явная
  ошибка. Ни mock, ни другой детектор не подставляются: скачанный D-FINE без явного выбора не используется, явный
  резерв даёт `status=degraded` и предупреждение.
- Таблица классов: все 80 классов COCO → 21 сопоставлен с метками контракта, 59 игнорируются; общий детектор не выдаёт
  нестандартные метки (cash, safe, door, keys…), даже если в его `id2label` есть похожие имена.
- Синтетические выходы моделей: нормированный бокс одинаков для портретного, горизонтального и квадратного кадра;
  пиксели = норм. × (W, H); кадры разных размеров в одном вызове; OWLv2 на горизонтальном кадре (паддинг снизу);
  bbox нормируется к фактическому кадру, даже если `clip.width/height` от него отличаются.
- Отсутствие детекции: на тёмных кадрах боксов нет, `coverage` считает кадры обработанными, но об отсутствии ничего не
  утверждается; `desk_inventory`/`safe_door_state` отвечают `unknown`.
- Совместимость с выгрузкой `det-labels` (`datasets/det/export_coco.py`): `info.images_root`, `contract_label`
  (`safe_closed` → `safe`), группы дубликатов и зафиксированный split.
- `desk_inventory`/`safe_door_state` на синтетике:
  - бывает только present или unknown, «нет» не выдаётся;
  - фильтр по зонам;
  - объект, мелькнувший на 1 из 3 кадров, считается unknown;
  - cash_bag выдаётся только после контрастной проверки;
  - стационарный телефон гасит COCO «cell phone» на всех кадрах окна;
  - кэш не переиспользуется между разными объектами клипа;
  - для сейфа выдаются open / closed / unknown;
  - предупреждение, если стол закрыт людьми;
  - битые кадры не роняют вызов;
  - горизонтальный кадр 848×464 обрабатывается так же.
- `probe` из папки кадров и из видео: отчёт, `.gitignore` в папке вывода, картинки только локально.
- `calibrate`: выбор порога по целевой точности, holdout по видео, отказ на псевдоразметке.
- `coverage`: битый кадр, увеличенный шаг из-за бюджета и отказ бэкенда дают `partial` с интервалами пробелов, а не «пусто».

**Не измерено:**
- скорость, VRAM и качество на RTX 4060 и реальных роликах — это делает L2 командой ниже;
- mAP — его нет и без ручной bbox-разметки не будет.

## Публичный API

```python
from ml.detect import detect, warmup, last_stats, available, desk_inventory, safe_door_state, DetectorUnavailable

dets = detect(clip, every_n=1)   # list[Detection]; нет локальных весов → DetectorUnavailable
last_stats()  # {"status": ok|degraded|partial|unavailable, "coverage": {...}, "timings", "backends", "warnings", ...}
inv = desk_inventory(clip, t, zones)      # DeskInventory; inv.items[kind].state ∈ {"present", "unknown"}
safe = safe_door_state(clip, t, zones)    # SafeDoorState; safe.state ∈ {"open", "closed", "unknown"}
inv.as_dict(); safe.as_dict()             # JSON-сериализуемо
```
`zones` — `list[Zone]` от A06, `[{"name", "polygon"}]` или `{name: polygon}`; используются `desk`/`cash_tray` и `safe`.

### `last_stats()["coverage"]` (для `meta.coverage` у A03)
`{"frames_planned", "frames_processed", "fraction", "gaps_s": [[t0, t1], ...]}`.
- `status="partial"`: часть кадров не проанализирована (кадр не декодировался, бюджет увеличил шаг, отказал бэкенд). **Отсутствие детекций в `gaps_s` означает «неизвестно», а не «нет».**
- `degraded`: кадры покрыты, но опции отрезаны бюджетом или отказала одна из стадий.
- `unavailable`: нет весов.

Ни один из этих статусов нельзя превращать в `no_violation` без проверки полноты.

### `desk_inventory(clip, t, zones)`
Проверяется окно `t ± window_s/2` (по умолчанию 3 точных кадра через `clip.frame_at`) в зонах `desk ∪ cash_tray`.

| kind | baseline (COCO) | rare_open_vocab (+OWLv2) |
|---|---|---|
| cup (чашка/стакан/бутылка) | cup, wine glass, bottle | + «cup», «drink bottle» |
| food | COCO-еда | + «food» |
| phone | COCO «cell phone»; в заметке: мобильный и стационарный не различаются | «mobile phone» должен перевесить «desk telephone»; где перевешивает стационарный, COCO-детекция гасится |
| bag (тип неизвестен) | handbag / backpack / suitcase | + обычные сумки и сумки, не прошедшие проверку на инкассаторскую |
| personal_item | **unknown** (в COCO нет курток; тип сумки неизвестен) | «jacket», «coat», обычная сумка (перевес над инкассаторской) |
| cash | **unknown** (нет класса в COCO) | «stack of banknotes», «cash money» |
| cash_bag | **unknown**; только дообученная на проверенной разметке модель | «инкассаторская сумка» должна перевесить «обычную сумку» на `margin` |

- `present` — предмет найден минимум на `min_frames_present` кадрах (по умолчанию 2), `conf = mean(score) × seen/checked`.
- Иначе `unknown`, а в `note` написано почему: «найдено на 1/3 кадрах», «не наблюдаемо в baseline», «контраст не пройден».
- `person_occlusion` — доля зоны стола, закрытая людьми. Если она ≥ 0.6, добавляется предупреждение «отсутствие не проверяемо».

### `safe_door_state(clip, t, zones)`
- baseline: `unknown` с предупреждением «не наблюдаемо». Исключение — дообученная модель с классом `safe_open`.
- rare_open_vocab: на патчах в зоне `safe` сравниваются «open safe / open safe door» и «closed metal safe / closed safe door» с запасом `margin`. По голосам кадров получается `open`, `closed` или `unknown`.
- `closed` выдаётся только при положительном свидетельстве, противоречивые кадры дают `unknown`.

## Абляция фазы 2 (ровно одна)
- **Сравниваем:** `inventory.mode=baseline` (по умолчанию) против `rare_open_vocab`. Включение: `SENTINEL_DETECT_CONFIG=ml/detect/ablations/rare_open_vocab.yaml` или `SENTINEL_DETECT_EXTRAS=rare_open_vocab`.
- **Почему выбран этот вариант, а не «второй проход по зонам с апскейлом»:** правило «ушла и оставила» опирается на наличные, инкассаторскую сумку и открытый сейф. В COCO таких классов нет, и апскейл тем же COCO-детектором их не добавит. Апскейл зон стола и сейфа остаётся следующей гипотезой, после замера этой.
- **Какие события должны улучшиться (гипотеза, не измерено):**
  - V02 — после ухода на столе остались наличные или сумка;
  - V03 — сейф открыт и оставлен;
  - V07 — инкассаторская сумка оставлена;
  - V05 — меньше ложных «мобильных» на стационарном телефоне;
  - V09 — куртка на столе.
- **Риск:** ложные «cash» (бумаги) и ложные «open» у сейфа. Его ограничивают `min_frames_present`, контрастные пары и `margin`.
- **Цена:** OWLv2 только по запросу правил, не больше `max_frames=2` кадров на вызов. Результаты кэшируются по `(клип, t)`, поэтому `desk_inventory` и `safe_door_state` в одной точке не считаются дважды.
- **Решение:** включать по умолчанию, только если у L2 падает FN по V02/V03/V07 без роста FP и с latency в бюджете. Нет выигрыша — режим остаётся выключенным.

## Замер L2 (одна команда)
```bash
HF_HUB_OFFLINE=1 HF_HUB_CACHE=weights/hf/hub \
python -m ml.detect.probe --video <ролик.mov> --fps 2 --zones <zones.json> \
       --modes baseline,rare_open_vocab --out runs/probe_a05/<ролик>
# или --frames <папка с кадрами>; zones.json — зоны A06: [{"name","polygon"}] | {"zones": [...]}
```
Выход — `runs/probe_a05/<ролик>/probe_report.json`:
- **detect:** `latency_ms_per_frame` (mean/p50/p95, без холодного первого батча — он отдельно в `cold_first_batch_ms_per_frame`), пик VRAM, по классам — n, доля кадров, квантили score;
- **inventory[mode]:** `latency_ms_per_call`, VRAM, present/unknown по видам, open/closed/unknown сейфа.

Картинки `vis/<режим>/*.jpg` с боксами и зонами лежат только на локальном диске (в папке есть `.gitignore` со `*`). По ним человек глазами проверяет ложные cash, cash_bag и open.

Метрики событий (FP/FN по видео, попадание таймкода, тип) появятся, когда A09 начнёт вызывать `desk_inventory`/`safe_door_state`. Тогда — тот же прогон eval (A16) дважды: по умолчанию и с `SENTINEL_DETECT_CONFIG=ml/detect/ablations/rare_open_vocab.yaml`. Все 42 ролика уже использованы в разведке, поэтому это сравнение режимов, а не независимый тест.

## Калибровка порогов (только по человеческой разметке)
```bash
python -m ml.detect.calibrate --gt datasets/det/annotations.json --annotator "<имя>" \
       --target-precision 0.9 [--with-open-vocab] --out calib/detect_thresholds.yaml --report calib/detect_calibration.json
SENTINEL_DETECT_CONFIG=calib/detect_thresholds.yaml python -m ml.detect.probe ...
```
- Кадры делятся по **видео** на calib и holdout.
- Для каждого класса берётся минимальный порог с precision ≥ цели на calib. Отчёт даёт P/R на calib и на holdout.
- Если класс недостижим или поддержки мало, остаётся приор из `config.yaml`.
- Пороги в `config.yaml` — **приоры, а не калибровка**. mAP — только через `eval_det.py` на человеческой разметке.

### Разметка Дамира (`datasets/det/annotations.json`, COCO из ветки `det-labels`)
- **Принимается:** ручная разметка с `--annotator <имя>`, либо проверенные предложения после `autolabel mark-verified`.
- **Отклоняется:** псевдоразметка (`info.kind=proposals`, `verified=false`, `score` без `verified`) и файл неизвестного происхождения без `--annotator`. Затрагивает `finetune`, `calibrate` и `quality_claim` в `eval_det`.
- **Формат `det-labels/export_coco.py` поддержан:**
  - кадры ищутся в `info.images_root` (`data/frames`, от корня репо), иначе в `<dir>/images`;
  - категория → метка контракта через `contract_label` (`safe_closed` → `safe`), иначе по имени;
  - train/val берётся из поля `split` каждого кадра (зафиксирован в `split.json`); без него деление идёт по `group`, поэтому дубликаты по md5 не разъезжаются.
- **Классы дообучения по умолчанию** — нестандартные метки разметки: `cash, cash_bag, safe, safe_open, money_counter, cell_phone, door_open`. person, cup и food покрывает COCO; их можно добавить через `--classes`.
- **Предложения для разметки:** `python -m ml.detect.autolabel propose --frames <кадры> --out datasets/det/proposals.json` (`verified=false`) → правка человеком → экспорт в `annotations.json`. У `det-labels` есть свой поток через Label Studio.

## Таблица классов: какой бэкенд реально даёт метку

Конфиг по умолчанию: `closed_set.hf_model_id = PekingU/rtdetr_v2_r18vd`, `backends: [hf]`,
`hf_fallback_ids: []`, open-vocab, TTA, тайлы и finetuned выключены, `inventory.mode = baseline`.

| Метка | По умолчанию (RT-DETRv2, COCO) | Опционально | Что проверить локально (L2) |
|---|---|---|---|
| person | COCO `person` | — | полнота на виде сверху, порог 0.25 — приор |
| cell_phone | COCO `cell phone` | OWLv2 (`mobile phone in hand`), finetuned | ложные срабатывания на стационарном телефоне |
| cup | COCO `cup`, `wine glass`, `bowl~proxy` | OWLv2 (`drink cup`) | — |
| bottle, laptop | одноимённые COCO | — | — |
| food | COCO-еда (banana … cake) | OWLv2 (`food`) | — |
| bag | COCO handbag/backpack/suitcase | — | любая сумка, **не инкассаторская** |
| document | COCO `book~proxy` | — | сходство, не «банковский документ» |
| **cash** | **нет** | OWLv2 (`stack of banknotes`, `cash money`), finetuned | нет ни одного проверенного бэкенда до замера/разметки |
| **cash_bag** | **нет** | OWLv2 (`green canvas bag`), finetuned | сходство с запросом ≠ инкассаторская сумка |
| **safe** | **нет** | OWLv2 (`metal safe`), finetuned (`safe_closed` → `safe`) | вид сверху не проверен |
| **safe_open** | **нет** | OWLv2 (`open safe door`), finetuned | открыт/закрыт на виде сверху не проверен |
| **door / door_open** | **нет** | OWLv2 (`glass door`, `open door`), finetuned (door_open) | — |
| **money_counter** | **нет** | OWLv2, finetuned | — |
| **keys** | **нет** | OWLv2 (`keys`) | — |

- Общий COCO-детектор (RT-DETRv2 или явно выбранный D-FINE) **никогда** не выдаёт нестандартные метки, даже если в `id2label` модели есть похожие имена (`STANDARD_LABELS` в `labels.py`). Поддержка наличных и сейфа появляется только с OWLv2 (`SENTINEL_DETECT_EXTRAS=open_vocab` или `inventory.mode=rare_open_vocab`) или с моделью, дообученной на проверенной разметке (`SENTINEL_DETECT_EXTRAS=finetuned`). Ни один из этих путей не измерен на реальных роликах.
- В `Detection.source` всегда записан исходный класс или запрос: `rtdetr_v2_r18vd:book~proxy`, `owlv2:green canvas bag`.
- Роли людей детектор не определяет.

### Отсутствие детекции ≠ отсутствие объекта
- `detect()` возвращает только найденное. Пустой результат на кадре **не** доказывает, что объекта нет: темнота (свет выключили), перекрытие, ракурс или слабый для вида сверху класс тоже дают пустой ответ.
- `last_stats()["coverage"]` говорит лишь, какие кадры были обработаны, а не что на них было видно.
- `desk_inventory` / `safe_door_state` никогда не отвечают «нет»: только `present`/`unknown`. Для сейфа `closed` — лишь при положительном свидетельстве.

## Запуск и окружение
```bash
pip install -r ml/detect/requirements.txt            # версии — по lock L1 (torch 2.11+cu128, transformers 5.20.1, opencv<5)
python -m ml.detect.weights                          # где найдены веса (weights/hf/hub), без сети
python -m ml.detect.weights --download               # ТОЛЬКО setup (сеть); у L1 веса уже скачаны
python -m ml.detect.licenses --check                 # сверка заявленных лицензий чекпойнтов (setup, сеть)
HF_HUB_OFFLINE=1 python -m ml.detect video.mov --out dets.json
python -m pytest ml/detect/tests
```

### Офлайн-загрузка весов
- Порядок поиска снапшота: `HF_HUB_CACHE` → `HF_HOME/hub` → `<SENTINEL_WEIGHTS>/hf/hub` (раскладка L1) → `<SENTINEL_WEIGHTS>/hf` → папка `save_pretrained` в `<SENTINEL_WEIGHTS>/detect/<имя>`.
- Загрузка — `from_pretrained(..., cache_dir=<найденный кэш>, local_files_only=True)`: в сеть не ходит даже без `HF_HUB_OFFLINE`.
- Нет весов → `DetectorUnavailable` с id модели и списком проверенных путей (`on_unavailable: raise`).
- Резервный детектор подставляется только если явно задан в `closed_set.hf_fallback_ids`; тогда `status=degraded` и предупреждение.
- Mock включается только `SENTINEL_MOCK=1`.

| env | смысл |
|---|---|
| `SENTINEL_WEIGHTS` | корень весов (по умолчанию `<repo>/weights`); HF-кэш — `weights/hf/hub` |
| `HF_HUB_CACHE` / `HF_HOME` | тоже учитываются при поиске весов |
| `SENTINEL_DETECT_EXTRAS` | `rare_open_vocab`, `tta`, `tiles`, `open_vocab`, `finetuned`, `all` |
| `SENTINEL_DETECT_CONFIG` | YAML поверх `config.yaml` (абляции, калибровка) |
| `SENTINEL_DETECT_BUDGET_S` | мягкий бюджет `detect()` |
| `SENTINEL_MOCK=1` | только тесты и демо; production сам на mock не переключается |

## Модели и лицензии (`models.json`)
| Модель | Заявленная лицензия | Статус | Роль |
|---|---|---|---|
| `PekingU/rtdetr_v2_r18vd` | apache-2.0 | `declared_unverified` | baseline |
| `ustc-community/dfine-medium-obj2coco` | apache-2.0 | `declared_unverified` | явная альтернатива (не резерв по умолчанию) |
| `google/owlv2-base-patch16-ensemble` | apache-2.0 | `declared_unverified` | абляция / опции |
| дообученная ONNX | наследуется | `derived` | опционально |

Лицензия **веса** проверяется на setup командой `python -m ml.detect.licenses --check`; название семейства не заменяет проверку чекпойнта.
Permissive-модели упрощают соблюдение условий. AGPL/GPL сами по себе коммерцию не запрещают, но их исключает правило команды.

## Ограничения
- В baseline cash, cash_bag, personal_item и сейф всегда unknown: в COCO нет таких классов. Это честный ответ, а не ошибка.
- OWLv2 на виде сверху не проверен. Пороги и `margin` — приоры до калибровки.
- Стоимость `desk_inventory` зависит от `clip.frame_at` (точное декодирование у A04) и не измерена.
- Пиксельных координат, цвета одежды и яркости как признаков нет. Свет при уходе может выключаться — тогда предметы просто не находятся и остаются unknown.

## Для интегратора
- **A03:**
  - в `meta.coverage` и предупреждения класть `last_stats()["coverage"]` и `status`;
  - `partial`/`unavailable` не превращать в `no_violation`;
  - `DetectorUnavailable` перехватывать;
  - `warmup()` вызывать при старте.
- **A09 (rules):** после события «ушла» вызывать `desk_inventory(clip, t_leave + Δ, zones)` и `safe_door_state(...)`, по одному или нескольким `t` в окне после ухода.
  - Нарушение-кандидат: `cash`/`cash_bag` present или safe `open`.
  - `unknown` ≠ «чисто»: решение принимается вместе с другими источниками (A08 и т.д.).
  - Пока A09 этого не делает, абляцию можно мерить только на уровне модуля (`probe`).
- **A08 (scene):** `cell_phone` из `detect()` может быть стационарным телефоном. Для V05 нужны поза (рука) и неподвижность трека (A06) или `desk_inventory` в rare-режиме.
- **A06:** зоны `desk`, `cash_tray`, `safe` — именно эти имена использует inventory.
- **A01:** в корневом `.gitignore` — `__pycache__/`, `weights/`, `data/`, `datasets/`, `runs/`, `calib/`; pytest-маркеры `slow`/`gpu`.
- **Предложение к контракту v2:** поле `Detection.attrs` (`orig_class`, `proxy`, `query`) и стандартный тип для inventory-ответов.

## Решения, принятые автономно
- **Фаза 1:**
  - baseline — один closed-set детектор, остальное флагами;
  - без весов — исключение, а не пустой список;
  - mock только явно;
  - тайлы — 2 квадратных окна на портрет;
  - в WBF бокс из одного вида получает штраф ×0.85 и не исчезает.
- **Фаза 2:**
  - Одна абляция — `rare_open_vocab` (обоснование выше). Апскейл зон — следующая гипотеза.
  - `bag` — отдельный вид «сумка неизвестного типа»: COCO-сумка не становится ни `cash_bag`, ни заведомо `personal_item`.
  - `present` требует минимум 2 кадра окна: против единичных ложных срабатываний и дребезга.
  - Яркость не используется даже как «наблюдаемость». Риск утечки признака «свет выключили при уходе» хуже, чем лишний unknown.
  - Ручная разметка без `mark-verified` принимается только с `--annotator`. Ответственность фиксируется в `train_meta`/отчёте калибровки.
- **После аудита:**
  - Тихий переход на D-FINE убран: непроверенный детектор не должен подменять основной. D-FINE — только явный выбор.
  - Общий детектор ограничен стандартными метками (`STANDARD_LABELS`).
  - Импорты тестов совпадают с интеграцией (`contracts.sentinel_contracts`), поэтому слияние идёт без конфликтов. Отдельная ветка A05 без `contracts/` получает тестовую прокладку в `tests/conftest.py`.
  - Загрузчики разметки следуют формату `det-labels`: `images_root`, `contract_label`, `group` и `split`.
