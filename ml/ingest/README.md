# ml/ingest — декодирование видео, цепочка времени, качество, coarse-to-fine, нарезка съёмок (A04)

Модуль превращает `.mov`/`.mp4` (портрет или ландшафт, с поворотом или без, CFR или VFR) в `Clip` —
реализацию контрактного `VideoClip`: сэмплы (по умолчанию 5 fps) с **фактическими PTS**, точный
доступ к кадрам исходника для доказательств, метрики качества и `motion_energy`.
Моделей нет (`models.json = []`), GPU не нужен.

## Команда замера для L2 (одна)

```bash
python -m ml.ingest.bench --videos data/videos --labels data/labels.csv --out reports/a04_ingest_bench.json \
    --variant base --variant skip:decode.skip_nonref=true --variant cv:decode.backend=opencv --coarse-fine
```

* **Что сравниваем:** `base` (настройки по умолчанию) против `skip` (декод только опорных кадров —
  быстрее, но таймлайн неполный) и `cv` (OpenCV-фолбэк вместо PyAV). Любой другой вариант —
  `имя:ключ=значение,...` (например, `small:storage.max_side=640`).
* **Выход:** JSON в `--out`: на вариант — `summary` и записи по роликам. В stdout — таблица по вариантам.
  Только числа и id роликов, без кадров и описаний.
* **Метрики:**
  * coverage: `incomplete`, `with_gaps`, `coverage_min`, ошибки декодера;
  * PTS: шаг кадров min/median/max/std, VFR, `t0_pts_sec`, отклонение сэмплов от сетки;
  * точность цепочки времени: `exact_offset_max` (|PTS выданного кадра − запрошенное t| на 20 случайных t),
    `idempotent_all`, `sample_vs_exact_absdiff_max` (сэмпл `frame(i)` против точного кадра в `times[i]`;
    около 2 — это потери JPEG, больше 10 — подозрение на чужой кадр);
  * по разметке: размеченная секунда попадает в покрытие, PTS ближайшего кадра и смещение;
  * latency: `load_s`, `frame()` промах и попадание, `frames_at(3)`, `coarse_fine.run`;
  * память: `peak_rss_mb` (psutil или resource; на Windows без psutil — null);
  * VRAM: модуль CPU-only, null.
* FP/FN и тип ingest не меряет — он влияет на них косвенно (кадр в нужный момент, без пробелов).
  Код выхода 1, если хоть один ролик не загрузился.

## Публичный API

```python
from ml.ingest import load_video, IngestError, FrameGapError

clip = load_video("X-1.mov")                 # контракт: load_video(path, sample_fps=5.0)
clip.path, clip.duration, clip.fps, clip.width, clip.height   # width/height — ПОСЛЕ поворота
clip.times          # фактические PTS сэмплов, сек от первого кадра (первый = 0.0), без округления
clip.frame(i)       # BGR uint8 сэмпла i (из LRU — копия)
clip.frame_at(t)    # контракт: точный кадр исходника, ближайший к t, полный размер (только картинка)

# расширения (совместимы с контрактом)
img, ts = clip.frame_at_ts(t)              # + ФАКТИЧЕСКИЙ PTS кадра -> EvidenceFrame.timestamp_sec
ref = clip.read_exact(t, strict=False)     # FrameRef(image, timestamp_sec, requested_sec, in_gap, out_of_range, ok)
refs = clip.frames_at([t1, t2])            # пачкой; запросы в один кадр декодируются один раз
refs = clip.distinct_frames_at(ts, min_separation_s=0.5)  # без повторов одного кадра (для A12)
clip.window_indices(t0, t1)                # сэмплы окна кандидата
clip.snap_time(t), clip.index_at(t)        # PTS ближайшего кадра исходника / ближайший сэмпл
clip.source_times                          # PTS всех кадров потока
clip.coverage                              # {complete, coverage, decoded_span, declared_duration, gaps, source_times_complete}
clip.quality, clip.motion_curve(), clip.activity_segments(), clip.gated_indices()
clip.meta, clip.warnings, clip.close()     # или `with load_video(...) as clip:`

load_video(path, sample_fps, *, config=None, backend=None, progress=None)  # keyword-only расширения
```

Модули: `probe` (ffprobe → PyAV → OpenCV, валидация, CLI), `coarse_fine` (адаптер для A03),
`augment` (синтетические искажения, CLI), `contact_sheet` (CLI), `split_blinks` (нарезка наших
съёмок, CLI), `bench` (замер, CLI), `names` (`normalize_video_id`, `find_duplicates`),
`synthetic` (тестовые видео со штрих-кодом номера кадра; моки — только явно).

## Цепочка времени

`PTS` → `t = pts·time_base − t0` (float, `time_base` — Fraction; t0 — PTS первого кадра, поэтому
смещение старта и edit lists убираются) → сэмпл `times[i]` (ближайший по PTS кадр к узлу
`k/sample_fps`, без повторов) → окно кандидата `window_indices` → `frames_at(t)`. Внутри:
`snap_time(t)` даёт PTS кадра из таймлайна загрузки, декодируется ровно он, и в
`FrameRef.timestamp_sec` возвращается фактический PTS. Округление — только при показе (контакт-листы,
предупреждения, тайминги). `frame_index/fps` временем не считается нигде.

* **Длительность.** `clip.duration` — НОМИНАЛЬНАЯ длительность видеопотока (контейнер: ffprobe → PyAV → OpenCV),
  а не декодированный отрезок. Если декодирование не дошло до конца (обрезанный/битый файл),
  недекодированный хвост остаётся внутри `[0, duration]` и виден потребителю: у A03 `times` кончаются раньше
  `duration`, а в `clip.coverage["unknown_spans"]` лежит хвост. Если контейнер занижает длительность,
  берётся декодированный отрезок (максимум из двух). Для дедлайна и покрытия длительность использовать
  можно и нужно; как ML-признак — нельзя.
* **Начало и конец.** `t < 0` или `t > duration` → крайний кадр (первый или ПОСЛЕДНИЙ декодированный)
  с `out_of_range=True`. Момент внутри недекодированного хвоста → последний декодированный кадр с
  `in_gap=True`. Первый кадр при выходе за конец не возвращается.
* **Пробелы покрытия.** Внутренние: соседние PTS дальше, чем `max(min_gap_s, gap_factor·медианный шаг)`
  (не путать с дрожанием VFR). Голова и хвост: недекодированное начало или конец длиннее 1,5 медианного
  шага кадра. Всё это попадает в `clip.coverage["gaps"]` и `["unknown_spans"]` с предупреждением.
  Кадр дальше `frame_at.max_offset_s` (0,25 с) от запрошенного t помечается `in_gap=True`;
  при `strict=True` — `FrameGapError(code="frame_gap")`. Сэмплы внутри дыры не выдумываются.
* **unknown ≠ false.** Пробел — это «неизвестно», а не «нарушений нет». Пайплайн должен смотреть
  `clip.coverage` (и `WindowRead.complete` в coarse-to-fine), прежде чем выдавать `no_violation`.
* **Дубликаты кадров.** Несколько запросов, попавших в один кадр, дают один и тот же
  `timestamp_sec`; `distinct_frames_at` отбрасывает повторы, чтобы A12 не взял три копии одного кадра.

## Поля и единицы (передача L2 / A03 / A12 / A11)

Все времена — секунды float от первого кадра потока (PTS первого кадра = 0.0), без округления.
Поле `Clip` → `meta` → `coverage`:

| поле | тип, единицы | смысл |
|---|---|---|
| `clip.duration` | float, с | номинальная длительность видеопотока (≥ декодированного; см. выше) |
| `clip.times` | list[float], с | фактические PTS сэмплов (по возрастанию; первый = 0.0) |
| `clip.source_times` | list[float], с | PTS всех декодированных кадров |
| `clip.fps` | float, кадр/с | средний fps потока из контейнера (информативно, НЕ для вычисления времени) |
| `clip.frame_at_ts(t)` | (ndarray, float с) | точный кадр + фактический PTS |
| `clip.read_exact(t, strict)` | FrameRef | `timestamp_sec`, `requested_sec`, `offset_sec`, `in_gap`, `out_of_range`, `ok`; strict → `FrameGapError` |
| `clip.coverage["duration"]` | float, с | = `clip.duration` |
| `clip.coverage["declared_duration"]` | float, с | длительность видеопотока из контейнера (0.0 — неизвестна) |
| `clip.coverage["decoded_span"]` | float, с | последний декодированный PTS + медианный шаг кадра |
| `clip.coverage["coverage"]` | float 0..1 | `decoded_span / declared_duration` (1.0, если длительность неизвестна) |
| `clip.coverage["complete"]` | bool | `coverage >= decode.warn_coverage` (0.98). При True малые `unknown_spans` (< 2%) допустимы |
| `clip.coverage["unknown_spans"]` | list[[t0, t1]], с | не декодировано: голова, внутренние дыры, хвост (формат A01 `Coverage.unknown_spans`) |
| `clip.coverage["gaps"]` | list[(t0, t1)], с | то же, кортежами |
| `clip.coverage["source_times_complete"]` | bool | False при `skip_nonref`: внутренние дыры не считаются, голова и хвост — считаются |
| `clip.meta["t0_pts_sec"]` | float, с | абсолютный PTS первого кадра (смещение старта, уже вычтено) |
| `clip.meta["median_frame_dt"]` | float, с | медианный шаг между PTS кадров |
| `clip.meta["warnings"]` | list[str] | по-русски, в т.ч. «видео неполное: …», «пробелы покрытия: …» |

Ошибки: `ml.ingest.IngestError` (`.code`, `.message`, `.details`): `decode_failed`, `incomplete_video`
(декодировано < `decode.min_coverage`), `probe_failed`, `no_video_stream`, `duration_out_of_range`,
`file_not_found`, `frame_gap` (только strict).

## Решения (фаза 1, сохранены)

* **Поворот.** Display matrix или тег `rotate`, логика autorotate ffmpeg (включая зеркало).
  Размер берётся из декодированного кадра, 464×848 нигде не зашит.
* **Бэкенды.** PyAV основной, OpenCV фолбэк (его авто-поворот выключается, поворот применяем сами).
* **Память.** JPEG q=92 в RAM до `max_ram_mb` (512), сверх — временный файл. LRU: 96 кадров / 256 МБ;
  точные кадры: 32 / 128 МБ. Сэмплы ужимаются до `max_side=1280`; `frame_at` — полный размер.
* **Ошибки — не «пустой успех».**
  * Нет кадров, или ошибок декодера больше `max_decode_errors`, или декодировано меньше 50% длительности —
    `DecodeError` (`decode_failed` / `incomplete_video`).
  * 50–98% — клип с `complete=False` и предупреждением.
  * Битый контейнер — `VideoProbeError`; нет видеопотока — `NoVideoStreamError`;
    длительность вне 1–900 с — `InvalidDurationError`.
* **Моки только явно.** `load_video("synthetic://N")` работает лишь при `SENTINEL_MOCK=1` или
  `mock.allow_synthetic_uri: true`. Без них — `VideoNotFoundError`; production сам на мок не
  переключается. У `SyntheticClip` стоит `meta.mock=True`.
* **sample_fps.** Значение по умолчанию из контракта (5.0) не менялось; `sampling.default_fps` —
  для пайплайна, `sampling.max_samples` — предохранитель памяти.

## Фаза 2: что добавлено (каждое — отдельный коммит; новые режимы выключены)

1. **Цепочка времени** (см. выше): без округления; кэш точных кадров по фактическому PTS;
   `in_gap`/`out_of_range`, strict-режим; `coverage`; `distinct_frames_at`; `window_indices`.
2. **Любая ориентация.** Все функции и тесты гоняются на портрете и ландшафте. Метрики качества теперь
   уменьшают кадр по КОРОТКОЙ стороне (`quality.analysis_short_side=160`, раньше — по ширине).
   Для портрета результат тот же, ландшафт теперь в той же шкале резкости.
3. **Coarse-to-fine** (`coarse_fine.py`, флаг `coarse_fine.enabled: false`):
   * `coarse_indices` — равномерный обзор всей длительности по сохранённым сэмплам;
     первый и последний сэмпл всегда входят;
   * окна — из кандидатов (Event / dict / кортеж), склейка дребезга `merge_gap_s`; если кандидатов нет —
     окна из пиков `motion_energy` (`activity_top_k`; число окон не растёт с длиной ролика);
   * `read_window` / `run` — плотное точное чтение окон без дублей;
   * бюджет `budget_frames` делится на обзор (`coarse_share`) и окна; шаг и плотность — параметры для L2.
4. **`angle_jury`** — перспектива + поворот + кроп в ЛАНДШАФТ (`out_orientation="landscape"`).
   У всех пресетов `kind="perturbation"`, CLI пишет пометку «синтетическое искажение, не реальный
   новый ракурс». Геометрия учитывает выходной размер (`output_size`, точки и bbox нормируются к нему).
   ffmpeg-движок расходится с numpy меньше чем на 3 уровня. `random_aug(p_landscape=0)` по умолчанию
   ведёт себя как раньше.
5. **`split_blinks`** — нарезка наших съёмок (ниже).
6. **`bench`** — замер для L2 (выше).

## Нарезка наших съёмок: `split_blinks`

```bash
python -m ml.ingest.split_blinks --videos data/extra/incoming --journal data/extra/journal.csv \
    --out data/extra/clips --dry-run      # сначала отчёт сопоставления, затем без --dry-run
```

* **Метка** — короткий провал средней яркости кадра. Тёмный кадр: ниже `0.5 ×` опорного уровня
  (75-й перцентиль за ±5 с) и минимум на 20 уровней ниже него. Длительность провала 0,3–3 с;
  свет горит и до, и после (проверка симметричная). Мерцание внутри метки (< 0,5 с) склеивается.
* **Не метки.** Длинное затемнение по сценарию и края длинной темноты идут в `long_dark` отчёта.
  Если автодетект ошибся — `--marks marks.csv` (`camera,file,on_t[,off_t]`).
* **Клип** — от включения света − 0,5 с до выключения света следующей меткой (или до конца файла);
  отрезок до первой метки — подготовка. Выход: H.264, 30 fps CFR, короткая сторона 464 (портрет и ландшафт),
  без звука. Движок `--engine auto|ffmpeg|pyav` (PyAV — если нет ffmpeg CLI, кодировщик из колеса).
* **Сопоставление** по камерам (подпапки `C1..C4`, файлы в естественном порядке имён): сегменты ↔ сцены
  журнала с `done=да` в порядке `shot_at` (если формат смешанный — порядок строк, с предупреждением).
  Число не совпало → камера пропускается (`--force` — по порядку, первые min(N)). Длительности
  сегментов сравниваются между камерами: метка видна всем сразу, расхождение > 1,5 с — предупреждение.
* **labels.csv** (по умолчанию `<out>/../labels.csv`, слияние по `video_id`):
  `video_id,file,label,t_sec,type_id,camera,layout,light,scene_id`, где `video_id = <scene_id>-<камера>`.
  * `t_sec = t_onset + 0.5`: считаю, что `t_onset` — секунды от включения света.
  * Для `no_violation` поля `t_sec` и `type_id` пустые, как в `eval.labels` у L1; код сцены
    восстанавливается по `scene_id`.
  * `kind`: нарушение / норма / violation / clean / …; нераспознанное → `unknown` с предупреждением.
* **Приватность.** `title` и `notes` журнала не попадают ни в отчёт, ни в stdout — только id, коды, времена.
  Отчёт: `<out>/split_report.json`. Код выхода 1, если есть камеры с несовпадением.

## Синтетические искажения и контакт-листы

* `augment.py` — numpy (`apply`, `homography`, `output_size`, `transform_points`/`transform_bbox`,
  `ClipAugmenter`, `random_aug`) и CLI:
  `python -m ml.ingest.augment in.mov out.mov --preset angle_left|angle_right|high|low|zoom|mirror|dark|rot_cw|rot_ccw|bright|blur|noise|jpeg|fps25|angle_jury|all`.
  Это **perturbation**: нет параллакса и новых окклюзий. Устойчивость на пресетах — необходимое, но не
  достаточное условие для критерия «другой угол».
* `contact_sheet.py` — `python -m ml.ingest.contact_sheet video.mov --t 22 --window 4 --out sheet.jpg`;
  подписи — фактические PTS. **Только локальный файл** для человека или локальной VLM; CLI печатает лишь
  путь и таймкоды. Кадры и листы реальных роликов облачным моделям не передавать (включая расширения IDE).

## Проверки и замеры (фактически выполнено)

`python -m pytest ml/ingest/tests -q` → **164 passed** (~60 с). Linux-контейнер, 4 vCPU, Python 3.11.17,
версии из lock L1: av 18.1.0, opencv-python-headless 4.14.0.94, numpy 2.4.6, PyYAML 6.0.3, pytest 9.1.1;
ffmpeg 6.1.1. Фаза 1 проходила и на opencv 5.0.0.93.

Тесты генерируют синтетику сами: штрих-код номера кадра проверяет PTS, `frame_at` и seek; синтетическое
«мигание» проверяет `split_blinks`. Покрыто:
* VFR, в том числе дыра ~1 с;
* смещение старта 2,5 с;
* начало и конец видео;
* поворот −90/90/180/hflip против ffmpeg autorotate, на портрете и ландшафте;
* битые MOV;
* strict-режим и пробелы;
* дедупликация кадров;
* coarse-to-fine;
* `angle_jury` (numpy и ffmpeg);
* `split_blinks`: длинная темнота, мерцание, два файла, ландшафтная камера, несовпадение числа сцен,
  ручные метки, ffmpeg и PyAV, отсутствие title/notes в отчёте;
* `bench` от начала до конца.

**Замеры — только синтетика в этом контейнере** (testsrc2, 145 с, 30 fps, 464×848, 5 fps → 725 сэмплов;
вывод `bench`). На целевом ноутбуке и реальных роликах **не мерялось** — это задача L2 по команде выше.

| вариант | файл | load_s | frame() промах / попадание | frames_at(3) | coarse_fine.run |
|---|---|---|---|---|---|
| base | H.264 | 5.6 | 3.9 / 0.16 мс | 65 мс | 0.83 с |
| base | HEVC + display matrix −90 | 6.9 | 2.0 / 0.09 мс | 275 мс | 0.67 с |
| skip_nonref | H.264 | 4.4 | 1.9 / 0.09 мс | 50 мс | 0.72 с |
| skip_nonref | HEVC + display matrix −90 | 5.4 | 2.0 / 0.09 мс | 266 мс | 0.69 с |

Во всех строках: `exact_offset_max` = 0,016 с (≤ полкадра), `sample_vs_exact_absdiff_max` ≈ 2,1,
`idempotent_all` = true, coverage = 1,0, peak RSS процесса ≈ 300 МБ. Цель «< 5 с на CPU» здесь
выполняется только со `skip_nonref`. Обещать время на проде не могу — нужен замер L2.

## Зависимости и лицензии

| пакет | лицензия | зачем |
|---|---|---|
| PyAV (`av`) | BSD-3-Clause; в колёсах FFmpeg (LGPL) | основной декодер; кодировщик H.264 в `split_blinks --engine pyav` |
| opencv-python-headless (< 5, lock: 4.14.0.94) | Apache-2.0 | фолбэк, JPEG, метрики, warp |
| numpy | BSD-3-Clause | |
| PyYAML | MIT | config.yaml, `--variant` |
| ffmpeg/ffprobe (бинарь, опционально) | LGPL/GPL по сборке | probe, augment CLI, `split_blinks`; отдельный процесс |
| psutil (опционально) | BSD-3-Clause | peak RSS в `bench` на Windows |

Моделей и весов нет; сеть не нужна.

## Ограничения

* Повороты не кратные 90° округляются до 90°; анаморфный SAR ≠ 1 не учитывается; HDR/10-бит — без тон-маппинга.
* При `skip_nonref` таймлайн неполный: `coverage_gaps` не считаются, `snap_time` приблизителен.
  `frame_at_ts` всё равно возвращает фактический PTS.
* OpenCV-фолбэк выбирает сэмпл правилом «первый с t ≥ узел − полкадра» (точно для CFR); PyAV — честный ближайший.
* `split_blinks`:
  * пороги (ratio, min_drop, длительность провала) подобраны на синтетике; реальные съёмки — сначала `--dry-run`;
  * сцена целиком в темноте может «съесть» метку — тогда `--marks`;
  * порядок `shot_at` — единственный ключ сопоставления (абсолютное время записи файла не используется).

## Для интегратора

* **A01.** Предлагаю добавить в `VideoClip` необязательные `frame_at_ts(t) -> (ndarray, float)` и
  `source_times` — это обратно совместимо. `_contract_fallback.py` удалить после мерджа контракта.
  В корневой `.gitignore` нужны `__pycache__/` и `*.pyc`; маркеры pytest `slow`, `gpu`.
* **A12.** `EvidenceFrame.timestamp_sec = ref.timestamp_sec` из `clip.read_exact(t)` или `frame_at_ts`,
  не запрошенное t. Кадры брать через `clip.distinct_frames_at(times, min_separation_s=...)`.
  `ref.in_gap` / `ref.out_of_range` — не показывать как доказательство (или `strict=True`).
* **A03 — граница, проверено на `claude/integration-cloud` @ 2055fc4.**
  * `backend/pipeline/worker.py:165` (`_clip_info`) передаёт оркестратору только
    `duration/fps/width/height/times/source`. `clip.meta["warnings"]` («видео неполное: …») и
    `clip.coverage` теряются. С исправленной `duration` хвост теперь виден и так. Синтетика: файл, обрезанный
    до 96%, раньше давал в `canonical_coverage` `ratio=1.0, complete=True`, теперь `ratio=0.96, complete=False`.
    Чтобы были видны и хвосты меньше шага сэмплов, и тексты предупреждений, нужно пробросить
    `getattr(clip, "coverage", None)` и `getattr(clip, "warnings", [])` в `_clip_info`. Тогда
    `coverage["unknown_spans"]` идут в `compute_coverage(unknown_spans=…)`. Ожидание: для обрезанного файла
    `meta.coverage.unknown_spans` содержит `[decoded_span, duration]`.
  * `backend/pipeline/coverage.py` (`covered_intervals`) считает покрытым хвост короче 1,5 шага сэмплов
    (0,3 с при 5 fps). Такой хвост ingest всё равно перечисляет в `unknown_spans`.
  * Загрузка: `load_video(path, sample_fps, progress=...)`; ловить `IngestError` (`.code`, `.message`).
  * Перед `no_violation` проверять `clip.coverage`: `complete` и `gaps`. При пробелах — частичный результат
    с предупреждением, а не «нарушений нет».
  * Coarse-to-fine: `from ml.ingest import coarse_fine as CF; if CF.enabled(cfg): plan, reads = CF.run(clip, candidates)`.
    Обзорные кадры — `clip.frame(i)` по `plan.coarse_indices`; окна — `reads[k].frames` (точные), `reads[k].complete`.
* **A11.** `ml/vlm/frames.py:55` (`get_frame`) берёт `clip.frame_at(t)` — только картинку. В
  недекодированном хвосте это последний декодированный кадр без пометки. Нужен фактический момент — брать
  `clip.read_exact(t)` и пропускать `not ref.ok`.
* **A02.** Коды ошибок в HTTP 422: `file_not_found`, `probe_failed`, `no_video_stream`,
  `duration_out_of_range`, `decode_failed`, `incomplete_video`; `frame_gap` — внутренний.
* **A08.** `clip.quality[i]["motion_energy"]`; метрики не зависят от ориентации. Яркость — не признак
  нарушения (свет выключают).
* **A10 / A05.** `ClipAugmenter`, `random_aug(p_landscape=...)`, `transform_bbox` / `output_size` для разметки.
* **A16 / L1.** `split_blinks` пишет `data/extra/labels.csv` в схеме, совместимой с `eval.labels`
  (плюс `type_id`, `camera`, `layout`, `light`, `scene_id`). Ролики `<scene>-C1..C4` одной сцены — одна
  группа для LOO, иначе утечка между камерами. `names.find_duplicates` — md5-дубликаты.
* **A17.** ffmpeg/ffprobe не обязательны: `$SENTINEL_FFPROBE`, `$SENTINEL_FFMPEG`, `$SENTINEL_INGEST_CONFIG`;
  `SENTINEL_MOCK=1` — только для тестов и демо.
