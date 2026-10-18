# ml/track — трекинг, роли без лиц, авто-зоны (A06)

`track(clip, dets) -> (list[Track], list[Zone])` — сигнатура по замороженному контракту v1.
Моделей и весов нет: своя реализация ByteTrack на numpy и геометрические эвристики.
`models.json` поэтому пустой (`[]`).

## Граница с A03 / A08 / A09 (зафиксировано после аудита `2055fc4`)

**Единицы и координаты**
- `t` — секунды от начала ролика по PTS (`Detection.t`, `clip.times`). В окне refine (`WindowClip` A03) время тоже **абсолютное**, а не от начала окна.
- `bbox` и полигоны зон нормированы 0..1 к кадру после поворота.
- `Track.boxes` = `[(t, bbox), ...]` по возрастанию `t`. Это только **реальные наблюдения** в моменты детекций: интерполяции нет (`fill_gaps_s: 0`), за последним боксом трек не продлевается.
- Нормировки (доли присутствия, «статичный предмет», `conf` зон) считаются по сэмплированному отрезку `clip.times`, а не по `clip.duration`. В окне refine `duration` — это весь ролик (исправлено, см. ниже).

**observed / unknown**
- Есть бокс в момент `t` → человек **наблюдается**. Нет бокса → **не наблюдается**, а не «отсутствует».
- Каждый пропуск классифицируют `gaps()` и `presence_state()`:
  - `lost_in_view` — окклюзия или пропуск модели, **не уход**;
  - `left_view_edge` — вышел из поля зрения, но не обязательно из помещения;
  - `left_via_door` — последний раз виден в зоне двери.
- Конец трека или новый ID после потери в глубине кадра **не означают** ни ухода, ни входа, ни нарушения. Новый фрагмент получает роль по собственному поведению.
- `TrackQueries`:
  - `Answer.value = None` — «неизвестно»; `source` объясняет причину;
  - роль с `conf < 0,5` и зона с `conf < 0,35` считаются неопределёнными;
  - во время `coverage.visibility_loss` все ответы — `None`.
- Роли — `cashier`, `visitor`, `collector`, `unknown` плюс `role_conf`. `unknown` ≠ «посторонний»; non-cashier ≠ «неуполномоченный».

**Поля, которые отдаёт ml.track**

| Объект | Поля | Замечания |
|---|---|---|
| `Track` | `track_id`, `label`, `role`, `role_conf`, `boxes` | `track_id` уникален в пределах вызова: сначала люди (1..N по времени появления), потом предметы. Группы меток: `safe_open` → `safe`, `door_open` → `door`, `bag` → `cash_bag`, если ≥30% голосов за `cash_bag`. У предметов `role="unknown"`, `role_conf=0`. |
| `Zone` | `name`, `polygon`, `conf` | `name` ∈ `cashier_area`, `desk`, `safe`, `door`, `client_area`, `cash_tray`. Нет зоны — не оценена: «весь кадр» не создаётся. Fallback-зоны имеют `conf ≤ 0,3`. |
| `Detection.track_id` | проставляется **на месте** внутри `track()` | Через границу процесса ProcessExecutor (spawn) A03 **не переходит**, см. «Для L2». |
| `track_full().info` | `roles[tid]` = {`role`, `conf`, `reasons`, `features`, …}; `zones[name]` = {`method`, `independent`, …}; `coverage` = {`frames_total`, `frames_processed`, `frames_with_detections`, `t_first_detection`, `t_last_detection`, `t_window`, `declared_duration`, `span_frac`, `visibility_loss`, `visibility_check`}; `stitch`; `cashier_selection`; `warnings`; `timings` | Provenance вне контракта: в `Track` и `Zone` нет поля `extra` (предложение A01). |
| `person_signals(extended=True)` | + `x_cashier_lost_in_view` | Префикс `x_` принят `contracts/COMPAT.md`. |
| `model_info()` | `config_sha`, `config_path`, `config_from_env` | Для ключа кэша A03 (`backend/pipeline/stages.py: module_identity`). |

Пример выходного объекта (**только синтетика**, `scene_v02`): `ml/track/examples/synthetic_output.json`.

**Для L2: несовместимости на границе** (чужой код не менял)
1. **A08 не учитывает `Zone.conf`.**
   - Где: `ml/scene/core.py:153-161`. Рабочее место — первая `cashier_area` без проверки `conf`.
   - Вход: `track()` на сцене без предметов стола → `cashier_area` с `conf ≤ 0,3`.
   - Ожидается: при `conf < 0,35` A08 считает рабочее место неизвестным (сигналы `unknown`) или взвешивает по `conf`.
2. **`Detection.track_id` не доходит до A08 при ProcessExecutor.**
   - Где: `backend/pipeline/worker.py`, spawn. `track()` меняет копию детекций в воркере, а scene получает исходный список A03.
   - Следствие: `ml/scene/timeline.py:133-137` — `det_score` пуст, score по умолчанию 0,8.
   - Ожидается: A08 сопоставляет детекции с треком по паре (`t`, `bbox`) — `Track.boxes` содержит те же боксы детекций (5 знаков) в те же `t`. Либо A03 передаёт `track_id` сам.
3. **A09 в запасном пути считает разрыв трека отсутствием.**
   - Где: `ml/rules/signals.py: derive_from_tracks` (строки ~300–345). Путь работает, только если сигнала нет в состояниях A08 (`ml/rules/engine.py:812-824`).
   - Что происходит: между боксами трека кассира и между его фрагментами `cashier_present = 0`.
   - Вход: трек кассира с пропуском в глубине кадра (`gaps()` → `lost_in_view`).
   - Ожидается: `NaN` или unknown на этом отрезке, а не 0. Классификация пропусков — `ml.track.events.gaps` / `presence_state`.
4. **A09 ждёт треки с метками `safe_open` и `door_open`, а `ml.track` объединяет состояния одного предмета в трек `safe` / `door`.**
   - Где: `ml/rules/signals.py: derive_from_tracks`, `label_to_signal`.
   - Следствие: в запасном пути `safe_open` и `door_open` из треков не появляются никогда. Состояние открыт/закрыт есть только в `Detection.label`, и A08 берёт его оттуда.
   - Ожидается: запасной путь A09 не выводит открытость из треков, а оставляет сигнал отсутствующим (unknown).
5. **`coverage` трекера не попадает в `meta.coverage`.**
   - Где: `backend/pipeline/orchestrator.py:314` вызывает `track()`, а не `track_full()`.
   - Ожидается: при необходимости A03 вызывает `ml.track.track_full` и передаёт `info["coverage"]["visibility_loss"]` как `unknown_spans` в `contracts/extensions.py: compute_coverage`.

**Исправлено после аудита** (каждое — с воспроизведением на синтетике до правки)
| Ошибка | Было | Стало | Тест |
|---|---|---|---|
| Окно refine: нормировка на `clip.duration` (весь ролик) | `conf` зон занижен (`safe` 0,50 вместо 1,0; `door` 0,43 вместо 0,8), стационарный телефон выпадал из опор стола, `span_frac` 0,17 | нормировка по `clip.times`; `coverage.t_window` и `declared_duration` | `test_refine_window_normalizes_by_sampled_span_not_full_duration` |
| Подмена ID при проходе посетителя перед сидящим кассиром | потерянный трек кассира забирал детекцию посетителя | `tracker.lost_cost_penalty: 0.1` — штраф потерянным трекам только при выборе пар (0 — как было) | `test_crossing_in_front_of_cashier_keeps_cashier_and_no_leave[0-4]` |
| Конфиг не перечитывался в долгоживущем воркере; внешний `$SENTINEL_TRACK_CONFIG` не менял ключ кэша A03 | правка YAML не действовала, кэш стадии мог вернуть результат со старым конфигом | YAML кэшируется по (путь, `mtime`, размер); `ml.track.model_info()` отдаёт `config_sha` | `test_config_reload_and_model_info_follow_external_yaml` |
| `replay --real`, `report` и `viz` не читали кэш стадий A03: `cache/<sha256>/dets.jsonl`, `tracks.json` + `zones.json`, `manifest.json` | команда замера L2 находила 0 роликов **молча**; метки не сопоставлялись, потому что папки названы по sha256 | читаются `dets.jsonl`, `tracks.json` + `zones.json`, `manifest.json["video"]`; `--videos-dir` сопоставляет sha256 с именем (считается локально); 0 файлов → явная ошибка | `test_replay_real_reads_a03_stage_cache_and_maps_sha_to_names`, `test_replay_real_without_files_fails_loudly`, `test_report_and_loaders_read_a03_tracks_zones` |
| Метки в схеме `eval.labels` (`t_sec`, `t_all`, `conflict`) не читались | таймкод не попадал в метрику попадания | читаются `t_sec`, `t_all` (все таймкоды дубликатов), `video_id`, `conflict` | `test_replay_real_reads_eval_labels_schema` |

Синтетический replay, 9 сценариев × 8 возмущений × 1 сид. Сравнение `baseline` с `no_lost_penalty` (прежнее поведение): подмены ID 1 против 6, фрагментация 31 против 39, ложные тревоги 8 против 10. Пропуски (3) и входы без изменений.

H-SPEED (`tracker.speed_gate`, **ВЫКЛ**): трек не «прыгает» на детекцию соседа без перекрытия боксов дальше `0,2 + 1,5·dt` диагоналей.
- На синтетике «двое сидят рядом + выпадение детекций» подмена в 0 из 8 сидов против 1 из 8.
- В полном replay эффект мал: фрагментация 31 → 30, подмены 1 → 1.
- Ждёт замера L2: `--variant speed_gate`.

## Фаза 2 — что изменилось (решения записаны здесь)

### Решения по спецификации команды (включены по умолчанию: это требования, а не гипотезы)
| Что | Было | Стало |
|---|---|---|
| «Кто первый» | `w_start = 0.25` | `w_start = 0`. Кассир — тот, кто **сидит у стола со стороны кассы**. Может войти через дверь в первые секунды. |
| Сторона кассы | из позиции выбранного кассира | направление «центр стола → сейф», то есть из предметов, а не из людей (`side_support = "safe"`). Без сейфа сторона не подтверждена: `conf` кассира ≤ 0,7. |
| Неоднозначность кассира | среди всех кандидатов | только среди **одновременно** присутствующих. Фрагменты одного кассира (ушёл и вернулся) друг другу не конкуренты. |
| Возврат после разрыва трека | `unknown` без внешности | `cashier` (`cashier_by_behavior`, `conf ≤ 0,65`), если сторона кассы подтверждена сейфом и человек сам сидит там. Иначе `unknown` (`possible_cashier_return`): цикла «зона ↔ роль» нет. |
| Клиент напротив | `visitor` | `visitor`, `conf ≥ 0,8` и причина `seated_at_client_side`: это норма. |
| Цвет одежды | гистограмма одежды включена | **выключена** (`appearance.enabled: false`): «цвет одежды и яркость — не признаки». Код остаётся гипотезой H-APP. |
| Время | индекс кадра / fps в `viz` | только PTS (`CAP_PROP_POS_MSEC`); в трекере — `Detection.t`. |
| Неуверенное | — | `TrackQueries` считает роль с `conf < 0,5` и зону с `conf < 0,35` неизвестными: ответ `value=None`. |
| `cashier_area` | оболочка всех позиций кассира у стола | только по наблюдениям **«сидит у стола»**: зона не растягивается туда, где кассир стоял, уйдя с места. |
| Тесты | `conftest` ставил `SENTINEL_MOCK=1` на весь процесс | убрано: мок-флаг не утекает в чужие тесты, модуль моков не имеет. |
| Покрытие | — | `info["coverage"]`: `frames_total`, `frames_processed`, `frames_with_detections`, `span_frac`, `visibility_loss`, `visibility_check`. Пустые детекции ≠ «людей нет». |

### Явные функции-события для правил (`ml/track/queries.py`)
```python
from ml.track import track_full, TrackQueries
res = track_full(clip, dets); q = TrackQueries(res.tracks, res.zones, res.info)
q.cashier_left_workplace(t)   # Answer(value=True|False|None, conf, source, track_ids, t_event=момент начала, detail)
q.cashier_left_room(t)        # last_seen_in_door_zone (conf выше) | last_seen_at_frame_edge (вышел из кадра — не обязательно из помещения)
q.person_entered(t, role=None, window_s=1.0)   # вход = появление в двери / у края кадра
q.n_outsiders(t)              # value = число visitor; detail = {"unknown", "upper_bound", "low_quality"}
q.outsider_at_cashier_side(t) # True — visitor у кассы/сейфа; None — там unknown; False — никого; collector не считается
q.events()                    # все переходы: person_entered, cashier_left_workplace, cashier_returned_workplace,
                              # cashier_left_room, outsider_at_cashier_side, unknown_at_cashier_side
```
- `value=None` означает **«неизвестно»**: окклюзия, пропуск детектора, потеря видимости, не определены зона или кассир. Это не `False` и не «нарушения нет».
- `unknown` никогда не превращается в «постороннего»: такие люди уходят в `detail["unknown"]` и в событие `unknown_at_cashier_side` («на проверку»).
- `t_event` — момент **начала** перехода: вышла из зоны рабочего места, появился в двери, последний раз видна в двери.
- **Устранённый источник ложных срабатываний.** Раньше обрыв трека в глубине кадра выглядел как «вошёл человек»: это давало ложные V01/V04 после окклюзии, пропуска детектора или включения света. Теперь новое появление в глубине кадра не считается входом. Наивный режим оставлен для сравнения: `queries.entry_requires_boundary: false`. Тест: `test_fragment_birth_in_view_is_not_entry_eliminated_fp_source`.
- Replay нашёл и устранил ещё два источника ложных событий:
  - кассир, входящий в первые секунды, получал «ушёл с рабочего места» — теперь уход возможен только после того, как он там был;
  - `cashier_area` растягивалась на место, где кассир стоял, уйдя.
- **Потеря видимости** (например, выключили свет) определяется **без яркости**: одновременно пропали все статичные предметы-якоря. Во время потери все ответы — `None`, а появления сразу после неё не считаются входами.

### Флаги-гипотезы (по умолчанию ВЫКЛ до замера L2 на реальных данных)
| Флаг | По умолчанию | Гипотеза | Чем мерить |
|---|---|---|---|
| `queries.smoothing: hysteresis` (`min_state 2 с`, `merge 3 с`) | `off` | **H-SMOOTH**: гистерезис и объединение дребезга событий уменьшают ложные тревоги без потери таймкода | `replay --variant smooth` (ниже) |
| `visibility.bridge_tracks: true` | `false` | **H-VIS**: склейка треков через потерю видимости по положению уменьшает фрагментацию ID | `replay --variant vis_bridge` |
| `appearance.enabled: true` | `false` | **H-APP**: склейка по гистограмме одежды (противоречит спецификации «одежда — не признак», только для замера) | `replay --variant appearance` |
| `queries.entry_requires_boundary: false` | `true` | наивные входы — эталон «до исправления» | `replay --variant naive_entries` |
| `tracker.speed_gate: true` | `false` | **H-SPEED**: ограничение скорости без перекрытия боксов уменьшает подмены ID у сидящих рядом | `replay --variant speed_gate` |
| `tracker.lost_cost_penalty: 0` | `0.1` (исправление ошибки) | откат к прежнему сопоставлению | `replay --variant no_lost_penalty` |

### Replay: текущий режим против одного изменения сглаживания
Синтетика: 8 сценариев × 8 возмущений (none, dropout, jitter, occlusion, blackout, mirror, perspective, noise) × 2 сида = 128 прогонов на вариант. Замерено в облаке на CPU; это **поведение кода на синтетике, не точность на роликах**.

| Метрика (сумма по 128 прогонам) | baseline | smooth |
|---|---|---|
| fragmentation / switches / spurious (ID) | 53 / 0 / 0 | 53 / 0 / 0 |
| ложные тревоги (уход, уход из помещения, посторонний у кассы) | 17 | 3 |
| пропущенные тревоги | 6 | 6 |
| `cashier_left_workplace` tp/fp/fn, средняя \|Δt\| | 32/12/0, 0,48 с | 32/0/0, 0,48 с |
| `person_entered` tp/fp/fn | 80/3/0 | 80/3/0 |
| latency p50, мс (вся стадия + события) | 239 | 243 |

Что видно в таблице:
- Фрагментация ID — в основном от сценариев blackout и occlusion, где после потери видимости трек начинается заново. Её снимает H-VIS (`test_blackout_no_false_tracks_and_bridge_flag`).
- Остаточные пропуски тревог — `cashier_left_room` в сценариях blackout и occlusion. Это честный `None`, а не ошибка.
- Сглаживание событийное, поэтому ID не меняются.

```bash
python -m ml.track.replay --variant smooth --seeds 2            # синтетика, около 1,5 мин CPU; ml/track/_out/replay/
python -m ml.track.report --demo v02_bounce [--smoothing hysteresis]   # PNG: СИНТЕТИЧЕСКОЕ демо
python -m ml.track.report --cache cache/<sha>                   # PNG-таймлайн по реальному кэшу (локально)
```

### Команда замера для L2 (одна; реальные детекции, локально)
```bash
python -m ml.track.replay --real --dets-glob "cache/*/dets.jsonl" --videos-dir data \
    --labels data/labels.csv --variant smooth --tol 2.0 --out ml/track/_out/replay_real
```
- **Сравниваем**: `baseline` (текущие значения) против `smooth` (H-SMOOTH). Другие флаги — `--variant vis_bridge`, `appearance` или `naive_entries`, по одному за раз.
- **Вход**:
  - кэш стадий A03 (`backend/pipeline/cache.py`): `$SENTINEL_CACHE_DIR` или `<repo>/cache`, внутри `<sha256>/dets.jsonl` и `manifest.json["video"]` (`width`, `height`, `duration`). Он появляется после прогона пайплайна с кэшем `read_write`. Подходят и дампы `detections.json` + `meta.json` (тогда `--dets-glob` указывает на них);
  - `--videos-dir` — локальная папка роликов: только для соответствия sha256 → имя; видео никуда не передаются;
  - метки `data/labels.csv` в схеме `eval.labels` (`tools/dataprep/build_labels.py`): `video_id`, `file`, `label`, `t_sec`, `t_all`, `conflict`, … Поддерживаются и простые CSV `video|file`, `label`, `second|t` (и необязательно `sha256`). Имена нормализуются: латинская X = кириллическая Х. Конфликтные дубликаты (`conflict=1`) считаются в `label_conflicts` — решает человек.
- **Выход**: `replay.json` и `replay.md`.
- **Метрики для решения**:
  - «чистые ролики с тревогой» — прокси ложных срабатываний;
  - «ролики с нарушением без тревоги» — прокси пропусков;
  - попадание тревоги в ±2 с от размеченной секунды;
  - среднее число треков людей;
  - latency p50 стадии;
  - VRAM = 0 (стадия только на CPU).
- **Правило**: включать флаг, только если прокси ложных срабатываний снизился, а прокси пропусков и попадания не ухудшились. Тип нарушения на этой стадии не меряется: тип не размечен.
- **Оговорка**: все 42 строки уже использованы в разведке, поэтому независимым тестом этот замер не является.

## Что сделано
| Часть | Файл | Суть |
|---|---|---|
| ByteTrack | `bytetrack.py` | Своя реализация (идея ByteTrack: двухэтапное сопоставление high/low score). Фильтр Калмана, учитывающий dt, с затуханием скорости. Стоимость = 1 − [(1−w)·IoU + w·близость центров]: при 5 fps у идущего человека IoU между кадрами бывает 0. Время в секундах, пропуск кадров детектором (`every_n`) определяется автоматически. Отдельные трекеры по группам классов: `person`; `safe`+`safe_open` → трек `safe`; `door`+`door_open` → `door`; `cash_bag`+`bag` → `cash_bag`, если ≥30% голосов за неё, иначе `bag`; остальные метки — каждая своя группа. |
| Склейка | `stitch.py`, `appearance.py` | Короткий разрыв (≤2,5 с, окклюзия или пропуск детектора) склеивается по положению с растущим допуском, внешность может запретить склейку. Длинный разрыв (ушёл и вернулся) — **только** при совпадении HSV-гистограммы **одежды** (кроп без верхних 25% бокса и без краёв; лица не используются) и правдоподобном месте: вернулся на своё место или появился там же, где исчез. Без внешности длинные разрывы не склеиваются. Статичные предметы склеиваются по IoU медианных боксов (мерцание до 15 с). |
| Авто-зоны | `zones.py` | Без единой захардкоженной координаты; выпуклая оболочка и масштаб от центроида аффинно-эквивариантны. **Независимые опоры** (строятся из предметов, не из людей): `desk` — оболочка медианных боксов laptop/money_counter и статичных cell_phone/keys/document; `safe` — медианный бокс safe/safe_open; `door` — детекции door/door_open и кластер точек появления/исчезновения людей; `cash_tray` — кластер cash (fallback — окрестность money_counter, conf 0,2). **Производные**: `cashier_area` — места кассира ∪ «отпечаток» стола, сдвинутый на его сторону; `client_area` — люди по другую сторону стола, иначе зеркало `cashier_area` через центр стола (conf ≤ 0,3). Если опор нет, зоны нет или у неё низкий `conf`. Зона «весь кадр по умолчанию» не создаётся никогда. |
| Роли | `roles.py` | См. ниже: `cashier` / `collector` / `visitor` / `unknown` + `role_conf` + provenance. |
| Утилиты A08/A09 | `events.py` | `entries_exits`, `dwell`, `proximity`, `gaps`, `presence_state`, `person_signals`, `box_at`, `zone_by_name`. |
| Отладка | `viz.py`, `serialize.py` | `python -m ml.track.viz video.mov --cache cache/<sha>`; кэш стадии `cache/<sha>/track.json`. |

### Роли — принципы
1. **Кассир** выбирается по признакам, не зависящим от зон людей: близость к объектам стола и к их центру, присутствие с начала ролика, неподвижность (скорость по точкам с шагом ~1 с), сторона сейфа. `cashier_area` строится **после** выбора кассира и для повторного «назначения кассиром» сама по себе не используется. Цикла «кассир, потому что в cashier_area; cashier_area, потому что там кассир» нет.
2. Нет объектов стола → fallback: кассир — самый неподвижный человек, присутствующий с начала. `role_conf ≤ 0,45`, зоны `cashier_area`, `desk` и `client_area` получают `conf ≤ 0,3`, `visitor ≤ 0,6`, в причинах — `fallback_no_desk_objects`.
3. **non-cashier ≠ посторонний.** Второй человек в зоне кассира одновременно с кассиром получает `unknown` (`co_present_in_cashier_area`, плюс `entered_mid_clip`, если пришёл по ходу ролика): у него может быть допуск. Близкий по score второй кандидат в кассиры → `unknown` (`ambiguous_cashier_candidate`), а conf кассира снижается.
4. **Сумка сама по себе роль не подтверждает.** `collector` = контакт с `cash_bag`/`bag` **и** независимый признак прихода: вошёл по ходу ролика или сумка появилась вместе с ним. Только контакт с сумкой → `unknown` (`bag_contact_only`). Спутник инкассатора без своей сумки → `unknown` (`arrived_with_collector`).
5. **Уход и возврат.** Возврат с совпадающей одеждой — тот же трек (склейка) или `cashier` (`clothes_appearance_match`). Только «то же место» без внешности → `unknown` (`possible_cashier_return`, `position_only_no_appearance`). Другая одежда на месте кассира → `unknown` (`possible_shift_change`, если остался до конца ролика).
6. Короче `min_presence_s` (1 с) → `unknown` (`too_short`). У предметов всегда `role="unknown"`, `role_conf=0`.

### Пропал из кадра ≠ ушёл
- `entries_exits()` по умолчанию **не** создаёт `exit` из-за пропуска детекций. Выход фиксируется только по наблюдаемому пересечению границы зоны с гистерезисом 0,4 с. Режим «пропуск > N с = выход» включается явно: `gap_as_exit_s=N`.
- `gaps(track, zones, t_end=...)` классифицирует каждый пропуск:
  - `lost_in_view` — пропал в глубине кадра: окклюзия или пропуск модели, **не уход**;
  - `left_view_edge` — последний раз виден у края кадра: вышел из поля зрения, но не обязательно из помещения;
  - `left_via_door` — последний раз виден в зоне `door`.
- `person_signals()`: кассир не виден, но `lost_in_view` → `cashier_present = 0.5` («неизвестно»), а не 0.
- В `Track.boxes` только **реальные** наблюдения с реальными timestamps детекций (`fill_gaps_s: 0`). Интерполяция включается в конфиге, но выключена по умолчанию.

## Публичные функции
```python
from ml.track import track, track_full, entries_exits, dwell, proximity, gaps, presence_state, person_signals
tracks, zones = track(clip, dets)          # контракт; dets[i].track_id проставляется на месте
res = track_full(clip, dets, cfg=None)     # TrackResult(tracks, zones, info) — info = provenance
res.info["roles"][tid]   # {"role","conf","reasons":[...],"features":{...},"in_cashier_area",...}
res.info["zones"][name]  # {"method","independent","evidence"/"tracks",...}
res.info["stitch"]       # журнал склеек: kind short_gap|long_gap_appearance, gap_s, app_dist
res.info["cashier_selection"], res.info["appearance"], res.info["timings"], res.info["warnings"]

entries_exits(tracks, zones, *, anchor="center", min_dwell_s=0.4, enter_at_start=True, gap_as_exit_s=None)
    -> list[(t, track_id, zone_name, "enter"|"exit")]
dwell(track, zone, t0=None, t1=None, *, max_dt=1.0) -> float          # секунды наблюдённого присутствия
proximity(track_a, track_b, t, *, max_dt=0.6, aspect=1.0) -> float    # 0..1; 0, если кого-то нет в момент t
gaps(track, zones=None, *, min_gap_s=0.6, edge_margin=0.06, t_end=None) -> list[dict]
presence_state(track, t, zones=None) -> "observed"|"not_yet"|"lost_in_view"|"left_view_edge"|"left_via_door"
person_signals(tracks, zones, times, *, extended=False) -> list[SceneState]
    # cashier_present, cashier_in_workplace, n_persons, outsider_in_cashier_area, collector_present
    # outsider_in_cashier_area: visitor -> role_conf, unknown -> 0.5, collector/cashier -> 0
```
Вход принимает как `Detection`, так и dict с теми же ключами. Битые боксы пропускаются, в `info.warnings` пишется предупреждение. Пустые или `None` детекции дают `([], [])`.

## Запуск
```bash
pip install -r ml/track/requirements.txt -c <lock L1>   # версии не менять вслепую (opencv < 5)
python -m pytest ml/track/tests -q                     # 88 тестов, без GPU/весов/данных (~25 с)
python -m pytest ml/track/tests -q -m perturbation     # только возмущения ракурса (perspective/mirror/rotation)
python -m ml.track.viz video.mov --cache cache/<sha> [--out out.mp4] [--recompute] [--blur-heads]
```
`viz` берёт детекции из `cache/<sha>/detections.json` (или `dets.json`, `detect.json`, `*.pkl`), а если их нет — из `ml.detect.detect(clip)`. Кадры читает через `ml.ingest.load_video`, иначе через встроенный `SimpleClip` на cv2 с автоповоротом. Пишет `track.json` и `track_viz.mp4`. Видео не коммитить: в `ml/track/.gitignore` они уже исключены.

## Проверки (реально выполнено)
- `pytest ml/track/tests` — 35 passed. Покрыто:
  - пустые и битые детекции;
  - стабильность ID при дрожании и выпадении 20% детекций;
  - детекция через кадр (`every_n=3`);
  - продление трека «слабыми» детекциями;
  - два пересекающихся человека без смены ID;
  - сценарии clean, V01, V02 (с внешностью и без), «на место кассира сел другой», V07, «сумка без прихода», «два сотрудника» (3 сида), отсутствие объектов стола;
  - зоны под аффинным «поворотом камеры» (±25…35°, масштаб, зеркало), поворот на 90° в альбомную ориентацию;
  - события, кэш, viz CLI.
- Скорость на синтетике (облачный CPU, без GPU): 145 с, 725 кадров, 4371 детекция → **1,43 с** на всю стадию. Реальные ролики не измерялись: данных здесь нет.

## Ограничения (честно)
- Фаза 2 проверена только на синтетике: в облаке нет реальных роликов и GPU. Все пороги ждут замера L2 (команда выше).
- Без сейфа в кадре сторона кассы не подтверждена: кассир — тот, кто дольше сидит у стола, а при двух сидящих по разные стороны выбор слабее (`conf ≤ 0,7`).
- Возврат кассира без сейфа в кадре → `unknown` (`possible_cashier_return`): цвет одежды выключен спецификацией.
- `n_outsiders` считает `visitor`; `unknown` — только в `upper_bound`. Отличить «второго сотрудника» от «постороннего» по траектории нельзя: это решают правила и VLM.
- Пороги не тюнены на реальных данных: синтетика ≠ реальная сцена. Все пороги — в `config.yaml`.
- Смена ID при плотном контакте людей (передача сумки «из рук в руки») в ассоциации не исправляется: внешность используется только при склейке треклетов.
- Без кадров (`clip.frame` недоступен) или без cv2 внешности нет. Тогда длинные разрывы не склеиваются, возврат кассира → `unknown`.
- Гистограмма одежды не различает людей в одинаковой форме, например двух кассиров в одинаковой униформе.
- Сторону стола определяет позиция кассира относительно центра объектов стола. При сильно наклонном ракурсе (камера сбоку) «отпечаток» стола — грубое приближение.
- `door` без детекции двери строится по кластеру появлений и исчезновений; при одном входе получается низкий conf или зоны нет.
- Роли — подсказки для правил, не утверждения о допуске или личности.

## Для интегратора
- **Подключение** в `backend.pipeline`: `tracks, zones = ml.track.track(clip, dets)`. Для provenance вызывать `ml.track.track_full(...)` и класть `res.info` в `AnalysisResult.meta["track"]` или в кэш через `ml.track.serialize.save_track_cache(cache_dir, tracks, zones, info)`.
- `track()` **проставляет `track_id` в переданные `Detection`** (in-place) — A08 может связывать детекции с треками.
- **A08/A09**: `entries_exits`, `dwell`, `proximity`, `gaps`, `presence_state` и `person_signals` — чистые функции над контрактными `Track`/`Zone`. Рекомендация: не считать `unknown` автоматически посторонним (вес 0,5 в `person_signals`), а `lost_in_view` не считать уходом.
- **A09: события в кандидаты нарушений** (решение принимают правила, здесь только наблюдения):
  - **V02**: `cashier_left_workplace(t).value is True`, а в окне после `t_event` по A08 на столе `cash` или `cash_bag` либо сейф открыт. Таймкод — `t_event`. Уход с убранными ценностями и закрытым сейфом — норма.
  - **V01**: `outsider_at_cashier_side(t).value is True` — кандидат. `None` (там `unknown`) — на проверку VLM, не автоматическое нарушение. `n_outsiders(t).detail["upper_bound"] >= 2` во время обслуживания — кандидат «второй посторонний».
  - **V04**: `person_entered(t)` при открытом сейфе или наличных на столе (A08).
  - **V07**: роль `collector` плюс состояние сумки по A08: сумка оставлена или ушла не в сейф.
- **A03**: класть `res.info["coverage"]` в `meta.coverage`. Если `frames_processed == 0` или есть `visibility_loss`, ответ «нарушений нет» без проверки полноты недопустим.
- **ENV**:
  - `SENTINEL_TRACK_CONFIG=/path.yaml` — свой конфиг;
  - `SENTINEL_TRACK_APPEARANCE=0` — выключить гистограммы одежды;
  - мок-режима нет и не будет: модуль без моделей; синтетика (`ml/track/synth.py`) используется только в тестах, replay и демо и помечена.
- **Контракт**: импорт через `ml/track/contract.py`: `contracts.sentinel_contracts`, а если его нет — `_contract_fallback.py`. Fallback удалить после мерджа A01.
- **Предложение A01 (не реализовано самостоятельно)**: добавить в `Track` поле `extra: dict` (provenance роли: `reasons`, `features`) и в `Zone` поле `extra: dict` (`method`, `independent`). Сейчас это только в `track_full().info`.
- **TODO на реальных данных**:
  1. Подобрать `roles.*` и `zones.*` по ~40 роликам: где реально сидит кассир, как детектор видит дверь.
  2. Проверить `cash_bag` против `bag` у A05.
  3. Проверить, нужен ли якорь `torso` или `bottom` вместо `center` для другого ракурса.
