# ml/pose — поза COCO-17 и поза-сигналы (агент A07)

Поза — **дополнительный источник подсказок**, а не доказательство. Рука у уха не доказывает телефон,
рука у рта — еду, пропавшие точки — сокрытие рук. Предметные выводы делаются только при наличии предмета
(объект-трек `cell_phone` / `cup` / `bottle` / `food` в аргументе `tracks`) и достаточной видимости.
Низкая уверенность или окклюзия → **unknown**, а не ложный ноль и не нарушение. Production никогда
не переключается на mock сам; неполный анализ позы виден в `status()["coverage"]`.

## Доступность сейчас (честно; аудит integration-cloud 2055fc4)

Импорт `ml.pose` **не означает**, что поза доступна. Бэкенд готов, только если одновременно найден файл весов
и установлен пакет — это показывает `ml.pose.availability()`, без загрузки моделей.

| бэкенд | нужно | у L1/L2 сейчас | итог |
|---|---|---|---|
| `rtmpose_onnx` (порядок по умолчанию) | `*rtmpose-m*.onnx` в `$SENTINEL_WEIGHTS_DIR/pose/` или `SENTINEL_POSE_ONNX` + пакет `onnxruntime` | веса **не скачаны** | недоступен → `stage_status: skipped` |
| `mediapipe` (opt-in) | `pose_landmarker_full.task` + пакет `mediapipe` (в lock L1 нет, сейчас не добавляем) | нет | недоступен |
| `mock` | только `SENTINEL_MOCK=1` / `SENTINEL_POSE_BACKEND=mock` | — | тесты/демо, в статусе `mock` |

**Текущий безопасный режим без новых весов** (ничего включать не нужно): `enabled: auto` → `estimate()` = `[]`,
`status()["state"] == "unavailable"`, `stage_status == "skipped"`, причина «веса не найдены локально»;
`pose_signals()` — все сигналы позы **unknown** (ключей нет), никаких координат рук/головы и скелетов.
Чтобы и `meta` у A03 честно показал пропуск (пока A03 не читает статус позы, см. «Для интегратора»):
`SENTINEL_PIPELINE_CONFIG=ml/pose/presets/a03_pose_off.yaml` (`stages.pose: false` → `disabled` → `skipped`).

**Включение настоящего бэкенда — только после локальной проверки у L2:**

```bash
# 1) положить веса (один раз, офлайн), проверить готовность без загрузки модели
python -c "import ml.pose as p, json; print(json.dumps(p.availability(), ensure_ascii=False, indent=1))"
# 2) проверить качество на своих кадрах локально (картинки — только на локальный диск)
python -m ml.pose.ab --video <ролик> --out <локальная папка> --backends rtmpose_onnx
# 3) включить: поза обязательна в этом запуске (нет бэкенда → failed, а не тихий skipped)
SENTINEL_POSE_ENABLED=1 SENTINEL_POSE_BACKEND=rtmpose_onnx SENTINEL_POSE_ONNX=<путь к .onnx> <команда запуска>
```

## Публичные функции (сигнатуры контракта v1 не менялись)

```python
from ml.pose import estimate, pose_signals, warmup, status, availability, model_info

poses  = estimate(clip, tracks)               # -> list[PoseFrame] (фактически PoseFrameX, наследник)
states = pose_signals(poses, tracks, zones)   # -> list[SceneState] (фактически PoseSceneState, наследник)
warmup()        # -> bool: загрузить и прогреть бэкенд; False — выключен или недоступен
status()        # -> dict: итог последнего estimate(): state, stage_status, available, error, warnings, coverage
availability()  # -> dict: готов ли реальный бэкенд сейчас (веса + пакет), без загрузки моделей
model_info()    # -> dict: хук A03 (identity и ключ кэша): checkpoint, available, stage_status, флаги, config_sha
```

`estimate` никогда не бросает исключение из-за позы. Нет весов или сбой → `[]` (или позы до сбоя), причина
в `status()` и в логгере `ml.pose`. Импорт пакета без побочных эффектов: numpy, cv2, onnxruntime и
mediapipe импортируются лениво.

| `status()["state"]` | `stage_status` | когда |
|---|---|---|
| `disabled` | `skipped` | `enabled: false` / `SENTINEL_POSE_ENABLED=0` |
| `unavailable` | `skipped` | `auto`: весов/пакета нет |
| `unavailable` | `failed` | веса есть, но бэкенд не поднялся; или `enabled: 1`, а бэкенда нет |
| `no_persons` | `ok` | нет person-треков |
| `ok` / `partial` | `ok` | поза посчитана (`partial` — шаг по сэмплам > 1 из-за бюджета, см. `coverage.stride`) |
| `partial` | `timeout` | упёрлись в `max_crops_total`: конец видео без позы (`coverage.analyzed_until_sec`) |
| `no_poses` | `ok` | бэкенд отработал, но ни одной уверенной позы — сигналы unknown |
| `failed` | `failed` | сбой инференса посреди прогона (позы до сбоя — настоящие, `coverage.stopped_early`) |
| `mock` | `mock` | явный mock-режим |

Что модуль **не выдаёт** как позу: нулевой скелет «поза не найдена» (max score < `min_pose_score`), точки
далеко за bbox своего трека (score 0, `geometry_margin`), позу из невалидного bbox (NaN, перевёрнутый,
не 0..1 — в `status()["invalid_boxes"]`), битые точки (не 3 числа, NaN — score 0).

Инструмент замера: `python -m ml.pose.ab` (см. раздел «Замер для L2»).

## Флаги (каждый новый режим по умолчанию выключен до замера L2)

| флаг | где | по умолчанию | смысл |
|---|---|---|---|
| `enabled` | config / `SENTINEL_POSE_ENABLED=0\|1\|auto` | `auto` | `false` → `estimate()` = `[]`, `stage_status: skipped`, все сигналы позы unknown. `auto` → нет весов = `skipped` (не `failed`). `1` → нет бэкенда = `failed` |
| бэкенд | `backend.order` / `SENTINEL_POSE_BACKEND=rtmpose_onnx\|mediapipe\|mock` | `[rtmpose_onnx]` | один бэкенд на запуск. MediaPipe — переключаемый вариант; mock — только явно (тесты и демо) |
| `signals.enabled` | config | `phone_in_hand, eating_drinking, hands_hidden, handover_motion` | какие ключи SIGNALS считать. Хрупкие `phone_to_ear` и `cashier_facing_away` — opt-in, код сохранён |
| `signals.extras.motion` | config | `false` | `extra["motion"]`: sitting / standing / walking + направление |
| `signals.extras.item_placed_on_desk` | config | `false` | `extra["placements"]`: кто и когда положил предмет на стол |
| `signals.extras.bending_to_safe` | config | `false` | хрупкий жест «наклон к сейфу» |
| пресеты | `SENTINEL_POSE_CONFIG=ml/pose/presets/<x>.yaml` | — | `extras_on.yaml` (motion + items), `fragile_on.yaml` (жесты фазы 1) |

Все флаги читаются при каждом вызове, а не при импорте (работает и `eval.ablate --inproc-arms`).

## Сигналы и семантика

| значение в `SceneState.signals` | смысл |
|---|---|
| ключа нет | **unknown**: окклюзия, низкий score, нет нужного предмета, нет кассира в кадре |
| `< 0.5` | наблюдаемое «нет»: нужные точки видны; для предметных сигналов предмет есть, но не у руки/рта |
| `≥ 0.5` | сигнал активен после гистерезиса (`thr_on`/`thr_off`), требований к длительности и склейки дребезга (`max_gap_s`) |

| сигнал (по умолчанию) | условие активности | unknown, когда |
|---|---|---|
| `phone_in_hand` | кисть кассира у bbox трека `cell_phone` ≥ 1 с | телефона в кадре нет |
| `eating_drinking` | повторяющееся «кисть ко рту» (кисть идёт к голове, а не голова к рукам) **и** `cup`/`bottle`/`food` у рта, пока кисть у рта | жест без предмета у рта; рот или кисти не видны |
| `hands_hidden` | только `0.0` при видимых кистях; позитив из позы выключен | кисти не видны: это окклюзия, а не умысел |
| `handover_motion` | вытянутые кисти двух людей сближаются (над `desk`/`cash_tray`, если зоны есть) | у кого-то из пары не видно кистей |
| `phone_to_ear` (opt-in) | кисть у уха ≥ 1.5 с **и** `cell_phone` у головы/кисти | жест без телефона |
| `cashier_facing_away` (opt-in) | нормаль к линии плеч направлена от зоны `desk` ≥ 2 с | нет зоны `desk`, плечи не видны |

«Кассирские» сигналы считаются только по трекам с `role == "cashier"`. Значения по всем людям — в
`extra["by_track"]`. Роль задаёт A06 по поведению и зоне, а не по порядку появления.

### `PoseSceneState.extra`

* `runs[signal] = {t_start, t_end, t_peak}` — для активного отрезка. `t_start` — начало перехода
  (кандидат таймкода: «поднесла», «подошёл»), `t_peak` — максимум (для еды — первый контакт кисти со
  ртом). Какую точку отдавать жюри, решает замер, а не глаз.
* `track_ids[signal]`, `boxes[signal]` — кто и где (bbox норм.) дал активный сигнал.
* `gestures` — подсказки без предмета, **не доказательства**: `hand_to_mouth`, `hands_not_visible`
  (+ `hand_at_head`, если включён `phone_to_ear`).
* `motion` (если `extras.motion`) — `{state: walking|standing|sitting|unknown, speed, run}` кассира.
  `run = {t_start, t_end, direction, toward, away_from}`: `t_start` — момент «отошла», `toward`/`away_from` —
  зона (`door`, `desk`, `safe`, …) по направлению движения. `None` — данных нет.
* `item_placed_on_desk` / `placements` (если `extras.item_placed_on_desk`) — `1.0` в окне события, иначе
  `0.0`; `None` — нет зоны `desk` или объект-треков. Событие: `{t_start (кисть у предмета), t_settle
  (предмет лёг), person_track_id, role, object_track_id, label, bbox}`. Событием не считаются: предмет,
  лежавший с начала ролика; новый трек на месте старого (смена id); появление предмета без кисти рядом.
* `by_track`, `staff_track_ids`, `handover_pairs`, `n_posed`, `pose_status`, `enabled`, `unknown`.

### `PoseFrameX.extra`

`bbox` (норм.), `source` (`rtmpose-onnx` | `mediapipe` | `mock`), `frame_wh`, `role`, `smoothed`,
`schema: "coco17"`. `keypoints` — всегда 17 точек COCO-17, `(x, y)` нормированы к кадру после поворота,
`score ∈ 0..1`. Горизонтальное видео «под другим углом» обрабатывается так же: геометрия изотропная
по `frame_wh`, расстояния — в масштабах тела.

## Бэкенды

* **RTMPose-m ONNX** (`rtmpose_onnx`, порядок по умолчанию): onnxruntime CUDA → CPU, батчинг кропов
  через кадры, SimCC-декод как в rtmlib. У L2 веса не скачаны: при `enabled: auto` стадия честно
  `skipped` и не блокирует сервис.
* **MediaPipe Pose Landmarker** (`mediapipe`, Tasks API, переключаемый вариант): кроп кассира
  (bbox × 1.4, за краем кадра — чёрное поле) с апскейлом короткой стороны до 384 px (не больше ×3).
  `num_poses: 2` — из кропа берётся поза, точки которой лежат в bbox трека. 33 landmark-а **явно**
  перекладываются в COCO-17 (`MP33_TO_COCO17`, тест сверяет имена). `score = min(visibility, presence)`.
  Поза не найдена → нулевые score (unknown). Отбор: только `role == cashier`, один кроп на кадр; если
  роли нет ни у кого — top-1 по приоритету. Старый `mp.solutions` в mediapipe 0.10.35 отсутствует —
  использую только Tasks API.
* **mock** — только `SENTINEL_MOCK=1` / `SENTINEL_POSE_BACKEND=mock`; в отчётах помечен `mock`/`MOCK`.

Лимиты: `max_persons_per_frame: 3` (MediaPipe — 1), `max_crops_total: 2500`, бюджет `budget_s: 10`
с адаптивным шагом по сэмплам. Шаг > 1 или упор в лимит → `state: partial`, это видно в `coverage`.

## Замер для L2 (видео, кадры и описания сцен — только локально)

Новую сеть поз не обучаем. Каждая гипотеза — отдельная команда, одно изменение за раз.

**H-A07-1. Вклад позы в итог пайплайна** (FP/FN по видео, попадание таймкода, тип, latency, VRAM — по
протоколу A16):

```bash
python -m eval.ablate --name H-A07-pose-mp \
  --baseline-env SENTINEL_POSE_ENABLED=0 \
  --candidate-env SENTINEL_POSE_ENABLED=1 --candidate-env SENTINEL_POSE_BACKEND=mediapipe \
  --candidate-env SENTINEL_POSE_MP_TASK=weights/pose/pose_landmarker_full.task \
  --videos data/raw --labels data/labels.csv --manifest data/manifest.csv --out runs/ablate/H-A07-pose-mp
```

Решение: нет выигрыша по FN/FP или растёт latency → оставить `enabled: false` (код сохраняется).
Эффект будет ненулевым, только если A08/A09 читают сигналы позы.

**H-A07-2. Качество бэкендов на одинаковых кадрах** (уверенность и видимость запястий, локтей и головы,
дрожание, мс/кроп, VRAM):

```bash
python -m ml.pose.ab --video data/raw/<ролик>.mov [--video ...] --out <локальная папка вне репо> \
  --backends mediapipe,mediapipe_noup[,rtmpose_onnx]
# или на готовых кадрах: --frames <dir> --boxes boxes.json
```

Выход: `ab_report.md` / `ab_report.json` (только числа и имена файлов), `per_crop.csv`, а также
`overlays/<вариант>/*.jpg` — скелеты, только на локальный диск (в папку кладётся `.gitignore` с `*`;
`--no-overlays` отключает картинки). Голова размывается только по найденным точкам головы; если их нет,
место головы не угадывается — размывается весь bbox человека (счётчики — `overlay_blur` в JSON). Недоступный вариант (нет весов) помечается
`unavailable` и не прерывает прогон. `mediapipe` vs `mediapipe_noup` измеряет пользу апскейла.
Согласие между бэкендами (PCK@0.5) — не точность: эталона нет.

**H-A07-3. Новые extra** (`motion`, `item_placed_on_desk`) — только после того, как A09 начнёт их читать:
`--baseline-env SENTINEL_POSE_ENABLED=1 --candidate-env SENTINEL_POSE_CONFIG=ml/pose/presets/extras_on.yaml`
(остальные аргументы — как в H-A07-1).

## Установка

```bash
pip install -r ml/pose/requirements.txt
# MediaPipe (opt-in): без зависимостей, иначе приедет opencv-contrib-python 5.x и сломает lock opencv<5
pip install --no-deps "mediapipe==0.10.35" && pip install absl-py flatbuffers certifi matplotlib
# Linux-контейнер: apt-get install libgles2 libegl1 (иначе PoseLandmarker не создаётся). Windows не проверялся.
```

Веса (один раз, дальше офлайн), см. `models.json`:

* MediaPipe — `pose_landmarker_full.task` в `$SENTINEL_WEIGHTS_DIR/pose/` или `SENTINEL_POSE_MP_TASK=<путь>`.
* RTMPose — `*rtmpose-m*.onnx` в `$SENTINEL_WEIGHTS_DIR/pose/` или `SENTINEL_POSE_ONNX=<путь>`.

| env | назначение |
|---|---|
| `SENTINEL_POSE_ENABLED` | `0` / `1` / `auto` — главный флаг позы |
| `SENTINEL_POSE_BACKEND` | `rtmpose_onnx` / `mediapipe` / `mock` |
| `SENTINEL_POSE_CONFIG` | YAML поверх `config.yaml` (deep-merge), например пресеты из `presets/` |
| `SENTINEL_POSE_ONNX`, `SENTINEL_POSE_MP_TASK`, `SENTINEL_WEIGHTS_DIR` | пути к офлайн-весам |
| `SENTINEL_MOCK=1`, `SENTINEL_POSE_MOCK_SCRIPT` | mock-позы и скрипт действий — только тесты/демо |

## Проверено / не проверено

* Аудит 2055fc4 (песочница, синтетика): `python -m pytest ml/pose/tests` — 105 passed, 1 skipped
  (real-smoke MediaPipe без модели). Тот же набор внутри `integration-cloud` с каноническим контрактом A01 —
  105 passed, 1 skipped; тесты A03 `backend/pipeline/tests -m "not gpu and not slow"` с этой версией позы —
  54 passed, 2 skipped (нет пакета `av`), 12 deselected. `testkit.e2e_stub --no-backend` (mock): стадии
  pose/pose_signals — real, без проблем; итоговый `status=failed` у стаба одинаков и с прежней версией
  `ml/pose` — причина вне модуля позы, не исследовалась.
* Ранее (фаза 2; CPU, Python 3.11, numpy 2.4, opencv-python-headless 4.14, PyYAML 6.0, pytest 9.1):
  78 passed, 1 skipped.
  С моделью (`SENTINEL_POSE_MP_TASK`) проходит и он: настоящий mediapipe 0.10.35 Tasks API создаёт
  PoseLandmarker и обрабатывает синтетический кадр. Установка `--no-deps` с opencv < 5 проверена в чистом
  venv. `python -m ml.pose.ab` прогнан с настоящим MediaPipe на синтетических кадрах (без людей, поэтому
  поза не найдена — проверялся только путь кода).
* Синтетические тесты: позитивы и негативы на каждый сигнал. Негативы: пересчёт денег; голова к рукам;
  стакан на столе + кисть ко рту; поправила волосы (стакан и телефон на столе); почесала ухо 2 с
  (телефон на столе); рука у уха без телефона; пропавшие кисти; низкая уверенность.
* **Не проверено** (нет GPU, весов RTMPose и данных): качество поз на потолочной камере, находит ли
  MediaPipe людей сверху, скорость и VRAM на RTX 4060, пороги на реальных роликах, работа на Windows.
  Цифр скорости не привожу: миллисекунды из песочницы на синтетике не отражают целевую машину.

## Ограничения

* Сверху ноги и часть кистей закрыты столом — много unknown, это ожидаемо и честно.
* Детектор BlazePose (MediaPipe) опирается на голову/лицо; при виде сверху он может не находить людей —
  это проверит H-A07-2.
* `sitting/standing` — грубая эвристика; `walking` надёжнее (скорость корпуса в масштабах тела).
* При выключенном свете точки пропадают — сигналы unknown, а не «нет нарушения».

## Для интегратора

1. **История ветки.** Ветка создана до появления `main` (несвязанная история), поэтому PR в `main` GitHub
   не открывает. Вливать по `scripts/merge_order.md` A01:
   `git merge --no-ff --allow-unrelated-histories origin/claude/nice-gates-fjt926`. Конфликт возможен
   только по пустому `ml/__init__.py` — берите любую сторону.
2. **A03 (pipeline) — несовместимости на границе, воспроизведены на `integration-cloud` 2055fc4** (синтетика,
   без весов, `SENTINEL_POSE_*` не заданы):
   * **B1. Пропуск позы выглядит как `ok`.** `backend/pipeline/orchestrator.py::_stage` ставит
     `rec.status = "ok"` при любом успешном вызове; `estimate()` без весов честно возвращает `[]`, а
     `coverage.py::stage_coverage` (поза в `_CLIP_WIDE`) приравнивает покрытие позы к покрытию кадров.
     Вход: 1 person-трек, нет RTMPose. Сейчас: `meta.stages.pose.status = ok`, `coverage.by_stage.pose ≈ 1.0`.
     Ожидается: `skipped` с причиной «веса не найдены локально», покрытие позы 0/None. Данные у A03 уже есть:
     `ctx.identities["pose"]["model_info"]["stage_status"]` (мой `model_info()`, собирается до вызова) или
     `ml.pose.status()["stage_status"]` после вызова через executor. До исправления — пресет
     `SENTINEL_PIPELINE_CONFIG=ml/pose/presets/a03_pose_off.yaml`.
   * **B2. `extra` теряется.** `backend/pipeline/serde.py::load_pose/load_state` собирают голые
     `PoseFrame`/`SceneState` (при `cache.normalize_outputs: true`), поэтому `PoseSceneState.extra`
     (`pose_status`, `runs`, `gestures`, `motion`, `placements`, `track_ids`, `boxes`) и `PoseFrameX.extra`
     (`bbox`, `frame_wh`, `source`) не доходят до A08/A09/A12. Ожидается: сохранять `extra`, как `load_event`.
   * **B3. Provenance заявляет неиспользованные модели.** `orchestrator.py::_provenance` берёт
     `meta.provenance.models` из `models.json` модуля — для позы это RTMPose-m и MediaPipe даже без весов.
     Ожидается: для позы брать `model_info()["models_loaded"]` (пусто — моделей нет).
   * Уже закрыто с моей стороны: `model_info()` даёт A03 `checkpoint` (→ `checkpoint_sig`), флаги
     `SENTINEL_POSE_*` и `config_sha` — ключ кэша стадии меняется при появлении весов и при смене флагов
     (иначе плечи A/B «поза вкл/выкл» с общим кэшем получали бы один и тот же результат).
   * Позу **не** включать в `required_stages`; `ml.pose.warmup()` возвращает False без весов — это норма.
   * Размытие голов на кадрах-доказательствах (A12) возможно только при найденных точках головы; когда поза
     пропущена, точек нет — A12 честно пишет warning «нет ключевых точек головы», и обещать размытие нельзя.
3. **A06:** возвращать в `tracks` объект-треки (`cell_phone`, `cup`, `bottle`, `food`, `bag`, `cash_bag`,
   `cash`, `document`, `keys`) — без них предметные сигналы и `placements` остаются unknown. Роль
   `cashier` — по поведению и зоне.
4. **A08/A09:** отсутствующий ключ — unknown (не подставлять 0). `extra["gestures"]` не считать
   доказательством. Таймкод брать из `extra["runs"][...]["t_start"]` / `motion.run.t_start` /
   `placements[].t_start` (политику t_peak выбирает замер). Кандидаты по спецификации: уход кассира —
   `motion.run.away_from == "desk"` + состояние стола и сейфа (A08) в окне после ухода; личные вещи на
   столе — `placements` с `label == "bag"`; передача сумки из рук в руки — `handover_motion` (норма, если
   сразу в сейф).
5. **A17:** офлайн-модель `pose_landmarker_full.task` (sha256 в `models.json`), mediapipe ставить
   `--no-deps` (см. «Установка»).
6. **A01 (контракт v2, предложение):** поле `extra: dict` в `PoseFrame`/`SceneState` (сейчас — наследники)
   и правило «нет ключа в signals = unknown». После мерджа A01 удалить `ml/pose/_contract_fallback.py`
   (`scripts/check_fallbacks.py`).
7. `ml/pose/tests/__init__.py` есть намеренно: тесты импортируются как `ml.pose.tests.*` и не конфликтуют
   с одноимёнными файлами других модулей.

## Журнал решений

Фаза 1:
* Unknown — это отсутствие ключа: контракт не меняется, а «нет данных» не путается с «нет нарушения».
* `hands_hidden` из позы позитивом не бывает: пропавшие точки сверху — обычная окклюзия.
* Нос не используется для ориентации (`nose_weight: 0`): на наклонной камере смещение носа от плеч
  определяется высотой головы.
* Ключи порогов `thr_on`/`thr_off`: в YAML 1.1 `on`/`off` превращаются в `True`/`False`.

Фаза 2:
* Главный флаг `enabled: auto`. Нет весов — это `skipped`, а не `failed`: RTMPose у L2 не скачан, и это не
  должно портить полноту. `enabled: 1` делает отсутствие бэкенда явным `failed` для замера.
* MediaPipe переписан на Tasks API (старого `mp.solutions` в 0.10.35 нет), кроп кассира + апскейл.
  Включается только явно, до замера H-A07-2.
* Немного надёжных сигналов: `phone_to_ear` и `cashier_facing_away` выключены по умолчанию (сверху ухо
  почти всегда закрыто, телефон и так ловит `phone_in_hand`; ориентация зависит от L/R и зон).
* Еда и питьё: предмет должен быть **у рта во время контакта кисти со ртом**. Стакан у руки в начале
  движения не считается: «стакан на столе + поправила волосы» иначе дало бы ложное питьё.
* Таймкод — начало перехода (`t_start`), пик — отдельно (`t_peak`); выбор политики — замером.
* `item_placed_on_desk` и `motion` — новые режимы, по умолчанию выключены. Признаки — только геометрия в
  масштабах тела и зоны из `zones`; без пиксельных координат, цвета одежды и яркости.
* `ml.pose.ab` пишет картинки только в локальную папку с `.gitignore` `*`.

Аудит (2055fc4):
* «Модуль импортируется» ≠ «поза доступна»: `availability()`/`model_info()` проверяют файл весов и пакет
  (`find_spec`, без загрузки); `verified` — только после реальной загрузки в процессе.
* Не выдаём «скелет» там, где позы нет: нулевые/неуверенные позы, точки за bbox трека, позы из невалидных
  bbox не попадают в выход; битые точки → score 0. Сбой посреди прогона — `failed`, а не `ok`.
* `item_placed_on_desk = 0.0` только при наличии позы в момент t; без поз — `None` (unknown).
* Оверлеи `ab`: без точек головы размывается весь человек — «верх bbox» на потолочной камере не голова.
