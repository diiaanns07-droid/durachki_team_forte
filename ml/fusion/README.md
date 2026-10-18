# ml/fusion — слияние, вердикт, кадры-доказательства, описания (A12)

Собирает события `rules` / `temporal` / `vlm` в итоговый `AnalysisResult`. Модуль:
- группирует и сливает события;
- выносит вердикт с проверкой полноты анализа;
- подбирает 1–3 кадра-доказательства с фактическим PTS;
- пишет описание только из наблюдаемых фактов;
- подставляет нормы из `norms/catalog.yaml`. Нормы не выдумываются; если пункт не найден, нарушение выдаётся честно «без пункта».

Моделей и весов нет (`models.json` = `[]`), GPU не нужен. Результат — подсказка проверяющему, а не юридическое заключение.

## Публичные функции

| Функция | Что делает |
|---|---|
| `ml.fusion.fuse(clip, events, risk, catalog, job_id) -> AnalysisResult` | Сигнатура заморожена контрактом v1. Всегда `status="done"` (`failed` — только аварийный режим без найденных нарушений). При неполном или неизвестном покрытии без нарушений `verdict=None`. |
| `ml.fusion.apply_completeness(result, stages) -> AnalysisResult` | Учитывает статусы стадий A03: упала обязательная стадия → `failed`, а не «чистый» вердикт. |
| `ml.fusion.result_to_jury(result) -> dict` | Строгий ответ жюри: `contracts.serialize.result_to_api` от A01, иначе fallback ровно по ТЗ. |
| `ml.fusion.result_to_ui_report(result) -> dict` | Полный отчёт для UI: подсказки, оверлеи, нормы со статусом сверки, полнота. |
| `ml.fusion.fuse_scores(...)` | Чистое слияние без кадров; используется в `tune`, `ab` и eval A16. |
| `python -m ml.fusion.tune ...` | Подбор порогов с калибровкой внутри фолдов (см. ниже). |
| `python -m ml.fusion.ab ...` | **Команда замера для L2**: сравнение вариантов флагов (см. ниже). |

## Фаза 2: что изменено и почему

Дисциплина: каждая гипотеза — отдельный коммит и флаг в `config.yaml`. **Новые режимы выключены**, пока L2 не измерит
выигрыш на реальных данных командой `ml.fusion.ab`. По умолчанию включены только исправления совместимости
и требования, сформулированные как правила («неполный анализ ≠ нет нарушений», «без выдуманных норм»).

| Изменение | Флаг | По умолчанию | Почему |
|---|---|---|---|
| Норма обязательна только для V99. Остальные типы с фактами выдаются «без пункта» (`rules: []`). | `report_gate.require_norm_types` | `[V99]` | В каталоге A13 у V01–V07 `rules: []`. Гейт фазы 1 «нужна норма» обнулил бы детекцию. По FAQ 32 пункт ищем, но не выдумываем. |
| Оценка A09 `extra.assessment`. Правило вида violation без проверенной нормы может стать нарушением, уверенность восстанавливается из `confidence_raw` / `trace.confidence_terms`. Правило-наблюдение и `insufficient_context` становятся подсказкой. | `report_gate.upstream_assessment`, `observation_policy`, `insufficient_policy`, `uncap_norm_review` | вкл, `hint`, `hint`, вкл | A09 срезает уверенность до 0.45 из-за отсутствия нормы, а это теряет ранжирование. Телефон, еда или дверь сами по себе — не нарушение (исключения A13). Неизвестно ≠ нет. |
| Покрытие глазами fusion (`meta.fusion.coverage`): по `clip.meta["coverage"/"complete"]` (A04) и сетке `clip.times`. Неполное или неизвестное покрытие без нарушений → `verdict=None`, `status="done"`. Канонический `meta.coverage` fusion не пишет. | `completeness.coverage_gate`, `min_decoded`, `min_sampled`, `max_gap_s`, `unknown_coverage` | вкл, 0.98, 0.9, 2 с, `no_verdict` | Failed или partial upstream не должны давать clean done. Подробнее — раздел «Полнота и вердикт». |
| Temporal (события и риск-кривая) не повышает скор кластера с фактическими источниками. | `temporal_mode` | `off` (`bonus`, `score` — для замера) | Разведка L1: SigLIP2 по кадру + max дал AUC 0.30–0.34, такой скор не отсеивает видео. Кластер только из temporal — подсказка. |
| «Ушла и оставила»: решение по столу и сейфу в окне после ухода. | `policies.v02_after_window.enabled` | **выкл** | Гипотеза. Ценности убраны и сейф закрыт → норма. Не видно (свет выключен) → без изменений. |
| Клиент напротив со своей стороны — норма, если нет второго постороннего и ценности закрыты. | `policies.v01_client_side.enabled` | **выкл** | Гипотеза. Неизвестная сторона → без изменений. |
| Отрицательные свидетельства (`extra.negative`) снижают скор один раз на кластер. | `policies.negative_evidence.enabled` | **выкл** | Гипотеза. |
| Кадры: ключевой момент и «до» / «после» реального действия. Для V02 — стол после ухода с подсветкой предмета. | `evidence.policy` | `sharpest` (`action` — для замера) | Резкость — только локальный выбор в ±0.2 с, не подмена ключевого момента. |
| Таймкод: `peak` — фактический кадр доказательства A09; `onset` — начало кластера. | `timecode.policy` | `peak` | Политика выбирается замером (`ab` считает попадание для обеих), не на глаз. |
| Подписи в UI-оверлее — легендой под кадром, у рамки только номер. | `evidence.ui_label_mode` | `legend` | Оверлей не должен закрывать действие. Кадр жюри всегда чистый. |
| Дедупликация кадров по максимуму блочной разности, а не по среднему. | `evidence.dup_block` | 4.0 | Локальное изменение (ушёл человек, осталась пачка) — не дубль. |
| Описания: стоп-лист суммы, подлинности, допуска и умысла. Названия типов из каталога A13. | `description.forbidden` | вкл | Не утверждаем невидимое. Дисклеймер «нельзя установить допуск» сохраняется. |

Исправлено попутно:
- подписи рамок в UI-оверлее рисовались на высоту шапки выше своих плашек;
- точный кадр берётся через A04 `frame_at_ts` / `read_exact` (фактический PTS);
- для ключей боксов `zone:*` и `visitor#7` (формат A09) — русские подписи.

## Как работает (кратко)

1. **Нормализация.**
   - Времена приводятся к границам видео, `confidence` — к 0..1, источник — к `rules|temporal|vlm|fusion|other`.
   - Неизвестный тип → `V99`; исходный тип сохраняется в `extra.orig_type_ids`.
   - Битые события пропускаются с предупреждением.
   - Учитывается оценка A09 `extra.assessment`.
2. **Группировка.** События одного типа с зазором ≤ `grouping.merge_gap_s` (3 с) объединяются — это дребезг одного события.
3. **Скор.** Источники коррелированы, поэтому скор — эвристика 0..1, а **не вероятность**.
   - `combine: max` от взвешенных скоров плюс ограниченный бонус согласия (≤ 0.10).
   - Temporal влияет на скор по `temporal_mode`.
   - VLM-отказ умножает скор на `vlm.reject_factor`.
4. **Политики** (флаги выше) меняют скор или переводят кандидата в подсказку с причиной; их след хранится в `extra.policies`.
5. **Гейт «нарушение».** Кандидат становится нарушением, если выполнено всё:
   - есть ≥ 1 наблюдаемый факт от rules/VLM без домыслов;
   - кластер не только из temporal;
   - не правило-наблюдение и не `insufficient_context`;
   - для типов из `require_norm_types` есть норма;
   - не подавлен более сильным нарушением другого типа;
   - скор ≥ τ[тип] из `thresholds.yaml`.

   Остальное попадает в `meta.ui.review_hints` с `reasons` / `reasons_ru`. Риск-кривая без фактов **никогда** не становится V99.
6. **Вердикт fusion** (окончательный статус решают A03 и сериализатор A01 — см. «Полнота и вердикт»).
   - `violation`, если есть хотя бы одно нарушение (в ответе ≤ 10: самые сильные, по времени).
   - `no_violation` — нарушений нет и покрытие глазами fusion полное.
   - `None` — нарушений нет, но покрытие неполное или неизвестно: «нет нарушения» fusion не утверждает.
7. **Таймкод** — по `timecode.policy`, привязан к фактическому PTS кадра (сетка `clip.times`), 0.1 с, в границах видео.
   Первый кадр-доказательство — ровно этот момент.
8. **Кадры.**
   - Жюри получает чистый исходный кадр, `timestamp_sec` = фактический PTS.
   - UI отдельно: `meta.ui.evidence_overlays[i]` с `frames[{timestamp_sec, role: key|before|after, image_base64}]`, `shapes`, `facts`, `norms`.
   - Головы размываются только по ключевым точкам или `head_boxes`.
   - JPEG ≤ 200 КБ.
9. **Описание.**
   - Шаблон каталога (`{t}`, `{facts}`…) или fallback: что видно → признаки возможного нарушения → «требуется проверка человеком».
   - В текст попадает только **сверенная** норма (`verified: true`). В текущем каталоге A13 таких нет, поэтому ссылок в тексте нет.
10. **Детерминизм.** При одинаковых входах отчёт побайтно одинаков, кроме `meta.fusion.timings`; это проверяет тест.

## Команда замера для L2

```bash
# 1) прогон пайплайна по датасету с сохранением ответов (A03 batch) → preds_raw/<id>.json с meta.scores_raw
# 2) замер вариантов флагов на тех же входах — без GPU, без повторного inference:
python -m ml.fusion.ab --preds preds_raw/ --labels data/labels.csv --cv 5 --out outputs/fusion_ab.json
# 3) (по желанию) кадры вариантов evidence для просмотра глазами — только локально, не в git и не в облако:
python -m ml.fusion.ab --preds preds_raw/ --labels data/labels.csv --only evidence_action \
       --videos data/videos --frames-out outputs/fusion_ab_frames
```

| Что | Где |
|---|---|
| Сравниваем | Варианты из `ml/fusion/experiments.yaml`: `baseline`, `temporal_bonus`, `temporal_score`, `v02_after_window`, `v01_client_side`, `negative_evidence`, `all_policies`, `observation_report`, `no_uncap`, `timecode_onset`, `evidence_action`. |
| Выход | `outputs/fusion_ab.json` и `.md` (таблица); кадры — в `outputs/fusion_ab_frames/<вариант>/<видео>/`. |
| Метрики | По уникальным видео (дубли — одна группа): FN, FP, полнота, доля FP, попадание таймкода \|Δt\| ≤ 1 с (peak и onset), точность типа (если в CSV есть колонка `type`). Считаются при порогах `thresholds.yaml` и OOF с калибровкой порогов внутри фолдов. Плюс задержка fusion (мс), задержка и VRAM пайплайна из meta A03, разница с baseline. |
| Правило включения | Вариант включается по умолчанию, если на OOF нет новых пропусков, FP не растёт, а таймкод или тип не хуже. Решает L2 / человек. |

Все 42 строки уже использованы в разведочной оценке: OOF-метрики — оптимистичная оценка, **не** независимый тест.

## Подбор порогов

```bash
python -m ml.fusion.tune --preds preds_raw/ --labels data/labels.csv --cv 5 --out ml/fusion/thresholds.yaml --report tune.json
```

- **`--mode cv` (по умолчанию).** Grouped stratified K-fold: пороги подбираются только на train-фолдах, OOF-метрики считаются на отложенных фолдах.
  Итоговые пороги — медиана порогов фолдов. Есть ещё `holdout` и `--fit-all` (последний только явно, с пометкой).
- **Цель.** Пропусков не больше допустимого (`--target-recall 1.0`), затем минимум `FP + dt_weight·(TP с |Δt| > dt_tol)`
  (по умолчанию 0.5 и 1 с). Внутри плато выбирается точка ближе к нижней границе.
- **Единица оценки — уникальное видео.** Дубли с одинаковыми входами объединяются в одну группу.
  Разные таймкоды у дублей (57 и 68 с) — конфликт: в штраф не идёт, в отчёте помечен «решает человек».
- **Разбор CSV гибкий:** `,` / `;` / таб, UTF-8 или cp1251, колонки видео / нарушение / секунда / тип, `Х` → `X`.

## Золотые случаи

`tests/golden/cases.json` (данные) и `tests/test_golden.py` покрывают:
- несколько событий и дребезг;
- конфликт источников (отказ VLM; temporal против rules);
- неизвестный тип или норму (V99 без нормы → подсказка; тип без нормы с фактами → «без пункта»);
- отсутствие фактов;
- крайний PTS;
- нечитаемые кадры;
- дубль JPEG;
- неполное покрытие;
- норм-ревью A09;
- правило-наблюдение A09.

Каждый случай дополнительно проверяется на детерминизм.

## Полнота и вердикт (согласовано с актуальными A01/A03)

- **Канонический `meta.coverage`** (`contracts.extensions.Coverage`: `ratio`, `analyzed_sec`, `duration_sec`, `unknown_spans`,
  `stages`, `required`, `complete`, `source`) считает A03 по статусам стадий. Сериализатор A01 (`resolve_outcome`) проверяет его
  перед `no_violation`; `done` без вердикта и без нарушений A01 превращает в `failed`. Fusion этот ключ **не пишет**:
  стадий fusion не видит, и подтверждать полноту за оркестратор было бы выдумкой.
- **Наблюдение fusion** — `meta.fusion.coverage`: `{checked, complete: true|false|null, reasons, decoded, sampled, max_gap_s,
  n_samples}`. При неполноте добавляются предупреждение и `meta.ui.incomplete`.
- **Почему не `status="failed"`.** `backend/pipeline/orchestrator.py::_fusion` (A03) принимает `status == "failed"` от fusion
  за падение модуля и подменяет весь результат встроенным `fallback_fuse`. Поэтому fusion отдаёт `done` + `verdict=None`.
  `failed` остаётся только для аварийного режима: исключение внутри fusion и ничего не найдено. Подмена fallback-ом там уместна.
- **Проверено тестами:** `test_upstream.py::test_partial_coverage_blocks_clean_verdict`, `test_fusion_never_returns_failed_for_coverage`,
  golden `partial_coverage_clean`. Ответ жюри для такого результата — `failed`, `verdict=null`.

## Каталог норм: None / пустой / fallback

| Что передали в `catalog` | Что делает fusion |
|---|---|
| Каталог с непустым `types` | Использует как есть, без перезагрузки и слияния с A13. |
| `None`, `{}`, `{"version": 1, "types": {}}` (так A03 передаёт недоступный каталог), мусор | `ml.fusion.catalog.load_norms_catalog()` → `norms.load_catalog()` (A13). |
| A13 недоступен или пуст | Встроенный fallback: только названия, шаблоны, серьёзность; `rules: []` (нормы не выдумываются). Предупреждение `norms catalog unavailable`. |

`load_norms_catalog` — единственная точка загрузки, тесты подменяют её monkeypatch-ем. Так тесты не зависят от наличия `norms`
в окружении. `tests/test_a13_catalog.py` проверяет fusion на **настоящем** каталоге A13 (в ветке A12 без `norms/` — skip):
- имена и нормы нарушений ровно из каталога, `verified` строго как в каталоге;
- в тексте цитируется только сверенная норма;
- V99 без нормы — подсказка;
- temporal-only сигнал не становится нарушением ни для одного типа.

## API подбора порогов (`ml.fusion.tune`) — схема для адаптера A16

Сигнатуры (не меняются):

```python
Cand = tuple            # (type_id: str, score: float, t_peak: float, t_start: float); старые 3-кортежи без t_start допустимы
@dataclass
class Sample:           # единица оценки — уникальное видео (группа дублей)
    video: str
    positive: bool                 # label == "violation"
    seconds: list[float]           # размеченные секунды события (для |Δt|); [] — нет таймкода
    cands: list[Cand]              # кандидаты, ПРОШЕДШИЕ гейт fusion (факт/норма/оценка A09/политики), без порогов
    group: str                     # группа дублей
    conflict: bool = False         # разные таймкоды у дублей → не штрафуется по Δt, «решает человек»
    type_id: Optional[str] = None  # тип из разметки (если есть) → type_accuracy
    members: list[str] = []

fit(samples, start, default, target_recall=1.0, bias=0.35, passes=4, dt_tol=1.0, dt_weight=0.5) -> dict[str, float]
    # start: {type_id: τ} — стартовые пороги (thresholds.yaml); default: τ для типов вне словаря
predict(sample, thr, default) -> list[Cand]          # кандидаты с score ≥ τ[type]; bool(...) == «видео — violation»
evaluate(samples, thr, default, dt_tol=1.0, policy="peak") -> dict  # fn/fp/recall/fp_rate/timecode_hit_rate/type_accuracy
candidates_for(VideoPred(video, raw=scores_raw, fingerprint=group), cfg, types) -> (list[Cand], seconds_spent)
```

Корректный вызов на строках `eval.run --raw` (`scores_raw.jsonl`: `video_id, group, label, gt_times, scores_raw`).
Этот рецепт проверяет тест `test_tune.py::test_documented_adapter_recipe_on_a16_rows`:

```python
from ml.fusion.catalog import resolve_catalog
from ml.fusion.config import load_config, load_thresholds, threshold_for
from ml.fusion.tune import Sample, VideoPred, candidates_for, fit, predict

cfg, thr0 = load_config(), load_thresholds()
default = float(thr0["default"])
start = {t: threshold_for(thr0, t) for t in thr0["per_type"]}
types, _ = resolve_catalog(None)            # каталог A13 (или fallback без норм)

def to_sample(row):
    pred = VideoPred(row["video_id"], row["scores_raw"], fingerprint=row.get("group") or row["video_id"])
    cands, _ = candidates_for(pred, cfg, types)
    return Sample(video=row["video_id"], positive=row["label"] == "violation",
                  seconds=[float(t) for t in row.get("gt_times") or []], cands=cands,
                  group=row.get("group") or row["video_id"])

thr = fit([to_sample(r) for r in train_rows], start, default)     # → {type_id: τ}
is_violation = bool(predict(to_sample(row), thr, default))
```

Для `eval/loo.py` с его `fit(rows) -> params` и `predict(params, row) -> bool` адаптер в зоне A16 — обёртки вида
`lambda rows: fit([to_sample(r) for r in rows], start, default)` и `lambda thr, row: bool(predict(to_sample(row), thr, default))`.

**Поля, нужные корректному адаптеру:**
- `scores_raw.inputs.events`, `scores_raw.inputs.risk`, `scores_raw.duration` — для пересчёта слияния текущим конфигом
  (пишутся fusion с фазы 1). Без `inputs` используется `scores_raw.candidates[]`: `type_id, confidence, t_peak, t_start,
  gate, suppressed_by`. Это решения гейта на момент прогона, смена флагов на них не влияет.
- `label` ∈ {violation, no_violation}; строки без метки — не обучающие.
- `gt_times` — секунды события; `group` — группа дублей.
- **Отсутствует, нужно для полной метрики:** тип из разметки (`gt_type` → `Sample.type_id`) и флаг конфликта таймкодов
  у дублей (`conflict`; `tune.build_samples` выводит его сам из разных секунд в группе).

**Ограничения преобразования сырых скоров:**
1. `candidates_for` пересчитывает слияние **текущими** `config.yaml` и каталогом. Для воспроизводимости фиксируйте их версии.
2. Скоры эвристические по коррелированным источникам. Пороги — точки отсечения, не «вероятность ≥ τ».
3. Полнота не моделируется. `predict` отвечает «есть ли нарушение»; пустой ответ ≠ `no_violation`, если покрытие неполное
   (API даст `failed`). Для метрик вердикта сверяйте со `status`/`verdict` ответа.
4. `fuse_scores` не видит кадров: таймкод кандидата не привязан к PTS сэмпла. Отличие от `timecode_sec` ответа API — не больше
   половины шага сэмплирования (0.1 с при 5 fps).
5. В `cands` только кандидаты, прошедшие гейт: подсказки (`review_hints`) в подбор порогов не входят.

## Запуск и проверки

```bash
pip install -r ml/fusion/requirements.txt
python -m pytest ml/fusion/tests -q     # без GPU, видео и весов: синтетические клипы
```

136 тестов (+1 skip без `norms`) проходят на Python 3.13 и на Python 3.11.17 с версиями из lock L1
(`pillow 13.0.0`, `numpy 2.4.6`, `opencv-python-headless 4.14.0.94`, `PyYAML 6.0.3`, `pytest 9.1.1`; облачный контейнер, CPU).
На копии `claude/integration-cloud` (2055fc4) с изменениями A12 — 158 passed (Python 3.13 и 3.11.17 с lock L1),
включая настоящий каталог A13.
Измерений на реальных данных в этой сессии **нет**. Пороги, веса и окна — стартовые, их меряет L2.

## Конфиги и env

- `config.yaml` — все флаги и пороги. Отсутствующие ключи берутся из `config.py`, синхронность проверена.
- `thresholds.yaml` — τ по типам.
- `experiments.yaml` — варианты для `ab`.
- Env:
  - `SENTINEL_FUSION_CONFIG`, `SENTINEL_FUSION_THRESHOLDS` — пути к конфигам;
  - `SENTINEL_FUSION_NO_FRAMES=1` — без кадров (eval/тюнинг);
  - `SENTINEL_FONT` — TTF с кириллицей.

## Форматы, которые fusion понимает в `Event.extra`

| Ключ | Кто | Смысл |
|---|---|---|
| `assessment`, `assessment_reasons`, `norm_status`, `trace.confidence_terms` | A09 | Оценка правила и уверенность до потолка; предложение A09: добавить `confidence_raw`. |
| `evidence_t` | A09 | ≤ 3 фактических момента для кадров (политика `action`). |
| `vlm = {"verdict": confirm\|reject\|unsure, "score"}` / `vlm_verdict` / `vlm_score` | A11 | Мнение VLM; `rejected` тоже понимается. |
| `state_after = {cash_visible, cash_bag_visible, safe_open, t_after, valuables_bboxes}` | A08/A09 | Стол и сейф после ухода (V02). `None` = неизвестно. Fallback — `trace.signals`. |
| `person_side` / `zone`, `n_outsiders`, `valuables_open` | A06/A09 | Клиент напротив (V01). |
| `negative = {"сейф закрыт": 0.9}` или `[{"fact", "strength"}]` | A08/A09/A11 | Отрицательные свидетельства. |
| `keypoints` / `poses` / `head_boxes` | A07 | Размытие голов. |
| `type_probs` | A10 | «Лучший тип» подсказки по риск-кривой. |

## Ограничения

- `confidence` — эвристический скор коррелированных источников, не вероятность.
- Политики фазы 2 работают только при наличии полей upstream. Без них поведение не меняется: неизвестно ≠ нет.
- Кадры: боксы относятся к `t_peak`; на кадре «после» рисуются только `state_after.valuables_bboxes`.
- Без ключевых точек головы на кадрах не размываются.

## Для интегратора

1. **A03 (`backend/pipeline/orchestrator.py`).** Канонический `meta.coverage` и `coverage_gate` теперь у A03, и fusion
   с этим согласован: `failed` из-за покрытия больше не отдаёт, так как `_fusion` принял бы его за падение и включил `fallback_fuse`.
   - Остаётся: при `complete=True` `_finalize` превращает `verdict=None` от fusion в `no_violation`. Если fusion увидел
     неполное покрытие, которого не видит `canonical_coverage` (`meta.fusion.coverage.complete is False`, например разрыв
     в `clip.times` > 2 с), это стоит добавить в `ctx.incomplete`.
2. **A09:** класть в `extra.confidence_raw` уверенность до потолка review/insufficient. Сейчас она восстанавливается из `trace`.
   `state_after` / `person_side` / `negative` — по таблице выше, по мере готовности сигналов A08.
3. **A04:** используются `frame_at_ts`, `read_exact` и `clip.meta["coverage"/"complete"]`. Предложение A01 из README A04 поддерживаю.
4. **A02:** ответ жюри — `contracts.serialize.result_to_jury_api` (A01; `ml.fusion.result_to_jury` делегирует ему).
   `failed` показывать как технический отказ, не как «нарушений нет».
   **A01, граница:** `contracts/serialize.py::_jury_description` дописывает в описание жюри «Норма: … — URL» из `rules[0]`,
   не проверяя `verified`. В каталоге A13 у V08/V09 нормы с `verified: false`, и они уйдут жюри без пометки «требует сверки».
   - Вход: нарушение V08 с `rules=[{verified: false, url: …}]`.
   - Ожидаемо: ссылка только при `verified: true`, либо с пометкой «(не подтверждено, требует сверки)».
5. **A15:** `meta.ui.evidence_overlays[i].frames[*].role` (`key|before|after`), `shapes[*].frame_role`; новые причины подсказок:
   `observation_only`, `insufficient_context`, `valuables_removed`, `client_side`; `meta.ui.incomplete`, `meta.fusion.coverage`.
   Ключи фазы 1 сохранены.
6. **A16:** `eval/loo.py` вызывает `tune.fit(rows)` / `tune.predict(params, row)` — сигнатуры другие, отсюда `TypeError`
   в `eval/tests/test_eval_loo.py::test_logo_exploratory_and_nested`. Падение воспроизводится и на чистом 2055fc4.
   Адаптер пишет A16 по схеме из раздела «API подбора порогов». `scores_raw` совместим с фазой 1 (добавлены
   `candidates[].assessment`). Для сравнения вариантов fusion есть `ml.fusion.ab`.
7. **A13:** fusion читает `types.<id>.{name, description_template, severity, rules[]}`. Норма в тексте описания — только при `verified: true`.
8. **A17:** в Docker нужен `fonts-dejavu-core`, иначе подписи UI-оверлея будут транслитом.
9. **История веток.** Ветка начинается от коммита контракта A01, `main` — пустой отдельный корень, PR в `main` GitHub не создаёт.
   Ветка влита в `claude/integration-cloud` (там интегратор удалил `_contract_fallback.py` и поправил `_types.py`). В ветке A12
   эти два файла не менялись после `9ca9a0e`, поэтому слияние не конфликтует.
10. **Зависимости:** `Pillow>=10.0,<14` совместим с lock L1 (`pillow==13.0.0`).
    `python deploy/merge_requirements.py --check`: на 2055fc4 exit 1 (конфликт pillow), с изменением A12 — exit 0, конфликтов нет.
