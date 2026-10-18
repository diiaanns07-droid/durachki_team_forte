# INTEGRATION_REPORT — облачная интеграция (A03), ветка `claude/integration-cloud`

Дата: 2026-10-17. Ветка создана от `origin/main` (`02b6e4d`, пустой корень A01). Все ветки вливались через
`git merge --no-ff --allow-unrelated-histories` в порядке ТЗ скриптом `scripts/integration_merge.py`
(повторяемо: `git fetch origin && python scripts/integration_merge.py`). Конфликты, если появятся,
разрешаются по владению папками (владелец папки побеждает; корень и `contracts/` — A01).

## 0. Третий проход (последний; ~09:10 UTC) — актуальное состояние

Все ветки снова влиты (A01 0209b98, A04 a97f69f, A05 dfee603, A06 353c826, A07 c790b45, A08 22f47e7, A13 d2a38db,
A09 3cb3002, A10 09a28a4, A11 5e802bc, A12 9ca9a0e, A03 938e658+, A02 10fcb27, A14 dca3b5b, A15 4e90ec2, A16 c298da3,
A17 88e985d, L1 ca6dc3d) — **конфликтов 0**; ещё 4 новых импорта `_contract_fallback` переведены на `contracts`.

**`pytest -m "not gpu and not slow"`: 1806 passed, 20 failed, 7 skipped, 22 deselected (219 с).**

Glue этого прохода — корневой `conftest.py` (изоляция окружения между модулями): `eval/tests/conftest.py` (A16)
при импорте ставит `SENTINEL_MOCK=1` на всю сессию, `testkit/e2e_stub.py` (A01) — во время теста; в общем прогоне
это включало mock у чужих модулей → защитный `mock_not_allowed` A02 (+29 ложных падений). Снимок `SENTINEL_*`
берётся при импорте корневого conftest и восстанавливается вокруг каждого теста вне `eval/`. Было 49 → стало 20.
Владельцам: A16 — ставить `SENTINEL_MOCK` через `monkeypatch`, A01 — `e2e_stub` не должен менять `os.environ`.

Найдено и исправлено в A03 по ходу интеграции: (1) `run()` в mock-конфиге мутировал `os.environ` (утечка mock в
процесс API) — убрано; (2) регрессия пробной интеграции A01 (дедлайн 40 с → ответ 110 с) — все RPC воркера
ограничены дедлайном; (3) `meta.coverage` переведён на канон A01 `compute_coverage` — без него A01 отдавал `failed`
вместо `no_violation`. Проверка в интеграции: `python -m backend.pipeline run <synth> --mock` → `done/no_violation`,
`coverage.ratio=1.0`, сериализатор A01 принимает; `compare --mock` проходит оба режима через процесс-воркер.

| Падающие тесты (20) | Чьё | Суть |
|---|---|---|
| `deploy/tests/test_merge_requirements.py::test_repo_merge_is_clean` | **A11**, A12 | `transformers<5` (ml/vlm) и `pillow<13` (ml/fusion) против лока L1 (5.20.1 / 13.0.0) — блокер VLM на L2 |
| `ml/vlm/tests/test_vlm_prompts.py` ×2, `ml/vlm/tests/test_vlm_compare.py` ×2 | A11 | промпт verify без type_id/названия; сравнение fixes/breaks |
| `copilot/tests/*` ×7 (падают и изолированно) | A14 | доступность/501 без весов, выбор llama.cpp, remote URL |
| `backend/api/tests/test_phase2.py::test_outcome_policy[no_violation-*]` ×2, `test_violation_without_timecode_is_dropped`, `test_mock_result_refused`; `test_jobs.py::test_pipeline_reporting_failed_or_garbage_is_failed` | A02 (+A01) | фикстуры без `meta.coverage` (канон A01 теперь требует его для `no_violation`); тексты ошибок |
| `eval/tests/test_eval_loo.py::test_logo_exploratory_and_nested` | A16 | `fit()` missing 'start', 'default' |
| `eval/tests/test_eval_run_split.py::test_run_inproc_end_to_end` | A16 (+A01) | фейковый пайплайн без `meta.coverage` → `failed` у сериализатора A01 |
| `ml/fusion/tests/test_core.py::test_no_catalog_reports_with_warning` | A12 | без каталога ожидается `violation` |

## 1. Что слилось (второй проход; история)

| # | Агент | Ветка | Голова | Конфликты |
|---|---|---|---|---|
| 1 | A01 | claude/gifted-bardeen-i404eh | a3b2b55 | 0 |
| 2 | A04 | claude/zealous-keller-occtqr | 731392a | 0 |
| 3 | A05 | claude/hopeful-hypatia-csegbf | 8c4d523 | 0 |
| 4 | A06 | claude/wizardly-pascal-w4vt3o | dd8c51c | 0 |
| 5 | A07 | claude/nice-gates-fjt926 | a98d0f8 | 0 |
| 6 | A08 | claude/practical-cerf-0dm6cc | 04b36ac | 0 |
| 7 | A13 | claude/vigilant-lovelace-ssecv4 | 9b3c239 | 0 |
| 8 | A09 | claude/wonderful-pasteur-wbu36i | 41c1407 | 0 |
| 9 | A10 | claude/upbeat-ride-3bylsq | 564859b | 0 |
| 10 | A11 | claude/inspiring-turing-2fm7rq | a4b4217 | 0 |
| 11 | A12 | claude/serene-bell-hycc21 | f0db43c | 0 |
| 12 | A03 | claude/elegant-knuth-yszqkw | 0146570 | 0 |
| 13 | A02 | claude/laughing-faraday-xe89m9 | 657424f | 0 |
| 14 | A14 | claude/compassionate-wright-9gha0v | d24e8a8 | 0 |
| 15 | A15 | claude/busy-pasteur-y15o46 | cc6ef09 | 0 |
| 16 | A16 | claude/wizardly-galileo-hg6mhp | 5df21d5 | 0 |
| 17 | A17 | claude/pensive-wright-5l5q0v | 91c8f6e | 0 |
| — | **A18** | **нет ветки** с коммитами `[A18]` | — | docs/ и pitch/ пусты (кроме docs/L1_LOCAL_SETUP.md от L1) |
| 18 | L1 | data/labels | ca6dc3d | 0 (только код tools/dataprep и docs, данных нет) |

Ветка → агент определена по префиксу `[Axx]` в сообщениях коммитов.

## 2. Glue-правки интегратора

* Удалены 14 fallback-копий контракта (`python scripts/check_fallbacks.py --drop`): все **побайтно совпадали**
  с `contracts/sentinel_contracts.py`.
* 26 строк импорта `from ._contract_fallback import …` / `from <пакет>._contract_fallback import …`
  (включая 3 новые после второго прохода: `ml/ingest/tests/test_ingest_coarse_fine.py`,
  `ml/rules/tests/test_interop.py`, `norms/validate.py`) заменены на `from contracts.sentinel_contracts import …`.
  Структура `try/except` владельцев сохранена (ветка `except` теперь импортирует тот же канон).
* `backend/pipeline/contract.py` (A03) — прямой импорт `contracts`.
* Запрещённых файлов (видео, веса, кадры датасета) нет. Есть `frontend/screenshots/*.png` (14 файлов, ≈5 МБ) —
  скриншоты UI на mock-данных (A15), не кадры датасета.

## 3. Тесты второго прохода (история): `pytest -m "not gpu and not slow"`

Окружение облака: CPU, Python 3.11.17, **без torch/transformers/весов**; numpy 2.4.6, **opencv 5.0.0** (лок L1:
`opencv < 5` — L2 должен перепрогнать на локе), PyYAML, jsonschema, fastapi, httpx, pillow, scipy, PyAV.

**Итог: 1323 passed, 9 failed, 6 skipped, 18 deselected (112 с).** Пропуски — тесты, требующие torch.

| Тест | Чьё | Причина (кратко) | Что делать |
|---|---|---|---|
| `deploy/tests/test_merge_requirements.py::test_repo_merge_is_clean` | **A11**, A12 (+A17 сторож) | `ml/vlm/requirements.txt`: `transformers<5` ↔ лок L1 **5.20.1**; `ml/fusion/requirements.txt`: `pillow<13` ↔ лок 13.0.0 | A11: проверить VLM на transformers 5.x API и снять `<5`; A12: снять `<13` |
| `ml/vlm/tests/test_vlm_prompts.py::test_render_verify_substitutes_everything[ru,en]` | A11 | в промпт verify не подставляются `type_id`/название типа (V05) | A11 |
| `ml/rules/tests/test_replay.py::test_replay_cli_end_to_end` | A09 | формат таблицы replay-отчёта изменился (`TP=1 FP=0 FN=0 TN=2` не найдено) | A09 |
| `ml/fusion/tests/test_core.py::test_no_catalog_reports_with_warning` | A12 | без каталога ожидается `violation`, получено `no_violation` (в интеграции каталог A13 есть — тест/логика) | A12 |
| `eval/tests/test_eval_loo.py::test_logo_exploratory_and_nested` | A16 | `TypeError: fit() missing 2 required positional arguments: 'start' and 'default'` — расхождение API внутри eval | A16 |
| `eval/tests/test_eval_run_split.py::test_run_inproc_end_to_end` | A16 (+A01) | с фейковым пайплайном 5 из 8 видео `failed` вместо 1 — в интеграции присутствует `contracts.serialize` (строгая полнота/вердикт) | A16: прогнать с каноном A01 |
| `backend/api/tests/test_media.py::test_copilot_501_then_router` | A02 | тест ожидает отсутствие copilot (501), а в интеграции A14 есть → 200 | A02: сделать тест независимым от наличия copilot |
| `backend/api/tests/test_jobs.py::test_pipeline_reporting_failed_or_garbage_is_failed` | A02 | текст ошибки для «мусорного» результата: ожидается `unusable result`, получено `pipeline reported status=failed` | A02 |

Тесты A03 (`backend/pipeline`) в интеграции зелёные (обе первоначальные ошибки исправлены в ветке A03).

## 4. Что L2 делать первым (по порядку)

1. **Зависимости по локу L1** (`tools/dataprep/requirements-l1.lock.txt`: torch 2.11.0+cu128, transformers 5.20.1,
   bitsandbytes 0.50.2, opencv<5). Конфликт `transformers<5` в `ml/vlm` — блокер для VLM (A11).
2. `pytest -m "not gpu and not slow"` на локе → сверить с таблицей выше; затем `-m gpu` и `-m slow`.
3. Сквозной mock: `python -m testkit.e2e_stub` и `python -m backend.pipeline run <видео> --mock --out out.json`.
4. Профиль реального пайплайна на 2–3 роликах (холодный и тёплый прогон):
   `python -m backend.pipeline profile <видео> --json > profile_<имя>.json` — источник бюджетов стадий
   (сейчас в `backend/pipeline/config.yaml` — плейсхолдеры, не измерения).
5. Сравнение режимов пайплайна (baseline vs coarse→fine) — команда в `backend/pipeline/README.md`
   (раздел «Замер для L2»).
6. `python -m eval.run --mode inproc …` (A16) на реальном пайплайне.

## 5. Как обновить

```bash
git fetch origin '+refs/heads/*:refs/remotes/origin/*'
git checkout claude/integration-cloud
python scripts/integration_merge.py          # все ветки по порядку; или подмножество: ... A03 A02
python scripts/check_fallbacks.py --drop      # если владельцы снова добавили копии
git grep -n "_contract_fallback import" -- '*.py'   # переписать на contracts.sentinel_contracts
pytest -m "not gpu and not slow" -q
```
