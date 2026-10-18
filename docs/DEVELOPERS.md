# Kassa Sentinel

Локальный AI-сервис для проверки видео с потолочной камеры кассы (Datathon ForteBank 2026).

Проверяющий загружает `.mov` и получает:

* вердикт `violation` / `no_violation`;
* список возможных нарушений: тип, таймкод, описание «что видно и почему это может быть нарушением», 1–3 кадра-доказательства (JPEG);
* ссылку на норму РК (документ, пункт, URL adilet.zan.kz).

Результат — подсказка для человека, а не юридическое заключение. Вся обработка видео идёт локально: облачные LLM/vision API не используются, в рантайме нет интернета. Разрешены только модели под Apache-2.0, MIT или BSD.

Подробная документация (архитектура, модели и лицензии, нормы, запуск, раздел 8 ТЗ) — в [`docs/`](docs/) (A18). Питч — в [`pitch/`](pitch/).

## API (как в ТЗ)

| Метод | Что делает |
|---|---|
| `POST /api/v1/analyze` | multipart-поле `file`, заголовок `X-API-Key` → `{"job_id": "..."}` |
| `GET /api/v1/jobs/{job_id}` | `{"job_id", "status", "verdict", "violations": [{"type", "timecode_sec", "description", "evidence_frames": [{"timestamp_sec", "image_base64"}]}]}` |

Без верного ключа — 401. Схема ответа: [`contracts/api_schema.json`](contracts/api_schema.json). Проверка ответа: `python -m contracts.validate --jury resp.json`.

## Быстрый старт

```bash
make setup-dev          # контракт, тест-кит, pytest, ruff
make test-fast          # тесты без GPU и без весов (без глобального SENTINEL_MOCK)
make e2e                # сквозной mock-пайплайн на синтетическом видео
make e2e-full           # полная сборка: backend.pipeline.run + сериализатор жюри (нужны все модули)
make setup && make weights && make run   # полный запуск (веса качаются один раз, дальше офлайн)
```

Windows / Docker / Cloudflare Tunnel — см. [`deploy/`](deploy/) (A17).

## Структура и владение (каждый агент пишет только в свою папку)

| ID | Папка | Модуль |
|---|---|---|
| A01 | `contracts/`, `testkit/`, `.github/`, `scripts/`, корневые файлы | Контракты, скелет, тест-кит, CI |
| A02 | `backend/api/` | FastAPI, API-ключ, очередь задач |
| A03 | `backend/pipeline/` | Оркестратор, бюджет времени, кэш, CLI |
| A04 | `ml/ingest/` | Декодирование видео, сэмплинг, качество, аугментации ракурса |
| A05 | `ml/detect/` | Детекция объектов (closed-set и open-vocab) |
| A06 | `ml/track/` | Трекинг, роли людей, авто-зоны |
| A07 | `ml/pose/` | Поза и жесты рук |
| A08 | `ml/scene/` | Состояния сцены: сейф, дверь, наличные, присутствие |
| A09 | `ml/rules/` | Движок правил → события-нарушения |
| A10 | `ml/temporal/` | Временная модель (SigLIP2 + temporal head) |
| A11 | `ml/vlm/` | Локальная VLM: верификация и поиск новых типов |
| A12 | `ml/fusion/` | Слияние, вердикт, кадры-доказательства, описания |
| A13 | `norms/` | Нормативка РК, каталог нарушений |
| A14 | `copilot/` | Локальный LLM-ассистент проверяющего |
| A15 | `frontend/` | Веб-интерфейс |
| A16 | `eval/` | Метрики как у жюри, LOO, стресс-тест ракурса |
| A17 | `deploy/` | Docker, офлайн-веса, Windows, Cloudflare Tunnel |
| A18 | `docs/`, `pitch/` | Документы для сдачи, питч |

Общие пустые файлы `ml/__init__.py` и `backend/__init__.py` создавать может любой агент.

## Для разработчиков модулей

* **Контракт:** [`contracts/README.md`](contracts/README.md). Там описано, как подключиться, оба сериализатора (ответ жюри и UI-report) и политика вердикта. Что изменилось — в [`contracts/MIGRATION.md`](contracts/MIGRATION.md).
* **Тест-кит без реальных видео:** [`testkit/`](testkit/README.md). Там синтетическое видео кассы, `FakeClip` и фикстуры Detection/Track/Zone/SceneState по сценариям.
* **Тесты:**
  * `pytest <папка>` из корня, `pythonpath=["."]`;
  * тяжёлые тесты помечать `@pytest.mark.gpu` / `@pytest.mark.slow`;
  * mock включается только явно: `SENTINEL_MOCK=1` в команде прогона или `monkeypatch.setenv` в тесте. На всю сессию pytest его не ставят ни CI, ни `make test`: иначе результат зависит от порядка тестов.
* **CI (`.github/workflows/ci.yml`):**
  * `core` — контракт, тест-кит, скрипты: тесты без `SENTINEL_MOCK` и повторно с ним (политика mock не зависит от окружения), примеры ТЗ, лицензии, fallback-копии и висячие импорты, медиа;
  * `modules` — матрица только по модулям, которые есть в ветке; модуль без тестов — ошибка;
  * `e2e-stages` — контракт каждой стадии (`--strict-stages`), частичная сборка допускается и так и подписана;
  * `e2e-full` — полная сборка: `backend.pipeline.run` + `result_to_jury_api`, явный тестовый mock → `done`, без mock → `failed` (весов в CI нет). Запускается, только если в ветке есть оркестратор и модули критического пути, иначе job пропущен (серый), а не зелёный;
  * необязательные: `frontend`; `ownership` для PR без тега `[Axx]` только информирует.
* **Интеграция:** порядок мерджа и чек-лист — [`scripts/merge_order.md`](scripts/merge_order.md). Проверка чужих файлов — `scripts/check_ownership.py`. Лицензии, облачные API и файлы в git — `scripts/check_licenses.py`. Сверка fallback-копий контракта с каноном — `scripts/check_fallbacks.py`.
* **В git не коммитим:** видео, кадры из датасета, веса (`*.mov`, `*.mp4`, `*.onnx`, `*.pt`, `*.safetensors`, `*.gguf`…), ключи и `.env`. См. `.gitignore`.

## Решения A01 (фаза 2)

- **«Нет нарушения» — только при подтверждённой полноте `meta.coverage`.** Без неё, при неполном или упавшем анализе ответ — `failed` с причиной. Это инвариант, а не гипотеза. Оркестратору — `contracts.extensions.compute_coverage`.
- **Production сам на mock не переключается.** Результат с `meta.mock` или стадией `mock` без явного `SENTINEL_MOCK=1` отдаётся как `failed`. Фикстуры и e2e тест-кита помечены как mock явно.
- **Неизвестно ≠ нет:** ненаблюдавшийся сигнал отсутствует в `signals`, `0.0` означает «видели, что нет».
- **Гипотезы — за флагами в `contracts/config.yaml`, по умолчанию выключены**, пока L2 не измерит выигрыш. Сейчас это `jury.embed_rule_in_description: false` (в фазе 1 было включено).
- **`result_to_api` = UI-report**, как в контракте; ответ жюри — `result_to_jury_api`. Строгий ответ по умолчанию ломал кэш A03 и `eval` A16 — это показала пробная интеграция.
- **Таймкод** — начало действия; если `timecode_sec` не конечен, берётся `t_start`. Сдвиг `t_peak` подбирается замером L2 (`signed_median` в отчёте), а не фиксированным +0.5 с.
- **`.gitignore` — только якорные шаблоны каталогов данных** (`/data/` и т.п.). Медиа, веса и файлы > 5 МБ блокирует `scripts/guard_media.py`: в pre-commit и в CI.
- **Зависимости — от lock L1:** `scripts/resolve_deps.py` показывает конфликты и не подставляет максимальные версии (`scripts/DEPS_REPORT.md`).
- **Совместимость веток** — `contracts/COMPAT.md`: матрица генерируется `scripts/compat_matrix.py`, к ней приложены записки владельцам.
- **CI строгий:** e2e с `--strict-stages` — модуль, нарушивший контракт, роняет обязательную проверку. Полная сборка проверяется отдельно (`e2e-full`, `--require-full`): без `backend.pipeline.run` это «NOT RUN» (код 3), а не зелёный пропуск.

### Команда замера для L2 (без GPU, по готовым результатам пайплайна)

```bash
python scripts/l2_contract_check.py --preds <папка JSON-результатов прогона, напр. preds_raw от eval A16> \
    --labels <CSV video,label,second> --out reports/a01_l2_contract.json \
    --compare-set jury.embed_rule_in_description=true
```

| Что сравниваем | Метрики |
|---|---|
| База (флаги по умолчанию) против одной гипотезы `--compare-set` | доля валидных строгих ответов жюри; статусы и причины `failed` (сколько «нет нарушения» не подтверждено `meta.coverage`); TP/FP/FN/TN и `failed` по разметке; таймкод (\|Δ\| медиана и p90, попадание ±1/±2/±3 с, знаковый сдвиг); latency (`meta.timings.total`); пик VRAM (`meta.vram_events`); воспроизводимость (`--repeat` со вторым прогоном) |

Цифр без этого замера A01 не приводит.
