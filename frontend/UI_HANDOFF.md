# UI — передача L2 (U01, единственный исполнитель UI)

Ветка `claude/laughing-faraday-xe89m9`. Во фронтенде здесь только изменённые файлы; полной копии `frontend/` нет.
Только синтетические данные (`?demo=1` и мок-бэкенд e2e), реальные видео и кадры не использовались.

## 1. Что уже запушено по UI (проверено `git fetch` 2026-10-17 ~12:12 UTC)

| Роль | Ветка | SHA | Файлы |
|---|---|---|---|
| UI-база | `claude/busy-pasteur-y15o46` (A15) | `90083a0` | весь `frontend/` (на ней построены U07 и U11) |
| U01 | `claude/laughing-faraday-xe89m9` | см. п. 3 | App, Header, ui, index.css, tailwind.config.js + экраны из п. 3 |
| U07 | `claude/serene-bell-hycc21` | `bfa711d` | ViolationCard.tsx, ViolationCard.module.css, NormCard.tsx |
| U08 | `claude/vigilant-lovelace-ssecv4` | `667a6cc` | CopilotPanel.tsx, CopilotPanel.module.css, tests/copilotPanel.test.tsx |
| U11 | `claude/pensive-wright-5l5q0v` | `b84edc2` | CompletenessPanel.tsx, RetryUpload.tsx, u11.module.css, tests/u11.test.tsx |

U02–U06, U09, U10, U12, U13: изменений во `frontend/` нет ни в одной удалённой ветке.
`origin/integration@33eebd7`: фронтенд такой же, как в `922cb8a`. В нём нет компонентов A15 (CompletenessPanel, EpisodeNavigator,
Evidence, RetryUpload, ReplayScreen, uiConfig), поэтому финальный UI собирается на базе A15 `90083a0`.

## 2. Стартовая сцена

Отдельной стартовой сцены нет ни в одной ветке: integration `33eebd7`, integration-cloud `2055fc4`, A15 `90083a0` и все `claude/*`.
U07, U08 и U11 под «стартовой сценой» понимали текущий визуальный язык: сетку `bg-grid`, тонкие линии, Manrope, бренд в знаке.
Замену я не придумывал. Как она защищена:
- все стили рабочей области действуют только внутри `.ks-ws` (корень App и порталы модалок); глобальные base-правила
  и прежние классы (`.card`, `.btn-*`, `.chip`, `.label`, `.input`) не изменены;
- стартовую сцену нужно рендерить **вне** `.ks-ws` (соседом корня App). Тогда она выглядит как раньше.
  Если положить её внутрь `.ks-ws`, переопределения рабочей области (кнопки, чипы) коснутся и её.

## 3. Сборка (поверх A15 `90083a0`)

```bash
git checkout 90083a0 -- frontend                       # UI-база A15
git checkout bfa711d -- frontend/src/components/ViolationCard.tsx frontend/src/components/NormCard.tsx
git checkout 667a6cc -- frontend/src/components/CopilotPanel.tsx frontend/src/components/CopilotPanel.module.css frontend/tests/copilotPanel.test.tsx
git checkout b84edc2 -- frontend/src/components/CompletenessPanel.tsx frontend/src/components/RetryUpload.tsx frontend/src/components/u11.module.css frontend/tests/u11.test.tsx
U=origin/claude/laughing-faraday-xe89m9               # ViolationCard.module.css здесь = U07 bfa711d + замена цветов на токены
git checkout $U -- frontend/src/App.tsx frontend/src/components/Header.tsx frontend/src/components/ui.tsx frontend/src/index.css \
  frontend/tailwind.config.js frontend/src/lib/format.ts frontend/src/components/RiskTimeline.tsx frontend/src/components/VideoPlayer.tsx \
  frontend/src/components/ResultView.tsx frontend/src/components/ProcessingView.tsx frontend/src/components/UploadScreen.tsx \
  frontend/src/components/ViolationCard.module.css frontend/scripts/e2e-mock.mjs frontend/scripts/ui-mobile.mjs
cd frontend && npm ci && npm run build && npx vitest run && npm run e2e:mock && node scripts/ui-mobile.mjs
```

Результат на этом наборе: tsc и vite build OK; vitest 111/111; e2e:mock «Все мок-e2e проверки прошли» (36 шагов);
ui-mobile 20/20 (375 и 768 px × светлая/тёмная × загрузка, обработка, результат, чистый результат, «О системе»).
Сообщение `ECONNREFUSED … /video` в логе e2e ожидаемо: видео в моке не отдаётся.

## 4. Токены (`frontend/src/index.css`, внутри `.ks-ws`; тёмная тема — `.dark .ks-ws`)

| Группа | Токены (светлая → тёмная) |
|---|---|
| Поверхности (RGB) | `--ks-bg` slate-50 → ink-950 · `--ks-surface` white → ink-900 · `--ks-surface-2` ink-50 → 24 33 56 · `--ks-surface-3` ink-100 → ink-800 |
| Текст (RGB) | `--ks-text` slate-900 → slate-100 · `--ks-text-2` slate-600 → slate-300 · `--ks-text-3` slate-500 → slate-400 |
| Границы (цвет) | `--ks-border` slate-200 → white/8% · `--ks-border-strong` slate-300 → white/14% |
| Акцент (RGB) | `--ks-accent` ink-800 → slate-100 (действия, активная навигация) · `--ks-accent-hover` · `--ks-accent-contrast` · `--ks-accent-soft` · `--ks-focus` ink-500 → ink-300 |
| Событие/ошибка | `--ks-event` rose-600 → rose-400 · `--ks-event-text` · `--ks-event-contrast` (white → ink-950) · `--ks-event-soft` · `--ks-event-border` |
| Успех (только подтверждённый) | `--ks-success` emerald-600 → emerald-400 · `-text` · `-soft` · `-border` |
| Предупреждение / справка | `--ks-warning*` amber · `--ks-info*` ink (нейтральный) · `--ks-overlay` |
| Радиусы | `--ks-radius-sm` 8 · `-md` 12 (кнопки, поля) · `-lg` 16 (карточки, модалки) · `-pill` |
| Расстояния | `--ks-space-1…16` = 4, 8, 12, 16, 20, 24, 32, 40, 48, 64 px |
| Компоновка | `--ks-content-max` 1320 · `--ks-gutter` 16/24/32 (от 640/1024 px) · `--ks-header-h` 64 · `--ks-section-gap` 24 · `--ks-control-h` 40 (sm 32, lg 48) |
| Шрифты, тени | `--ks-font-sans` Inter Variable · `--ks-font-display` Manrope Variable (локальные @fontsource) · `--ks-shadow-card/lift/action` |

Классы Tailwind на токенах: `bg-ks-*`, `text-ks-*` (bg, surface, surface-2/3, text, text-2/3, accent, accent-contrast, focus,
event, event-text, event-contrast, success…, warning…, info…), `border-ks`, `border-ks-strong`,
`border-ks-{event|success|warning|info}` (это **мягкая** граница; для сплошного цвета: `border-[rgb(var(--ks-event))]`),
`rounded-ks-{sm|md|lg}`, `max-w-ks-content`, `shadow-ks-{card|lift|action}`.

## 5. Правила компоновки и цвета

- Корень: `.ks-ws.ks-ws-page`; контент в `.ks-container` (1320 px, поля 16/24/32), `main.ks-main` (сверху 24, снизу 64); шапка `.ks-header` 64 px.
- Кнопки: на блок одно главное действие `btn-primary`/`btn-brand` (акцент, сплошная); вторичное `btn-ghost`; третичное `btn-quiet`;
  удаление `btn-danger`; действие над событием `btn-event`; размеры `btn-sm`/`btn-lg`; иконка `ks-icon-btn` (36 px).
- Статусы: `chip` + `chip-event|success|warning|info|muted` (или `StatusChip`); встроенные уведомления — `Notice`/`.ks-notice[data-tone]`;
  тосты `.ks-toast[data-tone]`, на экране не больше трёх.
- Цвет: красный — только события (эпизоды, маркеры, вердикт «нарушение») и ошибки; зелёный — только подтверждённый успех
  (полнота 100 %, завершённая стадия, сверенная цитата, чистый вердикт с полнотой); янтарный — предупреждения, демо, неполный
  или незавершённый анализ; нейтральный — справка («локально»); малиновый бренд — только знак и градиент заголовка на главной.
- Шкала индекса риска: нейтральный → янтарный → красный, без зелёного (низкий индекс не означает «всё в порядке»).
- Свой CSS внутри рабочей области: селектор `:where(.ks-ws) .x` (в тёмной теме `.dark :where(.ks-ws) .x`). Утилиты на элементе важнее.
  Не включайте `.btn` в такие селекторы: `@apply btn` в прежних классах копирует каждое правило с `.btn`.
  Порталу нужен класс `ks-ws` на корне (Modal это делает).

## 6. Оставшиеся визуальные расхождения (не правил: чужая готовая работа или второстепенное)

- U08 `CopilotPanel.module.css`: бренд в аватаре (69), чипах (168–170, 184), линии ответа (228), 349, 363, 368. Это не событие;
  при желании заменить на `--ks-accent`.
- U11 `u11.module.css:106`: прогресс градиентом бренда; лучше `rgb(var(--ks-accent))`. `CompletenessPanel.tsx:96` — иконка в бренде.
- U07: `ViolationCard.tsx:280` иконка в бренде; в `ViolationCard.module.css` ссылки, `.docLink:hover` и линия цитаты в бренде (акцент, не событие).
- Второстепенные экраны (работают, не трогал): `AboutScreen.tsx` brand-300 (39, 55, 92) и зелёный чип лицензии (120) — это не «успех»;
  `BatchList.tsx:43` — зелёный нужно сверить с подтверждённым исходом.
- Незавершённый анализ (failed) остаётся янтарным, а не красным: так решили A15 и аудит A02, на это опирается e2e.

## 7. Не запускалось (NOT RUN)

- `npm run e2e:real`: нужен живой backend с моделями и реальное видео, это локально у L2.
- Сверка со стартовой сценой: в ветках её нет (п. 2).
