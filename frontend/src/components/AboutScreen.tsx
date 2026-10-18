import { AlertTriangle, BookOpen, Cpu, EyeOff, FileSearch, Info, ListChecks, Lock, Scale, Terminal } from "lucide-react";
import type { ReactNode } from "react";
import { useApp } from "../context";
import { DEFAULT_MODELS, TYPE_CATALOG } from "../lib/catalog";
import { cx, LocalBadge } from "./ui";
import s from "./SecondaryScreens.module.css";

// U10: «О системе» — назначение, существующий процесс анализа и реальные ограничения.
// Без обещаний точности, скорости и готовности: только то, что делает текущий код.

const PRINCIPLES: [typeof Lock, string, string][] = [
  [Lock, "Локальная обработка", "Видео и кадры анализируются на сервере в локальной сети. Облачные AI-сервисы не используются."],
  [EyeOff, "Без распознавания лиц", "Роли людей определяются по зоне, поведению и предметам. Головы на кадрах размываются, если их положение определено надёжно."],
  [FileSearch, "Только видимое", "Описание — наблюдаемые факты. Умысел, суммы, подлинность и личность по видео не устанавливаются."],
  [Scale, "Нормы из каталога", "Ссылка на норму берётся из каталога норм РК. Если подходящего пункта нет, в отчёте так и написано."],
];

const STEPS: { title: string; text: string; optional?: boolean }[] = [
  { title: "Загрузка", text: "Запись с потолочной камеры кассы (.mov или .mp4) ставится в очередь сервера." },
  { title: "Подготовка видео", text: "Кадры отбираются до 5 раз в секунду (для длинных записей — реже) с учётом поворота; время берётся из меток кадров." },
  { title: "Люди, предметы и сцена", text: "Детекция и трекинг людей и предметов, положение рук, состояние сейфа и двери, наличные на столе." },
  { title: "Правила и временная модель", text: "Правила находят эпизоды-кандидаты; временная модель строит кривую риска по ходу записи." },
  {
    title: "Проверка эпизодов локальной VLM",
    text: "Если модель включена и загружена, она перепроверяет эпизоды. Если нет, стадия помечается в отчёте как пропущенная.",
    optional: true,
  },
  { title: "Отчёт", text: "Вердикт, эпизоды с таймкодами, кадры-доказательства и ссылка на норму. Решение принимает проверяющий." },
];

const LIMITS: string[] = [
  "Результат — подсказка для проверяющего, а не юридическое заключение. Каждый эпизод нужно просмотреть по кадрам и видео.",
  "Если часть записи не проанализирована или анализ прервался, отчёт помечается как неполный или незавершённый — такое видео не считается «без нарушений».",
  "Тип нарушения и таймкод — предположение системы. Точный момент и квалификацию определяет человек.",
  "Качество зависит от ракурса камеры, освещения и перекрытий в кадре. На записях с другим углом обзора ошибки возможны чаще.",
  "Время обработки зависит от длины записи, оборудования сервера и очереди: задачи на видеокарте выполняются по одной.",
];

function SectionHead({ id, icon, title, lead }: { id: string; icon: ReactNode; title: string; lead?: ReactNode }) {
  return (
    <div className="mb-5">
      <h2 id={id} className="flex items-center gap-2.5 text-lg font-bold text-slate-900 dark:text-white">
        <span className="inline-flex h-8 w-8 flex-none items-center justify-center rounded-xl bg-brand-50 text-brand-600 dark:bg-brand-400/10 dark:text-brand-300" aria-hidden>
          {icon}
        </span>
        {title}
      </h2>
      {lead && <p className="mt-2 max-w-3xl text-sm leading-relaxed text-slate-600 dark:text-slate-300">{lead}</p>}
    </div>
  );
}

export default function AboutScreen() {
  const { lastResult, demo } = useApp();
  const origin = typeof window !== "undefined" ? window.location.origin : "http://<host>";
  const serverModels = Array.isArray(lastResult?.meta.models) ? (lastResult!.meta.models as unknown[]) : [];
  const models = serverModels.length
    ? serverModels.map((m) =>
        typeof m === "string"
          ? { name: m, license: "", purpose: "" }
          : {
              name: String((m as { name?: unknown }).name ?? "?"),
              license: String((m as { license?: unknown }).license ?? ""),
              purpose: String((m as { purpose?: unknown }).purpose ?? ""),
            },
      )
    : DEFAULT_MODELS;
  const modelsNote = serverModels.length
    ? demo
      ? "Синтетическое демо: список взят из демонстрационного отчёта, а не с сервера."
      : "По данным последнего анализа на этом сервере (meta.models)."
    : "Состав по архитектуре проекта. Это не подтверждение, что модель загружена на этом сервере: фактический список приходит в отчёте анализа.";

  return (
    <div className="animate-fade-in space-y-6">
      <section
        className={cx(
          s.hero,
          "rounded-3xl border border-slate-200/70 bg-white px-5 py-8 shadow-card dark:border-white/[.06] dark:bg-ink-900 sm:px-10 sm:py-12",
        )}
        aria-labelledby="about-title"
      >
        <div className={s.heroGrid} aria-hidden />
        <div className={s.heroGlow} aria-hidden />
        <div className="relative">
          <LocalBadge />
          <h1 id="about-title" className="mt-4 text-3xl font-extrabold leading-[1.1] text-slate-900 dark:text-white sm:text-[42px]">
            О системе{" "}
            <span className="bg-gradient-to-r from-brand-600 to-ink-600 bg-clip-text text-transparent dark:from-brand-300 dark:to-slate-200">
              Kassa Sentinel
            </span>
          </h1>
          <p className="mt-4 max-w-3xl text-[15px] leading-relaxed text-slate-600 dark:text-slate-300">
            Сервис помогает проверяющему просматривать записи потолочной камеры кассы. Он находит эпизоды с признаками возможных
            нарушений и показывает таймкод, кадры-доказательства и ссылку на норму законодательства РК. Вывод о нарушении делает
            проверяющий.
          </p>
          <ul className="mt-7 grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
            {PRINCIPLES.map(([Icon, title, text]) => (
              <li key={title} className="rounded-2xl border border-slate-200/80 bg-white/80 p-4 dark:border-white/[.07] dark:bg-white/[.02]">
                <Icon className="h-5 w-5 text-brand-600 dark:text-brand-300" aria-hidden />
                <h3 className="mt-2.5 text-sm font-bold text-slate-900 dark:text-white">{title}</h3>
                <p className="mt-1 text-[13px] leading-relaxed text-slate-600 dark:text-slate-400">{text}</p>
              </li>
            ))}
          </ul>
        </div>
      </section>

      <section className="card p-5 sm:p-7" aria-labelledby="about-process">
        <SectionHead
          icon={<ListChecks className="h-4 w-4" />}
          id="about-process" title="Как проходит анализ"
          lead="Шаги выполняются на сервере по порядку. Если стадия не выполнилась, это видно в отчёте."
        />
        <ol className={s.steps}>
          {STEPS.map((st, i) => (
            <li key={st.title} className={cx(s.step, st.optional && s.stepOptional)}>
              <span className={s.stepNum} aria-hidden>
                {i + 1}
              </span>
              <div className="min-w-0">
                <h3 className="text-sm font-bold text-slate-900 dark:text-white">
                  <span className="sr-only">Шаг {i + 1}. </span>
                  {st.title}
                  {st.optional && (
                    <span className="ml-2 inline-block rounded-full border border-slate-300 px-2 py-px align-middle text-[11px] font-semibold text-slate-500 dark:border-white/15 dark:text-slate-400">
                      если включена
                    </span>
                  )}
                </h3>
                <p className="mt-1 text-[13px] leading-relaxed text-slate-600 dark:text-slate-400">{st.text}</p>
              </div>
            </li>
          ))}
        </ol>
      </section>

      <section className="card p-5 sm:p-7" aria-labelledby="about-limits">
        <SectionHead icon={<AlertTriangle className="h-4 w-4" />} id="about-limits" title="Ограничения" />
        <ul className="-my-1 max-w-4xl">
          {LIMITS.map((t) => (
            <li key={t} className={cx(s.limit, "text-slate-700 dark:text-slate-300")}>
              <Info className="h-4 w-4 text-amber-600 dark:text-amber-300" aria-hidden />
              <span>{t}</span>
            </li>
          ))}
        </ul>
      </section>

      <section className="card p-5 sm:p-7" aria-labelledby="about-api">
        <SectionHead
          icon={<Terminal className="h-4 w-4" />}
          id="about-api" title="API для автоматической проверки"
          lead={
            <>
              Тот же анализ доступен программно. Нужен заголовок <code className="font-mono text-[13px]">X-API-Key</code>; без ключа сервер
              отвечает 401.
            </>
          }
        />
        <div className="grid gap-3 lg:grid-cols-2">
          {[
            ["Загрузить видео → job_id", `curl -H "X-API-Key: <API_KEY>" -F "file=@video.mov" ${origin}/api/v1/analyze`],
            ["Статус и результат", `curl -H "X-API-Key: <API_KEY>" ${origin}/api/v1/jobs/<job_id>`],
          ].map(([t, cmd]) => (
            <div key={t} className={s.code}>
              <div className="mb-2 text-[11px] font-semibold uppercase tracking-wider text-slate-400">{t}</div>
              <pre>{cmd}</pre>
            </div>
          ))}
        </div>
        <p className="mt-3 text-xs leading-relaxed text-slate-500 dark:text-slate-400">
          Ответ:{" "}
          <code className="break-words font-mono">
            {"{job_id, status, verdict, violations:[{type, timecode_sec, description, evidence_frames:[{timestamp_sec, image_base64}]}]}"}
          </code>{" "}
          и дополнительные поля (type_id, confidence, t_start, t_end, rules, timeline, meta). Пока анализ идёт, status = processing;
          при ошибке — failed с причиной.
        </p>
      </section>

      <div className="grid gap-6 lg:grid-cols-2">
        <section className="card p-5 sm:p-7" aria-labelledby="about-types">
          <SectionHead icon={<BookOpen className="h-4 w-4" />} id="about-types" title="Типы нарушений" lead="Справочник категорий, по которым размечаются эпизоды." />
          <ul className="divide-y divide-slate-100 dark:divide-white/5">
            {TYPE_CATALOG.map((t) => (
              <li key={t.id} className="flex gap-3 py-3">
                <span className="w-10 shrink-0 pt-px font-mono text-xs font-bold text-brand-700 dark:text-brand-300">{t.id}</span>
                <span className="min-w-0">
                  <span className="block text-sm font-semibold text-slate-900 dark:text-white">{t.name}</span>
                  <span className="mt-0.5 block text-xs leading-relaxed text-slate-500 dark:text-slate-400">{t.observable}</span>
                </span>
              </li>
            ))}
          </ul>
        </section>
        <section className="card p-5 sm:p-7" aria-labelledby="about-models">
          <SectionHead icon={<Cpu className="h-4 w-4" />} id="about-models" title="Модели и лицензии" lead={modelsNote} />
          <ul className="divide-y divide-slate-100 dark:divide-white/5">
            {models.map((m, i) => (
              <li key={i} className="flex items-center justify-between gap-3 py-3">
                <span className="min-w-0">
                  <span className="block break-words text-sm font-semibold text-slate-900 dark:text-white">{m.name}</span>
                  {m.purpose && <span className="mt-0.5 block text-xs text-slate-500 dark:text-slate-400">{m.purpose}</span>}
                </span>
                {m.license && (
                  <span className="chip shrink-0 border-emerald-200 bg-emerald-50 text-emerald-700 dark:border-emerald-400/20 dark:bg-emerald-400/10 dark:text-emerald-300">
                    {m.license}
                  </span>
                )}
              </li>
            ))}
          </ul>
        </section>
      </div>
    </div>
  );
}
