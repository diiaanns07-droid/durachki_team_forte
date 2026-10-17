import { useSyncExternalStore, type ComponentType } from "react";
import {
  AlertTriangle,
  CheckCircle2,
  ChevronRight,
  CircleDashed,
  Clock3,
  Download,
  Eraser,
  FlaskConical,
  HelpCircle,
  Layers,
  OctagonAlert,
  UploadCloud,
  XOctagon,
} from "lucide-react";
import { batch, type BatchItem } from "../lib/batch";
import { formatBytes, formatTimecode, plural } from "../lib/format";
import { stageLabel } from "../lib/stages";
import { cx, ProgressBar } from "./ui";
import s from "./BatchList.module.css";

// Пакетная очередь (U09). Всё на экране выводится из фактического состояния batch-стора (A15):
// зелёный — только у завершённого анализа без нарушений с подтверждённой полнотой; проценты — только от XHR/сервера.

type Completeness = "complete" | "partial" | "unknown";
/** Поля A15 (completeness, mock). Со старым batch.ts их нет: полнота считается неподтверждённой, без зелёного. */
type Item = BatchItem & { completeness?: Completeness; mock?: boolean; partial?: boolean };

type Phase = "pending" | "active" | "done" | "error";
type Tone = "pending" | "active" | "success" | "danger" | "warning" | "neutral";
type Outcome = "violation" | "clean" | "unknown" | "partial" | "noVerdict" | "mock";
type Icon = ComponentType<{ className?: string }>;

interface RowView {
  phase: Phase;
  tone: Tone;
  outcome?: Outcome;
  Icon?: Icon;
  label: string;
  /** Число рядом с подписью: только реальный прогресс. */
  value?: string;
  detail?: string;
  progress?: number;
  action?: { href: string; label: string };
}

const TONE_CLASS: Record<Tone, string> = {
  pending: s.tonePending,
  active: s.toneActive,
  success: s.toneSuccess,
  danger: s.toneDanger,
  warning: s.toneWarning,
  neutral: s.toneNeutral,
};
const ICON_CLASS: Record<Tone, string> = {
  pending: "text-slate-500 dark:text-slate-400",
  active: "text-brand-600 dark:text-brand-300",
  success: "text-emerald-600 dark:text-emerald-400",
  danger: "text-rose-600 dark:text-rose-400",
  warning: "text-amber-600 dark:text-amber-400",
  neutral: "text-slate-500 dark:text-slate-400",
};
/** Текст подписи: акцент только там, где нужно внимание проверяющего. */
const LABEL_CLASS: Record<Tone, string> = {
  pending: "text-slate-600 dark:text-slate-300",
  active: "text-slate-900 dark:text-white",
  success: "text-slate-900 dark:text-white",
  danger: "text-rose-700 dark:text-rose-300",
  warning: "text-amber-800 dark:text-amber-300",
  neutral: "text-slate-800 dark:text-slate-200",
};

function completenessOf(it: Item): Completeness {
  return it.completeness ?? (it.partial ? "partial" : "unknown");
}

function episodes(it: Item): string | undefined {
  const tc = (it.violations ?? []).map((v) => v.timecode_sec).filter((t) => Number.isFinite(t));
  if (tc.length === 0) return undefined;
  const shown = tc.slice(0, 3).map(formatTimecode).join(" · ");
  return tc.length > 3 ? `${shown} · ещё ${tc.length - 3}` : shown;
}

function rowView(it: Item, queuePos: number): RowView {
  const href = it.jobId ? `#/job/${encodeURIComponent(it.jobId)}` : undefined;
  const open = (label: string) => (href ? { href, label } : undefined);
  switch (it.state) {
    case "waiting":
      return { phase: "pending", tone: "pending", Icon: Clock3, label: "В очереди", detail: `ждёт загрузки · № ${queuePos}` };
    case "uploading":
      return {
        phase: "active",
        tone: "active",
        Icon: UploadCloud,
        label: "Загрузка",
        value: `${Math.round(it.upload * 100)} %`,
        progress: it.upload,
        detail: it.upload >= 1 ? "файл передан, создаётся задача" : undefined,
      };
    case "processing": {
      const p = typeof it.progress === "number" && it.progress > 0 ? Math.min(1, it.progress) : undefined;
      return {
        phase: "active",
        tone: "active",
        label: "Анализ",
        value: p !== undefined ? `${Math.round(p * 100)} %` : undefined,
        progress: p,
        detail: it.stage ? stageLabel(it.stage) : "идёт на сервере",
        action: open("Ход анализа"),
      };
    }
    case "failed":
      return {
        phase: "error",
        tone: "warning",
        Icon: XOctagon,
        label: "Анализ не завершён",
        detail: it.error || "сервер сообщил об ошибке обработки",
        action: open("Подробности"),
      };
    case "error":
      return {
        phase: "error",
        tone: "warning",
        Icon: AlertTriangle,
        label: it.jobId ? "Статус недоступен" : "Файл не загружен",
        detail: it.error || "ошибка",
        action: open("Открыть задачу"),
      };
    default:
      break;
  }
  const c = completenessOf(it);
  const action = open("Открыть отчёт");
  const done = { phase: "done" as const, action };
  if (it.mock)
    return { ...done, tone: "neutral", outcome: "mock", Icon: FlaskConical, label: "Тестовый результат (mock)", detail: "модели не запускались — это не анализ видео" };
  if (it.verdict === "violation") {
    const note = c === "partial" ? "анализ неполный" : c === "unknown" ? "полнота не подтверждена" : undefined;
    return {
      ...done,
      tone: "danger",
      outcome: "violation",
      Icon: OctagonAlert,
      label: `Признаки нарушений: ${it.violations?.length ?? 0}`,
      detail: [episodes(it), note].filter(Boolean).join(" · ") || undefined,
    };
  }
  if (it.verdict === "no_violation") {
    if (c === "complete")
      return { ...done, tone: "success", outcome: "clean", Icon: CheckCircle2, label: "Нарушение не обнаружено", detail: "анализ полный" };
    if (c === "partial")
      return { ...done, tone: "warning", outcome: "partial", Icon: AlertTriangle, label: "Анализ неполный", detail: "в проверенной части нарушений не найдено" };
    return {
      ...done,
      tone: "neutral",
      outcome: "unknown",
      Icon: CircleDashed,
      label: "Не обнаружено · полнота не подтверждена",
      detail: "уверенная отметка не ставится",
    };
  }
  return { ...done, tone: "warning", outcome: "noVerdict", Icon: HelpCircle, label: "Вердикт не получен", detail: "нужен ручной просмотр" };
}

const OUTCOMES: { key: Outcome; label: string; Icon: Icon; tone: Tone }[] = [
  { key: "violation", label: "Признаки нарушений", Icon: OctagonAlert, tone: "danger" },
  { key: "partial", label: "Анализ неполный", Icon: AlertTriangle, tone: "warning" },
  { key: "noVerdict", label: "Вердикт не получен", Icon: HelpCircle, tone: "warning" },
  { key: "unknown", label: "Полнота не подтверждена", Icon: CircleDashed, tone: "neutral" },
  { key: "mock", label: "Тестовые (mock)", Icon: FlaskConical, tone: "neutral" },
  { key: "clean", label: "Без нарушений", Icon: CheckCircle2, tone: "success" },
];

/** Середина длинного имени сокращается: начало и хвост (номер части, расширение) остаются видны. */
function FileName({ name }: { name: string }) {
  const dot = name.lastIndexOf(".");
  const ext = dot > 0 && name.length - dot <= 6 ? name.length - dot : 0;
  const tailLen = name.length > 28 ? Math.min(name.length - ext - 8, 10) + ext : 0;
  const head = tailLen > ext ? name.slice(0, name.length - tailLen) : name;
  const tail = tailLen > ext ? name.slice(name.length - tailLen) : "";
  return (
    <span className={cx(s.name, "text-sm font-medium text-slate-900 dark:text-slate-100")} title={name}>
      <span className={s.nameHead}>{head}</span>
      {tail && <span className={s.nameTail}>{tail}</span>}
    </span>
  );
}

function StatusIcon({ v }: { v: RowView }) {
  if (!v.Icon)
    return (
      <span className="relative flex h-4 w-4 shrink-0 items-center justify-center" aria-hidden>
        <span className="absolute h-3 w-3 animate-ping rounded-full bg-brand-400/60" />
        <span className="relative h-2 w-2 rounded-full bg-brand-500 dark:bg-brand-300" />
      </span>
    );
  const I = v.Icon;
  return <I className={cx("h-4 w-4 shrink-0", ICON_CLASS[v.tone])} aria-hidden />;
}

function Row({ it, v }: { it: Item; v: RowView }) {
  return (
    <li
      className={cx(
        s.row,
        TONE_CLASS[v.tone],
        "animate-fade-in",
        v.phase === "error" ? "bg-amber-50/70 dark:bg-amber-400/[.05]" : "hover:bg-slate-50/80 dark:hover:bg-white/[.02]",
      )}
    >
      <div className={s.file}>
        <FileName name={it.name} />
        <div className="mt-0.5 text-xs text-slate-500 tnum dark:text-slate-400">
          {formatBytes(it.size)}
          {it.jobId && (
            <>
              {" · задача "}
              <span className="font-mono" title={it.jobId}>
                {it.jobId.length > 8 ? `${it.jobId.slice(0, 8)}…` : it.jobId}
              </span>
            </>
          )}
        </div>
      </div>
      <div className={s.status}>
        <div className="flex items-center gap-2">
          <StatusIcon v={v} />
          <span className={cx("min-w-0 text-sm font-semibold leading-5", LABEL_CLASS[v.tone])}>{v.label}</span>
          {v.value && <span className="ml-auto pl-2 text-xs font-medium text-slate-500 tnum dark:text-slate-400">{v.value}</span>}
        </div>
        {v.progress !== undefined && <ProgressBar value={v.progress} className="mt-1.5" />}
        {v.detail && (
          <div className={cx(s.detail, "mt-0.5 pl-6 text-xs leading-5 text-slate-500 dark:text-slate-400")}>
            {v.detail}
          </div>
        )}
      </div>
      <div className={s.action}>
        {v.action && (
          <a
            className={cx(s.actionBtn, s.rowAction, "btn-ghost gap-1 px-3 py-1.5 text-[13px]")}
            href={v.action.href}
            aria-label={`${v.action.label}: ${it.name}`}
          >
            <span className={s.actionText}>{v.action.label}</span>
            <ChevronRight className="h-4 w-4 shrink-0 text-slate-400 sm:-mr-1" aria-hidden />
          </a>
        )}
      </div>
    </li>
  );
}

/** Пакетная проверка: рабочая очередь файлов — загрузка по очереди, статусы, переходы к задачам. */
export default function BatchList() {
  const items = useSyncExternalStore(batch.subscribe, batch.getSnapshot, batch.getSnapshot) as Item[];
  if (items.length === 0) return null;

  let waiting = 0;
  const views = items.map((it) => rowView(it, it.state === "waiting" ? ++waiting : 0));
  const phase = (p: Phase) => views.filter((v) => v.phase === p).length;
  const n = { done: phase("done"), error: phase("error"), active: phase("active"), pending: phase("pending") };
  const finished = n.done + n.error;
  const outcomes = OUTCOMES.map((o) => ({ ...o, count: views.filter((v) => v.outcome === o.key).length })).filter((o) => o.count > 0);
  const legend = [
    { key: "done", label: "Готово", count: n.done, cls: s.done },
    { key: "error", label: "Ошибки", count: n.error, cls: s.error },
    { key: "active", label: "В работе", count: n.active, cls: s.active },
    { key: "pending", label: "В очереди", count: n.pending, cls: s.pending },
  ];

  const exportSummary = () => {
    const blob = new Blob([JSON.stringify(batch.summary(), null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `kassa-sentinel_batch_${new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-")}.json`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 2000);
  };

  return (
    <section className="card relative mt-6 overflow-hidden" aria-labelledby="batch-title">
      <div className="bg-grid pointer-events-none absolute inset-x-0 top-0 h-36 [mask-image:radial-gradient(ellipse_at_top_right,black,transparent_70%)]" />
      <div className="relative px-4 pb-4 pt-4 sm:px-6 sm:pt-5">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div className="flex min-w-0 items-center gap-3">
            <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-white text-brand-600 shadow-card ring-1 ring-slate-200 dark:bg-ink-800 dark:text-brand-300 dark:ring-white/10">
              <Layers className="h-[18px] w-[18px]" aria-hidden />
            </span>
            <div className="min-w-0">
              <h2 id="batch-title" className="text-base font-bold text-slate-900 dark:text-white">
                Пакетная проверка
              </h2>
              <p className="text-xs text-slate-500 tnum dark:text-slate-400">
                {items.length} {plural(items.length, ["файл", "файла", "файлов"])} · завершено {finished} из {items.length}
              </p>
            </div>
          </div>
          <div className="flex w-full flex-wrap gap-2 sm:w-auto">
            <button className={cx(s.actionBtn, "btn-ghost flex-auto px-3 py-1.5 text-[13px]")} onClick={exportSummary} disabled={finished === 0}>
              <Download className="h-4 w-4" aria-hidden /> Сводка JSON
            </button>
            <button
              className={cx(s.actionBtn, "btn-ghost flex-auto px-3 py-1.5 text-[13px]")}
              onClick={() => batch.clearFinished()}
              disabled={finished === 0}
              title="Убрать из списка строки с результатом или ошибкой. Идущие загрузки и анализы останутся; задачи и отчёты на сервере не удаляются."
            >
              <Eraser className="h-4 w-4" aria-hidden /> Убрать завершённые
            </button>
          </div>
        </div>

        <div className={cx(s.meter, "mt-4")} aria-hidden>
          {legend
            .filter((x) => x.count > 0)
            .map((x) => (
              <span key={x.key} className={cx(s.seg, x.cls)} style={{ flexGrow: x.count }} />
            ))}
        </div>
        <ul className="mt-2.5 flex flex-wrap gap-x-4 gap-y-1.5 text-xs text-slate-600 dark:text-slate-300" aria-label="Состояние очереди">
          {legend.map((x) => (
            <li key={x.key} className="flex items-center gap-1.5">
              <span className={cx(s.swatch, x.cls)} aria-hidden />
              {x.label}
              <span className="font-semibold text-slate-900 tnum dark:text-white">{x.count}</span>
            </li>
          ))}
        </ul>
        {outcomes.length > 0 && (
          <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1.5 text-xs text-slate-600 dark:text-slate-300">
            <span className="label">Итоги готовых</span>
            {outcomes.map((o) => (
              <span key={o.key} className="flex items-center gap-1.5">
                <o.Icon className={cx("h-3.5 w-3.5", ICON_CLASS[o.tone])} aria-hidden />
                {o.label}
                <span className="font-semibold text-slate-900 tnum dark:text-white">{o.count}</span>
              </span>
            ))}
          </div>
        )}
        <p className="sr-only" aria-live="polite" aria-atomic="true">
          {`Пакет: завершено ${finished} из ${items.length}; ошибок ${n.error}; в работе ${n.active}; в очереди ${n.pending}.`}
        </p>
      </div>

      <div className={cx(s.head, "label border-t border-slate-100 dark:border-white/5")} aria-hidden>
        <span>Файл</span>
        <span>Статус</span>
        <span className="text-right">Действие</span>
      </div>
      <ul className="divide-y divide-slate-100 border-t border-slate-100 dark:divide-white/5 dark:border-white/5" aria-label="Файлы пакета">
        {items.map((it, i) => (
          <Row key={it.key} it={it} v={views[i]} />
        ))}
      </ul>
      <p className="border-t border-slate-100 px-4 py-3 text-xs leading-relaxed text-slate-500 dark:border-white/5 dark:text-slate-400 sm:px-6">
        Файлы загружаются по очереди, анализ идёт на сервере. Готовые отчёты можно открывать — пакет продолжит работу.
        «Убрать завершённые» только скрывает строки здесь: задачи и отчёты на сервере остаются.
      </p>
    </section>
  );
}
