import { useSyncExternalStore } from "react";
import { AlertTriangle, CheckCircle2, Download, Layers, Loader2, OctagonAlert, Upload } from "lucide-react";
import { batch, type BatchItem } from "../lib/batch";
import { formatBytes } from "../lib/format";
import { stageLabel } from "../lib/stages";
import { cx, ProgressBar } from "./ui";

function StatusCell({ it }: { it: BatchItem }) {
  if (it.state === "waiting") return <span className="text-xs text-slate-500">в очереди на загрузку</span>;
  if (it.state === "uploading")
    return (
      <div className="w-full">
        <div className="mb-1 flex justify-between text-xs text-slate-500 tnum">
          <span>загрузка</span>
          <span>{Math.round(it.upload * 100)} %</span>
        </div>
        <ProgressBar value={it.upload} />
      </div>
    );
  if (it.state === "processing")
    return (
      <span className="inline-flex items-center gap-1.5 text-xs text-slate-600 dark:text-slate-300">
        <Loader2 className="h-3.5 w-3.5 animate-spin" />
        {it.stage ? stageLabel(it.stage) : "анализ"}
        {typeof it.progress === "number" && it.progress > 0 && <span className="tnum opacity-70">· {Math.round(it.progress * 100)} %</span>}
      </span>
    );
  if (it.state === "error" || it.state === "failed")
    return (
      <span className="inline-flex items-center gap-1.5 text-xs text-amber-700 dark:text-amber-300">
        <AlertTriangle className="h-3.5 w-3.5 shrink-0" />
        {it.state === "failed" ? "анализ не завершён" : it.error ?? "ошибка"}
      </span>
    );
  if (it.mock)
    return (
      <span className="inline-flex items-center gap-1.5 text-xs text-slate-600 dark:text-slate-300">
        <AlertTriangle className="h-3.5 w-3.5" /> тестовый результат (mock)
      </span>
    );
  if (it.verdict === "violation")
    return (
      <span className="inline-flex items-center gap-1.5 text-xs font-semibold text-rose-700 dark:text-rose-300">
        <OctagonAlert className="h-3.5 w-3.5" /> признаки нарушений: {it.violations?.length ?? 0}
      </span>
    );
  if (it.verdict === "no_violation" && it.completeness === "unknown")
    return (
      <span className="inline-flex items-center gap-1.5 text-xs text-slate-600 dark:text-slate-300">
        <CheckCircle2 className="h-3.5 w-3.5" /> не обнаружено · полнота не подтверждена
      </span>
    );
  if (it.verdict === "no_violation" && it.completeness === "complete")
    return (
      <span className="inline-flex items-center gap-1.5 text-xs font-semibold text-emerald-700 dark:text-emerald-300">
        <CheckCircle2 className="h-3.5 w-3.5" /> нарушение не обнаружено
      </span>
    );
  return (
    <span className="inline-flex items-center gap-1.5 text-xs text-amber-700 dark:text-amber-300">
      <AlertTriangle className="h-3.5 w-3.5" /> анализ неполный
    </span>
  );
}

/** Таблица пакетной проверки (несколько файлов): загрузка по очереди, статусы, ссылки на отчёты. */
export default function BatchList() {
  const items = useSyncExternalStore(batch.subscribe, batch.getSnapshot, batch.getSnapshot);
  if (items.length === 0) return null;
  const done = items.filter((i) => i.state === "done" || i.state === "failed" || i.state === "error").length;

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
    <section className="card mt-6 p-5">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
        <h2 className="flex items-center gap-2 text-base font-bold">
          <Layers className="h-4 w-4 text-slate-400" /> Пакетная проверка
          <span className="text-sm font-medium text-slate-500 tnum">
            {done} из {items.length}
          </span>
        </h2>
        <div className="flex gap-2">
          <button className="btn-ghost px-3 py-1.5 text-xs" onClick={exportSummary} disabled={done === 0}>
            <Download className="h-3.5 w-3.5" /> Сводка JSON
          </button>
          <button className="btn-ghost px-3 py-1.5 text-xs" onClick={() => batch.clearFinished()} disabled={done === 0}>
            Убрать завершённые
          </button>
        </div>
      </div>
      <ul className="divide-y divide-slate-100 dark:divide-white/5">
        {items.map((it) => (
          <li key={it.key} className="grid grid-cols-1 items-center gap-2 py-3 sm:grid-cols-[minmax(0,1fr)_minmax(0,1fr)_auto] sm:gap-4">
            <div className="min-w-0">
              <div className="truncate text-sm font-medium">{it.name}</div>
              <div className="text-xs text-slate-500 tnum">{formatBytes(it.size)}</div>
            </div>
            <div className="min-w-0">
              <StatusCell it={it} />
            </div>
            <div className="sm:text-right">
              {it.jobId ? (
                <a
                  className={cx("btn-ghost px-3 py-1.5 text-xs", it.state === "processing" && "opacity-80")}
                  href={`#/job/${encodeURIComponent(it.jobId)}`}
                >
                  {it.state === "processing" ? "Ход анализа" : "Открыть отчёт"}
                </a>
              ) : (
                <span className="inline-flex items-center gap-1 text-xs text-slate-400">
                  <Upload className="h-3.5 w-3.5" /> —
                </span>
              )}
            </div>
          </li>
        ))}
      </ul>
      <p className="mt-3 text-xs text-slate-500">
        Файлы загружаются по очереди; анализ идёт на сервере. Можно открывать готовые отчёты — пакет продолжит работу.
      </p>
    </section>
  );
}
