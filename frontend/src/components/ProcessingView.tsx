import { useEffect, useRef, useState } from "react";
import { Check, Cpu, Loader2 } from "lucide-react";
import type { JobResult } from "../api/types";
import { formatDuration } from "../lib/format";
import { stageLabel, STEPS, stepIndex } from "../lib/stages";
import { cx, ProgressBar } from "./ui";

/**
 * Экран обработки. Прогресс — только из реального источника (поле progress сервера); без него —
 * неопределённый индикатор. Оставшееся время НЕ прогнозируется: показываем только фактическое время —
 * сколько прошло и сколько длились уже пройденные стадии (по наблюдаемым сменам stage).
 */
export default function ProcessingView({ job, fileName }: { job: JobResult; fileName?: string }) {
  const [now, setNow] = useState(() => Date.now());
  const watchStart = useRef(Date.now());
  const stageLog = useRef<{ stage: string; at: number; end?: number }[]>([]);

  useEffect(() => {
    const id = window.setInterval(() => setNow(Date.now()), 500);
    return () => window.clearInterval(id);
  }, []);

  const hasProgress = typeof job.progress === "number" && job.status === "processing";
  const p = hasProgress ? job.progress! : 0;

  // Журнал смен стадий (реальные наблюдения при опросе, точность ≈ интервал опроса).
  const cur = job.stage ?? "";
  const log = stageLog.current;
  if (cur && (log.length === 0 || log[log.length - 1].stage !== cur)) {
    const t = Date.now();
    if (log.length) log[log.length - 1].end = t;
    log.push({ stage: cur, at: t });
  }

  const idx = job.status === "queued" ? 0 : stepIndex(job.stage, hasProgress ? p : undefined);
  const known = !!job.stage || hasProgress || job.status === "queued";
  // «Прошло» — по серверу (elapsed_s, A02) + время с последнего ответа; иначе — с открытия страницы.
  const lastServer = useRef<{ at: number; el: number } | null>(null);
  if (typeof job.elapsedSec === "number" && (lastServer.current === null || lastServer.current.el !== job.elapsedSec))
    lastServer.current = { at: Date.now(), el: job.elapsedSec };
  const elapsed = lastServer.current
    ? lastServer.current.el + (now - lastServer.current.at) / 1000
    : (now - watchStart.current) / 1000;
  const queued = job.status === "queued" || (job.stage ?? "").toLowerCase().startsWith("queue");

  return (
    <div className="mx-auto max-w-3xl animate-fade-in">
      <div className="card overflow-hidden">
        <div className="relative overflow-hidden bg-ink-900 px-6 py-8 text-white sm:px-8">
          <div className="bg-grid absolute inset-0 opacity-40" />
          <div className="absolute inset-x-0 top-0 h-24 animate-scan bg-gradient-to-b from-transparent via-brand-400/25 to-transparent" />
          <div className="relative flex items-center gap-4">
            <div className="relative flex h-14 w-14 items-center justify-center">
              <span className="absolute inset-0 animate-pulse-ring rounded-full bg-brand-500/40" />
              <span className="relative flex h-12 w-12 items-center justify-center rounded-full bg-brand-600">
                <Cpu className="h-6 w-6" />
              </span>
            </div>
            <div className="min-w-0">
              <div className="text-xs font-semibold uppercase tracking-[.12em] text-white/60">
                {queued ? "Задача в очереди" : "Идёт анализ записи"}
              </div>
              <h1 className="truncate text-2xl font-extrabold">{fileName ?? "Видеозапись"}</h1>
              <div className="mt-1 font-mono text-xs text-white/50">задача {job.job_id}</div>
            </div>
          </div>
        </div>

        <div className="p-6 sm:p-8">
          <div className="mb-6 space-y-2">
            <div className="flex items-baseline justify-between gap-3 text-sm">
              <span className="font-semibold">
                {queued
                  ? `Ожидание свободного обработчика${job.queuePosition ? ` · позиция в очереди: ${job.queuePosition}` : ""}`
                  : job.stage
                    ? `Стадия: ${stageLabel(job.stage)}`
                    : "Обработка на сервере"}
              </span>
              <span className="text-slate-500 tnum">
                {hasProgress ? `${Math.round(p * 100)} %` : "прогресс не передаётся"}
              </span>
            </div>
            {hasProgress ? (
              <ProgressBar value={p} />
            ) : (
              <div className="h-2 overflow-hidden rounded-full bg-slate-100 dark:bg-white/[.06]">
                <div className="h-full w-full animate-pulse rounded-full bg-brand-500/30" />
              </div>
            )}
            <div className="flex justify-between text-xs text-slate-500 tnum">
              <span>прошло {formatDuration(elapsed)}</span>
              <span>{hasProgress ? "доля работы по стадиям; оставшееся время не прогнозируется" : ""}</span>
            </div>
          </div>

          <ol className="relative space-y-1">
            {STEPS.map((s, i) => {
              const state = !known ? (i === 0 ? "done" : i === 1 ? "active" : "pending") : i < idx ? "done" : i === idx ? "active" : "pending";
              return (
                <li key={s.id} className="relative flex gap-4 pb-4 last:pb-0">
                  {i < STEPS.length - 1 && (
                    <span className={cx("absolute left-[15px] top-8 h-[calc(100%-24px)] w-0.5", state === "done" ? "bg-emerald-500" : "bg-slate-200 dark:bg-white/10")} />
                  )}
                  <span
                    className={cx(
                      "relative z-10 flex h-8 w-8 shrink-0 items-center justify-center rounded-full text-xs font-bold",
                      state === "done" && "bg-emerald-500 text-white",
                      state === "active" && "bg-brand-600 text-white ring-4 ring-brand-500/20",
                      state === "pending" && "bg-slate-100 text-slate-400 dark:bg-white/5",
                    )}
                  >
                    {state === "done" ? <Check className="h-4 w-4" /> : state === "active" ? <Loader2 className="h-4 w-4 animate-spin" /> : i + 1}
                  </span>
                  <div className="pt-1">
                    <div className={cx("text-sm font-semibold", state === "pending" && "text-slate-400")}>{s.title}</div>
                    <div className="text-xs text-slate-500">{s.hint}</div>
                  </div>
                </li>
              );
            })}
          </ol>
          {log.filter((x) => x.end).length > 0 && (
            <div className="mt-5">
              <div className="label mb-1.5">Пройденные стадии (наблюдение)</div>
              <ul className="space-y-1 text-xs text-slate-600 dark:text-slate-300">
                {log
                  .filter((x) => x.end)
                  .map((x, i) => (
                    <li key={`${x.stage}-${i}`} className="flex justify-between gap-3">
                      <span className="truncate">{stageLabel(x.stage)}</span>
                      <span className="font-mono tnum text-slate-500">≈ {formatDuration((x.end! - x.at) / 1000)}</span>
                    </li>
                  ))}
              </ul>
            </div>
          )}
          <p className="mt-6 rounded-xl bg-slate-50 p-3 text-xs leading-relaxed text-slate-500 dark:bg-white/5">
            Видео обрабатывается на локальном сервере. Страницу можно закрыть — задача сохранится в «Недавних проверках».
          </p>
        </div>
      </div>
    </div>
  );
}
