import { CheckCircle2, CircleDashed, CircleSlash, Gauge, XCircle } from "lucide-react";
import type { CompletenessReport } from "../lib/completeness";
import { cx } from "./ui";

/** «Полнота анализа»: что реально проверено — покрытие записи, статусы и время стадий (только реальные данные). */
export default function CompletenessPanel({ report }: { report: CompletenessReport }) {
  const head = {
    complete: { text: "Анализ полный", cls: "text-emerald-700 dark:text-emerald-300" },
    partial: { text: "Анализ неполный", cls: "text-amber-700 dark:text-amber-300" },
    unknown: { text: "Полнота не подтверждена сервером", cls: "text-slate-600 dark:text-slate-300" },
  }[report.state];
  return (
    <section className="card p-4" aria-label="Полнота анализа">
      <div className="label mb-1 flex items-center gap-1.5">
        <Gauge className="h-3.5 w-3.5" /> Полнота анализа
      </div>
      <div className={cx("text-sm font-semibold", head.cls)}>{head.text}</div>
      {report.coverage !== undefined && (
        <div className="mt-2">
          <div className="mb-1 flex justify-between text-xs text-slate-500 tnum">
            <span>проанализировано записи</span>
            <span>{Math.floor(report.coverage * 100)} %</span>
          </div>
          <div className="h-1.5 overflow-hidden rounded-full bg-slate-100 dark:bg-white/[.06]">
            <div
              className={cx("h-full rounded-full", report.coverage >= 0.98 ? "bg-emerald-500" : "bg-amber-500")}
              style={{ width: `${Math.max(2, report.coverage * 100)}%` }}
            />
          </div>
        </div>
      )}
      {report.reasons.length > 0 && (
        <ul className="mt-2 space-y-1 text-xs text-slate-600 dark:text-slate-300">
          {report.reasons.map((r) => (
            <li key={r} className="flex gap-1.5">
              <span className="mt-1.5 h-1 w-1 shrink-0 rounded-full bg-amber-500" />
              {r}
            </li>
          ))}
        </ul>
      )}
      {report.stages.length > 0 && (
        <ul className="mt-3 divide-y divide-slate-100 text-xs dark:divide-white/5">
          {report.stages.map((s) => {
            const Icon = s.ok ? CheckCircle2 : s.status === "skipped" || s.status === "disabled" ? CircleSlash : s.status === "pending" ? CircleDashed : XCircle;
            return (
              <li key={s.name} className="flex items-center gap-2 py-1.5" title={s.error}>
                <Icon
                  className={cx(
                    "h-3.5 w-3.5 shrink-0",
                    s.ok ? "text-emerald-600 dark:text-emerald-400" : s.required ? "text-rose-600 dark:text-rose-400" : "text-slate-400",
                  )}
                />
                <span className="min-w-0 flex-1 truncate">
                  {s.label}
                  {s.required && <span className="text-slate-400"> · обяз.</span>}
                </span>
                <span className={cx("shrink-0", s.ok ? "text-slate-500" : "font-semibold text-amber-700 dark:text-amber-300")}>{s.statusLabel}</span>
                {s.seconds !== undefined && s.ok && <span className="w-12 shrink-0 text-right font-mono text-slate-500 tnum">{s.seconds.toFixed(1)} с</span>}
              </li>
            );
          })}
        </ul>
      )}
      {report.cached.length > 0 && (
        <p className="mt-2 text-[11px] text-slate-500">Из кэша (тот же файл, повторный запуск): {report.cached.join(", ")}.</p>
      )}
      <p className="mt-2 text-[11px] leading-relaxed text-slate-500">
        Время — фактическое время стадий на сервере. Неизвестное не считается отсутствием нарушения.
      </p>
    </section>
  );
}
