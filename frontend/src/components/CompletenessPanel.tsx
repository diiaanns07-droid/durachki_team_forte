import { useId, type ReactNode } from "react";
import {
  AlertTriangle,
  CheckCircle2,
  ChevronRight,
  CircleDashed,
  CircleHelp,
  CircleSlash,
  FlaskConical,
  Gauge,
  XCircle,
} from "lucide-react";
import { COVERAGE_COMPLETE, type Completeness, type CompletenessReport, type StageInfo } from "../lib/completeness";
import { cx } from "./ui";
import s from "./u11.module.css";

/**
 * «Полнота анализа» (U11): что произошло → как это влияет на результат → какое действие доступно.
 * Только представление: состояние, покрытие, причины и стадии считает lib/completeness.ts — здесь не меняются.
 * «Нет найденных нарушений» и «недостаточно данных» различаются явно; неизвестное покрытие не рисуется полосой.
 */
export default function CompletenessPanel({
  report,
  violationsFound,
  failed,
}: {
  report: CompletenessReport;
  /** Сколько нарушений в вердикте (если передано) — для точной формулировки влияния. */
  violationsFound?: number;
  /** Задача завершилась ошибкой (если не передано — по причинам из отчёта). */
  failed?: boolean;
}) {
  const titleId = useId();
  const reasons = groupReasons(report.reasons);
  const isFailed = failed ?? report.reasons.includes(FAILED_REASON);
  const head = headline(report);
  const tech = reasons.filter((r) => r.tech).flatMap((r) => r.tech ?? []);
  const actions = actionsFor(report.state, isFailed, tech.length > 0);

  return (
    <section className={cx("card animate-fade-in p-4 pl-5", s.panel)} aria-labelledby={titleId}>
      <span aria-hidden className={cx(s.rail, RAIL[report.state])} />
      <span aria-hidden className={s.gridCorner} />

      <div className="relative">
        <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
          <div className="label flex items-center gap-1.5">
            <Gauge className="h-3.5 w-3.5" aria-hidden /> Полнота анализа
          </div>
          <StateChip state={report.state} />
        </div>

        {report.mock && (
          <div className="mb-3 flex gap-2 rounded-xl border border-amber-300 bg-amber-50 px-3 py-2 text-xs leading-relaxed text-amber-900 dark:border-amber-400/30 dark:bg-amber-400/10 dark:text-amber-200">
            <FlaskConical className="mt-0.5 h-3.5 w-3.5 shrink-0" aria-hidden />
            <span>
              <b>Тестовый результат: модели не запускались.</b> Для решения по записи он не подходит — нужна проверка
              реальной записи в рабочем режиме.
            </span>
          </div>
        )}

        <h3 id={titleId} className={cx("font-display text-base font-bold leading-snug", head.cls)}>
          {head.title}
        </h3>

        <Coverage report={report} />

        <Block label="Что произошло">
          {reasons.length === 0 ? (
            <p>{head.what}</p>
          ) : (
            <ul className="space-y-1">
              {reasons.map((r) => (
                <li key={r.text} className={cx("flex gap-2", s.wrapAnywhere)}>
                  <span aria-hidden className="mt-[7px] h-1 w-1 shrink-0 rounded-full bg-amber-500" />
                  <span>
                    {r.text}
                    {r.count > 1 && <span className="text-slate-500 tnum"> ×{r.count}</span>}
                  </span>
                </li>
              ))}
            </ul>
          )}
        </Block>

        <Block label="Как это влияет на результат">
          <p>{impact(report.state, violationsFound, report.mock)}</p>
        </Block>

        {actions.length > 0 && (
          <Block label="Что можно сделать">
            <ul className="space-y-1">
              {actions.map((a) => (
                <li key={a} className="flex gap-2">
                  <ChevronRight aria-hidden className="mt-[3px] h-3.5 w-3.5 shrink-0 text-brand-600 dark:text-brand-300" />
                  <span>{a}</span>
                </li>
              ))}
            </ul>
          </Block>
        )}

        <Stages report={report} openByDefault={report.state !== "complete" && hasRequiredFailure(report.stages)} />

        {tech.length > 0 && (
          <details className={cx("mt-3", s.details)}>
            <summary className="flex items-center gap-1.5 py-1 text-xs font-semibold text-slate-600 hover:text-slate-900 dark:text-slate-300 dark:hover:text-white">
              <ChevronRight aria-hidden className={cx("h-3.5 w-3.5", s.chevron)} />
              Технические подробности (для администратора)
            </summary>
            <ul className="mt-1.5 space-y-1.5">
              {[...new Set(tech)].map((t) => (
                <li key={t}>
                  <code className={cx("block rounded-lg bg-slate-50 px-2 py-1.5 font-mono text-[11px] text-slate-700 dark:bg-white/[.04] dark:text-slate-300", s.errorText)}>
                    {t}
                  </code>
                </li>
              ))}
            </ul>
          </details>
        )}

        <p className="mt-3 border-t border-slate-100 pt-2 text-[11px] leading-relaxed text-slate-500 dark:border-white/5 dark:text-slate-400">
          Время — фактическое время стадий на сервере. Неизвестное не считается отсутствием нарушения.
        </p>
      </div>
    </section>
  );
}

// ---------------------------------------------------------------- представление (без вычисления полноты)

/** Ровно та строка, которую lib/completeness.ts добавляет при status === "failed". */
const FAILED_REASON = "анализ завершился ошибкой";

const RAIL: Record<Completeness, string> = { complete: s.railComplete, partial: s.railPartial, unknown: s.railUnknown };

function pct(x: number): number {
  return Math.floor(x * 100); // как в lib/completeness: округление вниз, 99,6 % не превращается в «100 %»
}

function headline(r: CompletenessReport): { title: string; what: string; cls: string } {
  if (r.state === "complete")
    return {
      title: "Запись проверена целиком",
      what:
        r.coverage !== undefined
          ? `Проверено ${pct(r.coverage)} % длительности записи, обязательные стадии выполнены.`
          : "Сервер подтвердил полноту анализа (долю проверенной длительности он не сообщил).",
      cls: "text-emerald-700 dark:text-emerald-300",
    };
  if (r.state === "partial")
    return {
      title: r.coverage !== undefined && r.coverage < COVERAGE_COMPLETE ? "Проверена только часть записи" : "Анализ неполный",
      what: "Часть записи или обязательных стадий не проверена.",
      cls: "text-amber-700 dark:text-amber-300",
    };
  return {
    title: "Полнота не подтверждена",
    what:
      r.coverage !== undefined
        ? `Сервер сообщил долю ${pct(r.coverage)} %, но не подтвердил, что анализ полный.`
        : "Сервер не сообщил, какая часть записи проверена.",
    cls: "text-slate-800 dark:text-slate-100",
  };
}

function impact(state: Completeness, found: number | undefined, mock: boolean): string {
  if (mock) return "Модели не запускались, поэтому ни найденные эпизоды, ни их отсутствие ничего не говорят о записи.";
  if (state === "complete") {
    if (found === 0) return "Нарушений не найдено — это результат проверки всей записи, а не нехватка данных.";
    return "Вердикт относится ко всей записи: и найденные эпизоды, и отсутствие нарушений на остальных участках.";
  }
  if (state === "partial") {
    if (found === 0)
      return "Нарушений не найдено только в проверенной части. Это недостаток данных, а не подтверждение, что нарушений нет.";
    if (found && found > 0) return "Найденные эпизоды — из проверенной части; на непроверенных участках могут быть и другие.";
    return "Результат относится только к проверенной части. Отсутствие найденных нарушений не доказывает, что их нет.";
  }
  if (found === 0) return "Нарушений не найдено, но данных недостаточно: неизвестно, проверена ли запись целиком.";
  if (found && found > 0) return "Найденные эпизоды в силе, но неизвестно, проверена ли остальная часть записи.";
  return "Неизвестно, проверена ли запись целиком, поэтому отсутствие найденных нарушений не означает, что их нет.";
}

/** Только действия, которые уже есть в интерфейсе/API. Повторную отправку предлагаем лишь после ошибки:
 *  завершённую задачу с тем же файлом сервер вернёт ту же (дедупликация A02 по sha256). */
function actionsFor(state: Completeness, failed: boolean, hasTech: boolean): string[] {
  if (state === "complete") return [];
  const out = ["Просмотрите запись вручную в плеере отчёта — особенно участки вне найденных эпизодов."];
  if (failed) out.push("Отправьте тот же файл повторно: будет создана новая задача (кнопка — в сообщении об ошибке).");
  if (hasTech) out.push("Если причина неясна — передайте администратору технические подробности ниже.");
  return out;
}

interface Reason {
  text: string;
  count: number;
  tech?: string[];
}

const TECH = /meta\.|complete=|=false|=true|by_stage|coverage_problem|\bnull\b/i;
const JARGON: [RegExp, string][] = [[/оркестратор/gi, "сервер анализа"]];

function plain(text: string): string {
  let t = text.trim();
  for (const [re, to] of JARGON) t = t.replace(re, to);
  return t.charAt(0).toUpperCase() + t.slice(1);
}

/** Основная формулировка — понятна проверяющему; технический хвост («(нет meta.complete …)», сырые
 *  предупреждения сервера) уходит в «Технические подробности». Одинаковые формулировки схлопываются. */
function groupReasons(raw: string[]): Reason[] {
  const out = new Map<string, Reason>();
  for (const r0 of raw) {
    const r = r0.trim();
    let text: string;
    let tech: string | undefined;
    const paren = r.match(/^(.*?)\s*\(([^()]*)\)\s*$/);
    if (/^предупреждение сервера:/i.test(r)) {
      text = "Сервер передал предупреждение о неполной обработке";
      tech = r;
    } else if (paren && TECH.test(paren[2])) {
      text = plain(paren[1]);
      tech = r;
    } else if (TECH.test(r)) {
      text = "Сервер сообщил техническую причину неполноты";
      tech = r;
    } else {
      text = plain(r);
    }
    const cur = out.get(text);
    if (cur) {
      cur.count += 1;
      if (tech) cur.tech = [...(cur.tech ?? []), tech];
    } else out.set(text, { text, count: 1, tech: tech ? [tech] : undefined });
  }
  return [...out.values()];
}

function hasRequiredFailure(stages: StageInfo[]): boolean {
  return stages.some((x) => x.required && !x.ok);
}

// ---------------------------------------------------------------- части карточки

function StateChip({ state }: { state: Completeness }) {
  const v = {
    complete: {
      text: "полный",
      icon: CheckCircle2,
      cls: "border-emerald-200 bg-emerald-50 text-emerald-800 dark:border-emerald-400/25 dark:bg-emerald-400/10 dark:text-emerald-200",
    },
    partial: {
      text: "неполный",
      icon: AlertTriangle,
      cls: "border-amber-300 bg-amber-50 text-amber-900 dark:border-amber-400/30 dark:bg-amber-400/10 dark:text-amber-200",
    },
    unknown: {
      text: "не подтверждён",
      icon: CircleHelp,
      cls: "border-dashed border-slate-300 bg-white text-slate-700 dark:border-white/20 dark:bg-transparent dark:text-slate-200",
    },
  }[state];
  const Icon = v.icon;
  return (
    <span className={cx("chip", v.cls)}>
      <Icon aria-hidden className="h-3.5 w-3.5" />
      <span>
        <span className="sr-only">Анализ: </span>
        {v.text}
      </span>
    </span>
  );
}

function Coverage({ report }: { report: CompletenessReport }) {
  const c = report.coverage;
  const known = c !== undefined;
  const value = known ? `${pct(c)} %` : report.state === "complete" ? "не сообщена" : "неизвестно";
  return (
    <div className="mt-3">
      <div className="mb-1 flex items-baseline justify-between gap-3 text-xs text-slate-600 dark:text-slate-300">
        <span>Проверено записи</span>
        <span className={cx("font-semibold tnum", known ? "text-slate-900 dark:text-white" : "text-slate-500 dark:text-slate-400")}>{value}</span>
      </div>
      {known ? (
        <div
          role="meter"
          aria-label="Проверенная доля длительности записи"
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={pct(c)}
          aria-valuetext={`${pct(c)} %`}
          className="h-1.5 overflow-hidden rounded-full bg-slate-100 dark:bg-white/[.06]"
        >
          <div
            className={cx("h-full rounded-full", c >= COVERAGE_COMPLETE ? "bg-emerald-500" : "bg-amber-500")}
            style={{ width: `${Math.max(2, c * 100)}%` }}
          />
        </div>
      ) : (
        <div className={s.trackUnknown} aria-hidden />
      )}
      {!known && (
        <p className="mt-1 text-[11px] leading-relaxed text-slate-500 dark:text-slate-400">
          {report.state === "complete"
            ? "Полноту подтвердил сервер; долю длительности он не прислал."
            : "Сервер не прислал долю проверенной длительности — это не «100 %»."}
        </p>
      )}
    </div>
  );
}

function Block({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="mt-3">
      <div className="mb-1 text-[11px] font-semibold uppercase tracking-[.08em] text-slate-500 dark:text-slate-400">{label}</div>
      <div className="text-sm leading-relaxed text-slate-700 dark:text-slate-200">{children}</div>
    </div>
  );
}

function Stages({ report, openByDefault }: { report: CompletenessReport; openByDefault: boolean }) {
  const st = report.stages;
  if (st.length === 0)
    return (
      <p className="mt-3 rounded-xl border border-dashed border-slate-300 px-3 py-2 text-xs leading-relaxed text-slate-600 dark:border-white/15 dark:text-slate-300">
        Сведений о стадиях анализа сервер не прислал — какие проверки выполнены, неизвестно.
      </p>
    );
  const required = st.filter((x) => x.required);
  const reqOk = required.filter((x) => x.ok).length;
  const failedReq = required.filter((x) => !x.ok);
  const summary = required.length
    ? `Обязательные стадии: ${reqOk} из ${required.length} выполнены`
    : `Стадии анализа: ${st.filter((x) => x.ok).length} из ${st.length} выполнены`;
  return (
    <details className={cx("mt-3", s.details)} open={openByDefault}>
      <summary className="flex items-start gap-1.5 py-1 text-sm font-semibold text-slate-800 hover:text-slate-950 dark:text-slate-100 dark:hover:text-white">
        <ChevronRight aria-hidden className={cx("mt-0.5 h-4 w-4 shrink-0", s.chevron)} />
        <span className="min-w-0">
          {summary}
          {failedReq.length > 0 && (
            <span className={cx("block text-xs font-normal text-rose-700 dark:text-rose-300", s.wrapAnywhere)}>
              не выполнены: {failedReq.map((x) => `${x.label} — ${x.statusLabel}`).join("; ")}
            </span>
          )}
        </span>
      </summary>
      <ul className="mt-1 divide-y divide-slate-100 text-xs dark:divide-white/5">
        {st.map((x) => (
          <StageRow key={x.name} stage={x} />
        ))}
      </ul>
      {report.cached.length > 0 && (
        <p className="mt-2 text-[11px] leading-relaxed text-slate-500 dark:text-slate-400">
          Из кэша (тот же файл уже обрабатывался): {report.cached.join(", ")}.
        </p>
      )}
    </details>
  );
}

function StageRow({ stage: x }: { stage: StageInfo }) {
  const Icon = x.ok ? CheckCircle2 : x.status === "skipped" || x.status === "disabled" ? CircleSlash : x.status === "pending" ? CircleDashed : XCircle;
  const cov =
    x.coverage === null
      ? "доля неизвестна"
      : typeof x.coverage === "number" && x.coverage < 1
        ? `просмотрено ${pct(x.coverage)} %`
        : undefined;
  return (
    <li className="py-2">
      <div className="flex items-start gap-2">
        <Icon
          aria-hidden
          className={cx(
            "mt-px h-3.5 w-3.5 shrink-0",
            x.ok ? "text-emerald-600 dark:text-emerald-400" : x.required ? "text-rose-600 dark:text-rose-400" : "text-slate-400",
          )}
        />
        <span className={cx("min-w-0 flex-1 text-slate-700 dark:text-slate-200", s.wrapAnywhere)}>
          {x.label}
          {x.required && <span className="text-slate-500 dark:text-slate-400"> · обязательная</span>}
          {cov && <span className="block text-[11px] text-amber-700 dark:text-amber-300">{cov}</span>}
        </span>
        <span className={cx("shrink-0 text-right", x.ok ? "text-slate-500 dark:text-slate-400" : "font-semibold text-amber-800 dark:text-amber-300")}>
          {x.statusLabel}
          {x.seconds !== undefined && x.ok && <span className="block font-mono text-[11px] font-normal text-slate-500 tnum">{x.seconds.toFixed(1)} с</span>}
        </span>
      </div>
      {x.error && (
        <details className={cx("ml-5 mt-1", s.details)}>
          <summary className="inline-flex items-center gap-1 text-[11px] font-medium text-slate-600 hover:text-slate-900 dark:text-slate-300 dark:hover:text-white">
            <ChevronRight aria-hidden className={cx("h-3 w-3", s.chevron)} /> Текст ошибки сервера
          </summary>
          <pre className={cx("mt-1 rounded-lg bg-slate-50 px-2 py-1.5 font-mono text-[11px] text-slate-700 dark:bg-white/[.04] dark:text-slate-300", s.errorText)}>
            {x.error}
          </pre>
        </details>
      )}
    </li>
  );
}
